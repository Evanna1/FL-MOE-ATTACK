from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class A3FLSearchConfig:
    """Configuration for the text adaptation of A3FL.

    A3FL optimizes a trigger against both the downloaded global model and an
    adversarial model trained to unlearn that trigger.  Since token IDs are
    discrete, candidate tokens are proposed with a GCG-style embedding
    gradient instead of the image-domain PGD step used by the original paper.
    """

    outer_steps: int = 3
    trigger_steps: int = 4
    candidate_batch_size: int = 32
    top_k: int = 32
    adversarial_steps: int = 1
    adversarial_lr: float = 2e-5
    lambda0: float = 0.01
    early_stop_asr: float = 0.95
    max_length: int = 256
    seed: int = 42


@dataclass(frozen=True)
class A3FLMetrics:
    loss: float
    target_probability: float
    attack_success_rate: float


@dataclass(frozen=True)
class A3FLSearchResult:
    trigger_token_ids: Tuple[int, ...]
    trigger_text: str
    global_metrics: A3FLMetrics
    adversarial_metrics: A3FLMetrics
    search_time: float
    mutations: int


@dataclass(frozen=True)
class TriggeredBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    trigger_mask: torch.Tensor
    labels: torch.Tensor


def append_token_trigger(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    trigger_token_ids: Sequence[int],
    *,
    eos_token_id: int,
    pad_token_id: int,
    max_length: int,
) -> TriggeredBatch:
    """Append exact trigger IDs immediately before EOS in a padded batch."""

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have matching [batch, sequence] shapes.")
    if labels.ndim != 1 or labels.shape[0] != input_ids.shape[0]:
        raise ValueError("labels must have shape [batch].")
    trigger = [int(token_id) for token_id in trigger_token_ids]
    if not trigger:
        raise ValueError("A3FL requires at least one trigger token.")
    if len(trigger) + 1 >= max_length:
        raise ValueError("A3FL trigger must leave room for one input token and EOS.")

    rows: List[List[int]] = []
    masks: List[List[int]] = []
    trigger_masks: List[List[bool]] = []
    prompt_budget = max_length - len(trigger) - 1
    for row in range(input_ids.shape[0]):
        active = input_ids[row][attention_mask[row].bool()].detach().cpu().tolist()
        active = [int(token_id) for token_id in active]
        if active and active[-1] == int(eos_token_id):
            active = active[:-1]
        prompt = active[:prompt_budget]
        combined = prompt + trigger + [int(eos_token_id)]
        rows.append(combined)
        masks.append([1] * len(combined))
        trigger_masks.append(
            [False] * len(prompt) + [True] * len(trigger) + [False]
        )

    width = max(len(row) for row in rows)
    output_ids = torch.full(
        (len(rows), width), int(pad_token_id), dtype=input_ids.dtype, device=input_ids.device
    )
    output_attention = torch.zeros(
        (len(rows), width), dtype=attention_mask.dtype, device=attention_mask.device
    )
    output_trigger_mask = torch.zeros(
        (len(rows), width), dtype=torch.bool, device=input_ids.device
    )
    for row_index, (row_ids, row_attention, row_trigger_mask) in enumerate(
        zip(rows, masks, trigger_masks)
    ):
        length = len(row_ids)
        output_ids[row_index, :length] = torch.tensor(
            row_ids, dtype=input_ids.dtype, device=input_ids.device
        )
        output_attention[row_index, :length] = torch.tensor(
            row_attention, dtype=attention_mask.dtype, device=attention_mask.device
        )
        output_trigger_mask[row_index, :length] = torch.tensor(
            row_trigger_mask, dtype=torch.bool, device=input_ids.device
        )
    return TriggeredBatch(
        input_ids=output_ids,
        attention_mask=output_attention,
        trigger_mask=output_trigger_mask,
        labels=labels,
    )


def _snapshot_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu", copy=True)
        for name, value in model.state_dict().items()
    }


def _load_snapshot(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def _model_parameter_cosine_similarity(
    global_model: nn.Module,
    adversarial_state: Mapping[str, torch.Tensor],
) -> float:
    """Compute sim(theta'_t, theta_t) over model parameters on CPU.

    ``global_model`` must contain theta_t while ``adversarial_state`` contains
    theta'_t.  Parameters are processed one tensor at a time so the similarity
    calculation does not allocate another complete model-sized vector.
    """

    dot_product = 0.0
    global_squared_norm = 0.0
    adversarial_squared_norm = 0.0
    for name, parameter in global_model.named_parameters():
        if name not in adversarial_state:
            raise KeyError(f"Adversarial A3FL state is missing parameter '{name}'.")
        global_values = parameter.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        adversarial_values = (
            adversarial_state[name].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        )
        if global_values.shape != adversarial_values.shape:
            raise ValueError(
                f"A3FL parameter '{name}' has incompatible global/adversarial shapes: "
                f"{tuple(global_values.shape)} vs {tuple(adversarial_values.shape)}."
            )
        dot_product += float(torch.dot(global_values, adversarial_values).item())
        global_squared_norm += float(torch.dot(global_values, global_values).item())
        adversarial_squared_norm += float(
            torch.dot(adversarial_values, adversarial_values).item()
        )

    denominator = (global_squared_norm * adversarial_squared_norm) ** 0.5
    if denominator == 0.0:
        return 0.0
    similarity = dot_product / denominator
    if not torch.isfinite(torch.tensor(similarity)):
        raise FloatingPointError("A3FL produced a non-finite model cosine similarity.")
    return max(-1.0, min(1.0, float(similarity)))


def _candidate_tokens(
    current_ids: Sequence[int],
    gradient: torch.Tensor,
    embedding_matrix: torch.Tensor,
    *,
    top_k: int,
    candidate_batch_size: int,
    excluded_ids: Iterable[int],
    rng: random.Random,
) -> List[List[int]]:
    if gradient.ndim != 2 or gradient.shape[0] != len(current_ids):
        raise ValueError("Trigger gradient must have shape [trigger_length, hidden_size].")
    if embedding_matrix.ndim != 2 or embedding_matrix.shape[1] != gradient.shape[1]:
        raise ValueError("Embedding matrix is incompatible with the trigger gradient.")
    forbidden = {int(token_id) for token_id in excluded_ids}
    choices_by_position: List[List[int]] = []
    for position, current_id in enumerate(current_ids):
        scores = embedding_matrix.float().mv(gradient[position].float())
        invalid = set(forbidden)
        invalid.add(int(current_id))
        valid_invalid = [idx for idx in invalid if 0 <= idx < scores.numel()]
        if valid_invalid:
            scores[torch.tensor(valid_invalid, device=scores.device)] = torch.inf
        count = min(int(top_k), int(torch.isfinite(scores).sum().item()))
        if count <= 0:
            raise ValueError(f"No A3FL replacement token remains at position {position}.")
        choices_by_position.append(
            [int(value) for value in torch.topk(scores, k=count, largest=False).indices.cpu().tolist()]
        )

    candidates: List[List[int]] = [list(current_ids)]
    seen = {tuple(int(value) for value in current_ids)}
    for candidate_index in range(candidate_batch_size):
        position = min(
            (candidate_index * len(current_ids)) // candidate_batch_size,
            len(current_ids) - 1,
        )
        candidate = list(current_ids)
        candidate[position] = int(rng.choice(choices_by_position[position]))
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


class A3FLTriggerOptimizer:
    """Optimize a discrete text trigger with the A3FL adversarial objective.

    Only one model instance is used on the accelerator.  The adversarial model
    is represented by a CPU state snapshot and swapped in only when needed,
    avoiding a second full Switch Transformer on GPU.
    """

    def __init__(
        self,
        model: nn.Module,
        config: A3FLSearchConfig,
        *,
        eos_token_id: int,
        pad_token_id: int,
        target_label: int,
        restore_global: Callable[[], None],
        excluded_token_ids: Iterable[int] = (),
        decode_trigger: Optional[Callable[[Sequence[int]], str]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.target_label = int(target_label)
        self.restore_global = restore_global
        self.excluded_token_ids = tuple(int(value) for value in excluded_token_ids)
        self.decode_trigger = decode_trigger
        self.rng = random.Random(config.seed)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _gradient_embedding_module(self) -> nn.Module:
        """Return the encoder embedding module actually called by forward().

        The local Switch sequence-classification model exposes ``shared`` from
        ``get_input_embeddings()``, but its encoder calls
        ``encoder.embed_tokens``.  Their weights are tied, yet a forward hook on
        ``shared`` does not run when ``encoder.embed_tokens`` is invoked.  A3FL
        needs the module on the real encoder path so the injected leaf tensor is
        connected to the classification loss.
        """

        encoder = getattr(self.model, "encoder", None)
        encoder_embedding = getattr(encoder, "embed_tokens", None)
        if isinstance(encoder_embedding, nn.Module):
            return encoder_embedding

        embedding = self.model.get_input_embeddings()
        if not isinstance(embedding, nn.Module):
            raise TypeError(
                "A3FL requires get_input_embeddings() to return an nn.Module."
            )
        return embedding

    def _prepare(self, batch: Mapping[str, torch.Tensor], trigger_ids: Sequence[int]) -> TriggeredBatch:
        return append_token_trigger(
            batch["input_ids"].to(self.device),
            batch["attention_mask"].to(self.device),
            batch["labels"].to(self.device),
            trigger_ids,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            max_length=self.config.max_length,
        )

    def _forward_metrics(
        self, batch: Mapping[str, torch.Tensor], trigger_ids: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prepared = self._prepare(batch, trigger_ids)
        outputs = self.model(
            input_ids=prepared.input_ids,
            attention_mask=prepared.attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits.float()
        targets = torch.full(
            (logits.shape[0],), self.target_label, dtype=torch.long, device=logits.device
        )
        loss = F.cross_entropy(logits, targets)
        probabilities = torch.softmax(logits, dim=-1)[:, self.target_label].mean()
        hit_rate = logits.argmax(dim=-1).eq(self.target_label).float().mean()
        return loss, probabilities, hit_rate

    def _metrics(
        self, batch: Mapping[str, torch.Tensor], trigger_ids: Sequence[int]
    ) -> A3FLMetrics:
        self.model.eval()
        with torch.no_grad():
            loss, probability, hit_rate = self._forward_metrics(batch, trigger_ids)
        return A3FLMetrics(float(loss), float(probability), float(hit_rate))

    def _gradient(
        self, batch: Mapping[str, torch.Tensor], trigger_ids: Sequence[int]
    ) -> torch.Tensor:
        self.model.eval()
        self.model.requires_grad_(False)
        embedding = self._gradient_embedding_module()
        captured: List[torch.Tensor] = []

        def embedding_hook(_module: nn.Module, _args, output: torch.Tensor) -> torch.Tensor:
            leaf = output.detach().requires_grad_(True)
            leaf.retain_grad()
            captured.append(leaf)
            return leaf

        handle = embedding.register_forward_hook(embedding_hook)
        try:
            # Explicitly re-enable autograd in case the caller performs other
            # candidate evaluations under no_grad. Model parameters remain
            # frozen; only the hook-created embedding leaf receives gradients.
            with torch.enable_grad():
                prepared = self._prepare(batch, trigger_ids)
                outputs = self.model(
                    input_ids=prepared.input_ids,
                    attention_mask=prepared.attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                targets = torch.full(
                    (outputs.logits.shape[0],),
                    self.target_label,
                    dtype=torch.long,
                    device=outputs.logits.device,
                )
                loss = F.cross_entropy(outputs.logits.float(), targets)
                if not loss.requires_grad:
                    raise RuntimeError(
                        "A3FL could not connect the active encoder embedding "
                        "module to the classification loss."
                    )
                loss.backward()
        finally:
            handle.remove()
        if not captured or captured[0].grad is None:
            raise RuntimeError("A3FL classification loss did not reach the input embeddings.")
        encoder_embedding = captured[0]
        result = torch.zeros(
            len(trigger_ids), encoder_embedding.shape[-1], device=self.device, dtype=torch.float32
        )
        for row in range(encoder_embedding.shape[0]):
            row_gradient = encoder_embedding.grad[row][prepared.trigger_mask[row]]
            if row_gradient.shape != result.shape:
                raise RuntimeError(
                    f"A3FL trigger gradient shape {tuple(row_gradient.shape)} "
                    f"does not match {tuple(result.shape)}."
                )
            result += row_gradient.float() / float(encoder_embedding.shape[0])
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("A3FL produced a non-finite trigger gradient.")
        return result

    def _evaluate_candidates(
        self, batch: Mapping[str, torch.Tensor], candidates: Sequence[Sequence[int]]
    ) -> List[float]:
        return [self._metrics(batch, candidate).loss for candidate in candidates]

    def _train_adversarial_model(
        self, batch: Mapping[str, torch.Tensor], trigger_ids: Sequence[int]
    ) -> None:
        self.model.train()
        self.model.requires_grad_(True)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.config.adversarial_lr
        )
        prepared = self._prepare(batch, trigger_ids)
        try:
            for _ in range(self.config.adversarial_steps):
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(
                    input_ids=prepared.input_ids,
                    attention_mask=prepared.attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                # True labels deliberately teach the adversarial model to ignore
                # the trigger, simulating worst-case benign unlearning.
                loss = F.cross_entropy(outputs.logits.float(), prepared.labels.long())
                loss.backward()
                optimizer.step()
        finally:
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            self.model.requires_grad_(False)
            self.model.eval()

    def optimize(
        self,
        batch: Mapping[str, torch.Tensor],
        initial_trigger_token_ids: Sequence[int],
    ) -> A3FLSearchResult:
        current = [int(token_id) for token_id in initial_trigger_token_ids]
        if not current:
            raise ValueError("A3FL initial trigger must contain at least one token ID.")
        started = time.time()
        initial = tuple(current)
        adversarial_state: Optional[Dict[str, torch.Tensor]] = None
        try:
            self.restore_global()
            # Algorithm 1 initializes the adversarial model from the current
            # global model, then alternates trigger optimization and unlearning.
            adversarial_state = _snapshot_state(self.model)
            for outer_step in range(self.config.outer_steps):
                self.restore_global()
                model_similarity = _model_parameter_cosine_similarity(
                    self.model, adversarial_state
                )
                # Paper setting: lambda = lambda_0 * sim(theta'_t, theta_t).
                adversarial_weight = self.config.lambda0 * model_similarity
                for trigger_step in range(self.config.trigger_steps):
                    _load_snapshot(self.model, adversarial_state)
                    adversarial_gradient = self._gradient(batch, current)

                    self.restore_global()
                    global_gradient = self._gradient(batch, current)
                    embedding_module = self._gradient_embedding_module()
                    embedding_matrix = embedding_module.weight.detach()
                    combined_gradient = (
                        global_gradient
                        + adversarial_weight * adversarial_gradient
                    )
                    candidates = _candidate_tokens(
                        current,
                        combined_gradient,
                        embedding_matrix,
                        top_k=self.config.top_k,
                        candidate_batch_size=self.config.candidate_batch_size,
                        excluded_ids=self.excluded_token_ids,
                        rng=self.rng,
                    )
                    global_losses = self._evaluate_candidates(batch, candidates)
                    _load_snapshot(self.model, adversarial_state)
                    adversarial_losses = self._evaluate_candidates(batch, candidates)
                    combined_losses = [
                        global_loss
                        + adversarial_weight * adversarial_loss
                        for global_loss, adversarial_loss in zip(
                            global_losses, adversarial_losses
                        )
                    ]
                    selected = min(range(len(candidates)), key=combined_losses.__getitem__)
                    current = list(candidates[selected])

                    self.restore_global()
                    global_metrics = self._metrics(batch, current)
                    _load_snapshot(self.model, adversarial_state)
                    adversarial_metrics = self._metrics(batch, current)
                    if (
                        global_metrics.attack_success_rate >= self.config.early_stop_asr
                        and adversarial_metrics.attack_success_rate >= self.config.early_stop_asr
                    ):
                        break

                # Train theta'_t on triggered inputs with their true labels so
                # it explicitly attempts to erase the newly selected trigger.
                _load_snapshot(self.model, adversarial_state)
                self._train_adversarial_model(batch, current)
                # Drop the previous snapshot before allocating the next full CPU
                # state, limiting peak host memory for large Switch models.
                adversarial_state = None
                adversarial_state = _snapshot_state(self.model)

            self.restore_global()
            global_metrics = self._metrics(batch, current)
            if adversarial_state is None:
                adversarial_metrics = global_metrics
            else:
                _load_snapshot(self.model, adversarial_state)
                adversarial_metrics = self._metrics(batch, current)
            trigger_text = (
                self.decode_trigger(current) if self.decode_trigger is not None else str(current)
            )
            return A3FLSearchResult(
                trigger_token_ids=tuple(current),
                trigger_text=trigger_text,
                global_metrics=global_metrics,
                adversarial_metrics=adversarial_metrics,
                search_time=float(time.time() - started),
                mutations=sum(left != right for left, right in zip(initial, current)),
            )
        finally:
            self.restore_global()
            self.model.requires_grad_(False)

