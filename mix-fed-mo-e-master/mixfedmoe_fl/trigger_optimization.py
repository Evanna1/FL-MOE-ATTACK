from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from mixfedmoe_fl.client import MoeHandle, collect_moe_handles
from mixfedmoe_fl.data import build_tokenizer
from models.switch_transformers import SwitchTransformersForSequenceClassification


@dataclass(frozen=True)
class TriggerOptimizationConfig:
    checkpoint_path: str
    target_layer: str
    target_expert: int
    model_name_or_path: str = "model_ckpt/switch-base-8"
    trigger_length: int = 10
    initial_trigger: Optional[str] = None
    iterations: int = 50
    search_batch_size: int = 200
    top_k: int = 50
    beta: float = 0.1
    target_ppl: str = "auto"
    reference_text: Optional[str] = None
    ppl_model_name_or_path: str = "gpt2"
    ppl_batch_size: int = 16
    target_probability_threshold: float = 0.90
    calibration_file: Optional[str] = None
    calibration_samples: int = 100
    prompt: Optional[str] = None
    calibration_batch_size: int = 8
    eval_micro_batch_size: int = 8
    max_mutations_per_candidate: int = 3
    coordinate_candidates_per_position: int = 4
    plateau_patience: int = 5
    plateau_escape_objective: str = "probability"
    trigger_position: str = "append"
    fixed_position: Optional[int] = None
    max_length: int = 256
    output_dir: str = "outputs/trigger_optimization"
    device: str = "auto"
    seed: int = 42
    eps: float = 1e-8


@dataclass(frozen=True)
class TriggerRecord:
    trigger_ids: List[int]
    trigger_text: str
    routing_loss: float
    target_expert_probability: float
    iteration: int


@dataclass(frozen=True)
class PreparedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    trigger_mask: torch.Tensor


def _load_checkpoint_state(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: '{checkpoint_path}'.")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata: Dict[str, Any] = {}
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        state = payload["state_dict"]
        metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    elif isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        state = payload["model_state_dict"]
        metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    elif isinstance(payload, dict) and payload and all(torch.is_tensor(value) for value in payload.values()):
        state = payload
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected a state_dict, {'state_dict': ...}, "
            "or {'model_state_dict': ...}."
        )
    return dict(state), metadata


def load_victim_model(
    model_name_or_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[SwitchTransformersForSequenceClassification, PreTrainedTokenizerBase, Dict[str, Any]]:
    state_dict, metadata = _load_checkpoint_state(checkpoint_path)
    classifier_weight = state_dict.get("classification_head.out_proj.weight")
    num_labels = int(classifier_weight.shape[0]) if classifier_weight is not None else 2
    tokenizer = build_tokenizer(model_name_or_path, max_length=1_000_000)
    model = SwitchTransformersForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
        tie_word_embeddings=False,
        local_files_only=True,
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [name for name in incompatible.missing_keys if not name.endswith("position_ids")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible with the configured victim model. "
            f"Missing keys={missing[:20]}, unexpected keys={incompatible.unexpected_keys[:20]}."
        )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id
    model.config.use_cache = False
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return model, tokenizer, metadata


def resolve_moe_layer(model: nn.Module, target_layer: str) -> MoeHandle:
    handles = collect_moe_handles(model)
    exact = [handle for handle in handles if handle.layer_id == target_layer]
    if len(exact) == 1:
        return exact[0]
    available = "\n    ".join(handle.layer_id for handle in handles) or "<none>"
    raise ValueError(
        f"MoE layer '{target_layer}' was not found.\nAvailable MoE layers:\n    {available}"
    )


def get_router_module(handle: MoeHandle) -> nn.Linear:
    router = getattr(getattr(handle.mlp_module, "router", None), "classifier", None)
    if not isinstance(router, nn.Linear):
        path = handle.gate_linear_path
        if path:
            current: nn.Module = handle.mlp_module
            for part in path.split("."):
                current = getattr(current, part)
            router = current
    if not isinstance(router, nn.Linear):
        raise RuntimeError(
            f"Could not resolve a linear router classifier for MoE layer '{handle.layer_id}'."
        )
    return router


def _target_mask_for_stack(trigger_mask: torch.Tensor, stack_name: str) -> torch.Tensor:
    if stack_name == "encoder":
        return trigger_mask
    if stack_name == "decoder":
        shifted = torch.zeros_like(trigger_mask)
        shifted[:, 1:] = trigger_mask[:, :-1]
        return shifted
    raise ValueError(f"Unsupported stack '{stack_name}'.")


def compute_routing_loss(
    router_logits: torch.Tensor,
    trigger_mask: torch.Tensor,
    target: int | torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if router_logits.ndim == 2:
        expected_tokens = int(trigger_mask.numel())
        if router_logits.shape[0] != expected_tokens:
            raise ValueError(
                f"Flattened router logits have {router_logits.shape[0]} tokens, expected {expected_tokens}."
            )
        router_logits = router_logits.reshape(*trigger_mask.shape, router_logits.shape[-1])
    if router_logits.ndim != 3 or router_logits.shape[:2] != trigger_mask.shape:
        raise ValueError(
            f"Router logits shape {tuple(router_logits.shape)} is incompatible with "
            f"trigger mask shape {tuple(trigger_mask.shape)}."
        )
    probabilities = torch.softmax(router_logits.float(), dim=-1)
    if isinstance(target, int):
        if not 0 <= target < probabilities.shape[-1]:
            raise ValueError(f"target_expert={target} is outside [0, {probabilities.shape[-1] - 1}].")
        target_probabilities = probabilities[..., target]
        token_losses = -torch.log(target_probabilities + eps)
    else:
        target_vector = target.to(probabilities.device, dtype=probabilities.dtype)
        if target_vector.ndim != 1 or target_vector.shape[0] != probabilities.shape[-1]:
            raise ValueError("Target vector must have shape [num_experts].")
        if not torch.isclose(target_vector.sum(), target_vector.new_tensor(1.0), atol=1e-5):
            raise ValueError("Target vector must sum to 1.")
        token_losses = -(target_vector * torch.log(probabilities + eps)).sum(dim=-1)
        target_probabilities = (probabilities * target_vector).sum(dim=-1)
    selected = trigger_mask.bool()
    if not bool(selected.any()):
        raise ValueError("Trigger mask contains no optimizable token positions.")
    loss = token_losses[selected].mean()
    mean_probability = target_probabilities[selected].mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(
            "Non-finite routing loss. "
            f"logits={router_logits.detach().cpu()} probabilities={probabilities.detach().cpu()}"
        )
    return loss, mean_probability


def top_k_candidates_for_position(
    current_trigger_ids: Sequence[int],
    trigger_position: int,
    gradient: torch.Tensor,
    embedding_matrix: torch.Tensor,
    top_k: int,
    excluded_token_ids: Iterable[int] = (),
) -> List[int]:
    if not 0 <= trigger_position < len(current_trigger_ids):
        raise IndexError("trigger_position is out of range.")
    if gradient.ndim == 2:
        gradient = gradient[trigger_position]
    if gradient.ndim != 1 or embedding_matrix.ndim != 2:
        raise ValueError("Expected gradient [hidden] or [trigger_length, hidden] and embeddings [vocab, hidden].")
    if embedding_matrix.shape[1] != gradient.shape[0]:
        raise ValueError(
            f"Embedding width {embedding_matrix.shape[1]} != gradient width {gradient.shape[0]}."
        )
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")
    # First-order replacement delta is (E[new] - E[current]) dot grad.  The
    # current-token term is constant, so the smallest E[new] dot grad descends.
    scores = torch.mv(embedding_matrix.float(), gradient.float())
    forbidden = set(int(token_id) for token_id in excluded_token_ids)
    forbidden.add(int(current_trigger_ids[trigger_position]))
    if forbidden:
        valid = [token_id for token_id in forbidden if 0 <= token_id < scores.numel()]
        scores[torch.tensor(valid, device=scores.device, dtype=torch.long)] = torch.inf
    count = min(top_k, int(torch.isfinite(scores).sum().item()))
    return torch.topk(scores, k=count, largest=False).indices.detach().cpu().tolist()


def generate_candidate_triggers(
    current_trigger_ids: Sequence[int],
    candidates_by_position: Sequence[Sequence[int]],
    search_batch_size: int,
    rng: random.Random,
    max_mutations_per_candidate: int = 1,
    coordinate_candidates_per_position: int = 1,
) -> List[List[int]]:
    if search_batch_size <= 0:
        raise ValueError("search_batch_size must be > 0.")
    if max_mutations_per_candidate <= 0 or coordinate_candidates_per_position <= 0:
        raise ValueError("Mutation and coordinate candidate counts must be > 0.")
    generated = [list(current_trigger_ids)]  # Global-best/current candidate.
    seen = {tuple(current_trigger_ids)}

    # Deterministic coordinate coverage prevents random sampling from entirely
    # missing a promising trigger position or its highest-ranked token.
    for rank in range(coordinate_candidates_per_position):
        for position, choices in enumerate(candidates_by_position):
            if rank >= len(choices) or len(generated) >= search_batch_size:
                continue
            candidate = list(current_trigger_ids)
            candidate[position] = int(choices[rank])
            key = tuple(candidate)
            if key not in seen:
                seen.add(key)
                generated.append(candidate)

    # Multi-coordinate proposals can cross a barrier that no single-token
    # replacement can improve. They remain gradient guided at every position.
    attempts = 0
    max_attempts = max(search_batch_size * 20, 100)
    max_mutations = min(max_mutations_per_candidate, len(current_trigger_ids))
    while len(generated) < search_batch_size and attempts < max_attempts:
        attempts += 1
        mutation_count = rng.randint(1, max_mutations)
        positions = rng.sample(range(len(current_trigger_ids)), k=mutation_count)
        candidate = list(current_trigger_ids)
        for position in positions:
            choices = candidates_by_position[position]
            if not choices:
                raise ValueError(f"No candidate tokens for trigger position {position}.")
            candidate[position] = int(rng.choice(choices))
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            generated.append(candidate)
    return generated


def load_calibration_texts(
    calibration_file: Optional[str],
    prompt: Optional[str],
    calibration_samples: int,
) -> List[str]:
    if calibration_samples <= 0:
        raise ValueError("calibration_samples must be > 0.")
    if calibration_file:
        with Path(calibration_file).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("Calibration JSON must be a list of {'text': ...} objects.")
        texts = [row.get("text") for row in payload if isinstance(row, dict)]
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("Calibration JSON must contain non-empty string 'text' fields.")
        return list(texts[:calibration_samples])
    if prompt is not None:
        return [prompt]
    raise ValueError("Provide --calibration_file or --prompt.")


def build_triggered_batch(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    trigger_ids: Sequence[int],
    max_length: int,
    trigger_position: str = "append",
    fixed_position: Optional[int] = None,
) -> PreparedBatch:
    if max_length <= len(trigger_ids) + 1:
        raise ValueError("max_length must leave room for trigger tokens and EOS.")
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if eos_id is None or pad_id is None:
        raise ValueError("Victim tokenizer must define eos_token_id and pad_token_id.")
    rows: List[List[int]] = []
    masks: List[List[int]] = []
    trigger_masks: List[List[bool]] = []
    prompt_budget = max_length - len(trigger_ids) - 1
    for prompt in prompts:
        prompt_ids = list(tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"])
        # The classification wrapper requires an equal (single) EOS count in
        # every row; the pipeline appends that EOS explicitly below.
        prompt_ids = [token_id for token_id in prompt_ids if token_id != eos_id]
        prompt_ids = prompt_ids[:prompt_budget]
        if trigger_position == "append":
            insertion = len(prompt_ids)
        elif trigger_position == "prepend":
            insertion = 0
        elif trigger_position == "fixed_position":
            if fixed_position is None or fixed_position < 0:
                raise ValueError("--fixed_position >= 0 is required for trigger_position=fixed_position.")
            insertion = min(fixed_position, len(prompt_ids))
        else:
            raise ValueError("trigger_position must be append, prepend, or fixed_position.")
        ids = prompt_ids[:insertion] + list(trigger_ids) + prompt_ids[insertion:] + [eos_id]
        trig_mask = [False] * insertion + [True] * len(trigger_ids)
        trig_mask += [False] * (len(ids) - len(trig_mask))
        rows.append(ids)
        masks.append([1] * len(ids))
        trigger_masks.append(trig_mask)
    width = max(len(row) for row in rows)
    for ids, mask, trig_mask in zip(rows, masks, trigger_masks):
        padding = width - len(ids)
        ids.extend([pad_id] * padding)
        mask.extend([0] * padding)
        trig_mask.extend([False] * padding)
    return PreparedBatch(
        input_ids=torch.tensor(rows, dtype=torch.long),
        attention_mask=torch.tensor(masks, dtype=torch.long),
        trigger_mask=torch.tensor(trigger_masks, dtype=torch.bool),
    )


class RoutingAwareTriggerOptimizer:
    def __init__(
        self,
        model: SwitchTransformersForSequenceClassification,
        tokenizer: PreTrainedTokenizerBase,
        config: TriggerOptimizationConfig,
        calibration_texts: Sequence[str],
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.calibration_texts = list(calibration_texts)
        self.device = device
        self.handle = resolve_moe_layer(model, config.target_layer)
        self.router = get_router_module(self.handle)
        self.num_experts = int(self.router.out_features)
        if not 0 <= config.target_expert < self.num_experts:
            raise ValueError(
                f"target_expert={config.target_expert} is outside [0, {self.num_experts - 1}] "
                f"for layer {config.target_layer}."
            )
        stack = getattr(model, self.handle.stack_name)
        self.embedding: nn.Embedding = stack.embed_tokens
        if self.embedding.weight.ndim != 2:
            raise RuntimeError(f"Input embedding must be 2D, got {tuple(self.embedding.weight.shape)}.")
        self.rng = random.Random(config.seed)
        self.history: Dict[Tuple[int, ...], TriggerRecord] = {}
        self.satisfying: Dict[Tuple[int, ...], TriggerRecord] = {}
        self.iteration_log: List[TriggerRecord] = []

    def _capture_router_logits(self, batch: PreparedBatch, with_embedding_grad: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        captured: Dict[str, torch.Tensor] = {}

        def router_hook(_module: nn.Module, _args: Tuple[Any, ...], output: torch.Tensor) -> None:
            captured["router_logits"] = output

        def embedding_hook(_module: nn.Module, _args: Tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            leaf = output.detach().requires_grad_(True)
            leaf.retain_grad()
            captured["embedding_output"] = leaf
            return leaf

        handles = [self.router.register_forward_hook(router_hook)]
        if with_embedding_grad:
            handles.append(self.embedding.register_forward_hook(embedding_hook))
        try:
            self.model(
                input_ids=batch.input_ids.to(self.device),
                attention_mask=batch.attention_mask.to(self.device),
                output_router_logits=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        if "router_logits" not in captured:
            raise RuntimeError(f"Router hook for {self.config.target_layer} did not run.")
        return captured["router_logits"], captured.get("embedding_output")

    def _prepare(self, prompts: Sequence[str], trigger_ids: Sequence[int]) -> PreparedBatch:
        batch = build_triggered_batch(
            self.tokenizer,
            prompts,
            trigger_ids,
            self.config.max_length,
            self.config.trigger_position,
            self.config.fixed_position,
        )
        return PreparedBatch(
            input_ids=batch.input_ids.to(self.device),
            attention_mask=batch.attention_mask.to(self.device),
            trigger_mask=_target_mask_for_stack(batch.trigger_mask, self.handle.stack_name).to(self.device),
        )

    def evaluate_trigger(self, trigger_ids: Sequence[int]) -> Tuple[float, float]:
        loss_sum = 0.0
        probability_sum = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(self.calibration_texts), self.config.eval_micro_batch_size):
                prompts = self.calibration_texts[start : start + self.config.eval_micro_batch_size]
                batch = self._prepare(prompts, trigger_ids)
                logits, _ = self._capture_router_logits(batch, with_embedding_grad=False)
                loss, probability = compute_routing_loss(
                    logits, batch.trigger_mask, self.config.target_expert, self.config.eps
                )
                loss_sum += float(loss) * len(prompts)
                probability_sum += float(probability) * len(prompts)
                count += len(prompts)
        return loss_sum / count, probability_sum / count

    def compute_trigger_gradients(self, trigger_ids: Sequence[int]) -> torch.Tensor:
        total_examples = len(self.calibration_texts)
        gradient = torch.zeros(
            len(trigger_ids), self.embedding.embedding_dim, device=self.device, dtype=torch.float32
        )
        for start in range(0, total_examples, self.config.calibration_batch_size):
            prompts = self.calibration_texts[start : start + self.config.calibration_batch_size]
            batch = self._prepare(prompts, trigger_ids)
            logits, embedding_output = self._capture_router_logits(batch, with_embedding_grad=True)
            if embedding_output is None:
                raise RuntimeError("Embedding hook did not capture a differentiable tensor.")
            loss, _ = compute_routing_loss(logits, batch.trigger_mask, self.config.target_expert, self.config.eps)
            scaled_loss = loss * (len(prompts) / total_examples)
            scaled_loss.backward()
            if embedding_output.grad is None:
                raise RuntimeError("Routing loss did not produce a trigger embedding gradient.")
            for row in range(len(prompts)):
                row_gradient = embedding_output.grad[row][batch.trigger_mask[row]]
                if row_gradient.shape != gradient.shape:
                    raise RuntimeError(
                        f"Trigger gradient shape {tuple(row_gradient.shape)} != expected {tuple(gradient.shape)}."
                    )
                # scaled_loss already normalizes each token gradient by the
                # total number of calibration examples and trigger positions.
                gradient += row_gradient
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"Non-finite trigger gradient for IDs {list(trigger_ids)}.")
        return gradient

    def evaluate_candidate_triggers(self, candidates: Sequence[Sequence[int]]) -> Tuple[List[float], List[float]]:
        losses = [0.0] * len(candidates)
        probabilities = [0.0] * len(candidates)
        counts = [0] * len(candidates)
        pairs = [(idx, prompt) for idx in range(len(candidates)) for prompt in self.calibration_texts]
        with torch.no_grad():
            for start in range(0, len(pairs), self.config.eval_micro_batch_size):
                chunk = pairs[start : start + self.config.eval_micro_batch_size]
                prompts = [item[1] for item in chunk]
                # A microbatch may contain different candidate triggers, so build rows separately.
                prepared = [self._prepare([prompt], candidates[idx]) for idx, prompt in chunk]
                width = max(item.input_ids.shape[1] for item in prepared)
                pad_id = int(self.tokenizer.pad_token_id)
                ids = torch.full((len(prepared), width), pad_id, dtype=torch.long, device=self.device)
                attention = torch.zeros((len(prepared), width), dtype=torch.long, device=self.device)
                masks = torch.zeros((len(prepared), width), dtype=torch.bool, device=self.device)
                for row, item in enumerate(prepared):
                    size = item.input_ids.shape[1]
                    ids[row, :size] = item.input_ids[0]
                    attention[row, :size] = item.attention_mask[0]
                    masks[row, :size] = item.trigger_mask[0]
                batch = PreparedBatch(ids, attention, masks)
                logits, _ = self._capture_router_logits(batch, with_embedding_grad=False)
                shaped = logits.reshape(len(chunk), width, self.num_experts)
                probs = torch.softmax(shaped.float(), dim=-1)[..., self.config.target_expert]
                token_losses = -torch.log(probs + self.config.eps)
                for row, (candidate_idx, _prompt) in enumerate(chunk):
                    selected = masks[row]
                    value = token_losses[row][selected].mean()
                    probability = probs[row][selected].mean()
                    if not bool(torch.isfinite(value)):
                        raise FloatingPointError(
                            f"Non-finite candidate loss: trigger={list(candidates[candidate_idx])}, "
                            f"layer={self.config.target_layer}, logits={shaped[row].detach().cpu()}."
                        )
                    losses[candidate_idx] += float(value)
                    probabilities[candidate_idx] += float(probability)
                    counts[candidate_idx] += 1
        return (
            [losses[idx] / counts[idx] for idx in range(len(candidates))],
            [probabilities[idx] / counts[idx] for idx in range(len(candidates))],
        )

    def _decode(self, trigger_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(trigger_ids), skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

    def initial_trigger_ids(self) -> List[int]:
        if self.config.initial_trigger:
            ids = list(self.tokenizer(self.config.initial_trigger, add_special_tokens=False)["input_ids"])
            if not ids:
                raise ValueError("--initial_trigger tokenized to an empty sequence.")
            return (ids * math.ceil(self.config.trigger_length / len(ids)))[: self.config.trigger_length]
        bang_ids = list(self.tokenizer("!", add_special_tokens=False)["input_ids"])
        special = set(self.tokenizer.all_special_ids)
        usable = [token_id for token_id in bang_ids if token_id not in special and self.tokenizer.decode([token_id]).strip()]
        bang_id = usable[-1] if usable else bang_ids[-1]
        return [int(bang_id)] * self.config.trigger_length

    def _record(self, ids: Sequence[int], loss: float, probability: float, iteration: int) -> TriggerRecord:
        record = TriggerRecord(list(ids), self._decode(ids), loss, probability, iteration)
        self.iteration_log.append(record)
        key = tuple(ids)
        previous = self.history.get(key)
        if previous is None or loss < previous.routing_loss:
            self.history[key] = record
        if probability >= self.config.target_probability_threshold:
            previous = self.satisfying.get(key)
            if previous is None or loss < previous.routing_loss:
                self.satisfying[key] = record
        return record

    def optimize(self) -> Tuple[TriggerRecord, List[TriggerRecord]]:
        current = self.initial_trigger_ids()
        initial_loss, initial_probability = self.evaluate_trigger(current)
        self._record(current, initial_loss, initial_probability, 0)
        print(
            f"[Initial]\nTrigger: {self._decode(current)}\nRouting Loss: {initial_loss:.6f}\n"
            f"Target Expert Probability: {initial_probability:.6f}",
            flush=True,
        )
        excluded = set(self.tokenizer.all_special_ids)
        stagnant_iterations = 0
        for iteration in range(1, self.config.iterations + 1):
            gradient = self.compute_trigger_gradients(current)
            candidates_by_position = [
                top_k_candidates_for_position(
                    current,
                    position,
                    gradient,
                    self.embedding.weight.detach(),
                    self.config.top_k,
                    excluded,
                )
                for position in range(len(current))
            ]
            candidates = generate_candidate_triggers(
                current,
                candidates_by_position,
                self.config.search_batch_size,
                self.rng,
                self.config.max_mutations_per_candidate,
                self.config.coordinate_candidates_per_position,
            )
            losses, probabilities = self.evaluate_candidate_triggers(candidates)
            best_index = min(range(len(candidates)), key=lambda idx: losses[idx])
            old_loss = self.history[tuple(current)].routing_loss
            best_mutated_index = min(range(1, len(candidates)), key=lambda idx: losses[idx])
            best_probability_index = max(range(1, len(candidates)), key=lambda idx: probabilities[idx])
            escaped_plateau = False
            if best_index == 0:
                stagnant_iterations += 1
                if stagnant_iterations >= self.config.plateau_patience:
                    # The paper loss is a mean negative log probability, while
                    # the stopping condition uses arithmetic mean probability.
                    # They can disagree. During a plateau, optionally follow
                    # the threshold metric to enter a different search basin.
                    if self.config.plateau_escape_objective == "probability":
                        best_index = best_probability_index
                    else:
                        best_index = best_mutated_index
                    escaped_plateau = True
                    stagnant_iterations = 0
            else:
                stagnant_iterations = 0
            current = list(candidates[best_index])
            record = self._record(current, losses[best_index], probabilities[best_index], iteration)
            satisfying = record.target_expert_probability >= self.config.target_probability_threshold
            print(
                f"[Iteration {iteration:02d}]\nTrigger: {record.trigger_text}\n"
                f"Routing Loss: {record.routing_loss:.6f}\n"
                f"Target Expert Probability: {record.target_expert_probability:.6f}\n"
                f"Best Candidate: {best_index}\n"
                f"Best Mutated Loss: {losses[best_mutated_index]:.6f}\n"
                f"Best Candidate Probability: {probabilities[best_probability_index]:.6f}\n"
                f"Unique Candidates: {len(candidates)}\n"
                f"Plateau Escape: {escaped_plateau}\nSatisfying: {satisfying}",
                flush=True,
            )
            if escaped_plateau:
                print(
                    f"PLATEAU ESCAPE: accepted a mutated candidate using "
                    f"objective={self.config.plateau_escape_objective}; "
                    "the historical global best is still retained.",
                    flush=True,
                )
            elif record.routing_loss >= old_loss - 1e-12:
                print("WARNING: routing loss did not decrease in this iteration.", flush=True)
        pool = list(self.satisfying.values())
        if not pool:
            print(
                "No trigger satisfied target probability threshold. "
                "Falling back to minimum routing-loss trigger.",
                flush=True,
            )
            pool = list(self.history.values())
        pool.sort(key=lambda record: record.routing_loss)
        return pool[0], pool


def compute_perplexity(
    texts: Sequence[str],
    model_name_or_path: str,
    device: torch.device,
    batch_size: int = 16,
) -> List[float]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, local_files_only=True).to(device)
    except OSError as exc:
        raise FileNotFoundError(
            f"GPT-2 perplexity model '{model_name_or_path}' is not available locally. "
            "Download/cache it beforehand or pass --ppl_model_name_or_path to a local GPT-2 directory."
        ) from exc
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Perplexity tokenizer defines neither pad nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.requires_grad_(False)
    results: List[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = [
                text if text.strip() else (tokenizer.eos_token or "")
                for text in texts[start : start + batch_size]
            ]
            encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoded["input_ids"].to(device)
            attention = encoded["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention).logits
            shift_logits = logits[:, :-1].float()
            shift_labels = input_ids[:, 1:]
            shift_mask = attention[:, 1:].bool()
            losses = nn.functional.cross_entropy(
                shift_logits.transpose(1, 2), shift_labels, reduction="none"
            )
            for row in range(len(chunk)):
                selected = losses[row][shift_mask[row]]
                if selected.numel() == 0:
                    results.append(float("inf"))
                else:
                    results.append(float(torch.exp(selected.mean()).cpu()))
    model.cpu()
    return results


def select_final_trigger(
    records: Sequence[TriggerRecord],
    perplexities: Sequence[float],
    beta: float,
    target_ppl: float,
) -> Tuple[TriggerRecord, float, float]:
    if len(records) != len(perplexities) or not records:
        raise ValueError("records and perplexities must be non-empty and have equal length.")
    scores = [
        record.routing_loss + beta * abs(perplexity - target_ppl)
        for record, perplexity in zip(records, perplexities)
    ]
    index = min(range(len(records)), key=lambda idx: scores[idx])
    return records[index], float(perplexities[index]), float(scores[index])


def save_results(
    output_dir: Path,
    config: TriggerOptimizationConfig,
    final_record: TriggerRecord,
    perplexity: float,
    final_score: float,
    target_ppl: float,
    candidates: Sequence[TriggerRecord],
    history: Sequence[TriggerRecord],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint_path": config.checkpoint_path,
        "target_layer": config.target_layer,
        "target_expert": config.target_expert,
        "trigger_length": config.trigger_length,
        "trigger_ids": final_record.trigger_ids,
        "trigger_text": final_record.trigger_text,
        "routing_loss": final_record.routing_loss,
        "target_expert_probability": final_record.target_expert_probability,
        "perplexity": perplexity,
        "target_ppl": target_ppl,
        "beta": config.beta,
        "final_score": final_score,
        "iterations": config.iterations,
        "top_k": config.top_k,
        "search_batch_size": config.search_batch_size,
        "max_mutations_per_candidate": config.max_mutations_per_candidate,
        "coordinate_candidates_per_position": config.coordinate_candidates_per_position,
        "plateau_patience": config.plateau_patience,
        "plateau_escape_objective": config.plateau_escape_objective,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    # Compatibility artifact for the existing --selected_triggers_path BadNet
    # loader.  This does not start or modify FL training.
    selected_trigger = {
        "target": {"layer_id": config.target_layer, "expert_id": config.target_expert},
        "best_trigger": {
            "trigger": final_record.trigger_text,
            "trigger_token_ids": final_record.trigger_ids,
            "routing_loss": final_record.routing_loss,
            "mean_target_probability": final_record.target_expert_probability,
            "perplexity": perplexity,
            "combined_score": final_score,
        },
    }
    (output_dir / "selected_triggers.json").write_text(
        json.dumps(selected_trigger, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "candidates.json").write_text(
        json.dumps([asdict(record) for record in candidates], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "optimization_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["iteration", "trigger", "routing_loss", "target_expert_probability", "is_satisfying"],
        )
        writer.writeheader()
        for record in sorted(history, key=lambda item: item.iteration):
            writer.writerow(
                {
                    "iteration": record.iteration,
                    "trigger": record.trigger_text,
                    "routing_loss": record.routing_loss,
                    "target_expert_probability": record.target_expert_probability,
                    "is_satisfying": record.target_expert_probability >= config.target_probability_threshold,
                }
            )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{value}' requested, but CUDA is unavailable.")
    return device


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Routing-aware discrete trigger optimization.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--model_name_or_path", default="model_ckpt/switch-base-8")
    parser.add_argument("--target_layer", required=True)
    parser.add_argument("--target_expert", required=True, type=int)
    parser.add_argument("--trigger_length", type=int, default=10)
    parser.add_argument("--initial_trigger")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--search_batch_size", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--target_ppl", default="auto")
    parser.add_argument("--reference_text")
    parser.add_argument("--ppl_model_name_or_path", default="gpt2")
    parser.add_argument("--ppl_batch_size", type=int, default=16)
    parser.add_argument("--target_probability_threshold", type=float, default=0.90)
    parser.add_argument("--calibration_file")
    parser.add_argument("--calibration_samples", type=int, default=100)
    parser.add_argument("--prompt")
    parser.add_argument("--calibration_batch_size", type=int, default=8)
    parser.add_argument("--eval_micro_batch_size", type=int, default=8)
    parser.add_argument("--max_mutations_per_candidate", type=int, default=3)
    parser.add_argument("--coordinate_candidates_per_position", type=int, default=4)
    parser.add_argument("--plateau_patience", type=int, default=5)
    parser.add_argument(
        "--plateau_escape_objective",
        choices=["probability", "loss"],
        default="probability",
    )
    parser.add_argument("--trigger_position", choices=["append", "prepend", "fixed_position"], default="append")
    parser.add_argument("--fixed_position", type=int)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", default="outputs/trigger_optimization")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_trigger_optimization_config(argv: Optional[Sequence[str]] = None) -> TriggerOptimizationConfig:
    args = _build_parser().parse_args(argv)
    config = TriggerOptimizationConfig(**vars(args))
    if config.trigger_length <= 0 or config.iterations <= 0:
        raise ValueError("trigger_length and iterations must be > 0.")
    if config.top_k <= 0 or config.search_batch_size < 2:
        raise ValueError("top_k must be > 0 and search_batch_size must be >= 2.")
    if config.calibration_batch_size <= 0 or config.eval_micro_batch_size <= 0:
        raise ValueError("calibration_batch_size and eval_micro_batch_size must be > 0.")
    if config.max_mutations_per_candidate <= 0:
        raise ValueError("max_mutations_per_candidate must be > 0.")
    if config.coordinate_candidates_per_position <= 0 or config.plateau_patience <= 0:
        raise ValueError("coordinate_candidates_per_position and plateau_patience must be > 0.")
    if config.beta < 0 or not 0 <= config.target_probability_threshold <= 1:
        raise ValueError("beta must be >= 0 and target_probability_threshold must be in [0, 1].")
    return config


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_trigger_optimization_config(argv)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    calibration_texts = load_calibration_texts(
        config.calibration_file, config.prompt, config.calibration_samples
    )
    print(f"Loading victim checkpoint: {config.checkpoint_path}", flush=True)
    model, tokenizer, metadata = load_victim_model(
        config.model_name_or_path, config.checkpoint_path, device
    )
    print(f"Checkpoint metadata: {metadata}", flush=True)
    optimizer = RoutingAwareTriggerOptimizer(model, tokenizer, config, calibration_texts, device)
    print(
        f"Target layer={config.target_layer}, target expert={config.target_expert}, "
        f"num_experts={optimizer.num_experts}, device={device}",
        flush=True,
    )
    _routing_best, candidate_pool = optimizer.optimize()
    candidate_texts = [record.trigger_text for record in candidate_pool]
    if config.target_ppl.lower() == "auto":
        reference = [config.reference_text] if config.reference_text is not None else calibration_texts
        all_perplexities = compute_perplexity(
            candidate_texts + reference,
            config.ppl_model_name_or_path,
            device,
            config.ppl_batch_size,
        )
        perplexities = all_perplexities[: len(candidate_texts)]
        target_ppl = sum(all_perplexities[len(candidate_texts) :]) / len(reference)
    else:
        perplexities = compute_perplexity(
            candidate_texts, config.ppl_model_name_or_path, device, config.ppl_batch_size
        )
        target_ppl = float(config.target_ppl)
    print(f"target_ppl = {target_ppl:.6f}\nbeta = {config.beta:.6f}", flush=True)
    final, perplexity, score = select_final_trigger(
        candidate_pool, perplexities, config.beta, target_ppl
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / timestamp
    save_results(
        output_dir,
        config,
        final,
        perplexity,
        score,
        target_ppl,
        candidate_pool,
        optimizer.iteration_log,
    )
    print(
        "Optimized trigger:\n"
        f"    {final.trigger_text}\n\nTrigger token IDs:\n    {final.trigger_ids}\n\n"
        f"Target layer:\n    {config.target_layer}\n\nTarget expert:\n    {config.target_expert}\n\n"
        f"Mean target expert probability:\n    {final.target_expert_probability:.6f}\n\n"
        f"Routing loss:\n    {final.routing_loss:.6f}\n\nPerplexity:\n    {perplexity:.6f}\n\n"
        f"Final score:\n    {score:.6f}\n\nSaved to:\n    {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
