from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from mixfedmoe_fl.client import MoeHandle, collect_moe_handles
from mixfedmoe_fl.data import build_tokenizer
from models.switch_transformers import SwitchTransformersForSequenceClassification


@dataclass(frozen=True)
class IGCGTriggerConfig:
    checkpoint_path: str
    target_layer: str
    target_expert: int
    model_name_or_path: str = "model_ckpt/switch-base-8"
    trigger_length: int = 10
    initial_trigger: Optional[str] = None
    initial_trigger_path: Optional[str] = None
    iterations: int = 50
    candidate_batch_size: int = 256
    top_k: int = 256
    combine_top_p: int = 7
    calibration_file: str = "data/calibration.json"
    calibration_samples: int = 100
    heldout_samples: int = 20
    gradient_batch_size: int = 8
    evaluation_batch_size: int = 16
    trigger_position: str = "append"
    fixed_position: Optional[int] = None
    max_length: int = 256
    margin: float = 0.0
    margin_weight: float = 0.0
    target_probability_threshold: float = 0.97
    target_hit_rate_threshold: float = 0.97
    output_dir: str = "outputs/igcg_trigger_optimization"
    device: str = "auto"
    seed: int = 42
    eps: float = 1e-8


@dataclass(frozen=True)
class PreparedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    trigger_mask: torch.Tensor


@dataclass(frozen=True)
class TriggerMetrics:
    loss: float
    target_probability: float
    target_hit_rate: float
    target_margin: float


@dataclass(frozen=True)
class TriggerRecord:
    iteration: int
    trigger_ids: List[int]
    trigger_text: str
    routing_loss: float
    target_expert_probability: float
    target_expert_hit_rate: float
    target_expert_margin: float
    accepted_mutations: int
    heldout_routing_loss: Optional[float] = None
    heldout_target_probability: Optional[float] = None
    heldout_target_hit_rate: Optional[float] = None
    heldout_target_margin: Optional[float] = None


def heldout_selection_key(metrics: TriggerMetrics) -> Tuple[float, float, float, float]:
    """Rank validation results by actual Top-1 routing, then robustness margin."""
    return (
        metrics.target_hit_rate,
        metrics.target_margin,
        metrics.target_probability,
        -metrics.loss,
    )


def compute_routing_objective(
    router_logits: torch.Tensor,
    trigger_mask: torch.Tensor,
    target_expert: int,
    *,
    margin: float = 0.0,
    margin_weight: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return loss, target probability, Top-1 hit rate, and target logit margin."""
    if router_logits.ndim == 2:
        if router_logits.shape[0] != trigger_mask.numel():
            raise ValueError("Flattened router logits and trigger mask have different token counts.")
        router_logits = router_logits.reshape(*trigger_mask.shape, router_logits.shape[-1])
    if router_logits.ndim != 3 or router_logits.shape[:2] != trigger_mask.shape:
        raise ValueError(
            f"Router logits {tuple(router_logits.shape)} do not match mask {tuple(trigger_mask.shape)}."
        )
    if not 0 <= target_expert < router_logits.shape[-1]:
        raise ValueError("target_expert is outside the router output range.")
    selected = trigger_mask.bool()
    if not bool(selected.any()):
        raise ValueError("The trigger mask has no active token.")

    logits = router_logits.float()[selected]
    probabilities = torch.softmax(logits, dim=-1)
    target_probabilities = probabilities[:, target_expert]
    loss = -torch.log(target_probabilities + eps).mean()

    other_logits = logits.clone()
    other_logits[:, target_expert] = -torch.inf
    logit_margin = logits[:, target_expert] - other_logits.max(dim=-1).values
    if margin_weight:
        loss = loss + margin_weight * torch.relu(margin - logit_margin).mean()

    hit_rate = logits.argmax(dim=-1).eq(target_expert).float().mean()
    return loss, target_probabilities.mean(), hit_rate, logit_margin.mean()


def top_k_tokens_from_gradient(
    current_ids: Sequence[int],
    position: int,
    gradient: torch.Tensor,
    embedding_matrix: torch.Tensor,
    top_k: int,
    excluded_ids: Iterable[int] = (),
) -> List[int]:
    """Rank replacement tokens by the first-order loss change used by GCG."""
    if not 0 <= position < len(current_ids):
        raise IndexError("Trigger position is out of range.")
    row_gradient = gradient[position] if gradient.ndim == 2 else gradient
    if row_gradient.ndim != 1 or embedding_matrix.ndim != 2:
        raise ValueError("Expected a vector gradient and a [vocab, hidden] embedding matrix.")
    scores = embedding_matrix.float().mv(row_gradient.float())
    forbidden = {int(token_id) for token_id in excluded_ids}
    forbidden.add(int(current_ids[position]))
    valid_forbidden = [idx for idx in forbidden if 0 <= idx < scores.numel()]
    if valid_forbidden:
        scores[torch.tensor(valid_forbidden, device=scores.device)] = torch.inf
    count = min(int(top_k), int(torch.isfinite(scores).sum().item()))
    if count <= 0:
        raise ValueError(f"No valid replacement token remains at trigger position {position}.")
    return torch.topk(scores, k=count, largest=False).indices.cpu().tolist()


def sample_single_coordinate_candidates(
    current_ids: Sequence[int],
    candidates_by_position: Sequence[Sequence[int]],
    batch_size: int,
    rng: random.Random,
) -> List[List[int]]:
    """Sample one-token mutations across coordinates as in the original GCG."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if len(candidates_by_position) != len(current_ids):
        raise ValueError("A candidate list is required for every trigger position.")
    if any(not choices for choices in candidates_by_position):
        raise ValueError("Every trigger position must have at least one replacement token.")
    generated: List[List[int]] = []
    trigger_length = len(current_ids)
    for candidate_index in range(batch_size):
        # Integer equivalent of arange(0, length, length / batch_size).
        position = min((candidate_index * trigger_length) // batch_size, trigger_length - 1)
        choices = candidates_by_position[position]
        candidate = list(current_ids)
        candidate[position] = int(rng.choice(choices))
        generated.append(candidate)
    return generated


def filter_roundtrip_candidates(
    tokenizer: PreTrainedTokenizerBase,
    current_ids: Sequence[int],
    candidates: Sequence[Sequence[int]],
    *,
    require_single_coordinate: bool,
) -> List[List[int]]:
    """Canonicalize candidates through text and reject unstable tokenizations."""
    expected_length = len(current_ids)
    filtered: List[List[int]] = []
    seen = set()
    for candidate in candidates:
        text = tokenizer.decode(
            list(candidate),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        encoded = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(encoded) != expected_length or encoded == list(current_ids):
            continue
        changed = sum(old != new for old, new in zip(current_ids, encoded))
        if require_single_coordinate and changed != 1:
            continue
        key = tuple(int(token_id) for token_id in encoded)
        if key not in seen:
            seen.add(key)
            filtered.append(list(key))
    return filtered


def combine_top_single_coordinate_candidates(
    current_ids: Sequence[int],
    candidates: Sequence[Sequence[int]],
    losses: Sequence[float],
    top_p: int,
) -> List[List[int]]:
    """I-GCG cumulative merge of the best round-trip-stable candidates."""
    if len(candidates) != len(losses):
        raise ValueError("candidates and losses must have equal length.")
    if top_p <= 0:
        raise ValueError("top_p must be > 0.")
    ranked = sorted(range(len(candidates)), key=lambda idx: losses[idx])[:top_p]
    merged = list(current_ids)
    output: List[List[int]] = []
    seen = set()
    for candidate_index in ranked:
        candidate = list(candidates[candidate_index])
        changed = [idx for idx, (old, new) in enumerate(zip(current_ids, candidate)) if old != new]
        if not changed:
            continue
        # A raw proposal changes one token, but decode/encode can alter several
        # token IDs at a word boundary. The original I-GCG merges every changed
        # coordinate after this round trip, rather than rejecting the proposal.
        for position in changed:
            merged[position] = candidate[position]
        key = tuple(merged)
        if key != tuple(current_ids) and key not in seen:
            seen.add(key)
            output.append(list(merged))
    return output


def select_low_frequency_target(profile_path: str, layer_id: str) -> int:
    """Select the first ranked low-frequency expert from an LFE profile."""
    payload = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    try:
        experts = payload["layers"][layer_id]["top_low_frequency_experts"]
        expert = experts[0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Profile '{profile_path}' has no low-frequency ranking for layer '{layer_id}'."
        ) from exc
    if isinstance(expert, bool) or not isinstance(expert, int) or expert < 0:
        raise ValueError("The selected low-frequency expert ID is invalid.")
    return expert


def _load_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        return dict(payload["state_dict"]), {k: v for k, v in payload.items() if k != "state_dict"}
    if isinstance(payload, dict) and payload and all(torch.is_tensor(v) for v in payload.values()):
        return dict(payload), {}
    raise ValueError("Checkpoint must be a state_dict or contain a 'state_dict' object.")


def load_victim_model(
    model_name_or_path: str, checkpoint_path: str, device: torch.device
) -> Tuple[SwitchTransformersForSequenceClassification, PreTrainedTokenizerBase, Dict[str, Any]]:
    state, metadata = _load_checkpoint(checkpoint_path)
    classifier = state.get("classification_head.out_proj.weight")
    num_labels = int(classifier.shape[0]) if classifier is not None else 2
    tokenizer = build_tokenizer(model_name_or_path, max_length=1_000_000)
    model = SwitchTransformersForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
        tie_word_embeddings=False,
        local_files_only=True,
    )
    incompatible = model.load_state_dict(state, strict=False)
    missing = [name for name in incompatible.missing_keys if not name.endswith("position_ids")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing[:10]}, unexpected={incompatible.unexpected_keys[:10]}."
        )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = (
        model.config.decoder_start_token_id
        if model.config.decoder_start_token_id is not None
        else model.config.pad_token_id
    )
    model.config.use_cache = False
    model.eval().requires_grad_(False).to(device)
    return model, tokenizer, metadata


def resolve_moe_layer(model: nn.Module, layer_id: str) -> MoeHandle:
    matches = [handle for handle in collect_moe_handles(model) if handle.layer_id == layer_id]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(handle.layer_id for handle in collect_moe_handles(model))
    raise ValueError(f"Unknown MoE layer '{layer_id}'. Available layers: {available or '<none>'}.")


def get_router(handle: MoeHandle) -> nn.Linear:
    router = getattr(getattr(handle.mlp_module, "router", None), "classifier", None)
    if not isinstance(router, nn.Linear):
        raise RuntimeError(f"Layer '{handle.layer_id}' does not expose a linear router classifier.")
    return router


def build_triggered_batch(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    trigger_ids: Sequence[int],
    max_length: int,
    trigger_position: str,
    fixed_position: Optional[int],
) -> PreparedBatch:
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    if eos_id is None or pad_id is None:
        raise ValueError("Tokenizer must define EOS and padding token IDs.")
    if max_length <= len(trigger_ids) + 1:
        raise ValueError("max_length leaves no room for prompt tokens and EOS.")
    rows, attentions, trigger_masks = [], [], []
    budget = max_length - len(trigger_ids) - 1
    for prompt in prompts:
        prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        prompt_ids = [token for token in prompt_ids if token != eos_id][:budget]
        if trigger_position == "append":
            insertion = len(prompt_ids)
        elif trigger_position == "prepend":
            insertion = 0
        elif trigger_position == "fixed_position" and fixed_position is not None and fixed_position >= 0:
            insertion = min(fixed_position, len(prompt_ids))
        else:
            raise ValueError("fixed_position >= 0 is required when using fixed_position mode.")
        ids = prompt_ids[:insertion] + list(trigger_ids) + prompt_ids[insertion:] + [eos_id]
        mask = [False] * insertion + [True] * len(trigger_ids)
        mask.extend([False] * (len(ids) - len(mask)))
        rows.append(ids)
        attentions.append([1] * len(ids))
        trigger_masks.append(mask)
    width = max(map(len, rows))
    for ids, attention, mask in zip(rows, attentions, trigger_masks):
        padding = width - len(ids)
        ids.extend([pad_id] * padding)
        attention.extend([0] * padding)
        mask.extend([False] * padding)
    return PreparedBatch(
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(attentions, dtype=torch.long),
        torch.tensor(trigger_masks, dtype=torch.bool),
    )


def _stack_trigger_mask(mask: torch.Tensor, stack_name: str) -> torch.Tensor:
    if stack_name == "encoder":
        return mask
    if stack_name == "decoder":
        shifted = torch.zeros_like(mask)
        shifted[:, 1:] = mask[:, :-1]
        return shifted
    raise ValueError(f"Unsupported stack '{stack_name}'.")


def load_calibration_texts(path: str, limit: int) -> List[str]:
    calibration_path = Path(path)
    if not calibration_path.is_file() and not calibration_path.is_absolute():
        repository_relative = Path(__file__).resolve().parent.parent / calibration_path
        if repository_relative.is_file():
            calibration_path = repository_relative
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: '{path}'.")
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Calibration file must contain a JSON list.")
    texts = [row.get("text") for row in payload if isinstance(row, dict)]
    if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Calibration rows must contain non-empty 'text' strings.")
    return [str(text) for text in texts[:limit]]


def split_calibration_texts(
    texts: Sequence[str], heldout_samples: int, seed: int
) -> Tuple[List[str], List[str]]:
    """Deterministically split calibration contexts into search and held-out sets."""
    if heldout_samples <= 0:
        raise ValueError("heldout_samples must be > 0.")
    if len(texts) <= heldout_samples:
        raise ValueError(
            f"Calibration data has {len(texts)} samples, but heldout_samples={heldout_samples} "
            "must leave at least one optimization sample."
        )
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    heldout_indices = set(indices[:heldout_samples])
    optimization = [text for index, text in enumerate(texts) if index not in heldout_indices]
    heldout = [text for index, text in enumerate(texts) if index in heldout_indices]
    return optimization, heldout


class IGCGTriggerOptimizer:
    def __init__(
        self,
        model: SwitchTransformersForSequenceClassification,
        tokenizer: PreTrainedTokenizerBase,
        config: IGCGTriggerConfig,
        calibration_texts: Sequence[str],
        heldout_texts: Sequence[str],
        device: torch.device,
    ) -> None:
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.texts = list(calibration_texts)
        self.heldout_texts = list(heldout_texts)
        self.device = device
        if not self.texts or not self.heldout_texts:
            raise ValueError("Both optimization and held-out calibration sets must be non-empty.")
        self.handle = resolve_moe_layer(model, config.target_layer)
        self.router = get_router(self.handle)
        if not 0 <= config.target_expert < self.router.out_features:
            raise ValueError("target_expert is invalid for the selected layer.")
        self.embedding: nn.Embedding = getattr(model, self.handle.stack_name).embed_tokens
        self.rng = random.Random(config.seed)
        self.records: List[TriggerRecord] = []
        self.heldout_evaluations: List[Dict[str, Any]] = []

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
            batch.input_ids.to(self.device),
            batch.attention_mask.to(self.device),
            _stack_trigger_mask(batch.trigger_mask, self.handle.stack_name).to(self.device),
        )

    def _forward_router(
        self, batch: PreparedBatch, capture_embedding: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        router_outputs: List[torch.Tensor] = []
        embedding_outputs: List[torch.Tensor] = []

        def router_hook(_module: nn.Module, _args: Any, output: torch.Tensor) -> None:
            router_outputs.append(output)

        def embedding_hook(_module: nn.Module, _args: Any, output: torch.Tensor) -> torch.Tensor:
            leaf = output.detach().requires_grad_(True)
            leaf.retain_grad()
            embedding_outputs.append(leaf)
            return leaf

        hooks = [self.router.register_forward_hook(router_hook)]
        if capture_embedding:
            hooks.append(self.embedding.register_forward_hook(embedding_hook))
        try:
            self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                output_router_logits=False,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for hook in hooks:
                hook.remove()
        if len(router_outputs) != 1:
            raise RuntimeError(f"Expected one router call, observed {len(router_outputs)}.")
        selected_embedding = None
        if capture_embedding:
            # Switch/T5 commonly shares the embedding module. Its first call is
            # encoder input and its second call is shifted decoder input.
            index = 0 if self.handle.stack_name == "encoder" else -1
            if not embedding_outputs:
                raise RuntimeError("The embedding hook did not run.")
            selected_embedding = embedding_outputs[index]
        return router_outputs[0], selected_embedding

    def _objective(self, logits: torch.Tensor, mask: torch.Tensor):
        return compute_routing_objective(
            logits,
            mask,
            self.config.target_expert,
            margin=self.config.margin,
            margin_weight=self.config.margin_weight,
            eps=self.config.eps,
        )

    def evaluate(
        self, trigger_ids: Sequence[int], texts: Optional[Sequence[str]] = None
    ) -> TriggerMetrics:
        evaluation_texts = list(self.texts if texts is None else texts)
        if not evaluation_texts:
            raise ValueError("Evaluation texts must not be empty.")
        loss_sum = probability_sum = hit_sum = 0.0
        margin_sum = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(evaluation_texts), self.config.evaluation_batch_size):
                prompts = evaluation_texts[start : start + self.config.evaluation_batch_size]
                batch = self._prepare(prompts, trigger_ids)
                logits, _ = self._forward_router(batch, False)
                loss, probability, hit_rate, target_margin = self._objective(
                    logits, batch.trigger_mask
                )
                size = len(prompts)
                loss_sum += float(loss) * size
                probability_sum += float(probability) * size
                hit_sum += float(hit_rate) * size
                margin_sum += float(target_margin) * size
                count += size
        return TriggerMetrics(
            loss_sum / count,
            probability_sum / count,
            hit_sum / count,
            margin_sum / count,
        )

    def gradient(self, trigger_ids: Sequence[int]) -> torch.Tensor:
        total = len(self.texts)
        result = torch.zeros(
            len(trigger_ids), self.embedding.embedding_dim, device=self.device, dtype=torch.float32
        )
        for start in range(0, total, self.config.gradient_batch_size):
            prompts = self.texts[start : start + self.config.gradient_batch_size]
            batch = self._prepare(prompts, trigger_ids)
            logits, embedding_output = self._forward_router(batch, True)
            if embedding_output is None:
                raise RuntimeError("No differentiable embedding output was captured.")
            loss, _, _, _ = self._objective(logits, batch.trigger_mask)
            (loss * (len(prompts) / total)).backward()
            if embedding_output.grad is None:
                raise RuntimeError("Routing objective did not reach the input embeddings.")
            for row in range(len(prompts)):
                row_gradient = embedding_output.grad[row][batch.trigger_mask[row]]
                if row_gradient.shape != result.shape:
                    raise RuntimeError(
                        f"Trigger gradient shape {tuple(row_gradient.shape)} != {tuple(result.shape)}."
                    )
                result += row_gradient
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Non-finite trigger gradient.")
        return result

    def evaluate_many(self, candidates: Sequence[Sequence[int]]) -> List[TriggerMetrics]:
        if not candidates:
            raise ValueError("At least one candidate trigger is required.")
        loss_sums = [0.0] * len(candidates)
        probability_sums = [0.0] * len(candidates)
        hit_sums = [0.0] * len(candidates)
        margin_sums = [0.0] * len(candidates)
        counts = [0] * len(candidates)
        pairs = [(index, text) for index in range(len(candidates)) for text in self.texts]
        with torch.no_grad():
            for start in range(0, len(pairs), self.config.evaluation_batch_size):
                chunk = pairs[start : start + self.config.evaluation_batch_size]
                prepared = [self._prepare([text], candidates[index]) for index, text in chunk]
                width = max(item.input_ids.shape[1] for item in prepared)
                pad_id = int(self.tokenizer.pad_token_id)
                ids = torch.full(
                    (len(prepared), width), pad_id, dtype=torch.long, device=self.device
                )
                attention = torch.zeros(
                    (len(prepared), width), dtype=torch.long, device=self.device
                )
                masks = torch.zeros(
                    (len(prepared), width), dtype=torch.bool, device=self.device
                )
                for row, item in enumerate(prepared):
                    length = item.input_ids.shape[1]
                    ids[row, :length] = item.input_ids[0]
                    attention[row, :length] = item.attention_mask[0]
                    masks[row, :length] = item.trigger_mask[0]
                logits, _ = self._forward_router(PreparedBatch(ids, attention, masks), False)
                logits = logits.reshape(len(chunk), width, self.router.out_features)
                for row, (candidate_index, _text) in enumerate(chunk):
                    loss, probability, hit_rate, target_margin = self._objective(
                        logits[row : row + 1], masks[row : row + 1]
                    )
                    loss_sums[candidate_index] += float(loss)
                    probability_sums[candidate_index] += float(probability)
                    hit_sums[candidate_index] += float(hit_rate)
                    margin_sums[candidate_index] += float(target_margin)
                    counts[candidate_index] += 1
        return [
            TriggerMetrics(
                loss_sums[index] / counts[index],
                probability_sums[index] / counts[index],
                hit_sums[index] / counts[index],
                margin_sums[index] / counts[index],
            )
            for index in range(len(candidates))
        ]

    def _stable_text_ids(self, text: str) -> List[int]:
        """Return the fixed point of encode(decode(ids)) for a trigger string."""
        ids = list(self.tokenizer(text.strip(), add_special_tokens=False)["input_ids"])
        if not ids:
            raise ValueError("Initial trigger tokenized to an empty sequence.")
        for _ in range(8):
            decoded = self.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            encoded = list(self.tokenizer(decoded, add_special_tokens=False)["input_ids"])
            if encoded == ids:
                return [int(token_id) for token_id in ids]
            if not encoded:
                raise ValueError("Initial trigger became empty after tokenizer round trip.")
            ids = encoded
        raise RuntimeError("Initial trigger tokenization did not reach a stable round-trip encoding.")

    def _initial_ids(self) -> List[int]:
        initial = self.config.initial_trigger
        if self.config.initial_trigger_path:
            payload = json.loads(Path(self.config.initial_trigger_path).read_text(encoding="utf-8"))
            initial = payload.get("best_trigger", {}).get("trigger")
            if not isinstance(initial, str) or not initial.strip():
                raise ValueError("initial_trigger_path has no best_trigger.trigger string.")
        if initial:
            ids = self._stable_text_ids(initial)
            if len(ids) != self.config.trigger_length:
                print(
                    f"Initial trigger has {len(ids)} stable tokens; configured trigger_length="
                    f"{self.config.trigger_length}. Preserving the supplied trigger's stable length.",
                    flush=True,
                )
            return ids

        # Match the original I-GCG initialization: spaced exclamation marks as
        # text. Repeating the token ID for a bare '!' decodes to '!!!!', which
        # SentencePiece can re-encode into a shorter, unstable sequence.
        exact: Optional[List[int]] = None
        closest: Optional[List[int]] = None
        for count in range(1, max(4 * self.config.trigger_length, 64) + 1):
            candidate = self._stable_text_ids(" ".join(["!"] * count))
            if closest is None or abs(len(candidate) - self.config.trigger_length) < abs(
                len(closest) - self.config.trigger_length
            ):
                closest = candidate
            if len(candidate) == self.config.trigger_length:
                exact = candidate
                break
        ids = exact if exact is not None else closest
        if ids is None:
            raise RuntimeError("Could not construct a stable exclamation-mark trigger.")
        if len(ids) != self.config.trigger_length:
            print(
                f"Tokenizer cannot represent a stable spaced-'!' trigger with exactly "
                f"{self.config.trigger_length} tokens; using {len(ids)} tokens.",
                flush=True,
            )
        return ids

    def _record(self, iteration: int, ids: Sequence[int], metrics: TriggerMetrics, mutations: int):
        record = TriggerRecord(
            iteration,
            list(ids),
            self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip(),
            metrics.loss,
            metrics.target_probability,
            metrics.target_hit_rate,
            metrics.target_margin,
            mutations,
        )
        self.records.append(record)
        print(
            f"[I-GCG {iteration:03d}] loss={metrics.loss:.6f} "
            f"p(target)={metrics.target_probability:.4f} hit_rate={metrics.target_hit_rate:.4f} "
            f"margin={metrics.target_margin:.4f} "
            f"mutations={mutations} trigger={record.trigger_text!r}",
            flush=True,
        )
        return record

    def optimize(self) -> TriggerRecord:
        current = self._initial_ids()
        current_metrics = self.evaluate(current)
        best = self._record(0, current, current_metrics, 0)
        excluded = set(self.tokenizer.all_special_ids)
        for iteration in range(1, self.config.iterations + 1):
            gradient = self.gradient(current)
            per_position = [
                top_k_tokens_from_gradient(
                    current, position, gradient, self.embedding.weight.detach(), self.config.top_k, excluded
                )
                for position in range(len(current))
            ]
            sampled_singles = sample_single_coordinate_candidates(
                current, per_position, self.config.candidate_batch_size, self.rng
            )
            singles = filter_roundtrip_candidates(
                self.tokenizer,
                current,
                sampled_singles,
                require_single_coordinate=False,
            )
            if not singles:
                raise RuntimeError(
                    "All sampled candidates changed token length after decode/encode. "
                    "Increase candidate_batch_size or top_k."
                )
            single_metrics = self.evaluate_many(singles)
            combined = combine_top_single_coordinate_candidates(
                current,
                singles,
                [metrics.loss for metrics in single_metrics],
                self.config.combine_top_p,
            )
            combined = filter_roundtrip_candidates(
                self.tokenizer,
                current,
                combined,
                require_single_coordinate=False,
            )
            if not combined:
                raise RuntimeError(
                    "All combined I-GCG candidates changed token length after decode/encode."
                )
            combined_metrics = self.evaluate_many(combined)
            selected = min(range(len(combined)), key=lambda idx: combined_metrics[idx].loss)
            candidate, candidate_metrics = combined[selected], combined_metrics[selected]
            mutations = sum(left != right for left, right in zip(current, candidate))
            current, current_metrics = list(candidate), candidate_metrics
            record = self._record(iteration, current, current_metrics, mutations)
            if record.routing_loss < best.routing_loss:
                best = record
            if (
                current_metrics.target_probability >= self.config.target_probability_threshold
                and current_metrics.target_hit_rate >= self.config.target_hit_rate_threshold
            ):
                heldout_metrics = self.evaluate(record.trigger_ids, self.heldout_texts)
                if (
                    heldout_metrics.target_probability
                    >= self.config.target_probability_threshold
                    and heldout_metrics.target_hit_rate
                    >= self.config.target_hit_rate_threshold
                ):
                    break
        unique_records: Dict[Tuple[int, ...], TriggerRecord] = {}
        for candidate_record in self.records:
            key = tuple(candidate_record.trigger_ids)
            previous = unique_records.get(key)
            if previous is None or candidate_record.routing_loss < previous.routing_loss:
                unique_records[key] = candidate_record

        evaluated: List[Tuple[TriggerRecord, TriggerMetrics]] = []
        for candidate_record in unique_records.values():
            heldout_metrics = self.evaluate(candidate_record.trigger_ids, self.heldout_texts)
            evaluated.append((candidate_record, heldout_metrics))
            self.heldout_evaluations.append(
                {
                    "iteration": candidate_record.iteration,
                    "trigger_ids": candidate_record.trigger_ids,
                    "trigger_text": candidate_record.trigger_text,
                    "routing_loss": heldout_metrics.loss,
                    "target_probability": heldout_metrics.target_probability,
                    "target_hit_rate": heldout_metrics.target_hit_rate,
                    "target_margin": heldout_metrics.target_margin,
                }
            )
        best_record, heldout = max(
            evaluated,
            key=lambda item: heldout_selection_key(item[1]),
        )
        selected = replace(
            best_record,
            heldout_routing_loss=heldout.loss,
            heldout_target_probability=heldout.target_probability,
            heldout_target_hit_rate=heldout.target_hit_rate,
            heldout_target_margin=heldout.target_margin,
        )
        print(
            "[I-GCG held-out selection] "
            f"hit_rate={heldout.target_hit_rate:.4f} margin={heldout.target_margin:.4f} "
            f"p(target)={heldout.target_probability:.4f} loss={heldout.loss:.6f} "
            f"trigger={selected.trigger_text!r}",
            flush=True,
        )
        return selected


def save_results(
    path: Path,
    config: IGCGTriggerConfig,
    best: TriggerRecord,
    history: Sequence[TriggerRecord],
    heldout_evaluations: Sequence[Dict[str, Any]],
):
    path.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "routing_i_gcg",
        "target": {"layer_id": config.target_layer, "expert_id": config.target_expert},
        "best_trigger": {
            "trigger": best.trigger_text,
            "trigger_token_ids": best.trigger_ids,
            "routing_loss": best.routing_loss,
            "mean_target_probability": best.target_expert_probability,
            "target_hit_rate": best.target_expert_hit_rate,
            "mean_target_margin": best.target_expert_margin,
            "heldout_routing_loss": best.heldout_routing_loss,
            "heldout_mean_target_probability": best.heldout_target_probability,
            "heldout_target_hit_rate": best.heldout_target_hit_rate,
            "heldout_mean_target_margin": best.heldout_target_margin,
        },
        "config": asdict(config),
    }
    (path / "selected_triggers.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (path / "optimization_history.json").write_text(
        json.dumps([asdict(record) for record in history], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (path / "heldout_evaluations.json").write_text(
        json.dumps(list(heldout_evaluations), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _device(value: str) -> torch.device:
    resolved = torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="I-GCG-style MoE routing trigger optimization.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--model_name_or_path", default="model_ckpt/switch-base-8")
    parser.add_argument("--target_layer", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target_expert", type=int)
    target.add_argument("--lfe_profile")
    parser.add_argument("--trigger_length", type=int, default=10)
    parser.add_argument("--initial_trigger")
    parser.add_argument("--initial_trigger_path")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--candidate_batch_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=256)
    parser.add_argument("--combine_top_p", type=int, default=7)
    parser.add_argument("--calibration_file", default="data/calibration.json")
    parser.add_argument("--calibration_samples", type=int, default=100)
    parser.add_argument("--heldout_samples", type=int, default=20)
    parser.add_argument("--gradient_batch_size", type=int, default=8)
    parser.add_argument("--evaluation_batch_size", type=int, default=16)
    parser.add_argument("--trigger_position", choices=["append", "prepend", "fixed_position"], default="append")
    parser.add_argument("--fixed_position", type=int)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--margin_weight", type=float, default=0.0)
    parser.add_argument("--target_probability_threshold", type=float, default=0.97)
    parser.add_argument("--target_hit_rate_threshold", type=float, default=0.97)
    parser.add_argument("--output_dir", default="outputs/igcg_trigger_optimization")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_config(argv: Optional[Sequence[str]] = None) -> IGCGTriggerConfig:
    args = _parser().parse_args(argv)
    values = vars(args)
    profile = values.pop("lfe_profile")
    if values["target_expert"] is None:
        values["target_expert"] = select_low_frequency_target(profile, values["target_layer"])
    config = IGCGTriggerConfig(**values)
    positive = (
        config.trigger_length,
        config.iterations,
        config.candidate_batch_size,
        config.top_k,
        config.combine_top_p,
        config.calibration_samples,
        config.heldout_samples,
        config.gradient_batch_size,
        config.evaluation_batch_size,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Trigger/search/batch parameters must all be > 0.")
    if config.combine_top_p > config.candidate_batch_size:
        raise ValueError("combine_top_p cannot exceed candidate_batch_size.")
    if config.margin_weight < 0:
        raise ValueError("margin_weight must be >= 0.")
    if config.heldout_samples >= config.calibration_samples:
        raise ValueError("heldout_samples must be smaller than calibration_samples.")
    for value in (config.target_probability_threshold, config.target_hit_rate_threshold):
        if not 0 <= value <= 1:
            raise ValueError("Stopping thresholds must be in [0, 1].")
    return config


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_config(argv)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device(config.device)
    calibration_texts = load_calibration_texts(
        config.calibration_file, config.calibration_samples
    )
    optimization_texts, heldout_texts = split_calibration_texts(
        calibration_texts, config.heldout_samples, config.seed
    )
    model, tokenizer, metadata = load_victim_model(
        config.model_name_or_path, config.checkpoint_path, device
    )
    print(f"Loaded checkpoint metadata: {metadata}", flush=True)
    print(
        f"Calibration split: optimization={len(optimization_texts)}, "
        f"held_out={len(heldout_texts)}",
        flush=True,
    )
    optimizer = IGCGTriggerOptimizer(
        model, tokenizer, config, optimization_texts, heldout_texts, device
    )
    best = optimizer.optimize()
    output = Path(config.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(output, config, best, optimizer.records, optimizer.heldout_evaluations)
    print(f"Best routing trigger saved to {output / 'selected_triggers.json'}", flush=True)


if __name__ == "__main__":
    main()
