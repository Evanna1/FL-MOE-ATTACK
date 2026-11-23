"""Measure ASR under fixed-routing expert contribution ablations.

This experiment is intentionally an inference-only counterfactual.  It keeps
the router's original top-k indices fixed and changes only the selected expert
weights at one OLMoE/Granite layer:

* ``full``: the unmodified checkpoint;
* ``remove_target``: zero the target expert's contribution;
* ``remove_expert_<id>``: zero one selected high-frequency expert;
* ``keep_target_only``: keep only the target expert's contribution.

By default the remaining weights are renormalized after an ablation.  Thus the
experiment answers "how much does this selected expert contribute?" rather
than changing the router itself and selecting a different top-k set.  This is
useful for separating routing from expert-value dilution, but it is not a
full re-routing intervention. Both OLMoE and Granite-MoE are supported;
Switch is intentionally not included because its top-1 dispatch interface is
different.

Example::

    python -u experiments/moe_expert_ablation_asr.py \
      --checkpoint_path results/olmoe_mix_probe/.../checkpoints/round_0010.pt \
      --selected_triggers_path outputs/igcg_trigger_optimization/.../selected_triggers.json \
      --target_label 0 --num_high_frequency_experts 7

The script writes ``ablation_results.json`` and ``ablation_results.csv`` in
``--output_dir``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset

# Allow running as ``python experiments/...`` from the mixfed-moe directory or
# from its repository parent.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mixfedmoe_fl.attack import load_attack_checkpoint, load_selected_trigger  # noqa: E402
from mixfedmoe_fl.client import collect_moe_handles, get_handle_num_experts  # noqa: E402
from mixfedmoe_fl.config import SUPPORTED_DATASET_NAME_TO_HF  # noqa: E402
from mixfedmoe_fl.data import build_tokenizer  # noqa: E402
from mixfedmoe_fl.model_factory import build_sequence_classification_model  # noqa: E402


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    trigger_mask: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class RouteSummary:
    selection_rate: Tuple[float, ...]
    mean_gate_mass: Tuple[float, ...]
    trigger_token_count: int


@dataclass(frozen=True)
class VariantResult:
    name: str
    ablated_experts: Tuple[int, ...]
    kept_experts: Tuple[int, ...]
    asr: float
    target_hits: int
    total: int
    clean_accuracy: float
    clean_correct: int
    clean_total: int
    clean_target_rate: float
    target_label: int
    target_layer: str
    target_expert: int
    renormalize: bool


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_file():
        return path.resolve()
    repository_relative = REPO_ROOT / path
    if repository_relative.is_file():
        return repository_relative.resolve()
    raise FileNotFoundError(f"File does not exist: {raw}")


def _load_trigger_metadata(args: argparse.Namespace) -> Tuple[str, str, int, Tuple[int, ...]]:
    trigger = args.trigger
    target_layer = args.target_layer
    target_expert = args.target_expert
    trigger_ids: Tuple[int, ...] = tuple(args.trigger_token_ids or ())
    if args.selected_triggers_path:
        selected = load_selected_trigger(str(_resolve_path(args.selected_triggers_path)))
        trigger = trigger or selected.trigger
        target_layer = target_layer or selected.target_layer
        if target_expert is None:
            target_expert = selected.target_expert
        if not trigger_ids:
            trigger_ids = tuple(selected.trigger_token_ids)
    if not trigger:
        raise ValueError("Provide --selected_triggers_path or --trigger.")
    if not target_layer:
        raise ValueError("Provide --target_layer or include it in --selected_triggers_path.")
    if target_expert is None or target_expert < 0:
        raise ValueError("Provide a non-negative --target_expert or include it in the selected trigger file.")
    return str(trigger), str(target_layer), int(target_expert), trigger_ids


def _build_model(
    model_name_or_path: str,
    checkpoint_path: str,
    tokenizer: Any,
    device: torch.device,
    model_dtype: str,
) -> torch.nn.Module:
    state, metadata = load_attack_checkpoint(str(_resolve_path(checkpoint_path)))
    classifier = state.get("classification_head.out_proj.weight")
    num_labels = int(classifier.shape[0]) if isinstance(classifier, torch.Tensor) else 2
    model = build_sequence_classification_model(
        model_name_or_path,
        num_labels=num_labels,
        id2label={index: str(index) for index in range(num_labels)},
        label2id={str(index): index for index in range(num_labels)},
        tokenizer=tokenizer,
        model_backend="auto",
        model_dtype=model_dtype,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    incompatible = model.load_state_dict(state, strict=False)
    missing = [name for name in incompatible.missing_keys if not name.endswith("position_ids")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    model.config.use_cache = False
    model.eval().requires_grad_(False).to(device)
    print(f"Loaded checkpoint metadata: {metadata}", flush=True)
    return model


def _make_batches(
    rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    trigger: str,
    trigger_ids: Sequence[int],
    max_length: int,
    batch_size: int,
    device: torch.device,
    use_trigger: bool = True,
) -> List[Batch]:
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if eos_id is None or pad_id is None:
        raise ValueError("Tokenizer must define eos_token_id and pad_token_id.")
    ids_for_trigger = list(trigger_ids) if use_trigger else []
    if use_trigger and not ids_for_trigger:
        ids_for_trigger = list(tokenizer(trigger, add_special_tokens=False)["input_ids"])
    if use_trigger and not ids_for_trigger:
        raise ValueError("Trigger tokenization produced no tokens.")
    if use_trigger and len(ids_for_trigger) + 1 >= max_length:
        raise ValueError("max_length must leave room for prompt tokens and EOS.")

    tokenized: List[Tuple[List[int], int]] = []
    prompt_budget = max_length - len(ids_for_trigger) - 1
    for row in rows:
        prompt_ids = list(
            tokenizer(
                str(row["text"]),
                add_special_tokens=False,
                truncation=True,
                max_length=prompt_budget,
            )["input_ids"]
        )
        prompt_ids = [token_id for token_id in prompt_ids if token_id != eos_id]
        sequence = prompt_ids[:prompt_budget] + ids_for_trigger + [eos_id]
        tokenized.append((sequence, int(row["label"])))

    batches: List[Batch] = []
    for start in range(0, len(tokenized), batch_size):
        chunk = tokenized[start : start + batch_size]
        width = max(len(ids) for ids, _ in chunk)
        input_ids = torch.full((len(chunk), width), int(pad_id), dtype=torch.long)
        attention = torch.zeros((len(chunk), width), dtype=torch.long)
        trigger_mask = torch.zeros((len(chunk), width), dtype=torch.bool)
        labels = torch.tensor([label for _, label in chunk], dtype=torch.long)
        for row_index, (sequence, _label) in enumerate(chunk):
            length = len(sequence)
            input_ids[row_index, :length] = torch.tensor(sequence, dtype=torch.long)
            attention[row_index, :length] = 1
            if ids_for_trigger:
                trigger_start = length - 1 - len(ids_for_trigger)
                trigger_mask[row_index, trigger_start : trigger_start + len(ids_for_trigger)] = True
        batches.append(
            Batch(
                input_ids=input_ids.to(device),
                attention_mask=attention.to(device),
                trigger_mask=trigger_mask.to(device),
                labels=labels.to(device),
            )
        )
    return batches


class ExpertContributionIntervention:
    """Collect routes and optionally alter OLMoE selected expert weights."""

    def __init__(
        self,
        experts_module: torch.nn.Module,
        mode: str,
        expert_ids: Iterable[int],
        renormalize: bool,
        num_experts: Optional[int] = None,
    ) -> None:
        self.mode = mode
        self.expert_ids = tuple(sorted({int(value) for value in expert_ids}))
        self.renormalize = bool(renormalize)
        self.expected_num_experts = int(num_experts) if num_experts is not None else None
        self.trigger_mask: Optional[torch.Tensor] = None
        self.num_experts: Optional[int] = None
        self.selection_counts: Optional[torch.Tensor] = None
        self.mass_sums: Optional[torch.Tensor] = None
        self.hook = experts_module.register_forward_pre_hook(self._pre_hook)

    def set_trigger_mask(self, trigger_mask: torch.Tensor) -> None:
        self.trigger_mask = trigger_mask.detach().reshape(-1).bool()

    def _ensure_stats(self, num_experts: int, device: torch.device) -> None:
        if self.expected_num_experts is not None:
            num_experts = self.expected_num_experts
        if self.num_experts is None:
            self.num_experts = int(num_experts)
            self.selection_counts = torch.zeros(num_experts, dtype=torch.float64, device=device)
            self.mass_sums = torch.zeros(num_experts, dtype=torch.float64, device=device)
        elif self.num_experts != int(num_experts):
            raise RuntimeError("Router expert count changed during evaluation.")

    def _pre_hook(self, _module: torch.nn.Module, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
        if len(args) < 3:
            raise RuntimeError("Expected OlmoE experts inputs (hidden_states, top_k_index, top_k_weights).")
        hidden_states, top_k_index, top_k_weights = args[:3]
        if top_k_index.ndim != 2 or top_k_weights.ndim != 2:
            raise RuntimeError(
                f"Expected flattened OLMoE top-k tensors, got indices={tuple(top_k_index.shape)} "
                f"weights={tuple(top_k_weights.shape)}"
            )
        self._ensure_stats(int(top_k_index.max().item()) + 1 if top_k_index.numel() else 0, top_k_index.device)
        if self.trigger_mask is not None:
            if self.trigger_mask.numel() != top_k_index.shape[0]:
                raise RuntimeError(
                    f"Trigger mask has {self.trigger_mask.numel()} tokens, but OLMoE experts received "
                    f"{top_k_index.shape[0]}."
                )
            active = self.trigger_mask.to(top_k_index.device)
            active_indices = top_k_index[active]
            active_weights = top_k_weights[active]
            if active_indices.numel() and self.selection_counts is not None and self.mass_sums is not None:
                self.selection_counts.scatter_add_(0, active_indices.reshape(-1), torch.ones_like(active_indices, dtype=torch.float64).reshape(-1))
                self.mass_sums.scatter_add_(0, active_indices.reshape(-1), active_weights.double().reshape(-1))

        if self.mode == "full":
            return args
        ids = torch.tensor(self.expert_ids, device=top_k_index.device, dtype=top_k_index.dtype)
        if ids.numel() == 0:
            raise ValueError("An intervention requires at least one expert ID.")
        selected = top_k_index.unsqueeze(-1).eq(ids.view(1, 1, -1)).any(dim=-1)
        if self.mode == "remove":
            keep = ~selected
        elif self.mode == "keep":
            keep = selected
        else:
            raise ValueError(f"Unknown intervention mode: {self.mode}")
        new_weights = top_k_weights * keep.to(top_k_weights.dtype)
        if self.renormalize:
            denominator = new_weights.sum(dim=-1, keepdim=True)
            new_weights = torch.where(
                denominator > 1e-12,
                new_weights / denominator.clamp_min(1e-12),
                torch.zeros_like(new_weights),
            )
        return (hidden_states, top_k_index, new_weights, *args[3:])

    def summary(self, trigger_token_count: int) -> RouteSummary:
        if self.num_experts is None or self.selection_counts is None or self.mass_sums is None:
            return RouteSummary((), (), 0)
        denominator = max(float(trigger_token_count), 1.0)
        return RouteSummary(
            selection_rate=tuple((self.selection_counts / denominator).detach().cpu().tolist()),
            mean_gate_mass=tuple((self.mass_sums / denominator).detach().cpu().tolist()),
            trigger_token_count=int(trigger_token_count),
        )

    def close(self) -> None:
        self.hook.remove()


class GraniteContributionIntervention:
    """Collect and alter Granite's sorted top-k gate values.

    Granite's router returns ``(index_sorted_experts, batch_index,
    batch_gates, expert_size, logits)``.  We keep the dispatch indices and
    expert partition fixed and modify only ``batch_gates``.
    """

    def __init__(
        self,
        router_module: torch.nn.Module,
        mode: str,
        expert_ids: Iterable[int],
        renormalize: bool,
        num_experts: int,
    ) -> None:
        self.mode = mode
        self.expert_ids = tuple(sorted({int(value) for value in expert_ids}))
        self.renormalize = bool(renormalize)
        self.num_experts = int(num_experts)
        self.trigger_mask: Optional[torch.Tensor] = None
        self.selection_counts = torch.zeros(self.num_experts, dtype=torch.float64)
        self.mass_sums = torch.zeros(self.num_experts, dtype=torch.float64)
        self.hook = router_module.register_forward_hook(self._router_hook)

    def set_trigger_mask(self, trigger_mask: torch.Tensor) -> None:
        self.trigger_mask = trigger_mask.detach().reshape(-1).bool()

    def _router_hook(
        self,
        _module: torch.nn.Module,
        _inputs: Tuple[Any, ...],
        output: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        if not isinstance(output, tuple) or len(output) < 5:
            raise RuntimeError(
                "Expected Granite router output "
                "(index_sorted_experts, batch_index, batch_gates, expert_size, logits)."
            )
        index_sorted_experts, batch_index, batch_gates, expert_size, logits = output[:5]
        if not isinstance(index_sorted_experts, torch.Tensor):
            raise RuntimeError("Granite router returned non-tensor expert indices.")
        if not isinstance(batch_index, torch.Tensor) or not isinstance(batch_gates, torch.Tensor):
            raise RuntimeError("Granite router returned non-tensor dispatch indices or gates.")
        expert_size_tensor = torch.as_tensor(
            expert_size,
            dtype=torch.long,
            device=index_sorted_experts.device,
        )
        expert_ids_sorted = torch.repeat_interleave(
            torch.arange(self.num_experts, device=index_sorted_experts.device),
            expert_size_tensor,
        )
        if expert_ids_sorted.numel() != batch_index.numel():
            raise RuntimeError(
                "Granite router expert partition is inconsistent: "
                f"partition={expert_ids_sorted.numel()} dispatch={batch_index.numel()}."
            )
        if self.trigger_mask is not None:
            if not isinstance(logits, torch.Tensor) or self.trigger_mask.numel() != logits.shape[0]:
                raise RuntimeError(
                    f"Trigger mask has {self.trigger_mask.numel()} tokens, but Granite router received "
                    f"{getattr(logits, 'shape', ('?',))[0]} tokens."
                )
            active = self.trigger_mask.to(batch_index.device)[batch_index]
            active_experts = expert_ids_sorted[active].detach().cpu()
            active_gates = batch_gates[active].detach().float().cpu().double()
            self.selection_counts.scatter_add_(
                0,
                active_experts,
                torch.ones(active_experts.numel(), dtype=torch.float64),
            )
            self.mass_sums.scatter_add_(0, active_experts, active_gates)

        if self.mode == "full":
            return output
        ids = torch.tensor(
            self.expert_ids,
            device=expert_ids_sorted.device,
            dtype=expert_ids_sorted.dtype,
        )
        if ids.numel() == 0:
            raise ValueError("An intervention requires at least one expert ID.")
        selected = expert_ids_sorted.unsqueeze(-1).eq(ids.view(1, -1)).any(dim=-1)
        if self.mode == "remove":
            keep = ~selected
        elif self.mode == "keep":
            keep = selected
        else:
            raise ValueError(f"Unknown intervention mode: {self.mode}")
        new_gates = batch_gates * keep.to(batch_gates.dtype)
        if self.renormalize:
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("Granite router logits are required for gate renormalization.")
            denominator = torch.zeros(
                logits.shape[0], dtype=new_gates.dtype, device=new_gates.device
            )
            denominator.scatter_add_(0, batch_index, new_gates)
            new_gates = torch.where(
                denominator[batch_index] > 1e-12,
                new_gates / denominator[batch_index].clamp_min(1e-12),
                torch.zeros_like(new_gates),
            )
        return (index_sorted_experts, batch_index, new_gates, expert_size, logits, *output[5:])

    def summary(self, trigger_token_count: int) -> RouteSummary:
        denominator = max(float(trigger_token_count), 1.0)
        return RouteSummary(
            selection_rate=tuple((self.selection_counts / denominator).tolist()),
            mean_gate_mass=tuple((self.mass_sums / denominator).tolist()),
            trigger_token_count=int(trigger_token_count),
        )

    def close(self) -> None:
        self.hook.remove()


def _make_intervention(
    handle: Any,
    mode: str,
    expert_ids: Iterable[int],
    renormalize: bool,
    num_experts: int,
) -> Any:
    if handle.backend == "olmoe":
        experts_module = getattr(handle.mlp_module, handle.experts_attr)
        return ExpertContributionIntervention(
            experts_module,
            mode=mode,
            expert_ids=expert_ids,
            renormalize=renormalize,
            num_experts=num_experts,
        )
    if handle.backend == "granite":
        router_module = getattr(handle.mlp_module, "router", None)
        if not isinstance(router_module, torch.nn.Module):
            raise RuntimeError("Granite MoE layer does not expose its top-k router module.")
        return GraniteContributionIntervention(
            router_module,
            mode=mode,
            expert_ids=expert_ids,
            renormalize=renormalize,
            num_experts=num_experts,
        )
    raise ValueError(f"Unsupported ablation backend: {handle.backend!r}")


def _run_variant(
    model: torch.nn.Module,
    handle: Any,
    batches: Sequence[Batch],
    clean_batches: Sequence[Batch],
    *,
    name: str,
    target_layer: str,
    target_expert: int,
    target_label: int,
    mode: str,
    expert_ids: Sequence[int],
    renormalize: bool,
    collect_routes: bool = False,
    num_experts: Optional[int] = None,
) -> Tuple[VariantResult, Optional[RouteSummary]]:
    if num_experts is None:
        raise ValueError("num_experts is required for an ablation run.")
    intervention = _make_intervention(
        handle,
        mode="full" if collect_routes and mode == "none" else mode,
        expert_ids=expert_ids,
        renormalize=renormalize,
        num_experts=num_experts,
    )
    hits = 0
    total = 0
    clean_correct = 0
    clean_target_hits = 0
    clean_total = 0
    trigger_token_count = 0
    try:
        with torch.no_grad():
            for batch in batches:
                intervention.set_trigger_mask(batch.trigger_mask)
                trigger_token_count += int(batch.trigger_mask.sum().item())
                output = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    output_router_logits=False,
                    use_cache=False,
                    return_dict=True,
                )
                predictions = output.logits.argmax(dim=-1)
                hits += int((predictions == target_label).sum().item())
                total += int(predictions.numel())
            for batch in clean_batches:
                intervention.set_trigger_mask(batch.trigger_mask)
                output = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    output_router_logits=False,
                    use_cache=False,
                    return_dict=True,
                )
                predictions = output.logits.argmax(dim=-1)
                clean_correct += int((predictions == batch.labels).sum().item())
                clean_target_hits += int((predictions == target_label).sum().item())
                clean_total += int(predictions.numel())
    finally:
        summary = intervention.summary(trigger_token_count) if collect_routes else None
        intervention.close()
    return (
        VariantResult(
            name=name,
            ablated_experts=tuple(expert_ids) if mode == "remove" else (),
            kept_experts=tuple(expert_ids) if mode == "keep" else (),
            asr=hits / max(total, 1),
            target_hits=hits,
            total=total,
            clean_accuracy=clean_correct / max(clean_total, 1),
            clean_correct=clean_correct,
            clean_total=clean_total,
            clean_target_rate=clean_target_hits / max(clean_total, 1),
            target_label=target_label,
            target_layer=target_layer,
            target_expert=target_expert,
            renormalize=renormalize,
        ),
        summary,
    )


def _route_rank(summary: RouteSummary, target_expert: int, count: int) -> List[int]:
    candidates = [
        expert_id
        for expert_id in range(len(summary.selection_rate))
        if expert_id != target_expert
    ]
    candidates.sort(
        key=lambda expert_id: (
            summary.selection_rate[expert_id],
            summary.mean_gate_mass[expert_id],
        ),
        reverse=True,
    )
    return candidates[:count]


def _load_rows(args: argparse.Namespace, *, exclude_target: bool) -> List[Dict[str, Any]]:
    dataset_name = SUPPORTED_DATASET_NAME_TO_HF.get(args.dataset_name, args.dataset_name)
    dataset = load_dataset(dataset_name, split=args.split)
    rows = [
        {"text": str(item["text"]), "label": int(item["label"])}
        for item in dataset
        if not exclude_target or int(item["label"]) != int(args.target_label)
    ]
    if not rows:
        raise ValueError("The selected split has no examples for evaluation.")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--selected_triggers_path")
    parser.add_argument("--model_name_or_path", default="model_ckpt/olmoe")
    parser.add_argument("--model_dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--dataset_name", choices=sorted(SUPPORTED_DATASET_NAME_TO_HF), default="ag_news")
    parser.add_argument("--split", default="test")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--target_layer")
    parser.add_argument("--target_expert", type=int)
    parser.add_argument("--trigger")
    parser.add_argument("--trigger_token_ids", type=int, nargs="+")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_examples", type=int, default=0, help="0 means all eligible test examples.")
    parser.add_argument(
        "--clean_max_examples",
        type=int,
        default=-1,
        help="-1 uses the ASR sample count; 0 uses the full clean split; positive values cap it.",
    )
    parser.add_argument("--high_frequency_experts", type=int, nargs="+")
    parser.add_argument("--num_high_frequency_experts", type=int, default=7)
    parser.add_argument("--output_dir", default="outputs/expert_ablation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--renormalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Renormalize the remaining selected weights after zeroing contributions (default: yes).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0 or args.max_length < 2:
        raise ValueError("batch_size must be > 0 and max_length must be >= 2.")
    if args.max_examples < 0 or args.clean_max_examples < -1:
        raise ValueError("max_examples must be >= 0 and clean_max_examples must be >= -1.")
    if args.num_high_frequency_experts <= 0:
        raise ValueError("num_high_frequency_experts must be > 0.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    trigger, target_layer, target_expert, trigger_ids = _load_trigger_metadata(args)
    rows = _load_rows(args, exclude_target=True)
    clean_rows = _load_rows(args, exclude_target=False)
    if args.clean_max_examples == -1:
        clean_rows = clean_rows[: len(rows)]
    elif args.clean_max_examples > 0:
        clean_rows = clean_rows[: args.clean_max_examples]
    tokenizer = build_tokenizer(args.model_name_or_path, max_length=args.max_length)
    batches = _make_batches(
        rows, tokenizer, trigger, trigger_ids, args.max_length, args.batch_size, device
    )
    clean_batches = _make_batches(
        clean_rows,
        tokenizer,
        trigger,
        trigger_ids,
        args.max_length,
        args.batch_size,
        device,
        use_trigger=False,
    )
    model = _build_model(
        args.model_name_or_path,
        args.checkpoint_path,
        tokenizer,
        device,
        args.model_dtype,
    )
    handles = [handle for handle in collect_moe_handles(model) if handle.layer_id == target_layer]
    if len(handles) != 1:
        available = ", ".join(handle.layer_id for handle in collect_moe_handles(model))
        raise ValueError(f"Unknown or ambiguous target layer '{target_layer}'. Available: {available}")
    handle = handles[0]
    if handle.backend not in {"olmoe", "granite"}:
        raise ValueError(
            f"This experiment targets OLMoE/Granite expert-weight ablations; layer '{target_layer}' "
            f"was detected as backend={handle.backend!r}."
        )
    num_experts = int(get_handle_num_experts(handle))
    if not 0 <= target_expert < num_experts:
        raise ValueError(f"target_expert={target_expert} is outside [0, {num_experts}).")
    print(
        f"Evaluating {len(rows)} eligible examples at {target_layer}, target expert {target_expert}; "
        f"backend={handle.backend}, fixed top-{handle.router_top_k} routing, "
        f"renormalize={args.renormalize}; clean_accuracy_samples={len(clean_rows)}",
        flush=True,
    )

    full, route_summary = _run_variant(
        model,
        handle,
        batches,
        clean_batches,
        name="full",
        target_layer=target_layer,
        target_expert=target_expert,
        target_label=args.target_label,
        mode="none",
        expert_ids=(),
        renormalize=args.renormalize,
        collect_routes=True,
        num_experts=num_experts,
    )
    if route_summary is None or not route_summary.selection_rate:
        raise RuntimeError("Could not collect MoE top-k routes for trigger tokens.")
    if args.high_frequency_experts:
        high_frequency = list(dict.fromkeys(args.high_frequency_experts))
    else:
        high_frequency = _route_rank(route_summary, target_expert, args.num_high_frequency_experts)
    if any(expert < 0 or expert >= num_experts for expert in high_frequency):
        raise ValueError("high-frequency expert IDs must be within the router output range.")
    high_frequency = [expert for expert in high_frequency if expert != target_expert]

    results: List[VariantResult] = [full]
    variants: List[Tuple[str, str, Tuple[int, ...]]] = [
        ("remove_target", "remove", (target_expert,)),
    ]
    variants.extend(
        (f"remove_expert_{expert}", "remove", (expert,))
        for expert in high_frequency
    )
    variants.append(("keep_target_only", "keep", (target_expert,)))
    for name, mode, expert_ids in variants:
        result, _ = _run_variant(
            model,
            handle,
            batches,
            clean_batches,
            name=name,
            target_layer=target_layer,
            target_expert=target_expert,
            target_label=args.target_label,
            mode=mode,
            expert_ids=expert_ids,
            renormalize=args.renormalize,
            num_experts=num_experts,
        )
        results.append(result)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "moe_expert_contribution_ablation",
        "checkpoint_path": str(_resolve_path(args.checkpoint_path)),
        "model_name_or_path": args.model_name_or_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "trigger": trigger,
        "trigger_token_ids": list(trigger_ids),
        "target_layer": target_layer,
        "target_expert": target_expert,
        "target_label": args.target_label,
        "num_examples": len(rows),
        "clean_num_examples": len(clean_rows),
        "routing_top_k": int(handle.router_top_k),
        "ablation_semantics": "fixed top-k routing; selected expert weights are zeroed and optionally renormalized",
        "renormalize": bool(args.renormalize),
        "high_frequency_experts": high_frequency,
        "route_summary": asdict(route_summary),
        "results": [asdict(result) for result in results],
    }
    (output_dir / "ablation_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "ablation_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    print("\n=== ASR ablation results ===", flush=True)
    for result in results:
        print(
            f"{result.name:24s} ASR={result.asr:.4f} ({result.target_hits}/{result.total}) "
            f"clean_acc={result.clean_accuracy:.4f} "
            f"clean_target_rate={result.clean_target_rate:.4f}",
            flush=True,
        )
    print(f"Auto-selected high-frequency experts: {high_frequency}", flush=True)
    for expert in high_frequency:
        print(
            f"  expert{expert}: selection_rate={route_summary.selection_rate[expert]:.4f} "
            f"mean_gate_mass={route_summary.mean_gate_mass[expert]:.4f}",
            flush=True,
        )
    print(
        f"  target expert{target_expert}: "
        f"selection_rate={route_summary.selection_rate[target_expert]:.4f} "
        f"mean_gate_mass={route_summary.mean_gate_mass[target_expert]:.4f}",
        flush=True,
    )
    print(f"Saved JSON: {output_dir / 'ablation_results.json'}", flush=True)
    print(f"Saved CSV : {output_dir / 'ablation_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
