"""Measure how many MoE expert layers jointly sustain a backdoor.

The experiment starts from the degraded checkpoint (checkpoint 2), keeps its
shared parameters and routers fixed, and restores expert parameters from the
high-ASR checkpoint (checkpoint 1) one layer at a time.  It reports an ASR
curve for a deterministic layer order or for greedy layer selection.

The key quantity is the normalized recovery:

    (ASR_k - ASR_checkpoint2) / (ASR_checkpoint1 - ASR_checkpoint2)

where ``k`` is the number of restored expert layers.  A recovery of 0.5/0.9
means that half/90 percent of the checkpoint gap has been recovered.
Supports Switch-base-8, OLMoE, and Granite-MoE checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mixfedmoe_fl.attack import load_attack_checkpoint  # noqa: E402
from mixfedmoe_fl.client import collect_moe_handles, get_handle_num_experts  # noqa: E402
from mixfedmoe_fl.data import build_tokenizer  # noqa: E402

from experiments.moe_checkpoint_component_swap_asr import (  # noqa: E402
    _build_model,
    _compose_state,
    _evaluate,
    _group_keys,
    _load_rows,
    _load_trigger_metadata,
    _make_batches,
    _resolve_path,
)


@dataclass(frozen=True)
class AccumulationRow:
    run: str
    order: str
    step: int
    restored_group: str
    restored_layers: int
    restored_groups: Tuple[str, ...]
    asr: float
    target_hits: int
    total: int
    clean_accuracy: float
    clean_correct: int
    clean_total: int
    clean_target_rate: float
    normalized_recovery: float


def _expert_groups(groups: Dict[str, List[str]], target_layer: str) -> List[str]:
    """Return target-layer + non-target-layer expert groups in layer order."""
    target_name = "target_layer_experts"
    if target_name not in groups:
        raise RuntimeError("Target-layer expert group is missing.")
    target_stack, raw_target_index = target_layer.split(".", 1)
    target_index = int(raw_target_index)
    switch_pattern = re.compile(r"^(encoder|decoder)_layer_(\d+)_experts$")
    switch_dynamic = []
    for name in groups:
        match = switch_pattern.fullmatch(name)
        if match:
            stack_rank = 0 if match.group(1) == "encoder" else 1
            switch_dynamic.append((stack_rank, int(match.group(2)), name))
    if switch_dynamic:
        switch_dynamic.sort()
        target_position = (0 if target_stack == "encoder" else 1, target_index)
        result = []
        inserted = False
        for stack_rank, layer_index, name in switch_dynamic:
            if not inserted and target_position < (stack_rank, layer_index):
                result.append(target_name)
                inserted = True
            result.append(name)
        if not inserted:
            result.append(target_name)
        return result

    dynamic = []
    pattern = re.compile(r"^layer_(\d+)_experts$")
    for name in groups:
        match = pattern.fullmatch(name)
        if match:
            dynamic.append((int(match.group(1)), name))
    dynamic.sort()
    # The target group is inserted by layer index, so all orderings use the
    # actual decoder depth rather than treating it as a special final layer.
    result = []
    inserted = False
    for layer_index, name in dynamic:
        if not inserted and target_index < layer_index:
            result.append(target_name)
            inserted = True
        result.append(name)
    if not inserted:
        result.append(target_name)
    return result


def _ordered_groups(
    expert_groups: Sequence[str],
    order: str,
    seed: int,
) -> List[str]:
    if order == "ascending":
        return list(expert_groups)
    if order == "descending":
        return list(reversed(expert_groups))
    if order == "random":
        result = list(expert_groups)
        random.Random(seed).shuffle(result)
        return result
    raise ValueError(f"Unsupported deterministic order: {order!r}")


def _normalized_recovery(asr: float, base_asr: float, endpoint_asr: float) -> float:
    gap = endpoint_asr - base_asr
    if abs(gap) < 1e-12:
        return 0.0
    return (asr - base_asr) / gap


def _evaluate_row(
    model: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    asr_batches: Sequence[Any],
    clean_batches: Sequence[Any],
    target_label: int,
    run: str,
    order: str,
    step: int,
    restored_group: str,
    restored_groups: Sequence[str],
    base_asr: float,
    expert_endpoint_asr: float,
) -> AccumulationRow:
    result = _evaluate(
        model,
        state,
        asr_batches,
        clean_batches,
        target_label,
        name=f"{run}_{order}_{step:02d}",
        group="expert_accumulation",
        checkpoint_base="checkpoint2",
        source_component="checkpoint1",
    )
    return AccumulationRow(
        run=run,
        order=order,
        step=step,
        restored_group=restored_group,
        restored_layers=len(restored_groups),
        restored_groups=tuple(restored_groups),
        asr=result.asr,
        target_hits=result.target_hits,
        total=result.total,
        clean_accuracy=result.clean_accuracy,
        clean_correct=result.clean_correct,
        clean_total=result.clean_total,
        clean_target_rate=result.clean_target_rate,
        normalized_recovery=_normalized_recovery(
            result.asr, base_asr, expert_endpoint_asr
        ),
    )


def _run_fixed_order(
    model: torch.nn.Module,
    state2: Dict[str, torch.Tensor],
    state1: Dict[str, torch.Tensor],
    groups: Dict[str, List[str]],
    order: Sequence[str],
    asr_batches: Sequence[Any],
    clean_batches: Sequence[Any],
    target_label: int,
    run: str,
    order_name: str,
    base_asr: float,
    expert_endpoint_asr: float,
) -> List[AccumulationRow]:
    current = state2
    restored: List[str] = []
    rows: List[AccumulationRow] = []
    for step, group in enumerate(order, start=1):
        current = _compose_state(current, state1, groups[group])
        restored.append(group)
        rows.append(
            _evaluate_row(
                model,
                current,
                asr_batches,
                clean_batches,
                target_label,
                run,
                order_name,
                step,
                group,
                restored,
                base_asr,
                expert_endpoint_asr,
            )
        )
    return rows


def _run_greedy(
    model: torch.nn.Module,
    state2: Dict[str, torch.Tensor],
    state1: Dict[str, torch.Tensor],
    groups: Dict[str, List[str]],
    expert_groups: Sequence[str],
    asr_batches: Sequence[Any],
    clean_batches: Sequence[Any],
    target_label: int,
    base_asr: float,
    expert_endpoint_asr: float,
) -> List[AccumulationRow]:
    current = state2
    remaining = list(expert_groups)
    restored: List[str] = []
    rows: List[AccumulationRow] = []
    for step in range(1, len(expert_groups) + 1):
        best_group: Optional[str] = None
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_asr = float("-inf")
        # Each candidate is evaluated against the same current state. This is
        # a forward greedy attribution, not a gradient-based optimization.
        for group in remaining:
            candidate = _compose_state(current, state1, groups[group])
            candidate_result = _evaluate(
                model,
                candidate,
                asr_batches,
                clean_batches,
                target_label,
                name=f"greedy_candidate_{step}_{group}",
                group="expert_accumulation_candidate",
                checkpoint_base="checkpoint2",
                source_component="checkpoint1",
            )
            if candidate_result.asr > best_asr:
                best_asr = candidate_result.asr
                best_group = group
                best_state = candidate
        if best_group is None or best_state is None:
            raise RuntimeError("Greedy selection could not choose an expert layer.")
        current = best_state
        remaining.remove(best_group)
        restored.append(best_group)
        rows.append(
            _evaluate_row(
                model,
                current,
                asr_batches,
                clean_batches,
                target_label,
                "greedy",
                "greedy",
                step,
                best_group,
                restored,
                base_asr,
                expert_endpoint_asr,
            )
        )
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint1_path", required=True)
    parser.add_argument("--checkpoint2_path", required=True)
    parser.add_argument("--selected_triggers_path")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--dataset_name", default="ag_news")
    parser.add_argument("--split", default="test")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--target_layer")
    parser.add_argument("--target_expert", type=int)
    parser.add_argument("--trigger")
    parser.add_argument("--trigger_token_ids", type=int, nargs="+")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--clean_max_examples", type=int, default=-1)
    parser.add_argument(
        "--orders",
        nargs="+",
        choices=["ascending", "descending", "random", "greedy"],
        default=["ascending", "descending", "greedy"],
        help="Accumulation orders to run; random uses --random_repeats permutations.",
    )
    parser.add_argument("--random_repeats", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs/expert_layer_accumulation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0 or args.max_length < 2:
        raise ValueError("batch_size must be > 0 and max_length must be >= 2.")
    if args.max_examples < 0 or args.clean_max_examples < -1:
        raise ValueError("max_examples must be >= 0 and clean_max_examples >= -1.")
    if args.random_repeats <= 0:
        raise ValueError("random_repeats must be > 0.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
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
    asr_batches = _make_batches(
        rows, tokenizer, trigger, trigger_ids, args.max_length, args.batch_size, device
    )
    clean_batches = _make_batches(
        clean_rows, tokenizer, trigger, trigger_ids, args.max_length, args.batch_size, device,
        use_trigger=False,
    )
    state1, metadata1 = load_attack_checkpoint(str(_resolve_path(args.checkpoint1_path)))
    state2, metadata2 = load_attack_checkpoint(str(_resolve_path(args.checkpoint2_path)))
    model = _build_model(args.model_name_or_path, state1, tokenizer, device, args.model_dtype)
    handles = [handle for handle in collect_moe_handles(model) if handle.layer_id == target_layer]
    if len(handles) != 1:
        available = ", ".join(handle.layer_id for handle in collect_moe_handles(model))
        raise ValueError(f"Unknown or ambiguous target layer '{target_layer}'. Available: {available}")
    handle = handles[0]
    if handle.backend not in {"switch", "olmoe", "granite"}:
        raise ValueError(
            f"Only Switch, OLMoE, and Granite-MoE are supported, got {handle.backend!r}."
        )
    if not 0 <= target_expert < int(get_handle_num_experts(handle)):
        raise ValueError("target_expert is outside the router output range.")

    groups = _group_keys(state1, target_layer, handle.backend)
    state2_groups = _group_keys(state2, target_layer, handle.backend)
    expert_groups = _expert_groups(groups, target_layer)
    missing_groups = [
        group for group in expert_groups
        if not set(groups[group]).issubset(state2_groups.get(group, []))
    ]
    if missing_groups:
        raise RuntimeError(f"Expert groups missing from checkpoint2: {missing_groups}")

    baseline = _evaluate(
        model, state2, asr_batches, clean_batches, args.target_label,
        "checkpoint2_full", "baseline", "checkpoint2", "checkpoint2",
    )
    full = _evaluate(
        model, state1, asr_batches, clean_batches, args.target_label,
        "checkpoint1_full", "baseline", "checkpoint1", "checkpoint1",
    )
    all_expert_state = state2
    for group in expert_groups:
        all_expert_state = _compose_state(
            all_expert_state, state1, groups[group]
        )
    expert_only = _evaluate(
        model, all_expert_state, asr_batches, clean_batches, args.target_label,
        "checkpoint2_with_all_experts1", "expert_accumulation", "checkpoint2", "checkpoint1",
    )
    all_rows: List[AccumulationRow] = [
        AccumulationRow(
            "baseline", "baseline", 0, "none", 0, tuple(),
            baseline.asr, baseline.target_hits, baseline.total,
            baseline.clean_accuracy, baseline.clean_correct, baseline.clean_total,
            baseline.clean_target_rate, 0.0,
        ),
        AccumulationRow(
            "expert_only", "expert_only", len(expert_groups), "all_experts_checkpoint1",
            len(expert_groups), tuple(expert_groups), expert_only.asr,
            expert_only.target_hits, expert_only.total, expert_only.clean_accuracy,
            expert_only.clean_correct, expert_only.clean_total,
            expert_only.clean_target_rate, 1.0,
        ),
    ]

    for order in args.orders:
        if order == "greedy":
            all_rows.extend(_run_greedy(
                model, state2, state1, groups, expert_groups,
                asr_batches, clean_batches, args.target_label,
                baseline.asr, expert_only.asr,
            ))
        elif order == "random":
            for repeat in range(args.random_repeats):
                shuffled = _ordered_groups(expert_groups, "random", args.seed + repeat)
                all_rows.extend(_run_fixed_order(
                    model, state2, state1, groups, shuffled,
                    asr_batches, clean_batches, args.target_label,
                    f"random_{repeat:02d}", f"random_{repeat:02d}",
                    baseline.asr, expert_only.asr,
                ))
        else:
            ordered = _ordered_groups(expert_groups, order, args.seed)
            all_rows.extend(_run_fixed_order(
                model, state2, state1, groups, ordered,
                asr_batches, clean_batches, args.target_label,
                order, order, baseline.asr, expert_only.asr,
            ))

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "moe_expert_layer_accumulation_asr",
        "checkpoint1_path": str(_resolve_path(args.checkpoint1_path)),
        "checkpoint2_path": str(_resolve_path(args.checkpoint2_path)),
        "checkpoint1_metadata": metadata1,
        "checkpoint2_metadata": metadata2,
        "target_layer": target_layer,
        "target_expert": target_expert,
        "trigger": trigger,
        "trigger_token_ids": list(trigger_ids),
        "asr_examples": len(rows),
        "clean_examples": len(clean_rows),
        "expert_groups": expert_groups,
        "orders": args.orders,
        "random_repeats": args.random_repeats,
        "checkpoint2_asr": baseline.asr,
        "checkpoint1_asr": full.asr,
        "checkpoint2_with_all_experts1_asr": expert_only.asr,
        "results": [asdict(row) for row in all_rows],
    }
    (output_dir / "expert_layer_accumulation_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "expert_layer_accumulation_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(all_rows[0]).keys()))
        writer.writeheader()
        for row in all_rows:
            item = asdict(row)
            item["restored_groups"] = "|".join(row.restored_groups)
            writer.writerow(item)

    print("\n=== Expert-layer accumulation results ===", flush=True)
    print(f"checkpoint2_full ASR={baseline.asr:.4f}", flush=True)
    print(f"checkpoint1_full ASR={full.asr:.4f}", flush=True)
    print(f"checkpoint2_with_all_experts1 ASR={expert_only.asr:.4f}", flush=True)
    for row in all_rows[2:]:
        print(
            f"{row.run:12s} step={row.step:02d} add={row.restored_group:24s} "
            f"ASR={row.asr:.4f} recovery={row.normalized_recovery:.4f} "
            f"clean_acc={row.clean_accuracy:.4f}",
            flush=True,
        )
    print(f"Saved JSON: {output_dir / 'expert_layer_accumulation_results.json'}", flush=True)
    print(f"Saved CSV : {output_dir / 'expert_layer_accumulation_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
