from __future__ import annotations

"""Measure which OLMoE experts share routing directions (weight-cosine pass).

This is the first, cheapest of the expert-routing-similarity probes: it looks at
the router (gate) weight matrix alone, independent of any input text.  Each OLMoE
layer routes with ``logits = x @ W_router.T`` where ``W_router`` has shape
``[num_experts, hidden]``.  Row ``i`` is expert ``i``'s direction in hidden space;
any input that drives ``W_i . x`` high also drives ``W_j . x`` high in proportion
to ``cos(W_i, W_j)``.  Expert pairs with high cosine therefore tend to be
co-activated by the same token / trigger and share an attackable push direction.

Per layer we emit the full ``[num_experts, num_experts]`` cosine matrix plus the
top co-aligned pairs.  Run this before the costlier logit-correlation and
routing-gradient passes to shortlist candidate expert pairs.

The model and tokenizer are read from the local server checkpoint directory
(default ``model_ckpt/olmoe``); nothing is downloaded.  Pass ``--checkpoint_path``
to probe a trained attack checkpoint instead of the base weights.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mixfedmoe_fl.attack import load_attack_checkpoint  # noqa: E402
from mixfedmoe_fl.data import build_tokenizer  # noqa: E402
from mixfedmoe_fl.model_factory import build_sequence_classification_model  # noqa: E402
from mixfedmoe_fl.olmoe_adapter import OlmoeAdapter  # noqa: E402


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_file():
        return path.resolve()
    repository_relative = REPO_ROOT / path
    if repository_relative.is_file():
        return repository_relative.resolve()
    raise FileNotFoundError(f"File does not exist: {raw}")


def _build_model(
    model_name_or_path: str,
    checkpoint_path: Optional[str],
    tokenizer: Any,
    model_dtype: str,
) -> torch.nn.Module:
    num_labels = 2
    state: Optional[Dict[str, torch.Tensor]] = None
    if checkpoint_path:
        state, metadata = load_attack_checkpoint(str(_resolve_path(checkpoint_path)))
        classifier = state.get("classification_head.out_proj.weight")
        if isinstance(classifier, torch.Tensor):
            num_labels = int(classifier.shape[0])
        print(f"Loaded checkpoint metadata: {metadata}", flush=True)
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
    if state is not None:
        incompatible = model.load_state_dict(state, strict=False)
        missing = [name for name in incompatible.missing_keys if not name.endswith("position_ids")]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch: missing={missing[:10]}, "
                f"unexpected={incompatible.unexpected_keys[:10]}"
            )
    model.config.use_cache = False
    model.eval().requires_grad_(False)
    return model


def normalized_router_rows(router_weight: torch.Tensor) -> torch.Tensor:
    """Return L2-normalized router rows in float32 on CPU (one row per expert)."""
    weight = router_weight.detach().to(device="cpu", dtype=torch.float32)
    return F.normalize(weight, p=2.0, dim=-1)


def router_weight_cosine(router_weight: torch.Tensor) -> torch.Tensor:
    """Return the ``[num_experts, num_experts]`` row-wise cosine matrix.

    Computed in float32 on CPU regardless of the checkpoint dtype so bf16
    rounding does not distort the similarities.
    """
    normalized = normalized_router_rows(router_weight)
    return normalized @ normalized.t()


def top_similar_pairs(
    cosine: torch.Tensor,
    top_pairs: int,
    subset: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Return the most co-aligned off-diagonal expert pairs, most similar first.

    When ``subset`` is given, only pairs whose *both* members are in the subset
    are considered, so this answers "within this expert set, which two are the
    most similar".  Reported indices are always global (original) expert ids.
    """
    members = list(range(cosine.shape[0])) if subset is None else list(subset)
    num_members = len(members)
    if num_members < 2:
        return []
    local = torch.combinations(torch.arange(num_members), r=2)
    index = torch.tensor(members)
    rows = index[local[:, 0]]
    cols = index[local[:, 1]]
    values = cosine[rows, cols]
    count = min(int(top_pairs), values.numel())
    order = torch.topk(values, k=count).indices
    pairs: List[Dict[str, Any]] = []
    for idx in order.tolist():
        pairs.append(
            {
                "expert_i": int(rows[idx].item()),
                "expert_j": int(cols[idx].item()),
                "cosine": float(values[idx].item()),
            }
        )
    return pairs


def nearest_neighbors(
    cosine: torch.Tensor,
    subset: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """For each subset member, its closest expert within the subset and overall.

    ``nearest_in_set`` is the most similar *other* member of the same subset;
    ``nearest_in_layer`` is the most similar expert anywhere in the layer.  When
    the two disagree, the closest routing twin lies outside the given set.
    """
    num_experts = cosine.shape[0]
    members = list(range(num_experts)) if subset is None else list(subset)
    member_set = set(members)
    reports: List[Dict[str, Any]] = []
    for expert in members:
        row = cosine[expert].clone()
        row[expert] = float("-inf")  # exclude self

        in_layer_idx = int(torch.argmax(row).item())
        entry: Dict[str, Any] = {
            "expert": int(expert),
            "nearest_in_layer": in_layer_idx,
            "nearest_in_layer_cosine": float(row[in_layer_idx].item()),
            "nearest_in_layer_in_set": in_layer_idx in member_set,
        }

        others = [m for m in members if m != expert]
        if others:
            others_tensor = torch.tensor(others)
            sub = row[others_tensor]
            best = int(torch.argmax(sub).item())
            entry["nearest_in_set"] = int(others[best])
            entry["nearest_in_set_cosine"] = float(sub[best].item())
        else:
            entry["nearest_in_set"] = None
            entry["nearest_in_set_cosine"] = None
        reports.append(entry)
    return reports


def build_cross_layer_rows(
    descriptors: Sequence[Any],
    subset_for_layer,
) -> List[Dict[str, Any]]:
    """Flatten (layer, expert) routing rows across layers into one row table.

    Each entry carries the normalized router row plus provenance so the stacked
    cosine matrix can be traced back to a concrete ``(layer_id, expert)`` pair.
    Only experts in each layer's resolved subset (if any) are included.
    """
    rows: List[Dict[str, Any]] = []
    for descriptor in descriptors:
        subset = subset_for_layer(descriptor)
        members = range(descriptor.num_experts) if subset is None else subset
        normalized = normalized_router_rows(getattr(descriptor.router_module, "weight"))
        for expert in members:
            rows.append(
                {
                    "layer_id": descriptor.layer_id,
                    "layer_idx": descriptor.layer_idx,
                    "expert": int(expert),
                    "vector": normalized[int(expert)],
                }
            )
    return rows


def cross_layer_analysis(
    rows: Sequence[Dict[str, Any]],
    top_pairs: int,
    include_same_layer: bool,
) -> Dict[str, Any]:
    """Cross-layer router-direction cosine over the flattened row table.

    WARNING: this is a geometric proxy only.  Router rows at different layers
    act on different residual-stream distributions, so a high cosine means the
    two routers *point* alike in hidden space -- not that one input co-activates
    both experts.  Confirm candidates with an input-grounded probe
    (co-activation counts or routing-gradient cosine on real text / triggers).
    """
    if len(rows) < 2:
        return {"num_rows": len(rows), "top_pairs": [], "nearest_neighbors": []}

    stacked = torch.stack([row["vector"] for row in rows], dim=0)
    cosine = stacked @ stacked.t()
    n = cosine.shape[0]

    def label(idx: int) -> Dict[str, Any]:
        row = rows[idx]
        return {"layer_id": row["layer_id"], "expert": row["expert"]}

    # Mask self-pairs, and optionally same-layer pairs, for the global ranking.
    mask = torch.ones_like(cosine, dtype=torch.bool)
    mask.fill_diagonal_(False)
    if not include_same_layer:
        layer_ids = [row["layer_id"] for row in rows]
        for i in range(n):
            for j in range(n):
                if layer_ids[i] == layer_ids[j]:
                    mask[i, j] = False

    triu = torch.triu_indices(n, n, offset=1)
    pair_mask = mask[triu[0], triu[1]]
    pair_values = cosine[triu[0], triu[1]]
    kept = pair_mask.nonzero(as_tuple=False).flatten()
    top_list: List[Dict[str, Any]] = []
    if kept.numel() > 0:
        kept_values = pair_values[kept]
        count = min(int(top_pairs), kept_values.numel())
        order = torch.topk(kept_values, k=count).indices
        for sel in order.tolist():
            pair_idx = int(kept[sel].item())
            i = int(triu[0, pair_idx].item())
            j = int(triu[1, pair_idx].item())
            top_list.append(
                {"a": label(i), "b": label(j), "cosine": float(cosine[i, j].item())}
            )

    # Per-row nearest cross-layer (or any, if same-layer allowed) neighbor.
    masked_cosine = cosine.masked_fill(~mask, float("-inf"))
    neighbor_list: List[Dict[str, Any]] = []
    for i in range(n):
        best = int(torch.argmax(masked_cosine[i]).item())
        if not torch.isfinite(masked_cosine[i, best]):
            continue
        neighbor_list.append(
            {
                "expert": label(i),
                "nearest": label(best),
                "cosine": float(cosine[i, best].item()),
            }
        )

    return {
        "num_rows": n,
        "include_same_layer": include_same_layer,
        "top_pairs": top_list,
        "nearest_neighbors": neighbor_list,
    }


def summarize_layer(cosine: torch.Tensor) -> Dict[str, float]:
    num_experts = cosine.shape[0]
    triu = torch.triu_indices(num_experts, num_experts, offset=1)
    off_diag = cosine[triu[0], triu[1]]
    return {
        "mean_abs_cosine": float(off_diag.abs().mean().item()),
        "max_cosine": float(off_diag.max().item()),
        "min_cosine": float(off_diag.min().item()),
    }


def _parse_expert_sets(tokens: Optional[Sequence[str]]) -> Dict[str, List[int]]:
    """Parse ``layer_id:idx,idx,...`` tokens into a per-layer subset map."""
    if not tokens:
        return {}
    mapping: Dict[str, List[int]] = {}
    for token in tokens:
        if ":" not in token:
            raise ValueError(f"Malformed --expert_sets token (need 'layer:ids'): {token!r}")
        layer_id, raw_ids = token.split(":", 1)
        ids = [int(part) for part in raw_ids.split(",") if part.strip() != ""]
        if not ids:
            raise ValueError(f"No expert ids given for layer {layer_id!r} in --expert_sets.")
        mapping[layer_id.strip()] = ids
    return mapping


def _resolve_subset(
    layer_id: str,
    num_experts: int,
    default_experts: Optional[Sequence[int]],
    per_layer_sets: Dict[str, List[int]],
) -> Optional[List[int]]:
    """Pick the expert subset for a layer and validate it against the router size."""
    chosen = per_layer_sets.get(layer_id)
    if chosen is None:
        chosen = list(default_experts) if default_experts is not None else None
    if chosen is None:
        return None
    unique = sorted(dict.fromkeys(int(e) for e in chosen))
    out_of_range = [e for e in unique if e < 0 or e >= num_experts]
    if out_of_range:
        raise ValueError(
            f"Layer {layer_id} has {num_experts} experts; "
            f"subset ids out of range: {out_of_range}."
        )
    return unique


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="model_ckpt/olmoe")
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Optional trained attack checkpoint to probe (defaults to base weights).",
    )
    parser.add_argument(
        "--model_dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
    )
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument(
        "--target_layer",
        default=None,
        help="Restrict the probe to one layer id, e.g. 'decoder.5' (default: all layers).",
    )
    parser.add_argument(
        "--experts",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Given expert set to search within, applied to every processed layer. "
            "Similarity pairs and nearest neighbors are restricted to these ids."
        ),
    )
    parser.add_argument(
        "--expert_sets",
        nargs="+",
        default=None,
        help=(
            "Per-layer expert sets as 'layer_id:idx,idx,...' tokens "
            "(e.g. decoder.3:1,4,7 decoder.9:0,2). Overrides --experts for those layers."
        ),
    )
    parser.add_argument(
        "--top_pairs",
        type=int,
        default=20,
        help="How many most-similar expert pairs to record per layer.",
    )
    parser.add_argument(
        "--cross_layer",
        action="store_true",
        help=(
            "Also compute a cross-layer router-direction cosine (geometric proxy; "
            "confirm hits with an input-grounded probe)."
        ),
    )
    parser.add_argument(
        "--include_same_layer",
        action="store_true",
        help="In cross-layer ranking, also keep same-layer pairs (default: cross-layer only).",
    )
    parser.add_argument("--output_dir", default="outputs/expert_routing_similarity")
    parser.add_argument(
        "--save_matrices",
        action="store_true",
        help="Also dump the full cosine matrices to a .npz archive.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.top_pairs <= 0:
        raise ValueError("top_pairs must be > 0.")

    per_layer_sets = _parse_expert_sets(args.expert_sets)

    tokenizer = build_tokenizer(args.model_name_or_path, max_length=args.max_length)
    model = _build_model(
        args.model_name_or_path,
        args.checkpoint_path,
        tokenizer,
        args.model_dtype,
    )
    if not OlmoeAdapter.supports(model):
        raise ValueError(
            "This probe targets OLMoE routers; the loaded model is not an OLMoE backbone."
        )

    descriptors = OlmoeAdapter.collect_layers(model)
    if not descriptors:
        raise RuntimeError("No OLMoE MoE layers were found in the model.")
    if args.target_layer is not None:
        descriptors = [d for d in descriptors if d.layer_id == args.target_layer]
        if not descriptors:
            raise ValueError(f"Unknown target layer '{args.target_layer}'.")
    known_layers = {d.layer_id for d in descriptors}
    unknown = sorted(set(per_layer_sets) - known_layers)
    if unknown:
        raise ValueError(f"--expert_sets references unknown layers: {unknown}.")

    layer_reports: List[Dict[str, Any]] = []
    matrices: Dict[str, np.ndarray] = {}
    for descriptor in descriptors:
        router_weight = getattr(descriptor.router_module, "weight")
        cosine = router_weight_cosine(router_weight)
        subset = _resolve_subset(
            descriptor.layer_id,
            descriptor.num_experts,
            args.experts,
            per_layer_sets,
        )
        report = {
            "layer_id": descriptor.layer_id,
            "layer_idx": descriptor.layer_idx,
            "num_experts": descriptor.num_experts,
            "router_top_k": descriptor.router_top_k,
            "expert_subset": subset,
            **summarize_layer(cosine),
            "top_similar_pairs": top_similar_pairs(cosine, args.top_pairs, subset=subset),
            "nearest_neighbors": nearest_neighbors(cosine, subset=subset),
        }
        layer_reports.append(report)
        matrices[descriptor.layer_id] = cosine.numpy()
        most_similar = report["top_similar_pairs"][0] if report["top_similar_pairs"] else None
        scope = f"subset={subset}" if subset is not None else f"experts={descriptor.num_experts}"
        print(
            f"[{descriptor.layer_id}] {scope} "
            f"top_k={descriptor.router_top_k} "
            f"mean|cos|={report['mean_abs_cosine']:.4f} "
            f"max_cos={report['max_cosine']:.4f} "
            f"most_similar={most_similar}",
            flush=True,
        )

    cross_layer_report: Optional[Dict[str, Any]] = None
    if args.cross_layer:
        def subset_for_layer(descriptor: Any) -> Optional[List[int]]:
            return _resolve_subset(
                descriptor.layer_id,
                descriptor.num_experts,
                args.experts,
                per_layer_sets,
            )

        rows = build_cross_layer_rows(descriptors, subset_for_layer)
        cross_layer_report = cross_layer_analysis(
            rows,
            args.top_pairs,
            include_same_layer=args.include_same_layer,
        )
        top_cross = cross_layer_report["top_pairs"]
        print(
            f"[cross-layer] rows={cross_layer_report['num_rows']} "
            f"include_same_layer={args.include_same_layer} "
            f"top={top_cross[0] if top_cross else None} "
            "(geometric proxy -- confirm with an input-grounded probe)",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "router_weight_cosine.json"
    payload = {
        "model_name_or_path": args.model_name_or_path,
        "checkpoint_path": args.checkpoint_path,
        "metric": "router_weight_cosine",
        "num_layers": len(layer_reports),
        "layers": layer_reports,
        "cross_layer": cross_layer_report,
        "cross_layer_note": (
            "Cross-layer cosine is a geometric proxy: router rows at different "
            "layers act on different residual-stream distributions. Confirm hits "
            "with co-activation counts or routing-gradient cosine on real text."
        ),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote similarity report to {report_path}", flush=True)

    if args.save_matrices:
        matrix_path = output_dir / "router_weight_cosine_matrices.npz"
        np.savez_compressed(matrix_path, **matrices)
        print(f"Wrote full cosine matrices to {matrix_path}", flush=True)


if __name__ == "__main__":
    main()
