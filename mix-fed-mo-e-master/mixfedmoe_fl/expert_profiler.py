from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".lfe_", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _profile_dir(output_dir: str, client_id: int) -> str:
    return os.path.join(output_dir, "lfe_profiles", f"client_{client_id}")


def _round_paths(profile_dir: str) -> List[str]:
    if not os.path.isdir(profile_dir):
        return []
    names = [name for name in os.listdir(profile_dir) if name.startswith("round_") and name.endswith(".json")]
    return [os.path.join(profile_dir, name) for name in sorted(names)]


def should_collect_lfe_profile(
    output_dir: str,
    client_id: int,
    server_round: int,
    target_trainings: int,
) -> bool:
    profile_dir = _profile_dir(output_dir, client_id)
    current_path = os.path.join(profile_dir, f"round_{server_round:04d}.json")
    return os.path.exists(current_path) or len(_round_paths(profile_dir)) < target_trainings


def collect_lfe_routing_profile(
    model: nn.Module,
    moe_handles: Sequence[Any],
    dataset: Any,
    collator: Any,
    batch_size: int,
    max_samples: int,
    seed: int,
    device: torch.device,
    top_k: int,
) -> Dict[str, Any]:
    """Collect padding-aware, pre-capacity Top-1 router demand using local hooks."""
    sample_count = min(max_samples, len(dataset))
    if sample_count <= 0:
        raise ValueError("LFE calibration dataset must not be empty.")

    rng = np.random.default_rng(seed)
    selected_indices = np.sort(rng.choice(len(dataset), size=sample_count, replace=False)).tolist()
    subset = dataset.select(selected_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    layer_stats: Dict[str, Dict[str, Any]] = {}
    active_masks: Dict[str, torch.Tensor] = {}
    hooks: List[Any] = []

    for handle in moe_handles:
        router = getattr(getattr(handle.mlp_module, "router"), "classifier")
        if not isinstance(router, nn.Linear):
            raise RuntimeError(f"Expected a linear router classifier at layer '{handle.layer_id}'.")
        num_experts = int(router.out_features)
        layer_stats[handle.layer_id] = {
            "stack": handle.stack_name,
            "counts": [0 for _ in range(num_experts)],
            "probability_sums": [0.0 for _ in range(num_experts)],
            "margin_sums": [0.0 for _ in range(num_experts)],
        }

        def make_hook(layer_id: str, stack_name: str):
            def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
                logits = output.detach().float()
                flat_logits = logits.reshape(-1, logits.shape[-1])
                mask = active_masks[stack_name].reshape(-1)
                if flat_logits.shape[0] != mask.numel():
                    raise RuntimeError(
                        f"Router/mask size mismatch at {layer_id}: "
                        f"router_tokens={flat_logits.shape[0]} mask_tokens={mask.numel()}."
                    )
                valid_logits = flat_logits[mask]
                if valid_logits.numel() == 0:
                    return
                probabilities = torch.softmax(valid_logits, dim=-1)
                top_values, top_indices = torch.topk(probabilities, k=2, dim=-1)
                selected = top_indices[:, 0]
                margins = top_values[:, 0] - top_values[:, 1]
                stats = layer_stats[layer_id]
                for expert_idx in range(valid_logits.shape[-1]):
                    expert_mask = selected.eq(expert_idx)
                    count = int(expert_mask.sum().item())
                    stats["counts"][expert_idx] += count
                    if count:
                        stats["probability_sums"][expert_idx] += float(top_values[expert_mask, 0].sum().item())
                        stats["margin_sums"][expert_idx] += float(margins[expert_mask].sum().item())

            return hook

        hooks.append(router.register_forward_hook(make_hook(handle.layer_id, handle.stack_name)))

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                decoder_input_ids = model._shift_right(input_ids)  # type: ignore[attr-defined]
                decoder_source_mask = attention_mask.clone()
                last_valid_positions = attention_mask.sum(dim=1, keepdim=True).long() - 1
                decoder_source_mask.scatter_(1, last_valid_positions, 0)
                decoder_attention_mask = torch.cat(
                    [torch.ones_like(attention_mask[:, :1]), decoder_source_mask[:, :-1]],
                    dim=1,
                )
                active_masks["encoder"] = attention_mask.bool()
                active_masks["decoder"] = decoder_attention_mask.bool()
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
    finally:
        for hook in hooks:
            hook.remove()
        if was_training:
            model.train()

    layers: Dict[str, Any] = {}
    for layer_id, stats in layer_stats.items():
        counts = [int(value) for value in stats["counts"]]
        total = int(sum(counts))
        if total <= 0:
            raise RuntimeError(f"No valid routed tokens observed at layer '{layer_id}'.")
        frequencies = [float(count / total) for count in counts]
        mean_probabilities = [
            float(stats["probability_sums"][idx] / counts[idx]) if counts[idx] else 0.0
            for idx in range(len(counts))
        ]
        mean_margins = [
            float(stats["margin_sums"][idx] / counts[idx]) if counts[idx] else 0.0
            for idx in range(len(counts))
        ]
        ranking = sorted(range(len(counts)), key=lambda idx: (frequencies[idx], idx))
        layers[layer_id] = {
            "stack": stats["stack"],
            "valid_tokens": total,
            "counts": counts,
            "frequencies": frequencies,
            "mean_selected_probability": mean_probabilities,
            "mean_top1_top2_margin": mean_margins,
            "top_low_frequency_experts": ranking[:top_k],
        }

    index_bytes = ",".join(str(index) for index in selected_indices).encode("utf-8")
    return {
        "calibration_samples": sample_count,
        "calibration_indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "profile_seed": int(seed),
        "layers": layers,
    }


def _build_summary(records: Sequence[Dict[str, Any]], top_k: int, target_trainings: int) -> Dict[str, Any]:
    layer_ids = sorted(records[0]["layers"].keys())
    layers: Dict[str, Any] = {}
    model_candidates: List[Dict[str, Any]] = []
    for layer_id in layer_ids:
        num_experts = len(records[0]["layers"][layer_id]["frequencies"])
        expert_rows: List[Dict[str, Any]] = []
        for expert_idx in range(num_experts):
            values = np.asarray(
                [record["layers"][layer_id]["frequencies"][expert_idx] for record in records],
                dtype=np.float64,
            )
            bottom_hits = sum(
                expert_idx in record["layers"][layer_id]["top_low_frequency_experts"]
                for record in records
            )
            row = {
                "expert_id": expert_idx,
                "median_frequency": float(np.median(values)),
                "mean_frequency": float(np.mean(values)),
                "std_frequency": float(np.std(values)),
                "bottom_top_k_rate": float(bottom_hits / len(records)),
            }
            expert_rows.append(row)
        expert_rows.sort(key=lambda row: (row["median_frequency"], row["mean_frequency"], row["expert_id"]))
        top_rows = expert_rows[:top_k]
        layers[layer_id] = {
            "top_low_frequency_experts": [row["expert_id"] for row in top_rows],
            "experts": expert_rows,
        }
        for row in top_rows:
            model_candidates.append({"layer_id": layer_id, **row})

    model_candidates.sort(
        key=lambda row: (row["median_frequency"], row["mean_frequency"], row["layer_id"], row["expert_id"])
    )
    return {
        "scope": "malicious_client_local_only",
        "completed": len(records) >= target_trainings,
        "target_trainings": int(target_trainings),
        "observed_trainings": len(records),
        "observed_server_rounds": [int(record["server_round"]) for record in records],
        "top_k": int(top_k),
        "layers": layers,
        "model_wide_top_low_frequency_experts": model_candidates[:top_k],
    }


def save_local_lfe_profile(
    output_dir: str,
    client_id: int,
    server_round: int,
    profile: Dict[str, Any],
    target_trainings: int,
    top_k: int,
) -> str:
    profile_dir = _profile_dir(output_dir, client_id)
    os.makedirs(profile_dir, exist_ok=True)
    record = {"client_id": int(client_id), "server_round": int(server_round), **profile}
    round_path = os.path.join(profile_dir, f"round_{server_round:04d}.json")
    _atomic_write_json(round_path, record)

    records: List[Dict[str, Any]] = []
    for path in _round_paths(profile_dir)[:target_trainings]:
        with open(path, "r", encoding="utf-8") as file:
            records.append(json.load(file))
    summary = _build_summary(records=records, top_k=top_k, target_trainings=target_trainings)
    summary["client_id"] = int(client_id)
    summary_path = os.path.join(profile_dir, "lfe_summary.json")
    _atomic_write_json(summary_path, summary)

    csv_path = os.path.join(profile_dir, "lfe_top3.csv")
    fd, temp_path = tempfile.mkstemp(prefix=".lfe_", suffix=".csv", dir=profile_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "layer_id",
                    "rank",
                    "expert_id",
                    "median_frequency",
                    "mean_frequency",
                    "std_frequency",
                    "bottom_top_k_rate",
                ],
            )
            writer.writeheader()
            for layer_id, layer in summary["layers"].items():
                for rank, row in enumerate(layer["experts"][:top_k], start=1):
                    writer.writerow({"layer_id": layer_id, "rank": rank, **row})
        os.replace(temp_path, csv_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    if summary["completed"]:
        _atomic_write_json(os.path.join(profile_dir, "lfe_top3_final.json"), summary)
    return summary_path
