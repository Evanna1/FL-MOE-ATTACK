from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from xml.sax.saxutils import escape

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Accumulate padding-aware, pre-capacity Top-1 routing during training.
class TrainingLFERoutingTracker:

    def __init__(
        self,
        moe_handles: Sequence[Any],
        top_k: int,
        expert_id_maps: Optional[Dict[str, Sequence[int]]] = None,
    ) -> None:
        self.top_k = int(top_k)
        self.layer_stats: Dict[str, Dict[str, Any]] = {}
        self.active_masks: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []
        self.num_batches = 0
        self.num_examples = 0

        for handle in moe_handles:
            router = getattr(getattr(handle.mlp_module, "router"), "classifier")
            if not isinstance(router, nn.Linear):
                raise RuntimeError(f"Expected a linear router classifier at layer '{handle.layer_id}'.")
            num_experts = int(router.out_features)
            expert_ids = list(
                expert_id_maps.get(handle.layer_id, range(num_experts))
                if expert_id_maps is not None
                else range(num_experts)
            )
            if len(expert_ids) != num_experts:
                raise ValueError(
                    f"Expert ID map size mismatch at '{handle.layer_id}': "
                    f"ids={len(expert_ids)} router_outputs={num_experts}."
                )
            self.layer_stats[handle.layer_id] = {
                "stack": handle.stack_name,
                "expert_ids": [int(value) for value in expert_ids],
                "counts": [0 for _ in range(num_experts)],
                "probability_sums": [0.0 for _ in range(num_experts)],
                "margin_sums": [0.0 for _ in range(num_experts)],
            }
            self.hooks.append(
                router.register_forward_hook(self._make_hook(handle.layer_id, handle.stack_name))
            )

    def _make_hook(self, layer_id: str, stack_name: str):
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            if stack_name not in self.active_masks:
                raise RuntimeError(f"No active-token mask set for {stack_name} router '{layer_id}'.")
            logits = output.detach().float().reshape(-1, output.shape[-1])
            mask = self.active_masks[stack_name].reshape(-1)
            if logits.shape[0] != mask.numel():
                raise RuntimeError(
                    f"Router/mask size mismatch at {layer_id}: "
                    f"router_tokens={logits.shape[0]} mask_tokens={mask.numel()}."
                )
            valid_logits = logits[mask]
            if valid_logits.numel() == 0:
                return
            probabilities = torch.softmax(valid_logits, dim=-1)
            top_n = min(2, probabilities.shape[-1])
            top_values, top_indices = torch.topk(probabilities, k=top_n, dim=-1)
            selected = top_indices[:, 0]
            margins = top_values[:, 0] - top_values[:, 1] if top_n == 2 else top_values[:, 0]
            stats = self.layer_stats[layer_id]
            for expert_idx in range(valid_logits.shape[-1]):
                expert_mask = selected.eq(expert_idx)
                count = int(expert_mask.sum().item())
                stats["counts"][expert_idx] += count
                if count:
                    stats["probability_sums"][expert_idx] += float(
                        top_values[expert_mask, 0].sum().item()
                    )
                    stats["margin_sums"][expert_idx] += float(margins[expert_mask].sum().item())

        return hook

    # Set masks for the next model forward and count the observed batch.
    def begin_batch(self, attention_mask: torch.Tensor) -> None:
        encoder_mask = attention_mask.detach().bool()
        decoder_source_mask = encoder_mask.clone()
        last_valid_positions = encoder_mask.sum(dim=1, keepdim=True).long() - 1
        if torch.any(last_valid_positions < 0):
            raise ValueError("Each training sequence must contain at least one active token.")
        decoder_source_mask.scatter_(1, last_valid_positions, False)
        decoder_mask = torch.cat(
            [torch.ones_like(encoder_mask[:, :1]), decoder_source_mask[:, :-1]], dim=1
        )
        self.active_masks["encoder"] = encoder_mask
        self.active_masks["decoder"] = decoder_mask
        self.num_batches += 1
        self.num_examples += int(attention_mask.shape[0])

    def finish(self) -> Dict[str, Any]:
        self.close()
        return {
            "profile_source": "training_router_hooks",
            "training_batches": int(self.num_batches),
            "training_examples": int(self.num_examples),
            "layers": _finalize_layer_stats(self.layer_stats, self.top_k),
        }

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def _finalize_layer_stats(layer_stats: Dict[str, Dict[str, Any]], top_k: int) -> Dict[str, Any]:
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
        expert_ids = [int(value) for value in stats.get("expert_ids", range(len(counts)))]
        layers[layer_id] = {
            "stack": stats["stack"],
            "expert_ids": expert_ids,
            "valid_tokens": total,
            "counts": counts,
            "frequencies": frequencies,
            "mean_selected_probability": mean_probabilities,
            "mean_top1_top2_margin": mean_margins,
            "top_low_frequency_experts": [
                expert_ids[idx] for idx in ranking[: min(top_k, len(ranking))]
            ],
        }
    return layers


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


# def should_collect_lfe_profile(
#     output_dir: str,
#     client_id: int,
#     server_round: int,
#     target_trainings: int,
# ) -> bool:
#     profile_dir = _profile_dir(output_dir, client_id)
#     current_path = os.path.join(profile_dir, f"round_{server_round:04d}.json")
#     return os.path.exists(current_path) or len(_round_paths(profile_dir)) < target_trainings


# def collect_lfe_routing_profile(
#     model: nn.Module,
#     moe_handles: Sequence[Any],
#     dataset: Any,
#     collator: Any,
#     batch_size: int,
#     max_samples: int,
#     seed: int,
#     device: torch.device,
#     top_k: int,
# ) -> Dict[str, Any]:
#     """Collect padding-aware, pre-capacity Top-1 router demand using local hooks."""
#     sample_count = min(max_samples, len(dataset))
#     if sample_count <= 0:
#         raise ValueError("LFE calibration dataset must not be empty.")

#     rng = np.random.default_rng(seed)
#     selected_indices = np.sort(rng.choice(len(dataset), size=sample_count, replace=False)).tolist()
#     subset = dataset.select(selected_indices)
#     loader = DataLoader(
#         subset,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=collator,
#         num_workers=0,
#         pin_memory=False,
#     )

#     layer_stats: Dict[str, Dict[str, Any]] = {}
#     active_masks: Dict[str, torch.Tensor] = {}
#     hooks: List[Any] = []

#     for handle in moe_handles:
#         router = getattr(getattr(handle.mlp_module, "router"), "classifier")
#         if not isinstance(router, nn.Linear):
#             raise RuntimeError(f"Expected a linear router classifier at layer '{handle.layer_id}'.")
#         num_experts = int(router.out_features)
#         layer_stats[handle.layer_id] = {
#             "stack": handle.stack_name,
#             "counts": [0 for _ in range(num_experts)],
#             "probability_sums": [0.0 for _ in range(num_experts)],
#             "margin_sums": [0.0 for _ in range(num_experts)],
#         }

#         def make_hook(layer_id: str, stack_name: str):
#             def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
#                 logits = output.detach().float()
#                 flat_logits = logits.reshape(-1, logits.shape[-1])
#                 mask = active_masks[stack_name].reshape(-1)
#                 if flat_logits.shape[0] != mask.numel():
#                     raise RuntimeError(
#                         f"Router/mask size mismatch at {layer_id}: "
#                         f"router_tokens={flat_logits.shape[0]} mask_tokens={mask.numel()}."
#                     )
#                 valid_logits = flat_logits[mask]
#                 if valid_logits.numel() == 0:
#                     return
#                 probabilities = torch.softmax(valid_logits, dim=-1)
#                 top_values, top_indices = torch.topk(probabilities, k=2, dim=-1)
#                 selected = top_indices[:, 0]
#                 margins = top_values[:, 0] - top_values[:, 1]
#                 stats = layer_stats[layer_id]
#                 for expert_idx in range(valid_logits.shape[-1]):
#                     expert_mask = selected.eq(expert_idx)
#                     count = int(expert_mask.sum().item())
#                     stats["counts"][expert_idx] += count
#                     if count:
#                         stats["probability_sums"][expert_idx] += float(top_values[expert_mask, 0].sum().item())
#                         stats["margin_sums"][expert_idx] += float(margins[expert_mask].sum().item())

#             return hook

#         hooks.append(router.register_forward_hook(make_hook(handle.layer_id, handle.stack_name)))

#     was_training = model.training
#     model.eval()
#     try:
#         with torch.inference_mode():
#             for batch in loader:
#                 input_ids = batch["input_ids"].to(device)
#                 attention_mask = batch["attention_mask"].to(device)
#                 decoder_input_ids = model._shift_right(input_ids)  # type: ignore[attr-defined]
#                 decoder_source_mask = attention_mask.clone()
#                 last_valid_positions = attention_mask.sum(dim=1, keepdim=True).long() - 1
#                 decoder_source_mask.scatter_(1, last_valid_positions, 0)
#                 decoder_attention_mask = torch.cat(
#                     [torch.ones_like(attention_mask[:, :1]), decoder_source_mask[:, :-1]],
#                     dim=1,
#                 )
#                 active_masks["encoder"] = attention_mask.bool()
#                 active_masks["decoder"] = decoder_attention_mask.bool()
#                 model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     decoder_input_ids=decoder_input_ids,
#                     decoder_attention_mask=decoder_attention_mask,
#                     use_cache=False,
#                     return_dict=True,
#                 )
#     finally:
#         for hook in hooks:
#             hook.remove()
#         if was_training:
#             model.train()

#     layers = _finalize_layer_stats(layer_stats, top_k)

#     index_bytes = ",".join(str(index) for index in selected_indices).encode("utf-8")
#     return {
#         "calibration_samples": sample_count,
#         "calibration_indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
#         "profile_seed": int(seed),
#         "layers": layers,
#     }


# def _build_summary(records: Sequence[Dict[str, Any]], top_k: int, target_trainings: int) -> Dict[str, Any]:
#     layer_ids = sorted(records[0]["layers"].keys())
#     layers: Dict[str, Any] = {}
#     model_candidates: List[Dict[str, Any]] = []
#     for layer_id in layer_ids:
#         num_experts = len(records[0]["layers"][layer_id]["frequencies"])
#         expert_rows: List[Dict[str, Any]] = []
#         for expert_idx in range(num_experts):
#             values = np.asarray(
#                 [record["layers"][layer_id]["frequencies"][expert_idx] for record in records],
#                 dtype=np.float64,
#             )
#             bottom_hits = sum(
#                 expert_idx in record["layers"][layer_id]["top_low_frequency_experts"]
#                 for record in records
#             )
#             row = {
#                 "expert_id": expert_idx,
#                 "median_frequency": float(np.median(values)),
#                 "mean_frequency": float(np.mean(values)),
#                 "std_frequency": float(np.std(values)),
#                 "bottom_top_k_rate": float(bottom_hits / len(records)),
#             }
#             expert_rows.append(row)
#         expert_rows.sort(key=lambda row: (row["median_frequency"], row["mean_frequency"], row["expert_id"]))
#         top_rows = expert_rows[:top_k]
#         layers[layer_id] = {
#             "top_low_frequency_experts": [row["expert_id"] for row in top_rows],
#             "experts": expert_rows,
#         }
#         for row in top_rows:
#             model_candidates.append({"layer_id": layer_id, **row})

#     model_candidates.sort(
#         key=lambda row: (row["median_frequency"], row["mean_frequency"], row["layer_id"], row["expert_id"])
#     )
#     return {
#         "scope": "malicious_client_local_only",
#         "completed": len(records) >= target_trainings,
#         "target_trainings": int(target_trainings),
#         "observed_trainings": len(records),
#         "observed_server_rounds": [int(record["server_round"]) for record in records],
#         "top_k": int(top_k),
#         "layers": layers,
#         "model_wide_top_low_frequency_experts": model_candidates[:top_k],
#     }


# def save_local_lfe_profile(
#     output_dir: str,
#     client_id: int,
#     server_round: int,
#     profile: Dict[str, Any],
#     target_trainings: int,
#     top_k: int,
# ) -> str:
#     profile_dir = _profile_dir(output_dir, client_id)
#     os.makedirs(profile_dir, exist_ok=True)
#     record = {"client_id": int(client_id), "server_round": int(server_round), **profile}
#     round_path = os.path.join(profile_dir, f"round_{server_round:04d}.json")
#     _atomic_write_json(round_path, record)

#     records: List[Dict[str, Any]] = []
#     for path in _round_paths(profile_dir)[:target_trainings]:
#         with open(path, "r", encoding="utf-8") as file:
#             records.append(json.load(file))
#     summary = _build_summary(records=records, top_k=top_k, target_trainings=target_trainings)
#     summary["client_id"] = int(client_id)
#     summary_path = os.path.join(profile_dir, "lfe_summary.json")
#     _atomic_write_json(summary_path, summary)

#     csv_path = os.path.join(profile_dir, "lfe_top3.csv")
#     fd, temp_path = tempfile.mkstemp(prefix=".lfe_", suffix=".csv", dir=profile_dir)
#     try:
#         with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
#             writer = csv.DictWriter(
#                 file,
#                 fieldnames=[
#                     "layer_id",
#                     "rank",
#                     "expert_id",
#                     "median_frequency",
#                     "mean_frequency",
#                     "std_frequency",
#                     "bottom_top_k_rate",
#                 ],
#             )
#             writer.writeheader()
#             for layer_id, layer in summary["layers"].items():
#                 for rank, row in enumerate(layer["experts"][:top_k], start=1):
#                     writer.writerow({"layer_id": layer_id, "rank": rank, **row})
#         os.replace(temp_path, csv_path)
#     finally:
#         if os.path.exists(temp_path):
#             os.unlink(temp_path)

#     if summary["completed"]:
#         _atomic_write_json(os.path.join(profile_dir, "lfe_top3_final.json"), summary)
#     return summary_path


def _excel_column(index: int) -> str:
    result = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _worksheet_xml(rows: Sequence[Sequence[Any]]) -> str:
    xml_rows: List[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells: List[str] = []
        for column_index, value in enumerate(row):
            reference = f"{_excel_column(column_index)}{row_index}"
            if value is None:
                cells.append(f'<c r="{reference}"/>')
            elif isinstance(value, bool):
                cells.append(f'<c r="{reference}" t="b"><v>{int(value)}</v></c>')
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{reference}"><v>{value}</v></c>')
            else:
                text = escape(str(value))
                cells.append(f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>')
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )


def _atomic_write_xlsx(path: str, sheets: Sequence[tuple[str, Sequence[Sequence[Any]]]]) -> None:
    """Write a small dependency-free XLSX workbook atomically."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".lfe_", suffix=".xlsx", dir=directory)
    os.close(fd)
    try:
        sheet_entries = "".join(
            f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
            for idx, (name, _rows) in enumerate(sheets, start=1)
        )
        workbook_rels = "".join(
            '<Relationship '
            f'Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
            for idx in range(1, len(sheets) + 1)
        )
        overrides = "".join(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for idx in range(1, len(sheets) + 1)
        )
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/xl/workbook.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                f'{overrides}</Types>',
            )
            archive.writestr(
                "_rels/.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="xl/workbook.xml"/></Relationships>',
            )
            archive.writestr(
                "xl/workbook.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f'<sheets>{sheet_entries}</sheets></workbook>',
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f'{workbook_rels}</Relationships>',
            )
            for idx, (_name, rows) in enumerate(sheets, start=1):
                archive.writestr(f"xl/worksheets/sheet{idx}.xml", _worksheet_xml(rows))
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def save_training_lfe_profile(
    output_dir: str,
    client_id: int,
    server_round: int,
    mode: str,
    profile: Dict[str, Any],
) -> str:
    """Persist one client's round profile and rebuild its cumulative Excel workbook."""
    profile_dir = _profile_dir(output_dir, client_id)
    record = {
        "client_id": int(client_id),
        "server_round": int(server_round),
        "mode": str(mode),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        **profile,
    }
    _atomic_write_json(os.path.join(profile_dir, f"round_{server_round:04d}.json"), record)

    records: List[Dict[str, Any]] = []
    for round_path in _round_paths(profile_dir):
        with open(round_path, "r", encoding="utf-8") as file:
            records.append(json.load(file))

    round_rows: List[List[Any]] = [[
        "client_id", "server_round", "mode", "profile_source", "training_batches",
        "training_examples", "recorded_at_utc",
    ]]
    low_frequency_rows: List[List[Any]] = [[
        "client_id", "server_round", "mode", "layer_id", "stack", "rank",
        "expert_id", "frequency", "count", "valid_tokens",
    ]]
    detail_rows: List[List[Any]] = [[
        "client_id", "server_round", "mode", "layer_id", "stack", "expert_id",
        "frequency", "count", "valid_tokens", "mean_selected_probability",
        "mean_top1_top2_margin", "is_low_frequency",
    ]]
    for item in records:
        common = [item["client_id"], item["server_round"], item.get("mode", "")]
        round_rows.append(common + [
            item.get("profile_source", "calibration_inference"),
            item.get("training_batches", ""),
            item.get("training_examples", item.get("calibration_samples", "")),
            item.get("recorded_at_utc", ""),
        ])
        for layer_id, layer in sorted(item["layers"].items()):
            counts = layer["counts"]
            frequencies = layer["frequencies"]
            expert_ids = layer.get("expert_ids", list(range(len(counts))))
            probabilities = layer["mean_selected_probability"]
            margins = layer["mean_top1_top2_margin"]
            low_ids = layer["top_low_frequency_experts"]
            for rank, expert_id in enumerate(low_ids, start=1):
                local_idx = expert_ids.index(expert_id)
                low_frequency_rows.append(common + [
                    layer_id, layer["stack"], rank, expert_id, frequencies[local_idx],
                    counts[local_idx], layer["valid_tokens"],
                ])
            for local_idx, expert_id in enumerate(expert_ids):
                detail_rows.append(common + [
                    layer_id, layer["stack"], expert_id, frequencies[local_idx],
                    counts[local_idx], layer["valid_tokens"], probabilities[local_idx],
                    margins[local_idx], expert_id in low_ids,
                ])

    workbook_path = os.path.join(profile_dir, "lfe_rounds.xlsx")
    _atomic_write_xlsx(
        workbook_path,
        [
            ("rounds", round_rows),
            ("low_frequency", low_frequency_rows),
            ("all_experts", detail_rows),
        ],
    )
    return workbook_path
