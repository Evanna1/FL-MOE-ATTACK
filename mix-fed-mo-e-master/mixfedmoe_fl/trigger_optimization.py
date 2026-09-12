from __future__ import annotations

import csv
import json
import math
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


LAYER_ID_PATTERN = re.compile(
    r"^(encoder|decoder)(?:[._](?:layer|block))?[._](\d+)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, order=True)
class TargetExpert:
    layer_id: str
    expert_id: int


@dataclass(frozen=True)
class TriggerCandidate:
    candidate_id: int
    trigger: str
    trigger_token_ids: Tuple[int, ...]

    @property
    def trigger_token_count(self) -> int:
        return len(self.trigger_token_ids)


def set_experiment_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonical_layer_id(layer_id: str) -> str:
    normalized = layer_id.strip().replace("/", ".").replace("_", ".")
    match = LAYER_ID_PATTERN.match(normalized)
    if match is None:
        raise ValueError(
            f"Invalid layer_id='{layer_id}'. Expected encoder.1, encoder.layer.1, "
            "decoder.3, or an equivalent block form."
        )
    return f"{match.group(1).lower()}.{int(match.group(2))}"


def _target_from_mapping(row: Dict[str, Any]) -> TargetExpert:
    if "layer_id" not in row or "expert_id" not in row:
        raise ValueError("Every target entry must contain layer_id and expert_id.")
    expert_id = int(row["expert_id"])
    if expert_id < 0:
        raise ValueError(f"expert_id must be >= 0, got {expert_id}.")
    return TargetExpert(canonical_layer_id(str(row["layer_id"])), expert_id)


def _targets_from_json(payload: Any) -> List[TargetExpert]:
    if isinstance(payload, list):
        return [_target_from_mapping(dict(row)) for row in payload]
    if not isinstance(payload, dict):
        raise ValueError("Target JSON must be an object or a list of target objects.")

    for key in ["selected_low_frequency_experts", "targets", "model_wide_top_low_frequency_experts"]:
        value = payload.get(key)
        if isinstance(value, list):
            return [_target_from_mapping(dict(row)) for row in value]

    layers = payload.get("layers")
    if isinstance(layers, dict):
        targets: List[TargetExpert] = []
        for raw_layer_id, layer_payload in layers.items():
            layer_id = canonical_layer_id(str(raw_layer_id))
            if not isinstance(layer_payload, dict):
                continue
            expert_ids = layer_payload.get("top_low_frequency_experts")
            if isinstance(expert_ids, list):
                targets.extend(TargetExpert(layer_id, int(expert_id)) for expert_id in expert_ids)
        if targets:
            return targets

    if "layer_id" in payload and "expert_id" in payload:
        return [_target_from_mapping(payload)]
    raise ValueError("No (layer_id, expert_id) targets found in JSON profiling file.")


def load_target_experts(path: str) -> List[TargetExpert]:
    lower_path = path.lower()
    if lower_path.endswith(".csv"):
        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        targets = [_target_from_mapping(row) for row in rows]
    elif lower_path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as file:
            targets = _targets_from_json(json.load(file))
    else:
        raise ValueError("profiling_file must be a .csv or .json file.")

    unique_targets: List[TargetExpert] = []
    seen = set()
    for target in targets:
        key = (target.layer_id, target.expert_id)
        if key not in seen:
            seen.add(key)
            unique_targets.append(target)
    if not unique_targets:
        raise ValueError(f"No targets found in profiling file '{path}'.")
    return unique_targets


def _target_sort_key(target: TargetExpert) -> Tuple[int, int, int]:
    stack_name, layer_index = target.layer_id.split(".", 1)
    return (0 if stack_name == "encoder" else 1, int(layer_index), target.expert_id)


def load_per_layer_lowest_experts(path: str) -> List[TargetExpert]:
    """Read exactly the Rank-1 low-frequency expert from every profiled layer."""
    lower_path = path.lower()
    targets: List[TargetExpert] = []
    if lower_path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict) or not isinstance(payload.get("layers"), dict):
            raise ValueError("Per-layer target selection requires a JSON object containing 'layers'.")
        for raw_layer_id, layer_payload in payload["layers"].items():
            if not isinstance(layer_payload, dict):
                raise ValueError(f"Layer '{raw_layer_id}' payload must be an object.")
            ranked_ids = layer_payload.get("top_low_frequency_experts")
            if not isinstance(ranked_ids, list) or not ranked_ids:
                raise ValueError(
                    f"Layer '{raw_layer_id}' has no non-empty top_low_frequency_experts list."
                )
            targets.append(
                TargetExpert(
                    layer_id=canonical_layer_id(str(raw_layer_id)),
                    expert_id=int(ranked_ids[0]),
                )
            )
    elif lower_path.endswith(".csv"):
        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"No targets found in profiling file '{path}'.")
        best_by_layer: Dict[str, Tuple[int, int, TargetExpert]] = {}
        for row_index, row in enumerate(rows):
            target = _target_from_mapping(row)
            rank = int(row["rank"]) if str(row.get("rank", "")).strip() else row_index + 1
            current = best_by_layer.get(target.layer_id)
            candidate_key = (rank, row_index)
            if current is None or candidate_key < current[:2]:
                best_by_layer[target.layer_id] = (rank, row_index, target)
        targets = [entry[2] for entry in best_by_layer.values()]
    else:
        raise ValueError("profiling_file must be a .csv or .json file.")

    for target in targets:
        if target.expert_id < 0:
            raise ValueError(f"expert_id must be >= 0, got {target.expert_id} at {target.layer_id}.")
    targets.sort(key=_target_sort_key)
    if not targets:
        raise ValueError(f"No per-layer Rank-1 targets found in profiling file '{path}'.")
    return targets


def discover_sparse_routers(model: nn.Module) -> Dict[str, nn.Linear]:
    routers: Dict[str, nn.Linear] = {}
    for stack_name in ["encoder", "decoder"]:
        stack = getattr(model, stack_name, None)
        if stack is None or not hasattr(stack, "block"):
            raise RuntimeError(f"Model does not expose the expected '{stack_name}.block' structure.")
        for layer_idx, block in enumerate(stack.block):
            feed_forward = block.layer[-1]
            if not bool(getattr(feed_forward, "is_sparse", False)):
                continue
            mlp = getattr(feed_forward, "mlp", None)
            router = getattr(getattr(mlp, "router", None), "classifier", None)
            if not isinstance(router, nn.Linear):
                raise RuntimeError(f"Sparse layer {stack_name}.{layer_idx} has no linear router classifier.")
            routers[f"{stack_name}.{layer_idx}"] = router
    if not routers:
        raise RuntimeError("No sparse Switch Transformer routers were found.")
    return routers


def validate_target(target: TargetExpert, routers: Dict[str, nn.Linear]) -> None:
    if target.layer_id not in routers:
        available = ", ".join(sorted(routers))
        raise ValueError(f"Target layer '{target.layer_id}' not found. Available sparse layers: {available}.")
    num_experts = int(routers[target.layer_id].out_features)
    if target.expert_id >= num_experts:
        raise ValueError(
            f"Target expert {target.expert_id} is invalid for {target.layer_id}, "
            f"which has {num_experts} experts."
        )


def _is_natural_candidate_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 48:
        return False
    if "\ufffd" in stripped or not any(character.isalpha() for character in stripped):
        return False
    if any(unicodedata.category(character).startswith("C") for character in stripped):
        return False
    visible = sum(character.isalnum() or character.isspace() or character in "-'.," for character in stripped)
    return visible / len(stripped) >= 0.8


def generate_trigger_candidates(
    tokenizer: Any,
    candidate_size: int,
    trigger_length: int,
    seed: int,
) -> List[TriggerCandidate]:
    if candidate_size <= 0 or trigger_length <= 0:
        raise ValueError("candidate_size and trigger_length must be > 0.")
    special_ids = set(int(token_id) for token_id in tokenizer.all_special_ids)
    vocab_ids = sorted(set(int(token_id) for token_id in tokenizer.get_vocab().values()))
    usable_ids: List[int] = []
    for token_id in vocab_ids:
        if token_id in special_ids:
            continue
        decoded = tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if _is_natural_candidate_text(decoded):
            usable_ids.append(token_id)
    if not usable_ids:
        raise RuntimeError("No valid non-special candidate tokens remain after filtering.")

    rng = random.Random(seed)
    candidates: List[TriggerCandidate] = []
    seen: set[Tuple[int, ...]] = set()
    max_attempts = max(candidate_size * 500, 1000)
    for _ in range(max_attempts):
        sampled_ids = [rng.choice(usable_ids) for _ in range(trigger_length)]
        text = tokenizer.decode(sampled_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        if not _is_natural_candidate_text(text):
            continue
        encoded = tokenizer(
            " " + text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        normalized_ids = tuple(int(token_id) for token_id in encoded)
        if len(normalized_ids) != trigger_length or normalized_ids in seen:
            continue
        if any(token_id in special_ids for token_id in normalized_ids):
            continue
        normalized_text = tokenizer.decode(
            list(normalized_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if not _is_natural_candidate_text(normalized_text):
            continue
        seen.add(normalized_ids)
        candidates.append(
            TriggerCandidate(
                candidate_id=len(candidates),
                trigger=normalized_text,
                trigger_token_ids=normalized_ids,
            )
        )
        if len(candidates) == candidate_size:
            return candidates
    raise RuntimeError(
        f"Generated only {len(candidates)}/{candidate_size} unique candidates after {max_attempts} attempts. "
        "Reduce --candidate_size or --trigger_length."
    )


def _tokenize_clean_prefix(tokenizer: Any, text: str, budget: int) -> List[int]:
    if budget <= 0:
        raise ValueError("No token budget remains for clean text.")
    encoded = tokenizer(
        str(text),
        add_special_tokens=False,
        truncation=True,
        max_length=budget,
        return_attention_mask=False,
    )["input_ids"]
    return [int(token_id) for token_id in encoded]


def _prepare_switch_batch(
    texts: Sequence[str],
    tokenizer: Any,
    max_length: int,
    device: torch.device,
    trigger_token_ids: Optional[Sequence[int]] = None,
    reserved_trigger_tokens: int = 0,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Switch tokenizer must define eos_token_id.")
    trigger_ids = [int(token_id) for token_id in (trigger_token_ids or [])]
    clean_budget = max_length - len(trigger_ids) - reserved_trigger_tokens - 1
    encoded_rows: List[Dict[str, List[int]]] = []
    clean_masks: List[List[int]] = []
    trigger_masks: List[List[int]] = []
    for text in texts:
        clean_ids = _tokenize_clean_prefix(tokenizer, text, clean_budget)
        input_ids = clean_ids + trigger_ids + [int(eos_token_id)]
        encoded_rows.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
        clean_masks.append([1] * len(clean_ids) + [0] * (len(trigger_ids) + 1))
        trigger_masks.append([0] * len(clean_ids) + [1] * len(trigger_ids) + [0])

    batch = tokenizer.pad(encoded_rows, padding=True, return_tensors="pt")
    width = int(batch["input_ids"].shape[1])
    padded_clean_masks = [row + [0] * (width - len(row)) for row in clean_masks]
    padded_trigger_masks = [row + [0] * (width - len(row)) for row in trigger_masks]
    model_batch = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    return (
        model_batch,
        torch.tensor(padded_clean_masks, dtype=torch.bool, device=device),
        torch.tensor(padded_trigger_masks, dtype=torch.bool, device=device),
    )


def _decoder_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    source_mask = attention_mask.clone()
    last_valid_positions = attention_mask.sum(dim=1, keepdim=True).long() - 1
    source_mask.scatter_(1, last_valid_positions, 0)
    return torch.cat([torch.ones_like(attention_mask[:, :1]), source_mask[:, :-1]], dim=1)


def _shift_boolean_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1)


def _forward_and_capture_router(
    model: nn.Module,
    router: nn.Linear,
    stack_name: str,
    model_batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    captured: List[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        captured.append(output.detach().float())

    handle = router.register_forward_hook(hook)
    try:
        input_ids = model_batch["input_ids"]
        attention_mask = model_batch["attention_mask"]
        decoder_input_ids = model._shift_right(input_ids)  # type: ignore[attr-defined]
        decoder_mask = _decoder_attention_mask(attention_mask)
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one router output for {stack_name} target, captured {len(captured)}.")
    return captured[0]


def _routing_statistics(logits: torch.Tensor, mask: torch.Tensor, expert_id: int) -> Dict[str, float]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_mask = mask.reshape(-1)
    if flat_logits.shape[0] != flat_mask.numel():
        raise RuntimeError(
            f"Router/mask token mismatch: logits={flat_logits.shape[0]} mask={flat_mask.numel()}."
        )
    selected_logits = flat_logits[flat_mask]
    token_count = int(selected_logits.shape[0])
    if token_count == 0:
        raise RuntimeError("No tokens were available for routing evaluation.")
    probabilities = torch.softmax(selected_logits, dim=-1)
    top1 = torch.argmax(selected_logits, dim=-1)
    successes = int(top1.eq(expert_id).sum().item())
    return {
        "token_count": float(token_count),
        "success_count": float(successes),
        "probability_sum": float(probabilities[:, expert_id].sum().item()),
    }


def _merge_routing_statistics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    token_count = sum(row["token_count"] for row in rows)
    success_count = sum(row["success_count"] for row in rows)
    probability_sum = sum(row["probability_sum"] for row in rows)
    if token_count <= 0:
        raise RuntimeError("Routing evaluation produced zero target tokens.")
    return {
        "token_count": int(token_count),
        "routing_rate": float(success_count / token_count),
        "mean_target_probability": float(probability_sum / token_count),
    }


def evaluate_clean_routing(
    model: nn.Module,
    tokenizer: Any,
    router: nn.Linear,
    target: TargetExpert,
    texts: Sequence[str],
    trigger_token_count: int,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    stack_name = target.layer_id.split(".", 1)[0]
    rows: List[Dict[str, float]] = []
    for start in range(0, len(texts), batch_size):
        model_batch, clean_mask, _ = _prepare_switch_batch(
            texts=texts[start : start + batch_size],
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
            trigger_token_ids=None,
            reserved_trigger_tokens=trigger_token_count,
        )
        logits = _forward_and_capture_router(model, router, stack_name, model_batch)
        selected_mask = clean_mask if stack_name == "encoder" else _shift_boolean_mask(clean_mask)
        rows.append(_routing_statistics(logits, selected_mask, target.expert_id))
    return _merge_routing_statistics(rows)


def evaluate_trigger_routing(
    model: nn.Module,
    tokenizer: Any,
    router: nn.Linear,
    target: TargetExpert,
    texts: Sequence[str],
    candidate: TriggerCandidate,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    stack_name = target.layer_id.split(".", 1)[0]
    rows: List[Dict[str, float]] = []
    for start in range(0, len(texts), batch_size):
        model_batch, _, trigger_mask = _prepare_switch_batch(
            texts=texts[start : start + batch_size],
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
            trigger_token_ids=candidate.trigger_token_ids,
        )
        logits = _forward_and_capture_router(model, router, stack_name, model_batch)
        selected_mask = trigger_mask if stack_name == "encoder" else _shift_boolean_mask(trigger_mask)
        rows.append(_routing_statistics(logits, selected_mask, target.expert_id))
    return _merge_routing_statistics(rows)


def evaluate_candidate_routing(
    model: nn.Module,
    tokenizer: Any,
    router: nn.Linear,
    target: TargetExpert,
    texts: Sequence[str],
    candidates: Sequence[TriggerCandidate],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    if not texts or not candidates:
        raise ValueError("Routing evaluation requires non-empty texts and candidates.")
    trigger_lengths = {candidate.trigger_token_count for candidate in candidates}
    if len(trigger_lengths) != 1:
        raise ValueError("All candidates in one routing evaluation must have the same token length.")
    clean = evaluate_clean_routing(
        model=model,
        tokenizer=tokenizer,
        router=router,
        target=target,
        texts=texts,
        trigger_token_count=next(iter(trigger_lengths)),
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    results: List[Dict[str, Any]] = []
    for candidate in candidates:
        triggered = evaluate_trigger_routing(
            model=model,
            tokenizer=tokenizer,
            router=router,
            target=target,
            texts=texts,
            candidate=candidate,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        results.append(
            {
                "candidate_id": candidate.candidate_id,
                "trigger": candidate.trigger,
                "trigger_token_ids": json.dumps(list(candidate.trigger_token_ids)),
                "trigger_token_count": candidate.trigger_token_count,
                "target_layer": target.layer_id,
                "target_expert": target.expert_id,
                "routing_success_rate": triggered["routing_rate"],
                "mean_target_probability": triggered["mean_target_probability"],
                "clean_routing_rate": clean["routing_rate"],
                "delta_routing_rate": triggered["routing_rate"] - clean["routing_rate"],
                "routing_token_count": triggered["token_count"],
                "clean_token_count": clean["token_count"],
            }
        )
    return results


def rank_by_routing(rows: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -float(row["routing_success_rate"]),
            -float(row["mean_target_probability"]),
            -float(row["delta_routing_rate"]),
            int(row["candidate_id"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["routing_rank"] = rank
    return ranked[: min(top_k, len(ranked))]


def build_triggered_texts(texts: Sequence[str], trigger: str) -> List[str]:
    clean_trigger = trigger.strip()
    if not clean_trigger:
        raise ValueError("Trigger text must not be empty.")
    return [f"{str(text).rstrip()} {clean_trigger}" for text in texts]


def corpus_perplexity(
    model: nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> float:
    if not texts:
        raise ValueError("PPL evaluation requires at least one text.")
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].bool()
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)
        total_nll += float(token_losses[shift_mask].sum().item())
        total_tokens += int(shift_mask.sum().item())
    if total_tokens <= 0:
        raise RuntimeError("PPL evaluation produced zero next-token targets.")
    return float(math.exp(min(total_nll / total_tokens, 50.0)))


def add_perplexity_and_score(
    rows: Sequence[Dict[str, Any]],
    eval_texts: Sequence[str],
    ppl_model: nn.Module,
    ppl_tokenizer: Any,
    ppl_batch_size: int,
    ppl_max_length: int,
    lambda_ppl: float,
    device: torch.device,
) -> List[Dict[str, Any]]:
    if lambda_ppl < 0.0:
        raise ValueError("lambda_ppl must be >= 0.")
    clean_ppl = corpus_perplexity(
        model=ppl_model,
        tokenizer=ppl_tokenizer,
        texts=eval_texts,
        batch_size=ppl_batch_size,
        max_length=ppl_max_length,
        device=device,
    )
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        candidate_row = dict(row)
        triggered_ppl = corpus_perplexity(
            model=ppl_model,
            tokenizer=ppl_tokenizer,
            texts=build_triggered_texts(eval_texts, str(row["trigger"])),
            batch_size=ppl_batch_size,
            max_length=ppl_max_length,
            device=device,
        )
        candidate_row["clean_perplexity"] = clean_ppl
        candidate_row["perplexity"] = triggered_ppl
        candidate_row["delta_perplexity"] = triggered_ppl - clean_ppl
        enriched.append(candidate_row)

    ppl_values = np.asarray([float(row["perplexity"]) for row in enriched], dtype=np.float64)
    ppl_min = float(ppl_values.min())
    ppl_range = float(ppl_values.max() - ppl_min)
    for row in enriched:
        normalized = 0.0 if ppl_range == 0.0 else (float(row["perplexity"]) - ppl_min) / ppl_range
        row["normalized_perplexity"] = float(normalized)
        row["combined_score"] = float(row["routing_success_rate"]) - lambda_ppl * float(normalized)
    enriched.sort(
        key=lambda row: (
            -float(row["combined_score"]),
            -float(row["routing_success_rate"]),
            float(row["perplexity"]),
            int(row["candidate_id"]),
        )
    )
    for rank, row in enumerate(enriched, start=1):
        row["final_rank"] = rank
    return enriched


def pareto_frontier(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frontier: List[Dict[str, Any]] = []
    for candidate in rows:
        candidate_rsr = float(candidate["routing_success_rate"])
        candidate_ppl = float(candidate["perplexity"])
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_rsr = float(other["routing_success_rate"])
            other_ppl = float(other["perplexity"])
            if (
                other_rsr >= candidate_rsr
                and other_ppl <= candidate_ppl
                and (other_rsr > candidate_rsr or other_ppl < candidate_ppl)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(dict(candidate))
    frontier.sort(key=lambda row: (float(row["perplexity"]), -float(row["routing_success_rate"])))
    return frontier


def candidate_to_dict(candidate: TriggerCandidate) -> Dict[str, Any]:
    payload = asdict(candidate)
    payload["trigger_token_ids"] = json.dumps(list(candidate.trigger_token_ids))
    payload["trigger_token_count"] = candidate.trigger_token_count
    return payload


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV '{path}'.")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
