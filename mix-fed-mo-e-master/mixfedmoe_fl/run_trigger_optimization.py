from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from mixfedmoe_fl.config import SUPPORTED_DATASET_NAME_TO_HF
from mixfedmoe_fl.trigger_optimization import (
    TargetExpert,
    add_perplexity_and_score,
    canonical_layer_id,
    discover_sparse_routers,
    evaluate_candidate_routing,
    generate_trigger_candidates,
    load_per_layer_lowest_experts,
    load_target_experts,
    pareto_frontier,
    rank_by_routing,
    set_experiment_seed,
    validate_target,
    write_csv,
)
from models.switch_transformers import SwitchTransformersForSequenceClassification


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen-model low-frequency expert trigger optimization.")
    parser.add_argument("--model_name_or_path", default="model_ckpt/switch-base-8")
    parser.add_argument("--dataset_name", default="emotion")
    parser.add_argument("--dataset_file", default=None, help="Optional local CSV/JSON dataset; preferred for client-local data.")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--profiling_file", default=None)
    parser.add_argument("--target_layer", default=None)
    parser.add_argument("--target_expert", type=int, default=None)
    parser.add_argument("--all_targets", action="store_true")
    parser.add_argument(
        "--target_selection",
        choices=["profile_default", "per_layer_top1"],
        default="profile_default",
        help="Use the profiling file's default targets or exactly the Rank-1 expert from every layer.",
    )
    parser.add_argument("--candidate_size", type=int, default=100)
    parser.add_argument("--trigger_length", type=int, default=1)
    parser.add_argument("--optimization_samples", type=int, default=100)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--lambda_ppl", type=float, default=0.1)
    parser.add_argument("--ppl_model_name_or_path", default="gpt2")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--ppl_max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ppl_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete device such as cuda:0.")
    parser.add_argument("--output_dir", default="outputs/trigger_optimization")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of prior trigger results only.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_names = [
        "candidate_size",
        "trigger_length",
        "optimization_samples",
        "eval_samples",
        "top_k",
        "max_length",
        "ppl_max_length",
        "batch_size",
        "ppl_batch_size",
    ]
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be > 0.")
    if args.top_k > args.candidate_size:
        raise ValueError("--top_k cannot exceed --candidate_size.")
    if args.max_length <= args.trigger_length + 1:
        raise ValueError("--max_length must leave room for clean text, trigger tokens, and EOS.")
    if args.lambda_ppl < 0.0:
        raise ValueError("--lambda_ppl must be >= 0.")
    explicit_count = int(args.target_layer is not None) + int(args.target_expert is not None)
    if explicit_count == 1:
        raise ValueError("--target_layer and --target_expert must be provided together.")
    if explicit_count == 0 and not args.profiling_file:
        raise ValueError("Provide either an explicit target pair or --profiling_file.")
    if args.all_targets and not args.profiling_file:
        raise ValueError("--all_targets requires --profiling_file.")
    if args.target_selection == "per_layer_top1":
        if not args.profiling_file:
            raise ValueError("--target_selection=per_layer_top1 requires --profiling_file.")
        if not args.all_targets:
            raise ValueError("--target_selection=per_layer_top1 requires --all_targets.")
        if explicit_count:
            raise ValueError("Per-layer target selection cannot be combined with an explicit target pair.")


def _resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{raw_device}' requested, but CUDA is unavailable.")
    return device


def _load_checkpoint_state(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a dictionary.")
    has_wrapped_state = "state_dict" in payload
    state_dict = payload["state_dict"] if has_wrapped_state else payload
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Checkpoint does not contain a non-empty state_dict.")
    if not all(isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in state_dict.items()):
        raise ValueError("Checkpoint state_dict must map string names to tensors.")
    metadata = {key: value for key, value in payload.items() if key != "state_dict"} if has_wrapped_state else {}
    return dict(state_dict), metadata


def _infer_num_labels(state_dict: Dict[str, torch.Tensor]) -> int:
    key = "classification_head.out_proj.weight"
    if key not in state_dict:
        raise KeyError(
            f"Clean FL checkpoint is missing '{key}'; a sequence-classification checkpoint is required."
        )
    return int(state_dict[key].shape[0])


def _load_frozen_switch_model(
    model_name_or_path: str,
    checkpoint_path: str,
    tokenizer: Any,
    device: torch.device,
) -> Tuple[SwitchTransformersForSequenceClassification, Dict[str, Any]]:
    state_dict, checkpoint_metadata = _load_checkpoint_state(checkpoint_path)
    model = SwitchTransformersForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=_infer_num_labels(state_dict),
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
        tie_word_embeddings=False,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model state mismatch: "
            f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}."
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
    return model, checkpoint_metadata


def _load_raw_dataset(args: argparse.Namespace) -> Dataset:
    if args.dataset_file:
        extension = Path(args.dataset_file).suffix.lower()
        if extension == ".csv":
            dataset = load_dataset("csv", data_files=args.dataset_file, split="train")
        elif extension in {".json", ".jsonl"}:
            dataset = load_dataset("json", data_files=args.dataset_file, split="train")
        else:
            raise ValueError("--dataset_file must end in .csv, .json, or .jsonl.")
    else:
        dataset_id = SUPPORTED_DATASET_NAME_TO_HF.get(args.dataset_name, args.dataset_name)
        loaded = load_dataset(dataset_id)
        if isinstance(loaded, DatasetDict):
            if args.dataset_split not in loaded:
                raise KeyError(f"Dataset '{dataset_id}' has no split '{args.dataset_split}'.")
            dataset = loaded[args.dataset_split]
        else:
            dataset = loaded
    if args.text_column not in dataset.column_names:
        raise KeyError(f"Dataset is missing text column '{args.text_column}'. Columns={dataset.column_names}.")
    return dataset


def _split_clean_texts(
    dataset: Dataset,
    text_column: str,
    optimization_samples: int,
    eval_samples: int,
    seed: int,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    required = optimization_samples + eval_samples
    if len(dataset) < required:
        raise ValueError(f"Dataset has {len(dataset)} rows but {required} disjoint samples are required.")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=required, replace=False).tolist()
    optimization_indices = [int(index) for index in indices[:optimization_samples]]
    eval_indices = [int(index) for index in indices[optimization_samples:]]
    optimization_texts = [str(dataset[index][text_column]) for index in optimization_indices]
    eval_texts = [str(dataset[index][text_column]) for index in eval_indices]
    return optimization_texts, eval_texts, optimization_indices, eval_indices


def _resolve_targets(args: argparse.Namespace) -> List[TargetExpert]:
    if args.target_layer is not None:
        return [TargetExpert(canonical_layer_id(args.target_layer), int(args.target_expert))]
    if args.target_selection == "per_layer_top1":
        return load_per_layer_lowest_experts(args.profiling_file)
    targets = load_target_experts(args.profiling_file)
    return targets if args.all_targets else [targets[0]]


def _target_directory(base_output_dir: str, dataset_name: str, target: TargetExpert) -> str:
    layer_component = target.layer_id.replace(".", "_")
    return os.path.join(base_output_dir, dataset_name, f"{layer_component}_expert_{target.expert_id}")


def _configure_logger(output_dir: str) -> logging.Logger:
    logger_name = f"trigger_optimization.{os.path.abspath(output_dir)}"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(os.path.join(output_dir, "trigger_optimization.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def _prepare_target_output(args: argparse.Namespace, target: TargetExpert) -> str:
    output_dir = _target_directory(args.output_dir, args.dataset_name, target)
    result_path = os.path.join(output_dir, "selected_triggers.json")
    if os.path.exists(result_path) and not args.overwrite:
        raise FileExistsError(
            f"Trigger result already exists at '{result_path}'. Use a new --output_dir or explicitly pass --overwrite."
        )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _load_frozen_ppl_model(model_name_or_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("PPL tokenizer must define either pad_token_id or eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return model, tokenizer


def _merge_optimization_and_eval_rows(
    optimization_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    optimization_by_id = {int(row["candidate_id"]): row for row in optimization_rows}
    merged: List[Dict[str, Any]] = []
    for eval_row in eval_rows:
        candidate_id = int(eval_row["candidate_id"])
        optimization_row = optimization_by_id[candidate_id]
        row = dict(eval_row)
        for key in [
            "routing_success_rate",
            "mean_target_probability",
            "clean_routing_rate",
            "delta_routing_rate",
            "routing_token_count",
            "clean_token_count",
            "routing_rank",
        ]:
            row[f"optimization_{key}"] = optimization_row[key]
        merged.append(row)
    return merged


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    set_experiment_seed(args.seed)
    device = _resolve_device(args.device)

    switch_tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        model_max_length=args.max_length,
    )
    if switch_tokenizer.eos_token_id is None:
        raise ValueError("Switch tokenizer must define eos_token_id.")
    if switch_tokenizer.pad_token_id is None:
        switch_tokenizer.pad_token = switch_tokenizer.eos_token

    targets = _resolve_targets(args)
    target_output_dirs = {target: _prepare_target_output(args, target) for target in targets}
    dataset = _load_raw_dataset(args)
    optimization_texts, eval_texts, optimization_indices, eval_indices = _split_clean_texts(
        dataset=dataset,
        text_column=args.text_column,
        optimization_samples=args.optimization_samples,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )
    candidates = generate_trigger_candidates(
        tokenizer=switch_tokenizer,
        candidate_size=args.candidate_size,
        trigger_length=args.trigger_length,
        seed=args.seed,
    )
    switch_model, checkpoint_metadata = _load_frozen_switch_model(
        model_name_or_path=args.model_name_or_path,
        checkpoint_path=args.checkpoint_path,
        tokenizer=switch_tokenizer,
        device=device,
    )
    routers = discover_sparse_routers(switch_model)
    for target in targets:
        validate_target(target, routers)

    ppl_model, ppl_tokenizer = _load_frozen_ppl_model(args.ppl_model_name_or_path, device=device)

    for target in targets:
        output_dir = target_output_dirs[target]
        logger = _configure_logger(output_dir)
        logger.info("Starting frozen trigger search for target=%s expert=%d", target.layer_id, target.expert_id)
        logger.info("Device=%s candidates=%d optimization_samples=%d eval_samples=%d", device, len(candidates), len(optimization_texts), len(eval_texts))

        config_payload = dict(vars(args))
        config_payload.update(
            {
                "canonical_target_layer": target.layer_id,
                "canonical_target_expert": target.expert_id,
                "optimization_indices": optimization_indices,
                "eval_indices": eval_indices,
                "checkpoint_metadata": checkpoint_metadata,
                "model_frozen": True,
                "flower_started": False,
            }
        )
        _write_json(os.path.join(output_dir, "run_config.json"), config_payload)

        optimization_rows = evaluate_candidate_routing(
            model=switch_model,
            tokenizer=switch_tokenizer,
            router=routers[target.layer_id],
            target=target,
            texts=optimization_texts,
            candidates=candidates,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )
        write_csv(os.path.join(output_dir, "trigger_candidates.csv"), optimization_rows)
        routing_top_rows = rank_by_routing(optimization_rows, top_k=args.top_k)
        top_candidate_ids = {int(row["candidate_id"]) for row in routing_top_rows}
        top_candidates = [candidate for candidate in candidates if candidate.candidate_id in top_candidate_ids]
        logger.info("Stage 1 complete; evaluating PPL for routing Top-%d", len(top_candidates))

        eval_rows = evaluate_candidate_routing(
            model=switch_model,
            tokenizer=switch_tokenizer,
            router=routers[target.layer_id],
            target=target,
            texts=eval_texts,
            candidates=top_candidates,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )
        merged_rows = _merge_optimization_and_eval_rows(routing_top_rows, eval_rows)
        final_rows = add_perplexity_and_score(
            rows=merged_rows,
            eval_texts=eval_texts,
            ppl_model=ppl_model,
            ppl_tokenizer=ppl_tokenizer,
            ppl_batch_size=args.ppl_batch_size,
            ppl_max_length=args.ppl_max_length,
            lambda_ppl=args.lambda_ppl,
            device=device,
        )
        frontier_rows = pareto_frontier(final_rows)
        write_csv(os.path.join(output_dir, "trigger_optimization_results.csv"), final_rows)
        write_csv(os.path.join(output_dir, "pareto_frontier.csv"), frontier_rows)

        selected_payload = {
            "target": asdict(target),
            "best_trigger": final_rows[0],
            "ranked_triggers": final_rows,
            "pareto_candidate_ids": [int(row["candidate_id"]) for row in frontier_rows],
        }
        _write_json(os.path.join(output_dir, "selected_triggers.json"), selected_payload)
        logger.info(
            "Finished; best_trigger=%r RSR=%.6f PPL=%.6f score=%.6f",
            final_rows[0]["trigger"],
            final_rows[0]["routing_success_rate"],
            final_rows[0]["perplexity"],
            final_rows[0]["combined_score"],
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
