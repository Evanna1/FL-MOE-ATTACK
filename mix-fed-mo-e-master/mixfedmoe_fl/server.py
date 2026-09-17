from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from flwr.common import NDArrays, Parameters, Scalar, ndarrays_to_parameters
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from mixfedmoe_fl.client import _load_from_numpy_ndarrays, _state_dict_names
from mixfedmoe_fl.config import MixFedMoEConfig
from mixfedmoe_fl.data import MixFedMoEDataManager
from mixfedmoe_fl.logging_utils import RoundMetricsLogger
from models.switch_transformers import SwitchTransformersForSequenceClassification

_HF_LOGS_SILENCED = False


def _make_eval_loader(
    dataset: Optional[Any],
    batch_size: int,
    collator: DataCollatorWithPadding,
) -> Optional[DataLoader]:
    """Build an evaluation loader only when its dataset exists."""
    if dataset is None:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )


def _silence_hf_loading_logs_once() -> None:
    global _HF_LOGS_SILENCED
    if _HF_LOGS_SILENCED:
        return
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        from datasets.utils import logging as ds_logging

        ds_logging.set_verbosity_error()
    except Exception:
        pass
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    _HF_LOGS_SILENCED = True


def build_server_model(
    runtime_config: MixFedMoEConfig,
    data_manager: MixFedMoEDataManager,
) -> SwitchTransformersForSequenceClassification:
    _silence_hf_loading_logs_once()
    label_info = data_manager.get_label_info()
    model = SwitchTransformersForSequenceClassification.from_pretrained(
        runtime_config.model_name_or_path,
        num_labels=label_info.num_labels,
        id2label=label_info.id2label,
        label2id=label_info.label2id,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
        tie_word_embeddings=False,
    )
    tokenizer = data_manager.tokenizer
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id
    model.config.use_cache = False
    return model


def build_initial_parameters(
    runtime_config: MixFedMoEConfig,
    data_manager: MixFedMoEDataManager,
) -> Tuple[List[str], Parameters]:
    model = build_server_model(runtime_config=runtime_config, data_manager=data_manager)
    try:
        parameter_names = _state_dict_names(model)
        arrays: NDArrays = [model.state_dict()[name].detach().cpu().numpy().copy() for name in parameter_names]
    finally:
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return parameter_names, ndarrays_to_parameters(arrays)


def make_central_evaluate_fn(
    runtime_config: MixFedMoEConfig,
    data_manager: MixFedMoEDataManager,
    parameter_names: List[str],
    metrics_logger: RoundMetricsLogger,
):
    test_bundle = data_manager.load_server_test_dataset()
    collator = DataCollatorWithPadding(
        tokenizer=data_manager.tokenizer,
        padding=True,
        return_tensors="pt",
    )

    def evaluate_fn(
        server_round: int,
        parameters_ndarrays: NDArrays,
        eval_config: Dict[str, Scalar],
    ):
        model = build_server_model(runtime_config=runtime_config, data_manager=data_manager)
        clean_loader = None
        triggered_loader = None
        try:
            _load_from_numpy_ndarrays(model=model, param_names=parameter_names, parameters=parameters_ndarrays)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            clean_loader = _make_eval_loader(
                dataset=test_bundle.test_dataset,
                batch_size=runtime_config.eval_batch_size,
                collator=collator,
            )
            if clean_loader is None:
                raise RuntimeError("Server clean test dataset is missing.")
            triggered_loader = _make_eval_loader(
                dataset=(
                    test_bundle.triggered_test_dataset
                    if runtime_config.attack_enabled
                    else None
                ),
                batch_size=runtime_config.eval_batch_size,
                collator=collator,
            )

            total_loss = 0.0
            total_examples = 0
            total_correct = 0
            with torch.no_grad():
                for batch in clean_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                        return_dict=True,
                    )
                    if outputs.loss is None:
                        raise RuntimeError("Server evaluation got loss=None from model forward.")
                    batch_size = int(labels.shape[0])
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)
                    total_correct += int((preds == labels).sum().item())
                    total_loss += float(outputs.loss.item()) * batch_size
                    total_examples += batch_size

            eval_loss = float(total_loss / max(total_examples, 1))
            eval_accuracy = float(total_correct / max(total_examples, 1))

            attack_success_rate = None
            if triggered_loader is not None:
                triggered_examples = 0
                target_predictions = 0
                with torch.no_grad():
                    for batch in triggered_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            return_dict=True,
                        )
                        preds = torch.argmax(outputs.logits, dim=-1)
                        target_predictions += int((preds == runtime_config.target_label).sum().item())
                        triggered_examples += int(preds.shape[0])
                attack_success_rate = float(target_predictions / max(triggered_examples, 1))

            metrics: Dict[str, Scalar] = {
                "accuracy": eval_accuracy,
                "clean_accuracy": eval_accuracy,
                "effective_test_samples": int(test_bundle.effective_test_samples),
                "mode": runtime_config.mode,
                "attack_enabled": bool(runtime_config.attack_enabled),
            }
            if attack_success_rate is not None:
                metrics["asr"] = attack_success_rate
            for key, value in eval_config.items():
                metrics[key] = value

            log_row = {
                "round": int(server_round),
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
                "clean_accuracy": eval_accuracy,
                "asr": attack_success_rate if attack_success_rate is not None else "",
                "train_loss": float(metrics["train_loss"]) if "train_loss" in metrics else "",
                "round_time": float(metrics["round_time"]) if "round_time" in metrics else 0.0,
                "cumulative_time": float(metrics["cumulative_time"]) if "cumulative_time" in metrics else 0.0,
                "effective_test_samples": int(test_bundle.effective_test_samples),
                "num_clients_sampled": int(metrics["num_clients_sampled"]) if "num_clients_sampled" in metrics else 0,
                "mode": runtime_config.mode,
                "attack_enabled": bool(runtime_config.attack_enabled),
            }
            metrics_logger.log_round(log_row)
            if attack_success_rate is None:
                print(
                    f"[MixFedMoE][round={server_round}] Clean Accuracy={eval_accuracy:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[MixFedMoE][round={server_round}] "
                    f"Clean Accuracy={eval_accuracy:.6f} ASR={attack_success_rate:.6f}",
                    flush=True,
                )
            return eval_loss, metrics
        finally:
            if clean_loader is not None:
                del clean_loader
            if triggered_loader is not None:
                del triggered_loader
            model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return evaluate_fn
