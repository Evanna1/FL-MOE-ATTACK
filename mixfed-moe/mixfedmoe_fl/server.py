from __future__ import annotations

import json
import logging
import timeit
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from flwr.common import NDArrays, Parameters, Scalar, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server import History, Server
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from mixfedmoe_fl.attack import load_attack_checkpoint
from mixfedmoe_fl.client import _load_from_numpy_ndarrays, _state_dict_names, collect_moe_handles
from mixfedmoe_fl.config import MixFedMoEConfig
from mixfedmoe_fl.data import MixFedMoEDataManager
from mixfedmoe_fl.logging_utils import RoundMetricsLogger, UPDATE_DIAGNOSTIC_COLUMNS
from models.switch_transformers import SwitchTransformersForSequenceClassification

_HF_LOGS_SILENCED = False


class ExpertProbeEarlyStoppingServer(Server):
    """Flower server that exits cleanly once the strategy finds a target expert."""

    def fit(self, num_rounds: int, timeout: float | None) -> tuple[History, float]:
        history = History()

        log(logging.INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(logging.INFO, "Starting evaluation of initial global parameters")
        initial_evaluation = self.strategy.evaluate(0, parameters=self.parameters)
        if initial_evaluation is not None:
            loss, metrics = initial_evaluation
            history.add_loss_centralized(server_round=0, loss=loss)
            history.add_metrics_centralized(server_round=0, metrics=metrics)

        start_time = timeit.default_timer()
        for current_round in range(1, num_rounds + 1):
            log(logging.INFO, "")
            log(logging.INFO, "[ROUND %s]", current_round)
            fit_result = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if fit_result is not None:
                parameters_prime, fit_metrics, _ = fit_result
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round,
                    metrics=fit_metrics,
                )

            if bool(
                getattr(self.strategy, "expert_probe_stop_requested", False)
            ):
                found = getattr(
                    self.strategy,
                    "found_low_frequency_expert",
                    None,
                )
                expert = found.get("expert", "unknown") if found else "unknown"
                log(
                    logging.INFO,
                    "Expert probe found %s in round %s; stopping before the next round.",
                    expert,
                    current_round,
                )
                break

            centralized = self.strategy.evaluate(
                current_round,
                parameters=self.parameters,
            )
            if centralized is not None:
                loss, metrics = centralized
                history.add_loss_centralized(server_round=current_round, loss=loss)
                history.add_metrics_centralized(
                    server_round=current_round,
                    metrics=metrics,
                )

            distributed = self.evaluate_round(
                server_round=current_round,
                timeout=timeout,
            )
            if distributed is not None:
                loss, metrics, _ = distributed
                if loss is not None:
                    history.add_loss_distributed(
                        server_round=current_round,
                        loss=loss,
                    )
                    history.add_metrics_distributed(
                        server_round=current_round,
                        metrics=metrics,
                    )

        return history, timeit.default_timer() - start_time


def _resolve_target_router(model, layer_id: str):
    for handle in collect_moe_handles(model):
        if handle.layer_id == layer_id:
            return handle.mlp_module.router
    return None


class _NaturalTriggerRoutingTracker:
    """Measure natural routing only at the exact trigger-token positions.

    The classifier hook records the target expert's softmax probability and
    top-1 choice. The router hook reads the capacity-masked expert assignment,
    which distinguishes a top-1 choice from a token that was actually admitted
    to and processed by the expert.
    """

    def __init__(
        self,
        router: torch.nn.Module,
        target_expert: int,
        trigger_token_ids: Sequence[int],
        decoder: bool = False,
    ) -> None:
        self.target_expert = int(target_expert)
        self.trigger_token_ids = tuple(int(token_id) for token_id in trigger_token_ids)
        if not self.trigger_token_ids:
            raise ValueError("Natural trigger routing requires non-empty trigger_token_ids.")
        self.decoder = bool(decoder)
        self._trigger_mask: Optional[torch.Tensor] = None
        self.target_probability_sum = 0.0
        self.top1_hits = 0
        self.actual_hits = 0
        self.num_trigger_tokens = 0
        self._classifier_handle = router.classifier.register_forward_hook(self._classifier_hook)
        self._router_handle = router.register_forward_hook(self._router_hook)

    def begin_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have matching [batch, sequence] shapes.")
        trigger_length = len(self.trigger_token_ids)
        trigger_tensor = torch.tensor(self.trigger_token_ids, dtype=input_ids.dtype, device=input_ids.device)
        mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for row in range(input_ids.shape[0]):
            active_positions = torch.nonzero(attention_mask[row].bool(), as_tuple=False).flatten()
            if active_positions.numel() < trigger_length + 1:
                raise RuntimeError("Triggered test input is too short to contain trigger tokens followed by EOS.")
            trigger_positions = active_positions[-(trigger_length + 1) : -1]
            actual_ids = input_ids[row, trigger_positions]
            if not torch.equal(actual_ids, trigger_tensor):
                raise RuntimeError(
                    "Triggered test input does not contain the configured trigger_token_ids immediately before EOS."
                )
            mask[row, trigger_positions] = True
        if self.decoder:
            # The sequence-classification model feeds shift_right(input_ids) to
            # decoder routers, so token positions move one step to the right.
            shifted = torch.zeros_like(mask)
            shifted[:, 1:] = mask[:, :-1]
            mask = shifted
        self._trigger_mask = mask.reshape(-1)

    def _selected_rows(self, values: torch.Tensor) -> torch.Tensor:
        if self._trigger_mask is None:
            raise RuntimeError("begin_batch must be called before natural routing is measured.")
        flattened = values.reshape(-1, values.shape[-1])
        if flattened.shape[0] != self._trigger_mask.numel():
            raise RuntimeError(
                "Target router token count does not match the triggered evaluation batch: "
                f"router={flattened.shape[0]} batch={self._trigger_mask.numel()}."
            )
        return flattened[self._trigger_mask]

    def _classifier_hook(self, _module, _inputs, output: torch.Tensor) -> None:
        selected_logits = self._selected_rows(output)
        if not 0 <= self.target_expert < selected_logits.shape[-1]:
            raise ValueError("Target expert is outside the router output range.")
        probabilities = torch.softmax(selected_logits.float(), dim=-1)
        self.target_probability_sum += float(probabilities[:, self.target_expert].sum().item())
        self.top1_hits += int(selected_logits.argmax(dim=-1).eq(self.target_expert).sum().item())
        self.num_trigger_tokens += int(selected_logits.shape[0])

    def _router_hook(self, _module, _inputs, output) -> None:
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError("Target router did not return its capacity-masked expert assignment.")
        selected_experts = self._selected_rows(output[1])
        self.actual_hits += int(selected_experts[:, self.target_expert].bool().sum().item())
        self._trigger_mask = None

    def metrics(self) -> Dict[str, float | int]:
        denominator = max(self.num_trigger_tokens, 1)
        return {
            "trigger_target_probability": float(self.target_probability_sum / denominator),
            "trigger_target_hit_rate": float(self.top1_hits / denominator),
            "trigger_target_execution_rate": float(self.actual_hits / denominator),
            "trigger_routing_tokens": int(self.num_trigger_tokens),
        }

    def remove(self) -> None:
        self._classifier_handle.remove()
        self._router_handle.remove()


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


def _should_evaluate_asr(runtime_config: MixFedMoEConfig, server_round: int) -> bool:
    """Return whether this centralized evaluation should include triggered data."""
    return bool(
        runtime_config.attack_enabled
        or (runtime_config.monitor_asr and server_round >= 0)
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
        checkpoint_path = runtime_config.checkpoint_path or runtime_config.init_checkpoint
        if checkpoint_path:
            state, metadata = load_attack_checkpoint(checkpoint_path)
            model_state = model.state_dict()
            missing = [name for name in model_state if name not in state]
            if missing:
                raise RuntimeError(
                    f"--checkpoint_path '{checkpoint_path}' is missing {len(missing)} "
                    f"parameters (first: {missing[:5]})."
                )
            for name in model_state:
                model_state[name] = torch.as_tensor(state[name]).to(model_state[name].dtype)
            model.load_state_dict(model_state, strict=True)
            source_round = metadata.get("round")
            round_suffix = "" if source_round is None else f" (source round {source_round})"
            print(
                f"[MixFedMoE] Initialized global model from "
                f"{checkpoint_path}{round_suffix}",
                flush=True,
            )
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
            # Flower also invokes centralized evaluation for round 0. Monitoring is
            # intentionally limited to models produced by a completed benign round.
            evaluate_asr = _should_evaluate_asr(runtime_config, server_round)
            triggered_dataset = test_bundle.triggered_test_dataset if evaluate_asr else None
            if (
                evaluate_asr
                and runtime_config.attack_mode == "a3fl"
                and "a3fl_eval_trigger_token_ids_json" in eval_config
            ):
                raw_trigger_ids = json.loads(
                    str(eval_config["a3fl_eval_trigger_token_ids_json"])
                )
                if not isinstance(raw_trigger_ids, list) or not raw_trigger_ids:
                    raise ValueError("Evaluator received an invalid A3FL trigger.")
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in raw_trigger_ids
                ):
                    raise ValueError("Evaluator received invalid A3FL trigger token IDs.")
                trigger_ids = tuple(int(value) for value in raw_trigger_ids)
                trigger_text = str(
                    eval_config.get(
                        "a3fl_eval_trigger_text",
                        data_manager.tokenizer.decode(
                            list(trigger_ids),
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ).strip(),
                    )
                )
                triggered_dataset = data_manager.load_triggered_server_test_dataset(
                    trigger=trigger_text,
                    trigger_token_ids=trigger_ids,
                )
            triggered_loader = _make_eval_loader(
                dataset=triggered_dataset,
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
            trigger_routing_metrics: Optional[Dict[str, float | int]] = None
            if triggered_loader is not None:
                triggered_examples = 0
                target_predictions = 0
                routing_tracker: Optional[_NaturalTriggerRoutingTracker] = None
                if (
                    runtime_config.attack_mode != "a3fl"
                    and runtime_config.lfe_target_layer is not None
                    and runtime_config.lfe_target_expert is not None
                    and runtime_config.trigger_token_ids
                ):
                    target_router = _resolve_target_router(model, runtime_config.lfe_target_layer)
                    if target_router is None:
                        raise RuntimeError(
                            f"Could not resolve target router for layer '{runtime_config.lfe_target_layer}'."
                        )
                    routing_tracker = _NaturalTriggerRoutingTracker(
                        target_router,
                        int(runtime_config.lfe_target_expert),
                        runtime_config.trigger_token_ids,
                        decoder=runtime_config.lfe_target_layer.startswith("decoder."),
                    )
                elif (
                    runtime_config.attack_mode != "a3fl"
                    and runtime_config.lfe_target_layer is not None
                    and runtime_config.lfe_target_expert is not None
                ):
                    print(
                        f"[MixFedMoE][round={server_round}] Trigger routing metrics unavailable: "
                        "trigger_token_ids is empty. Regenerate selected_triggers.json with the current "
                        "iGCG optimizer and pass it via --selected_triggers_path.",
                        flush=True,
                    )
                try:
                    with torch.no_grad():
                        for batch in triggered_loader:
                            input_ids = batch["input_ids"].to(device)
                            attention_mask = batch["attention_mask"].to(device)
                            if routing_tracker is not None:
                                routing_tracker.begin_batch(input_ids, attention_mask)
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                                return_dict=True,
                            )
                            preds = torch.argmax(outputs.logits, dim=-1)
                            target_predictions += int((preds == runtime_config.target_label).sum().item())
                            triggered_examples += int(preds.shape[0])
                    if routing_tracker is not None:
                        trigger_routing_metrics = routing_tracker.metrics()
                finally:
                    if routing_tracker is not None:
                        routing_tracker.remove()
                attack_success_rate = float(target_predictions / max(triggered_examples, 1))

            metrics: Dict[str, Scalar] = {
                "accuracy": eval_accuracy,
                "clean_accuracy": eval_accuracy,
                "effective_test_samples": int(test_bundle.effective_test_samples),
                "mode": runtime_config.mode,
                "attack_enabled": bool(
                    eval_config.get("attack_enabled", runtime_config.attack_enabled)
                ),
                "monitor_asr": bool(
                    eval_config.get("monitor_asr", runtime_config.monitor_asr)
                ),
            }
            if attack_success_rate is not None:
                metrics["asr"] = attack_success_rate
            if trigger_routing_metrics is not None:
                metrics.update(trigger_routing_metrics)
            for key, value in eval_config.items():
                metrics[key] = value
            if attack_success_rate is not None and "local_asr_post" in metrics:
                local_pre = float(metrics.get("local_asr_pre", 0.0))
                local_post = float(metrics["local_asr_post"])
                metrics["local_global_asr_gap"] = float(
                    local_post - attack_success_rate
                )
                local_gain = local_post - local_pre
                if abs(local_gain) > 1e-12:
                    metrics["local_asr_aggregation_retention"] = float(
                        (attack_success_rate - local_pre) / local_gain
                    )

            log_row = {
                "round": int(server_round),
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
                "clean_accuracy": eval_accuracy,
                "asr": attack_success_rate if attack_success_rate is not None else "",
                "local_asr_pre": metrics.get("local_asr_pre", ""),
                "local_asr_post": metrics.get("local_asr_post", ""),
                "local_asr_gain": metrics.get("local_asr_gain", ""),
                "local_global_asr_gap": metrics.get("local_global_asr_gap", ""),
                "local_asr_aggregation_retention": metrics.get(
                    "local_asr_aggregation_retention", ""
                ),
                "local_asr_eval_examples": metrics.get("local_asr_eval_examples", ""),
                "local_asr_num_clients": metrics.get("local_asr_num_clients", ""),
                # Kept in JSONL (not the compact CSV) so multi-attacker runs
                # retain each client's individual pre/post measurement.
                "local_asr_clients_json": metrics.get("local_asr_clients_json", ""),
                "parameter_update_diagnostics_json": metrics.get(
                    "parameter_update_diagnostics_json", ""
                ),
                "trigger_target_probability": (
                    trigger_routing_metrics["trigger_target_probability"]
                    if trigger_routing_metrics is not None
                    else ""
                ),
                "trigger_target_hit_rate": (
                    trigger_routing_metrics["trigger_target_hit_rate"]
                    if trigger_routing_metrics is not None
                    else ""
                ),
                "trigger_target_execution_rate": (
                    trigger_routing_metrics["trigger_target_execution_rate"]
                    if trigger_routing_metrics is not None
                    else ""
                ),
                "trigger_routing_tokens": (
                    trigger_routing_metrics["trigger_routing_tokens"]
                    if trigger_routing_metrics is not None
                    else ""
                ),
                "train_loss": float(metrics["train_loss"]) if "train_loss" in metrics else "",
                "round_time": float(metrics["round_time"]) if "round_time" in metrics else 0.0,
                "cumulative_time": float(metrics["cumulative_time"]) if "cumulative_time" in metrics else 0.0,
                "effective_test_samples": int(test_bundle.effective_test_samples),
                "num_clients_sampled": int(metrics["num_clients_sampled"]) if "num_clients_sampled" in metrics else 0,
                "benign_lfe_activation_rate": metrics.get("benign_lfe_activation_rate", ""),
                "benign_lfe_activation_count": metrics.get("benign_lfe_activation_count", ""),
                "benign_lfe_valid_tokens": metrics.get("benign_lfe_valid_tokens", ""),
                "num_benign_clients_profiled": metrics.get("num_benign_clients_profiled", ""),
                "mode": runtime_config.mode,
                "attack_enabled": bool(
                    metrics.get("attack_enabled", runtime_config.attack_enabled)
                ),
                "monitor_asr": bool(
                    metrics.get("monitor_asr", runtime_config.monitor_asr)
                ),
            }
            for diagnostic_key in UPDATE_DIAGNOSTIC_COLUMNS:
                log_row[diagnostic_key] = metrics.get(diagnostic_key, "")
            metrics_logger.log_round(log_row)
            if attack_success_rate is None:
                print(
                    f"[MixFedMoE][round={server_round}] Clean Accuracy={eval_accuracy:.6f}",
                    flush=True,
                )
            else:
                routing = ""
                if trigger_routing_metrics is not None:
                    routing = (
                        f" TriggerRouteP={trigger_routing_metrics['trigger_target_probability']:.6f}"
                        f" TriggerRouteHit={trigger_routing_metrics['trigger_target_hit_rate']:.6f}"
                        f" TriggerExecHit={trigger_routing_metrics['trigger_target_execution_rate']:.6f}"
                    )
                print(
                    f"[MixFedMoE][round={server_round}] "
                    f"Clean Accuracy={eval_accuracy:.6f} ASR={attack_success_rate:.6f}{routing}"
                    + (
                        f" LocalASRPre={float(metrics['local_asr_pre']):.6f}"
                        f" LocalASRPost={float(metrics['local_asr_post']):.6f}"
                        f" LocalGlobalGap={float(metrics['local_global_asr_gap']):.6f}"
                        if "local_asr_post" in metrics
                        else ""
                    ),
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
