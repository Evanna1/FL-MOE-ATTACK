from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from mixfedmoe_fl.attack import load_selected_trigger
from mixfedmoe_fl.defenses import SUPPORTED_DEFENSES

SUPPORTED_DATASET_NAME_TO_HF: Dict[str, str] = {
    "ag_news": "ag_news",
    "imdb": "imdb",
    "20news": "SetFit/20_newsgroups",
    "sst2": "SetFit/sst2",
    "yelp_polarity": "yelp_polarity",
    "emotion": "dair-ai/emotion",
}

SUPPORTED_MODES: List[str] = ["full", "mix", "drop", "flex"]
SUPPORTED_ASSIGNMENT_POLICIES: List[str] = ["hot", "random"]
SUPPORTED_DATA_DISTRIBUTIONS: List[str] = ["iid", "non-iid"]
DEFAULT_BADNET_TRIGGER = "BadMagic"
# Tokenized with model_ckpt/switch-base-8 and add_special_tokens=False.
DEFAULT_BADNET_TRIGGER_TOKEN_IDS: Tuple[int, ...] = (3862, 22909, 447)


def supported_dataset_names() -> List[str]:
    return list(SUPPORTED_DATASET_NAME_TO_HF.keys())


def _normalize_data_distribution(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"noniid", "niid"}:
        return "non-iid"
    return normalized


@dataclass(frozen=True)
class MixFedMoEConfig:
    mode: Literal["full", "mix", "drop", "flex"]
    model_name_or_path: str
    dataset_name: str
    num_clients: int
    num_rounds: int
    k: int
    alpha: float
    fraction_fit: float
    fraction_evaluate: float
    local_epochs: float
    learning_rate: float
    weight_decay: float
    train_batch_size: int
    eval_batch_size: int
    max_length: int
    client_eval_ratio: float
    seed: int
    num_cpus_per_client: int
    num_gpus_per_client: float
    output_dir: str
    test_samples: int
    data_distribution: Literal["iid", "non-iid"] = "non-iid"
    assignment_policy: Literal["hot", "random"] = "hot"
    coverage_guarantee: bool = True
    checkpoint_path: Optional[str] = None
    init_checkpoint: Optional[str] = None
    attack_enabled: bool = False
    monitor_asr: bool = False
    measure_local_asr: bool = False
    measure_poison_gradient_flow: bool = False
    poison_gradient_samples: int = 64
    malicious_clients: Tuple[int, ...] = (0,)
    poison_rate: float = 0.1
    global_random_poison_source: bool = False
    iid_malicious_client_data: bool = False
    target_label: int = 0
    attack_start_round: int = 2
    attack_end_round: int = 0
    inject_target_expert: bool = False
    malicious_target_expert_router_only: bool = False
    isolate_poison_gradients_to_target: bool = False
    target_expert_poison_gradient_goal_ratio: float = 0.25
    target_expert_poison_gradient_max_gain: float = 3.0
    target_router_loss_weight: float = 0.1
    target_router_margin: float = 2.0
    target_router_kl_weight: float = 0.1
    target_router_full_weight_below: float = 0.5
    target_router_stop_median: float = 1.5
    target_router_stop_p10: float = 0.3
    attack_mode: str = "badnet"
    neurotoxin_mask_ratio: float = 0.95
    neurotoxin_aggregate_all_layers: bool = True
    neurotoxin_max_update_norm: float = 0.0
    neurotoxin_update_scale: float = 1.0
    a3fl_outer_steps: int = 2
    a3fl_trigger_steps: int = 2
    a3fl_candidate_batch_size: int = 32
    a3fl_top_k: int = 32
    a3fl_calibration_samples: int = 16
    a3fl_adv_steps: int = 1
    a3fl_adv_lr: float = 2e-5
    a3fl_lambda: float = 0.01
    a3fl_early_stop_asr: float = 0.95
    malicious_lr: float = 0.0
    malicious_local_epochs: float = 0.0
    trigger: str = DEFAULT_BADNET_TRIGGER
    trigger_token_ids: Tuple[int, ...] = DEFAULT_BADNET_TRIGGER_TOKEN_IDS
    selected_triggers_path: Optional[str] = None
    lfe_target_layer: Optional[str] = None
    lfe_target_expert: Optional[int] = None
    checkpoint_interval: int = 5
    checkpoint_rounds: Tuple[int, ...] = ()
    lfe_profile_enabled: bool = False
    lfe_profile_trainings: int = 20
    lfe_top_k: int = 3
    lfe_calibration_samples: int = 512
    defense: str = "none"
    defense_trim_ratio: float = 0.2
    defense_krum_m: int = 0
    defense_num_malicious: int = 0
    expert_probe_enabled: bool = False
    expert_probe_target_layer: Optional[str] = None
    expert_probe_auto_layer: bool = True
    expert_probe_layer_top_experts: int = 2
    expert_probe_max_layers: int = 0
    expert_probe_drift_window: int = 5
    expert_probe_min_layer_contrast: float = 1.0
    expert_probe_max_pilot_attempts: int = 2
    expert_probe_batch_size: int = 2
    expert_probe_screen_samples: int = 32
    expert_probe_trigger_bank_size: int = 4
    expert_probe_trigger_length: int = 6
    expert_probe_boundary_margin: float = 0.0
    expert_probe_boundary_quantile: float = 0.8
    expert_probe_warmup_observations: int = 3
    expert_probe_pairs: int = 8
    expert_probe_candidate_pool: int = 128
    expert_probe_relative_scale: float = 1e-4
    expert_probe_max_relative_scale: float = 1e-3
    expert_probe_noise_multiplier: float = 2.0
    expert_probe_repetitions: int = 5
    expert_probe_confirmation_rounds: int = 5
    expert_probe_single_retention_min: float = 0.9
    expert_probe_single_retention_max: float = 1.1
    expert_probe_max_consistency_error: float = 0.1
    expert_probe_spoof_activation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MixFedMoE Flower simulation configuration.")
    parser.add_argument("--mode", type=str, choices=SUPPORTED_MODES, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="model_ckpt/switch-base-8")
    parser.add_argument("--dataset_name", type=str, choices=supported_dataset_names(), default="ag_news")
    parser.add_argument(
        "--data_distribution",
        "--data_mode",
        type=_normalize_data_distribution,
        choices=SUPPORTED_DATA_DISTRIBUTIONS,
        default="non-iid",
        help=(
            "Client data partitioning mode: 'iid' for random balanced partitions, "
            "or 'non-iid' for Dirichlet label-skewed partitions (default)."
        ),
    )
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--K", "--k", dest="k", type=int, default=4)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration for label-skewed Non-IID partitioning; smaller is more skewed.",
    )
    parser.add_argument("--fraction_fit", type=float, default=1.0)
    parser.add_argument("--fraction_evaluate", type=float, default=0.0)
    parser.add_argument("--local_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--client_eval_ratio", type=float, default=0, help="Deprecated!")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_cpus_per_client", type=int, default=4)
    parser.add_argument("--num_gpus_per_client", type=float, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs/mixfedmoe")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a trained checkpoint used to initialize the global model before poisoning "
        "training. Accepts this project's Flower round checkpoints, model_state_dict wrappers, "
        "or a raw PyTorch state dict.",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Deprecated alias for --checkpoint_path.",
    )
    parser.add_argument(
        "--assignment_policy",
        type=str,
        choices=SUPPORTED_ASSIGNMENT_POLICIES,
        default="hot",
        help="Expert assignment policy for mix/drop/flex modes (flex requires hot).",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=1000,
        help="If >0, evaluate on at most this many test samples; if <=0, use full test split.",
    )
    parser.add_argument(
        "--coverage_guarantee",
        action="store_true",
        default=True,
        help="Enforce global expert coverage (orphan expert repair) for hot policy in mix/drop modes. "
        "Use --no_coverage_guarantee to disable.",
    )
    parser.add_argument(
        "--no_coverage_guarantee",
        action="store_false",
        dest="coverage_guarantee",
        help="Disable global expert coverage (orphan expert repair) for hot policy.",
    )
    parser.add_argument(
        "--attack_enabled",
        action="store_true",
        default=False,
        help="Enable the selected --attack_mode on malicious clients.",
    )
    parser.add_argument(
        "--monitor_asr",
        action="store_true",
        default=False,
        help=(
            "Run completely benign federated training and evaluate natural ASR on the "
            "triggered server test set after every completed round. This option is mutually "
            "exclusive with --attack_enabled."
        ),
    )
    parser.add_argument(
        "--measure_local_asr",
        action="store_true",
        default=False,
        help=(
            "In mix-mode attack rounds, evaluate each active malicious client before and "
            "after local training on the same triggered server test set used for global ASR."
        ),
    )
    parser.add_argument(
        "--measure_poison_gradient_flow",
        action="store_true",
        default=False,
        help=(
            "In active mix-mode malicious clients, measure clean and poisoned CE-loss "
            "gradient flow before local optimization without taking an optimizer step."
        ),
    )
    parser.add_argument(
        "--poison_gradient_samples",
        type=int,
        default=64,
        help="Maximum clean and poisoned local examples used by each gradient-flow probe.",
    )
    parser.add_argument(
        "--malicious_clients",
        type=int,
        nargs="*",
        default=[0],
        help="Partition IDs of malicious clients (default: 0).",
    )
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument(
        "--global_random_poison_source",
        action="store_true",
        default=False,
        help=(
            "Randomly draw poisoned examples from the complete training split and "
            "replace the same number of malicious-client rows. No label balancing is used."
        ),
    )
    parser.add_argument(
        "--iid_malicious_client_data",
        action="store_true",
        default=False,
        help=(
            "Replace each malicious client's complete local partition with an equally "
            "sized random sample from the full training split. Benign clients stay Non-IID."
        ),
    )
    parser.add_argument(
        "--attack_mode",
        type=str,
        choices=["badnet", "neurotoxin", "a3fl"],
        default="badnet",
        help="Malicious training objective: standard BadNet, Neurotoxin, or the independent "
        "A3FL adaptive-trigger baseline.",
    )
    parser.add_argument(
        "--neurotoxin_mask_ratio",
        type=float,
        default=0.95,
        help=(
            "Fraction of smallest global-update coordinates to keep. Used by classic "
            "Neurotoxin over its complete gradient, or by poison-gradient isolation over "
            "the combined non-target parameter pool. In poison-gradient isolation, 1.0 "
            "keeps every non-target coordinate and disables activity-based masking."
        ),
    )
    parser.add_argument(
        "--neurotoxin_aggregate_all_layers",
        action="store_true",
        default=True,
        help="Build one exact global bottom-k mask across all trainable tensors (BackFed default).",
    )
    parser.add_argument(
        "--neurotoxin_mask_per_layer",
        action="store_false",
        dest="neurotoxin_aggregate_all_layers",
        help="Build an independent exact bottom-k mask for each trainable tensor.",
    )
    parser.add_argument(
        "--neurotoxin_max_update_norm",
        type=float,
        default=0.0,
        help="Optional L2 radius for projected malicious local training; 0 disables projection.",
    )
    parser.add_argument(
        "--neurotoxin_update_scale",
        type=float,
        default=1.0,
        help="Scale the final masked malicious model delta before upload; 1 leaves it unchanged.",
    )
    parser.add_argument("--a3fl_outer_steps", type=int, default=2)
    parser.add_argument("--a3fl_trigger_steps", type=int, default=2)
    parser.add_argument("--a3fl_candidate_batch_size", type=int, default=32)
    parser.add_argument("--a3fl_top_k", type=int, default=32)
    parser.add_argument("--a3fl_calibration_samples", type=int, default=16)
    parser.add_argument("--a3fl_adv_steps", type=int, default=1)
    parser.add_argument(
        "--a3fl_adv_lr",
        type=float,
        default=2e-5,
        help="Learning rate for the adversarial model that unlearns the current trigger.",
    )
    parser.add_argument(
        "--a3fl_lambda",
        type=float,
        default=0.01,
        help="A3FL lambda_0 coefficient; the effective adversarial loss weight is "
        "lambda_0 * cosine_similarity(theta'_t, theta_t).",
    )
    parser.add_argument("--a3fl_early_stop_asr", type=float, default=0.95)
    parser.add_argument(
        "--malicious_lr",
        type=float,
        default=0.0,
        help="Learning rate the malicious client uses during attack rounds (0 => same as honest). "
        "A stronger attacker trains its target expert harder to write a durable payload.",
    )
    parser.add_argument(
        "--malicious_local_epochs",
        type=float,
        default=0.0,
        help="Local epochs the malicious client runs during attack rounds (0 => same as honest).",
    )
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--attack_start_round", type=int, default=2)
    parser.add_argument(
        "--attack_end_round",
        type=int,
        default=0,
        help="Last round the malicious client participates in the attack (<=0 means never leaves). "
        "Rounds after this measure backdoor persistence once the attacker has departed.",
    )
    parser.add_argument(
        "--inject_target_expert",
        action="store_true",
        default=False,
        help="Malicious-client deviation: promote the target low-frequency expert into the next "
        "round's hot top-k by swapping its reported activation count with the current k-th expert. "
        "The server remains authoritative over assignments.",
    )
    parser.add_argument(
        "--malicious_target_expert_router_only",
        action="store_true",
        default=False,
        help=(
            "During malicious attack rounds, freeze every parameter except the configured "
            "target expert and the router at its MoE layer. If the target expert is not in "
            "the current server assignment, only that router is trainable until promotion "
            "places the expert in a later assignment."
        ),
    )
    parser.add_argument(
        "--isolate_poison_gradients_to_target",
        action="store_true",
        default=False,
        help=(
            "Keep the ordinary per-example poison weight and one optimizer step per batch, "
            "leave target-expert/router poison CE gradients unmasked, adaptively raise the "
            "target-expert gradient share under a fixed total poison-gradient norm, and apply poison "
            "CE only to the globally least-active fraction of every other trainable "
            "coordinate selected by --neurotoxin_mask_ratio. Clean CE gradients retain "
            "the mode's ordinary trainable-parameter scope."
        ),
    )
    parser.add_argument(
        "--target_expert_poison_gradient_goal_ratio",
        type=float,
        default=0.25,
        help=(
            "Target share of the masked poison-gradient L2 norm. When the target "
            "expert's EMA share is lower, its gradient is amplified and the complete "
            "poison gradient is renormalized to its original norm. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--target_expert_poison_gradient_max_gain",
        type=float,
        default=3.0,
        help="Maximum adaptive pre-normalization gain for target-expert poison gradients.",
    )
    parser.add_argument(
        "--target_router_loss_weight",
        type=float,
        default=0.1,
        help="Weight of the adaptive trigger-to-target-expert margin objective.",
    )
    parser.add_argument(
        "--target_router_margin",
        type=float,
        default=2.0,
        help="Required target-logit lead over the strongest non-target expert.",
    )
    parser.add_argument(
        "--target_router_kl_weight",
        type=float,
        default=0.1,
        help=(
            "Relative clean-token KL weight inside the target-router objective; the "
            "whole objective is scaled by --target_router_loss_weight."
        ),
    )
    parser.add_argument(
        "--target_router_full_weight_below",
        type=float,
        default=0.5,
        help="Use full trigger margin weight while the batch median margin is at or below this value.",
    )
    parser.add_argument(
        "--target_router_stop_median",
        type=float,
        default=1.5,
        help="Stop the trigger margin term once its batch median reaches this value.",
    )
    parser.add_argument(
        "--target_router_stop_p10",
        type=float,
        default=0.3,
        help="Also require the batch P10 trigger margin to reach this value before stopping.",
    )
    parser.add_argument(
        "--lfe_target_layer",
        type=str,
        default=None,
        help="Target sparse MoE layer id (e.g. 'encoder.5') for cold-expert injection; "
        "overrides the value read from --selected_triggers_path.",
    )
    parser.add_argument(
        "--lfe_target_expert",
        type=int,
        default=None,
        help="Target (cold) expert index for cold-expert injection; overrides --selected_triggers_path.",
    )
    parser.add_argument("--trigger", type=str, default=DEFAULT_BADNET_TRIGGER)
    parser.add_argument(
        "--selected_triggers_path",
        "--selected_triggers_file",
        type=str,
        default=None,
        help=(
            "Enable a low-frequency-expert trigger source by reading best_trigger.trigger and target metadata "
            "from one trigger-optimization selected_triggers.json. This overrides --trigger while preserving "
            "the standard BadNet training and output pipeline."
        ),
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5,
        help=(
            "Save a server checkpoint every N rounds (default: 5); "
            "set to 0 to disable periodic checkpoints."
        ),
    )
    parser.add_argument(
        "--checkpoint_rounds",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Additional server rounds to checkpoint, kept for compatibility. "
            "Periodic checkpoints are controlled by --checkpoint_interval."
        ),
    )
    parser.add_argument(
        "--lfe_profile_enabled",
        action="store_true",
        default=False,
        help=(
            "Enable per-client, per-round low-frequency expert profiling during local training "
            "and write JSON/XLSX results under output_dir/lfe_profiles."
        ),
    )
    parser.add_argument(
        "--lfe_profile_trainings",
        type=int,
        default=20,
        help="Deprecated compatibility option; training-hook profiling now runs every participated round.",
    )
    parser.add_argument(
        "--lfe_top_k",
        type=int,
        default=3,
        help="Number of low-frequency experts reported for every sparse MoE layer.",
    )
    parser.add_argument(
        "--lfe_calibration_samples",
        type=int,
        default=512,
        help="Maximum local examples used by the fast no-gradient activation-profile pass, "
        "including the round-one warmup.",
    )
    parser.add_argument(
        "--defense",
        type=str,
        choices=SUPPORTED_DEFENSES,
        default="none",
        help="Server robust-aggregation defense applied in aggregate_fit (mix mode): "
        "norm_clip, coord_median, trimmed_mean, multi_krum.",
    )
    parser.add_argument("--defense_trim_ratio", type=float, default=0.2)
    parser.add_argument(
        "--defense_krum_m",
        type=int,
        default=0,
        help="Per-group Multi-Krum: contributors to keep in each shared/expert group "
        "(0 => group contributors - num_malicious).",
    )
    parser.add_argument(
        "--defense_num_malicious",
        type=int,
        default=0,
        help="Assumed number of malicious clients for Krum (0 => len(--malicious_clients)).",
    )
    parser.add_argument(
        "--expert_probe_enabled",
        action="store_true",
        default=False,
        help="Enable the malicious client's cross-round paired expert-retention probe.",
    )
    parser.add_argument(
        "--expert_probe_target_layer",
        type=str,
        default=None,
        help=(
            "Optional compatibility/debug restriction to one MoE layer. If omitted, "
            "the malicious client probes experts across all MoE layers in parallel."
        ),
    )
    parser.add_argument(
        "--expert_probe_auto_layer",
        action="store_true",
        default=True,
        help="Automatically discover a promising MoE layer when no target layer is specified.",
    )
    parser.add_argument(
        "--no_expert_probe_auto_layer",
        action="store_false",
        dest="expert_probe_auto_layer",
        help="Disable automatic layer discovery; requires --expert_probe_target_layer.",
    )
    parser.add_argument("--expert_probe_layer_top_experts", type=int, default=2)
    parser.add_argument("--expert_probe_max_layers", type=int, default=0)
    parser.add_argument("--expert_probe_drift_window", type=int, default=5)
    parser.add_argument("--expert_probe_min_layer_contrast", type=float, default=1.0)
    parser.add_argument("--expert_probe_max_pilot_attempts", type=int, default=2)
    parser.add_argument(
        "--expert_probe_batch_size",
        type=int,
        default=2,
        help=(
            "Maximum experts probed in parallel across the model. In automatic "
            "cross-layer mode, each layer is additionally capped at K-2 targets so "
            "at least two ordinary hot-expert slots remain."
        ),
    )
    parser.add_argument("--expert_probe_screen_samples", type=int, default=32)
    parser.add_argument("--expert_probe_trigger_bank_size", type=int, default=4)
    parser.add_argument("--expert_probe_trigger_length", type=int, default=6)
    parser.add_argument("--expert_probe_boundary_margin", type=float, default=0.0)
    parser.add_argument("--expert_probe_boundary_quantile", type=float, default=0.8)
    parser.add_argument("--expert_probe_warmup_observations", type=int, default=3)
    parser.add_argument("--expert_probe_pairs", type=int, default=8)
    parser.add_argument("--expert_probe_candidate_pool", type=int, default=128)
    parser.add_argument("--expert_probe_relative_scale", type=float, default=1e-4)
    parser.add_argument("--expert_probe_max_relative_scale", type=float, default=1e-3)
    parser.add_argument("--expert_probe_noise_multiplier", type=float, default=2.0)
    parser.add_argument("--expert_probe_repetitions", type=int, default=5)
    parser.add_argument("--expert_probe_confirmation_rounds", type=int, default=5)
    parser.add_argument("--expert_probe_single_retention_min", type=float, default=0.9)
    parser.add_argument("--expert_probe_single_retention_max", type=float, default=1.1)
    parser.add_argument("--expert_probe_max_consistency_error", type=float, default=0.1)
    parser.add_argument(
        "--expert_probe_spoof_activation",
        "--expert_probe_force_assignment",
        dest="expert_probe_spoof_activation",
        action="store_true",
        default=True,
        help=(
            "Promote the probe target into the next round's hot top-k by swapping "
            "reported expert activation counts. The old "
            "--expert_probe_force_assignment name is retained as an alias."
        ),
    )
    parser.add_argument(
        "--no_expert_probe_spoof_activation",
        "--no_expert_probe_force_assignment",
        action="store_false",
        dest="expert_probe_spoof_activation",
        help="Do not alter the probe client's reported expert activation counts.",
    )
    return parser


def validate_config(config: MixFedMoEConfig) -> None:
    if config.monitor_asr and config.attack_enabled:
        raise ValueError("--monitor_asr and --attack_enabled are mutually exclusive.")
    if config.measure_local_asr and not config.attack_enabled:
        raise ValueError("--measure_local_asr requires --attack_enabled.")
    if config.measure_local_asr and config.mode != "mix":
        raise ValueError("--measure_local_asr is currently supported only for --mode=mix.")
    if config.measure_poison_gradient_flow and not config.attack_enabled:
        raise ValueError("--measure_poison_gradient_flow requires --attack_enabled.")
    if config.measure_poison_gradient_flow and config.mode != "mix":
        raise ValueError(
            "--measure_poison_gradient_flow is currently supported only for --mode=mix."
        )
    if config.measure_poison_gradient_flow and (
        config.lfe_target_layer is None or config.lfe_target_expert is None
    ):
        raise ValueError(
            "--measure_poison_gradient_flow requires --lfe_target_layer and "
            "--lfe_target_expert."
        )
    if config.poison_gradient_samples <= 0:
        raise ValueError("--poison_gradient_samples must be > 0.")
    if (
        config.checkpoint_path is not None
        and config.init_checkpoint is not None
        and config.checkpoint_path != config.init_checkpoint
    ):
        raise ValueError("--checkpoint_path and --init_checkpoint must refer to the same file.")
    if config.num_clients <= 0:
        raise ValueError("--num_clients must be > 0.")
    if config.num_rounds <= 0:
        raise ValueError("--num_rounds must be > 0.")
    if config.data_distribution not in SUPPORTED_DATA_DISTRIBUTIONS:
        raise ValueError(
            f"--data_distribution must be one of {SUPPORTED_DATA_DISTRIBUTIONS}, "
            f"got '{config.data_distribution}'."
        )
    if config.alpha <= 0.0:
        raise ValueError("--alpha must be > 0 for Dirichlet Non-IID partitioning.")
    if config.k <= 0:
        raise ValueError("--K/--k must be > 0.")
    if not 0.0 < config.fraction_fit <= 1.0:
        raise ValueError("--fraction_fit must be in (0, 1].")
    if not 0.0 <= config.fraction_evaluate <= 1.0:
        raise ValueError("--fraction_evaluate must be in [0, 1].")
    if config.local_epochs <= 0:
        raise ValueError("--local_epochs must be > 0.")
    if config.learning_rate <= 0.0:
        raise ValueError("--learning_rate must be > 0.")
    if config.train_batch_size <= 0:
        raise ValueError("--train_batch_size must be > 0.")
    if config.eval_batch_size <= 0:
        raise ValueError("--eval_batch_size must be > 0.")
    if config.max_length < 2:
        raise ValueError("--max_length must be >= 2.")
    if config.num_cpus_per_client <= 0:
        raise ValueError("--num_cpus_per_client must be > 0.")
    if config.num_gpus_per_client < 0.0:
        raise ValueError("--num_gpus_per_client must be >= 0.")
    if config.assignment_policy not in SUPPORTED_ASSIGNMENT_POLICIES:
        raise ValueError(
            f"--assignment_policy must be one of {SUPPORTED_ASSIGNMENT_POLICIES}, " f"got '{config.assignment_policy}'."
        )
    if config.mode == "flex" and config.k != 1:
        raise ValueError("--mode=flex requires --K/--k to be exactly 1.")
    if config.mode == "flex" and config.assignment_policy != "hot":
        raise ValueError("--mode=flex requires --assignment_policy=hot.")
    if not 0.0 <= config.poison_rate <= 1.0:
        raise ValueError("--poison_rate must be in [0, 1].")
    if config.global_random_poison_source and not config.attack_enabled:
        raise ValueError("--global_random_poison_source requires --attack_enabled.")
    if config.iid_malicious_client_data and not config.attack_enabled:
        raise ValueError("--iid_malicious_client_data requires --attack_enabled.")
    if config.iid_malicious_client_data and config.global_random_poison_source:
        raise ValueError(
            "--iid_malicious_client_data and --global_random_poison_source are "
            "mutually exclusive experiment modes."
        )
    if config.target_label < 0:
        raise ValueError("--target_label must be >= 0.")
    if config.attack_start_round <= 0:
        raise ValueError("--attack_start_round must be >= 1.")
    if config.attack_end_round < 0:
        raise ValueError("--attack_end_round must be >= 0 (0 means the attacker never leaves).")
    if config.attack_mode not in {"badnet", "neurotoxin", "a3fl"}:
        raise ValueError("--attack_mode must be 'badnet', 'neurotoxin', or 'a3fl'.")
    if not 0.0 < config.neurotoxin_mask_ratio <= 1.0:
        raise ValueError("--neurotoxin_mask_ratio must be in (0, 1].")
    if config.attack_mode == "neurotoxin" and config.neurotoxin_mask_ratio >= 1.0:
        raise ValueError(
            "Classic --attack_mode=neurotoxin requires "
            "--neurotoxin_mask_ratio to be in (0, 1)."
        )
    if config.neurotoxin_max_update_norm < 0.0:
        raise ValueError("--neurotoxin_max_update_norm must be >= 0.")
    if config.neurotoxin_update_scale <= 0.0:
        raise ValueError("--neurotoxin_update_scale must be > 0.")
    if config.a3fl_outer_steps <= 0 or config.a3fl_trigger_steps <= 0:
        raise ValueError("--a3fl_outer_steps and --a3fl_trigger_steps must be > 0.")
    if config.a3fl_candidate_batch_size <= 0 or config.a3fl_top_k <= 0:
        raise ValueError("--a3fl_candidate_batch_size and --a3fl_top_k must be > 0.")
    if config.a3fl_calibration_samples <= 0 or config.a3fl_adv_steps <= 0:
        raise ValueError("--a3fl_calibration_samples and --a3fl_adv_steps must be > 0.")
    if config.a3fl_adv_lr <= 0.0:
        raise ValueError("--a3fl_adv_lr must be > 0.")
    if config.a3fl_lambda < 0.0:
        raise ValueError("--a3fl_lambda (lambda_0) must be >= 0.")
    if not 0.0 <= config.a3fl_early_stop_asr <= 1.0:
        raise ValueError("--a3fl_early_stop_asr must be in [0, 1].")
    if config.attack_mode == "a3fl" and config.inject_target_expert:
        raise ValueError("--attack_mode=a3fl is independent of --inject_target_expert.")
    if config.attack_end_round > 0 and config.attack_end_round < config.attack_start_round:
        raise ValueError("--attack_end_round must be >= --attack_start_round when set.")
    if config.target_router_loss_weight < 0.0:
        raise ValueError("--target_router_loss_weight must be >= 0.")
    if not 0.0 <= config.target_expert_poison_gradient_goal_ratio <= 1.0:
        raise ValueError(
            "--target_expert_poison_gradient_goal_ratio must be in [0, 1]."
        )
    if config.target_expert_poison_gradient_max_gain < 1.0:
        raise ValueError(
            "--target_expert_poison_gradient_max_gain must be >= 1."
        )
    if config.target_router_margin <= 0.0:
        raise ValueError("--target_router_margin must be > 0.")
    if config.target_router_kl_weight < 0.0:
        raise ValueError("--target_router_kl_weight must be >= 0.")
    if config.target_router_full_weight_below >= config.target_router_stop_median:
        raise ValueError(
            "--target_router_full_weight_below must be smaller than "
            "--target_router_stop_median."
        )
    if config.target_router_stop_p10 > config.target_router_stop_median:
        raise ValueError(
            "--target_router_stop_p10 must be <= --target_router_stop_median."
        )
    if config.inject_target_expert:
        if not config.attack_enabled:
            raise ValueError("--inject_target_expert requires --attack_enabled.")
        if config.mode == "full":
            raise ValueError(
                "--inject_target_expert requires --mode=mix, drop, or flex so the next-round "
                "assignment can be influenced by activation reporting."
            )
        if config.assignment_policy != "hot":
            raise ValueError(
                "--inject_target_expert requires --assignment_policy=hot because the target is "
                "promoted through reported activation counts."
            )
        if config.lfe_target_layer is None or config.lfe_target_expert is None:
            raise ValueError(
                "--inject_target_expert requires a target expert: set --lfe_target_layer and "
                "--lfe_target_expert (or provide --selected_triggers_path)."
            )
    if config.malicious_target_expert_router_only:
        if not config.attack_enabled:
            raise ValueError(
                "--malicious_target_expert_router_only requires --attack_enabled."
            )
        if config.attack_mode == "a3fl":
            raise ValueError(
                "--malicious_target_expert_router_only is not supported with --attack_mode=a3fl."
            )
        if config.lfe_target_layer is None or config.lfe_target_expert is None:
            raise ValueError(
                "--malicious_target_expert_router_only requires --lfe_target_layer and "
                "--lfe_target_expert (or --selected_triggers_path)."
            )
    if config.isolate_poison_gradients_to_target:
        if not config.attack_enabled:
            raise ValueError(
                "--isolate_poison_gradients_to_target requires --attack_enabled."
            )
        if config.attack_mode in {"a3fl", "neurotoxin"}:
            raise ValueError(
                "--isolate_poison_gradients_to_target is not supported with "
                f"--attack_mode={config.attack_mode}."
            )
        if config.lfe_target_layer is None or config.lfe_target_expert is None:
            raise ValueError(
                "--isolate_poison_gradients_to_target requires --lfe_target_layer and "
                "--lfe_target_expert (or --selected_triggers_path)."
            )
        if config.malicious_target_expert_router_only:
            raise ValueError(
                "--isolate_poison_gradients_to_target and "
                "--malicious_target_expert_router_only are mutually exclusive."
            )
    if not config.trigger.strip():
        raise ValueError("--trigger must not be empty.")
    if config.trigger_token_ids and len(config.trigger_token_ids) + 1 >= config.max_length:
        raise ValueError(
            "The selected trigger_token_ids must leave room for EOS and at least one text token "
            "within --max_length."
        )
    if (
        config.selected_triggers_path is not None
        and not config.attack_enabled
        and not config.monitor_asr
    ):
        raise ValueError("--selected_triggers_path requires --attack_enabled or --monitor_asr.")
    if (config.lfe_target_layer is None) != (config.lfe_target_expert is None):
        raise ValueError("Low-frequency expert target layer and expert must be set together.")
    if len(set(config.malicious_clients)) != len(config.malicious_clients):
        raise ValueError("--malicious_clients must not contain duplicates.")
    invalid_clients = [cid for cid in config.malicious_clients if cid < 0 or cid >= config.num_clients]
    if invalid_clients:
        raise ValueError(
            f"--malicious_clients contains IDs outside [0, {config.num_clients - 1}]: {invalid_clients}."
        )
    if config.expert_probe_enabled and not config.malicious_clients:
        raise ValueError("--expert_probe_enabled requires at least one --malicious_clients ID.")
    if config.expert_probe_enabled and config.mode == "full":
        raise ValueError(
            "--expert_probe_enabled requires --mode=mix, drop, or flex so an exact-K "
            "next-round expert assignment can be influenced by activation reporting."
        )
    if config.expert_probe_enabled and config.assignment_policy != "hot":
        raise ValueError(
            "--expert_probe_enabled requires --assignment_policy=hot because probe "
            "targets are promoted through reported activation counts."
        )
    if (
        config.expert_probe_enabled
        and config.expert_probe_target_layer is None
        and config.k <= 2
    ):
        raise ValueError(
            "Cross-layer multi-expert probing requires --K >= 3 because each layer "
            "reserves two ordinary hot-expert slots and probes at most K-2 experts."
        )
    if (
        config.expert_probe_enabled
        and config.expert_probe_target_layer is None
        and config.expert_probe_batch_size < 2
    ):
        raise ValueError(
            "Cross-layer multi-expert probing requires "
            "--expert_probe_batch_size >= 2."
        )
    if config.checkpoint_interval < 0:
        raise ValueError("--checkpoint_interval must be >= 0.")
    if any(round_number <= 0 for round_number in config.checkpoint_rounds):
        raise ValueError("--checkpoint_rounds values must be >= 1.")
    if config.lfe_profile_trainings <= 0:
        raise ValueError("--lfe_profile_trainings must be > 0.")
    if not 1 <= config.lfe_top_k <= 8:
        raise ValueError("--lfe_top_k must be in [1, 8] for switch-base-8.")
    if config.lfe_calibration_samples <= 0:
        raise ValueError("--lfe_calibration_samples must be > 0.")
    if config.defense not in SUPPORTED_DEFENSES:
        raise ValueError(f"--defense must be one of {SUPPORTED_DEFENSES}, got '{config.defense}'.")
    if config.defense != "none":
        if config.mode != "mix":
            raise ValueError("--defense is currently supported only for --mode=mix.")
        if not 0.0 <= config.defense_trim_ratio < 0.5:
            raise ValueError("--defense_trim_ratio must be in [0, 0.5).")
        if config.defense_krum_m < 0 or config.defense_num_malicious < 0:
            raise ValueError("--defense_krum_m and --defense_num_malicious must be >= 0.")
    if config.expert_probe_warmup_observations < 1:
        raise ValueError("--expert_probe_warmup_observations must be >= 1.")
    if config.expert_probe_pairs < 1:
        raise ValueError("--expert_probe_pairs must be >= 1.")
    if config.expert_probe_candidate_pool < 2 * config.expert_probe_pairs:
        raise ValueError("--expert_probe_candidate_pool must be at least 2 * --expert_probe_pairs.")
    if config.expert_probe_relative_scale <= 0.0:
        raise ValueError("--expert_probe_relative_scale must be > 0.")
    if config.expert_probe_max_relative_scale < config.expert_probe_relative_scale:
        raise ValueError(
            "--expert_probe_max_relative_scale must be >= --expert_probe_relative_scale."
        )
    if config.expert_probe_noise_multiplier < 0.0:
        raise ValueError("--expert_probe_noise_multiplier must be >= 0.")
    if config.expert_probe_repetitions < 1:
        raise ValueError("--expert_probe_repetitions must be >= 1.")
    if config.expert_probe_confirmation_rounds < 1:
        raise ValueError("--expert_probe_confirmation_rounds must be >= 1.")
    if not (
        0.0 <= config.expert_probe_single_retention_min
        <= config.expert_probe_single_retention_max
    ):
        raise ValueError(
            "Probe single-contributor retention bounds must satisfy "
            "0 <= min <= max."
        )
    if config.expert_probe_max_consistency_error < 0.0:
        raise ValueError("--expert_probe_max_consistency_error must be >= 0.")
    if (
        config.expert_probe_enabled
        and not config.expert_probe_auto_layer
        and config.expert_probe_target_layer is None
    ):
        raise ValueError(
            "Disabling automatic probe-layer discovery requires "
            "--expert_probe_target_layer."
        )
    if config.expert_probe_layer_top_experts < 1:
        raise ValueError("--expert_probe_layer_top_experts must be >= 1.")
    if config.expert_probe_max_layers < 0:
        raise ValueError("--expert_probe_max_layers must be >= 0.")
    if config.expert_probe_drift_window < 1:
        raise ValueError("--expert_probe_drift_window must be >= 1.")
    if config.expert_probe_min_layer_contrast < 1.0:
        raise ValueError("--expert_probe_min_layer_contrast must be >= 1.")
    if config.expert_probe_max_pilot_attempts < 1:
        raise ValueError("--expert_probe_max_pilot_attempts must be >= 1.")
    if config.expert_probe_batch_size < 1:
        raise ValueError("--expert_probe_batch_size must be >= 1.")
    if config.expert_probe_screen_samples < 1:
        raise ValueError("--expert_probe_screen_samples must be >= 1.")
    if config.expert_probe_trigger_bank_size < 1:
        raise ValueError("--expert_probe_trigger_bank_size must be >= 1.")
    if config.expert_probe_trigger_length < 1:
        raise ValueError("--expert_probe_trigger_length must be >= 1.")
    if config.expert_probe_boundary_margin < 0.0:
        raise ValueError("--expert_probe_boundary_margin must be >= 0.")
    if not 0.0 <= config.expert_probe_boundary_quantile <= 1.0:
        raise ValueError("--expert_probe_boundary_quantile must be in [0, 1].")


def parse_config(argv: Sequence[str] | None = None) -> MixFedMoEConfig:
    parser = _build_parser()
    args = parser.parse_args(argv)

    trigger = args.trigger
    lfe_target_layer: Optional[str] = None
    lfe_target_expert: Optional[int] = None
    trigger_token_ids = (
        DEFAULT_BADNET_TRIGGER_TOKEN_IDS if trigger == DEFAULT_BADNET_TRIGGER else ()
    )
    if args.selected_triggers_path is not None:
        selected = load_selected_trigger(args.selected_triggers_path)
        trigger = selected.trigger
        lfe_target_layer = selected.target_layer
        lfe_target_expert = selected.target_expert
        trigger_token_ids = selected.trigger_token_ids
    if args.lfe_target_layer is not None:
        lfe_target_layer = args.lfe_target_layer
    if args.lfe_target_expert is not None:
        lfe_target_expert = args.lfe_target_expert

    checkpoint_path = args.checkpoint_path or args.init_checkpoint
    config = MixFedMoEConfig(
        mode=args.mode,
        model_name_or_path=args.model_name_or_path,
        dataset_name=args.dataset_name,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        k=args.k,
        alpha=args.alpha,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        client_eval_ratio=args.client_eval_ratio,
        seed=args.seed,
        num_cpus_per_client=args.num_cpus_per_client,
        num_gpus_per_client=args.num_gpus_per_client,
        output_dir=args.output_dir,
        test_samples=args.test_samples,
        data_distribution=args.data_distribution,
        assignment_policy=args.assignment_policy,
        checkpoint_path=checkpoint_path,
        init_checkpoint=checkpoint_path,
        coverage_guarantee=args.coverage_guarantee,
        attack_enabled=args.attack_enabled,
        monitor_asr=args.monitor_asr,
        measure_local_asr=args.measure_local_asr,
        measure_poison_gradient_flow=args.measure_poison_gradient_flow,
        poison_gradient_samples=args.poison_gradient_samples,
        malicious_clients=tuple(args.malicious_clients),
        poison_rate=args.poison_rate,
        global_random_poison_source=args.global_random_poison_source,
        iid_malicious_client_data=args.iid_malicious_client_data,
        target_label=args.target_label,
        attack_start_round=args.attack_start_round,
        attack_end_round=args.attack_end_round,
        inject_target_expert=args.inject_target_expert,
        malicious_target_expert_router_only=args.malicious_target_expert_router_only,
        isolate_poison_gradients_to_target=(
            args.isolate_poison_gradients_to_target
        ),
        target_expert_poison_gradient_goal_ratio=(
            args.target_expert_poison_gradient_goal_ratio
        ),
        target_expert_poison_gradient_max_gain=(
            args.target_expert_poison_gradient_max_gain
        ),
        target_router_loss_weight=args.target_router_loss_weight,
        target_router_margin=args.target_router_margin,
        target_router_kl_weight=args.target_router_kl_weight,
        target_router_full_weight_below=args.target_router_full_weight_below,
        target_router_stop_median=args.target_router_stop_median,
        target_router_stop_p10=args.target_router_stop_p10,
        attack_mode=args.attack_mode,
        neurotoxin_mask_ratio=args.neurotoxin_mask_ratio,
        neurotoxin_aggregate_all_layers=args.neurotoxin_aggregate_all_layers,
        neurotoxin_max_update_norm=args.neurotoxin_max_update_norm,
        neurotoxin_update_scale=args.neurotoxin_update_scale,
        a3fl_outer_steps=args.a3fl_outer_steps,
        a3fl_trigger_steps=args.a3fl_trigger_steps,
        a3fl_candidate_batch_size=args.a3fl_candidate_batch_size,
        a3fl_top_k=args.a3fl_top_k,
        a3fl_calibration_samples=args.a3fl_calibration_samples,
        a3fl_adv_steps=args.a3fl_adv_steps,
        a3fl_adv_lr=args.a3fl_adv_lr,
        a3fl_lambda=args.a3fl_lambda,
        a3fl_early_stop_asr=args.a3fl_early_stop_asr,
        malicious_lr=args.malicious_lr,
        malicious_local_epochs=args.malicious_local_epochs,
        trigger=trigger,
        trigger_token_ids=trigger_token_ids,
        selected_triggers_path=args.selected_triggers_path,
        lfe_target_layer=lfe_target_layer,
        lfe_target_expert=lfe_target_expert,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_rounds=tuple(sorted(set(args.checkpoint_rounds))),
        lfe_profile_enabled=args.lfe_profile_enabled,
        lfe_profile_trainings=args.lfe_profile_trainings,
        lfe_top_k=args.lfe_top_k,
        lfe_calibration_samples=args.lfe_calibration_samples,
        defense=args.defense,
        defense_trim_ratio=args.defense_trim_ratio,
        defense_krum_m=args.defense_krum_m,
        defense_num_malicious=args.defense_num_malicious,
        expert_probe_enabled=args.expert_probe_enabled,
        expert_probe_target_layer=args.expert_probe_target_layer,
        expert_probe_auto_layer=args.expert_probe_auto_layer,
        expert_probe_layer_top_experts=args.expert_probe_layer_top_experts,
        expert_probe_max_layers=args.expert_probe_max_layers,
        expert_probe_drift_window=args.expert_probe_drift_window,
        expert_probe_min_layer_contrast=args.expert_probe_min_layer_contrast,
        expert_probe_max_pilot_attempts=args.expert_probe_max_pilot_attempts,
        expert_probe_batch_size=args.expert_probe_batch_size,
        expert_probe_screen_samples=args.expert_probe_screen_samples,
        expert_probe_trigger_bank_size=args.expert_probe_trigger_bank_size,
        expert_probe_trigger_length=args.expert_probe_trigger_length,
        expert_probe_boundary_margin=args.expert_probe_boundary_margin,
        expert_probe_boundary_quantile=args.expert_probe_boundary_quantile,
        expert_probe_warmup_observations=args.expert_probe_warmup_observations,
        expert_probe_pairs=args.expert_probe_pairs,
        expert_probe_candidate_pool=args.expert_probe_candidate_pool,
        expert_probe_relative_scale=args.expert_probe_relative_scale,
        expert_probe_max_relative_scale=args.expert_probe_max_relative_scale,
        expert_probe_noise_multiplier=args.expert_probe_noise_multiplier,
        expert_probe_repetitions=args.expert_probe_repetitions,
        expert_probe_confirmation_rounds=args.expert_probe_confirmation_rounds,
        expert_probe_single_retention_min=args.expert_probe_single_retention_min,
        expert_probe_single_retention_max=args.expert_probe_single_retention_max,
        expert_probe_max_consistency_error=args.expert_probe_max_consistency_error,
        expert_probe_spoof_activation=args.expert_probe_spoof_activation,
    )
    validate_config(config)
    return config
