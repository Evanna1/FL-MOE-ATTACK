from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from mixfedmoe_fl.attack import load_selected_trigger

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


def supported_dataset_names() -> List[str]:
    return list(SUPPORTED_DATASET_NAME_TO_HF.keys())


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
    assignment_policy: Literal["hot", "random"] = "hot"
    coverage_guarantee: bool = True
    attack_enabled: bool = False
    malicious_clients: Tuple[int, ...] = (0,)
    poison_rate: float = 0.1
    target_label: int = 0
    attack_start_round: int = 1
    trigger: str = "cf"
    selected_triggers_path: Optional[str] = None
    lfe_target_layer: Optional[str] = None
    lfe_target_expert: Optional[int] = None
    checkpoint_rounds: Tuple[int, ...] = (10, 20)
    lfe_profile_enabled: bool = False
    lfe_profile_trainings: int = 20
    lfe_top_k: int = 3
    lfe_calibration_samples: int = 512

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MixFedMoE Flower simulation configuration.")
    parser.add_argument("--mode", type=str, choices=SUPPORTED_MODES, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="model_ckpt/switch-base-8")
    parser.add_argument("--dataset_name", type=str, choices=supported_dataset_names(), default="ag_news")
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--K", "--k", dest="k", type=int, default=4)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Deprecated compatibility option; ignored because client data is partitioned IID.",
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
        help="Enable standard BadNet data poisoning on malicious clients.",
    )
    parser.add_argument(
        "--malicious_clients",
        type=int,
        nargs="*",
        default=[0],
        help="Partition IDs of malicious clients (default: 0).",
    )
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--attack_start_round", type=int, default=1)
    parser.add_argument("--trigger", type=str, default="cf")
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
        "--checkpoint_rounds",
        type=int,
        nargs="*",
        default=[10, 20],
        help="Server rounds to save under output_dir/checkpoints; pass no values to disable.",
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
        help="Deprecated compatibility option; training-hook profiling uses all locally trained batches.",
    )
    return parser


def validate_config(config: MixFedMoEConfig) -> None:
    if config.num_clients <= 0:
        raise ValueError("--num_clients must be > 0.")
    if config.num_rounds <= 0:
        raise ValueError("--num_rounds must be > 0.")
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
    if config.target_label < 0:
        raise ValueError("--target_label must be >= 0.")
    if config.attack_start_round <= 0:
        raise ValueError("--attack_start_round must be >= 1.")
    if not config.trigger.strip():
        raise ValueError("--trigger must not be empty.")
    if config.selected_triggers_path is not None and not config.attack_enabled:
        raise ValueError("--selected_triggers_path requires --attack_enabled.")
    if (config.lfe_target_layer is None) != (config.lfe_target_expert is None):
        raise ValueError("Low-frequency expert target layer and expert must be set together.")
    if len(set(config.malicious_clients)) != len(config.malicious_clients):
        raise ValueError("--malicious_clients must not contain duplicates.")
    invalid_clients = [cid for cid in config.malicious_clients if cid < 0 or cid >= config.num_clients]
    if invalid_clients:
        raise ValueError(
            f"--malicious_clients contains IDs outside [0, {config.num_clients - 1}]: {invalid_clients}."
        )
    if any(round_number <= 0 for round_number in config.checkpoint_rounds):
        raise ValueError("--checkpoint_rounds values must be >= 1.")
    if config.lfe_profile_trainings <= 0:
        raise ValueError("--lfe_profile_trainings must be > 0.")
    if not 1 <= config.lfe_top_k <= 8:
        raise ValueError("--lfe_top_k must be in [1, 8] for switch-base-8.")
    if config.lfe_calibration_samples <= 0:
        raise ValueError("--lfe_calibration_samples must be > 0.")


def parse_config(argv: Sequence[str] | None = None) -> MixFedMoEConfig:
    parser = _build_parser()
    args = parser.parse_args(argv)

    trigger = args.trigger
    lfe_target_layer: Optional[str] = None
    lfe_target_expert: Optional[int] = None
    if args.selected_triggers_path is not None:
        selected = load_selected_trigger(args.selected_triggers_path)
        trigger = selected.trigger
        lfe_target_layer = selected.target_layer
        lfe_target_expert = selected.target_expert

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
        assignment_policy=args.assignment_policy,
        coverage_guarantee=args.coverage_guarantee,
        attack_enabled=args.attack_enabled,
        malicious_clients=tuple(args.malicious_clients),
        poison_rate=args.poison_rate,
        target_label=args.target_label,
        attack_start_round=args.attack_start_round,
        trigger=trigger,
        selected_triggers_path=args.selected_triggers_path,
        lfe_target_layer=lfe_target_layer,
        lfe_target_expert=lfe_target_expert,
        checkpoint_rounds=tuple(sorted(set(args.checkpoint_rounds))),
        lfe_profile_enabled=args.lfe_profile_enabled,
        lfe_profile_trainings=args.lfe_profile_trainings,
        lfe_top_k=args.lfe_top_k,
        lfe_calibration_samples=args.lfe_calibration_samples,
    )
    validate_config(config)
    return config
