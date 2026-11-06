from __future__ import annotations

import pytest

from mixfedmoe_fl.config import (
    DEFAULT_BADNET_TRIGGER,
    DEFAULT_BADNET_TRIGGER_TOKEN_IDS,
    SUPPORTED_DATASET_NAME_TO_HF,
    parse_config,
)


def test_emotion_uses_namespaced_huggingface_dataset_id() -> None:
    assert SUPPORTED_DATASET_NAME_TO_HF["emotion"] == "dair-ai/emotion"


def test_data_distribution_defaults_to_non_iid() -> None:
    cfg = parse_config(["--mode", "full"])
    assert cfg.data_distribution == "non-iid"


@pytest.mark.parametrize("value", ["non-iid", "non_iid", "noniid", "niid"])
def test_parse_config_accepts_non_iid_aliases(value: str) -> None:
    cfg = parse_config(["--mode", "full", "--data_distribution", value])
    assert cfg.data_distribution == "non-iid"


def test_parse_config_accepts_iid_data_mode_alias() -> None:
    cfg = parse_config(["--mode", "full", "--data_mode", "iid"])
    assert cfg.data_distribution == "iid"


def test_parse_config_accepts_fractional_local_epochs() -> None:
    cfg = parse_config(["--mode", "full", "--client_eval_ratio", "0.2", "--local_epochs", "0.5"])
    assert cfg.local_epochs == 0.5


def test_parse_config_accepts_random_assignment_policy() -> None:
    cfg = parse_config(
        [
            "--mode",
            "mix",
            "--client_eval_ratio",
            "0.2",
            "--assignment_policy",
            "random",
        ]
    )
    assert cfg.assignment_policy == "random"


def test_parse_config_rejects_non_positive_local_epochs() -> None:
    with pytest.raises(ValueError, match="--local_epochs must be > 0"):
        parse_config(["--mode", "full", "--client_eval_ratio", "0.2", "--local_epochs", "0"])


def test_parse_config_accepts_flex_mode_with_k1_hot_policy() -> None:
    cfg = parse_config(
        [
            "--mode",
            "flex",
            "--client_eval_ratio",
            "0.2",
            "--k",
            "1",
            "--assignment_policy",
            "hot",
        ]
    )
    assert cfg.mode == "flex"
    assert cfg.k == 1
    assert cfg.assignment_policy == "hot"


def test_parse_config_rejects_flex_mode_when_k_is_not_one() -> None:
    with pytest.raises(ValueError, match="--mode=flex requires --K/--k to be exactly 1"):
        parse_config(["--mode", "flex", "--client_eval_ratio", "0.2", "--k", "2"])


def test_parse_config_rejects_flex_mode_with_random_policy() -> None:
    with pytest.raises(ValueError, match="--mode=flex requires --assignment_policy=hot"):
        parse_config(
            [
                "--mode",
                "flex",
                "--client_eval_ratio",
                "0.2",
                "--k",
                "1",
                "--assignment_policy",
                "random",
            ]
        )


def test_parse_config_accepts_badnet_options() -> None:
    cfg = parse_config(
        [
            "--mode",
            "full",
            "--num_clients",
            "4",
            "--attack_enabled",
            "--malicious_clients",
            "0",
            "2",
            "--poison_rate",
            "0.25",
            "--target_label",
            "1",
            "--attack_start_round",
            "3",
            "--trigger",
            "bb",
            "--checkpoint_rounds",
            "2",
            "4",
        ]
    )
    assert cfg.attack_enabled is True
    assert cfg.malicious_clients == (0, 2)
    assert cfg.poison_rate == 0.25
    assert cfg.target_label == 1
    assert cfg.attack_start_round == 3
    assert cfg.trigger == "bb"
    assert cfg.trigger_token_ids == ()
    assert cfg.checkpoint_interval == 5
    assert cfg.checkpoint_rounds == (2, 4)


def test_parse_config_accepts_checkpoint_interval() -> None:
    cfg = parse_config(
        [
            "--mode",
            "full",
            "--checkpoint_interval",
            "3",
        ]
    )
    assert cfg.checkpoint_interval == 3
    assert cfg.checkpoint_rounds == ()


def test_parse_config_accepts_independent_a3fl_options() -> None:
    cfg = parse_config(
        [
            "--mode",
            "full",
            "--attack_enabled",
            "--attack_mode",
            "a3fl",
            "--a3fl_outer_steps",
            "3",
            "--a3fl_trigger_steps",
            "4",
            "--a3fl_candidate_batch_size",
            "12",
            "--a3fl_top_k",
            "7",
            "--a3fl_calibration_samples",
            "9",
            "--a3fl_adv_steps",
            "2",
            "--a3fl_adv_lr",
            "0.0001",
            "--a3fl_lambda",
            "0.5",
        ]
    )

    assert cfg.attack_mode == "a3fl"
    assert cfg.a3fl_outer_steps == 3
    assert cfg.a3fl_trigger_steps == 4
    assert cfg.a3fl_candidate_batch_size == 12
    assert cfg.a3fl_top_k == 7
    assert cfg.a3fl_calibration_samples == 9
    assert cfg.a3fl_adv_steps == 2
    assert cfg.a3fl_adv_lr == 0.0001
    assert cfg.a3fl_lambda == 0.5
    assert cfg.inject_target_expert is False


def test_a3fl_lambda0_defaults_to_paper_value() -> None:
    cfg = parse_config(["--mode", "full"])

    assert cfg.a3fl_lambda == 0.01


def test_a3fl_rejects_cold_expert_injection() -> None:
    with pytest.raises(ValueError, match="independent"):
        parse_config(
            [
                "--mode",
                "full",
                "--attack_enabled",
                "--attack_mode",
                "a3fl",
                "--inject_target_expert",
                "--lfe_target_layer",
                "encoder.1",
                "--lfe_target_expert",
                "0",
            ]
        )


def test_cold_expert_profile_promotion_requires_sparse_hot_assignment() -> None:
    base = [
        "--attack_enabled",
        "--inject_target_expert",
        "--lfe_target_layer",
        "encoder.1",
        "--lfe_target_expert",
        "0",
    ]
    with pytest.raises(ValueError, match="mode=mix, drop, or flex"):
        parse_config(["--mode", "full", *base])

    with pytest.raises(ValueError, match="assignment_policy=hot"):
        parse_config(["--mode", "mix", "--assignment_policy", "random", *base])


def test_target_router_margin_configuration() -> None:
    cfg = parse_config(
        [
            "--mode",
            "mix",
            "--target_router_loss_weight",
            "0.2",
            "--target_router_margin",
            "2.5",
            "--target_router_kl_weight",
            "0.4",
            "--target_router_full_weight_below",
            "0.6",
            "--target_router_stop_median",
            "1.8",
            "--target_router_stop_p10",
            "0.4",
        ]
    )

    assert cfg.target_router_loss_weight == 0.2
    assert cfg.target_router_margin == 2.5
    assert cfg.target_router_kl_weight == 0.4
    assert cfg.target_router_full_weight_below == 0.6
    assert cfg.target_router_stop_median == 1.8
    assert cfg.target_router_stop_p10 == 0.4

    with pytest.raises(ValueError, match="full_weight_below"):
        parse_config(
            [
                "--mode",
                "mix",
                "--target_router_full_weight_below",
                "1.5",
                "--target_router_stop_median",
                "1.0",
            ]
        )


def test_parse_config_enables_fair_poison_gradient_isolation() -> None:
    cfg = parse_config(
        [
            "--mode",
            "mix",
            "--attack_enabled",
            "--isolate_poison_gradients_to_target",
            "--neurotoxin_mask_ratio",
            "0.9",
            "--target_expert_poison_gradient_goal_ratio",
            "0.3",
            "--target_expert_poison_gradient_max_gain",
            "4.0",
            "--lfe_target_layer",
            "encoder.1",
            "--lfe_target_expert",
            "0",
        ]
    )

    assert cfg.isolate_poison_gradients_to_target is True
    assert cfg.neurotoxin_mask_ratio == 0.9
    assert cfg.target_expert_poison_gradient_goal_ratio == 0.3
    assert cfg.target_expert_poison_gradient_max_gain == 4.0


def test_poison_gradient_isolation_allows_ratio_one_to_disable_mask() -> None:
    cfg = parse_config(
        [
            "--mode",
            "mix",
            "--attack_enabled",
            "--isolate_poison_gradients_to_target",
            "--neurotoxin_mask_ratio",
            "1.0",
            "--lfe_target_layer",
            "encoder.1",
            "--lfe_target_expert",
            "0",
        ]
    )

    assert cfg.neurotoxin_mask_ratio == 1.0


def test_classic_neurotoxin_rejects_ratio_one() -> None:
    with pytest.raises(ValueError, match="Classic --attack_mode=neurotoxin"):
        parse_config(
            [
                "--mode",
                "full",
                "--attack_enabled",
                "--attack_mode",
                "neurotoxin",
                "--neurotoxin_mask_ratio",
                "1.0",
            ]
        )


def test_target_expert_poison_gradient_redistribution_validation() -> None:
    with pytest.raises(ValueError, match="goal_ratio"):
        parse_config(
            [
                "--mode",
                "mix",
                "--target_expert_poison_gradient_goal_ratio",
                "1.1",
            ]
        )
    with pytest.raises(ValueError, match="max_gain"):
        parse_config(
            [
                "--mode",
                "mix",
                "--target_expert_poison_gradient_max_gain",
                "0.5",
            ]
        )


def test_poison_gradient_isolation_excludes_classic_neurotoxin() -> None:
    with pytest.raises(ValueError, match="attack_mode=neurotoxin"):
        parse_config(
            [
                "--mode",
                "mix",
                "--attack_enabled",
                "--attack_mode",
                "neurotoxin",
                "--isolate_poison_gradients_to_target",
                "--lfe_target_layer",
                "encoder.1",
                "--lfe_target_expert",
                "0",
            ]
        )


def test_poison_gradient_isolation_requires_target_and_excludes_global_freeze() -> None:
    with pytest.raises(ValueError, match="requires --lfe_target_layer"):
        parse_config(
            [
                "--mode",
                "mix",
                "--attack_enabled",
                "--isolate_poison_gradients_to_target",
            ]
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_config(
            [
                "--mode",
                "mix",
                "--attack_enabled",
                "--isolate_poison_gradients_to_target",
                "--malicious_target_expert_router_only",
                "--lfe_target_layer",
                "encoder.1",
                "--lfe_target_expert",
                "0",
            ]
        )


def test_badnet_defaults_to_badmagic_with_exact_token_ids() -> None:
    cfg = parse_config(["--mode", "full", "--attack_enabled"])

    assert cfg.trigger == DEFAULT_BADNET_TRIGGER == "BadMagic"
    assert cfg.trigger_token_ids == DEFAULT_BADNET_TRIGGER_TOKEN_IDS == (3862, 22909, 447)


def test_parse_config_enables_global_random_poison_source() -> None:
    cfg = parse_config(
        ["--mode", "full", "--attack_enabled", "--global_random_poison_source"]
    )

    assert cfg.global_random_poison_source is True


def test_global_random_poison_source_requires_attack() -> None:
    with pytest.raises(ValueError, match="requires --attack_enabled"):
        parse_config(["--mode", "full", "--global_random_poison_source"])


def test_parse_config_enables_iid_malicious_client_data() -> None:
    cfg = parse_config(
        ["--mode", "full", "--attack_enabled", "--iid_malicious_client_data"]
    )

    assert cfg.iid_malicious_client_data is True


def test_iid_malicious_client_data_requires_attack() -> None:
    with pytest.raises(ValueError, match="requires --attack_enabled"):
        parse_config(["--mode", "full", "--iid_malicious_client_data"])


def test_global_random_experiment_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_config(
            [
                "--mode",
                "full",
                "--attack_enabled",
                "--iid_malicious_client_data",
                "--global_random_poison_source",
            ]
        )


def test_parse_config_accepts_checkpoint_path_and_legacy_alias() -> None:
    cfg = parse_config(
        ["--mode", "full", "--attack_enabled", "--checkpoint_path", "trained.pt"]
    )
    assert cfg.checkpoint_path == "trained.pt"
    assert cfg.init_checkpoint == "trained.pt"

    legacy_cfg = parse_config(
        ["--mode", "full", "--attack_enabled", "--init_checkpoint", "legacy.pt"]
    )
    assert legacy_cfg.checkpoint_path == "legacy.pt"


def test_parse_config_accepts_monitor_asr_as_benign_mode() -> None:
    cfg = parse_config(
        [
            "--mode",
            "full",
            "--monitor_asr",
            "--checkpoint_path",
            "backdoored.pt",
            "--trigger",
            "cf",
        ]
    )
    assert cfg.monitor_asr is True
    assert cfg.attack_enabled is False


def test_parse_config_rejects_monitor_asr_with_attack_enabled() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_config(["--mode", "full", "--monitor_asr", "--attack_enabled"])


def test_parse_config_rejects_invalid_malicious_client() -> None:
    with pytest.raises(ValueError, match="outside"):
        parse_config(["--mode", "full", "--num_clients", "2", "--malicious_clients", "2"])


def test_parse_config_accepts_local_lfe_profile_options() -> None:
    cfg = parse_config(
        [
            "--mode",
            "full",
            "--lfe_profile_enabled",
            "--lfe_profile_trainings",
            "20",
            "--lfe_top_k",
            "3",
            "--lfe_calibration_samples",
            "64",
        ]
    )
    assert cfg.lfe_profile_enabled is True
    assert cfg.lfe_profile_trainings == 20
    assert cfg.lfe_top_k == 3
    assert cfg.lfe_calibration_samples == 64


def test_lfe_profile_is_disabled_by_default() -> None:
    cfg = parse_config(["--mode", "full"])
    assert cfg.lfe_profile_enabled is False


def test_parse_config_loads_lfe_trigger_without_changing_badnet_pipeline(tmp_path) -> None:
    path = tmp_path / "selected_triggers.json"
    path.write_text(
        '{"target":{"layer_id":"decoder.3","expert_id":5},'
        '"best_trigger":{"trigger":"optimized trigger"}}',
        encoding="utf-8",
    )
    cfg = parse_config(
        [
            "--mode",
            "full",
            "--attack_enabled",
            "--trigger",
            "ignored",
            "--selected_triggers_path",
            str(path),
        ]
    )

    assert cfg.attack_enabled is True
    assert cfg.trigger == "optimized trigger"
    assert cfg.selected_triggers_path == str(path)
    assert cfg.lfe_target_layer == "decoder.3"
    assert cfg.lfe_target_expert == 5


def test_parse_config_loads_exact_trigger_token_ids(tmp_path) -> None:
    path = tmp_path / "selected_triggers.json"
    path.write_text(
        '{"target":{"layer_id":"encoder.5","expert_id":2},'
        '"best_trigger":{"trigger":"optimized","trigger_token_ids":[7,8,9]}}',
        encoding="utf-8",
    )

    cfg = parse_config(
        ["--mode", "full", "--attack_enabled", "--selected_triggers_path", str(path)]
    )

    assert cfg.trigger_token_ids == (7, 8, 9)


def test_parse_config_requires_attack_for_selected_trigger(tmp_path) -> None:
    path = tmp_path / "selected_triggers.json"
    path.write_text(
        '{"target":{"layer_id":"encoder.1","expert_id":0},'
        '"best_trigger":{"trigger":"optimized"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires --attack_enabled or --monitor_asr"):
        parse_config(["--mode", "full", "--selected_triggers_path", str(path)])


def test_parse_config_allows_selected_trigger_for_monitoring(tmp_path) -> None:
    path = tmp_path / "selected_triggers.json"
    path.write_text(
        '{"target":{"layer_id":"encoder.1","expert_id":0},'
        '"best_trigger":{"trigger":"optimized"}}',
        encoding="utf-8",
    )
    cfg = parse_config(
        ["--mode", "full", "--monitor_asr", "--selected_triggers_path", str(path)]
    )
    assert cfg.monitor_asr is True
    assert cfg.attack_enabled is False
    assert cfg.trigger == "optimized"


def test_measure_local_asr_is_opt_in_and_mix_only() -> None:
    default_cfg = parse_config(["--mode", "mix"])
    assert default_cfg.measure_local_asr is False

    cfg = parse_config(
        ["--mode", "mix", "--attack_enabled", "--measure_local_asr"]
    )
    assert cfg.measure_local_asr is True

    with pytest.raises(ValueError, match="requires --attack_enabled"):
        parse_config(["--mode", "mix", "--measure_local_asr"])
    with pytest.raises(ValueError, match="only for --mode=mix"):
        parse_config(
            ["--mode", "full", "--attack_enabled", "--measure_local_asr"]
        )


def test_poison_gradient_flow_requires_mix_attack_and_target() -> None:
    cfg = parse_config(
        [
            "--mode",
            "mix",
            "--attack_enabled",
            "--measure_poison_gradient_flow",
            "--poison_gradient_samples",
            "12",
            "--lfe_target_layer",
            "encoder.0",
            "--lfe_target_expert",
            "1",
        ]
    )
    assert cfg.measure_poison_gradient_flow is True
    assert cfg.poison_gradient_samples == 12

    with pytest.raises(ValueError, match="requires --attack_enabled"):
        parse_config(
            [
                "--mode",
                "mix",
                "--measure_poison_gradient_flow",
                "--lfe_target_layer",
                "encoder.0",
                "--lfe_target_expert",
                "1",
            ]
        )

    with pytest.raises(ValueError, match="requires --lfe_target_layer"):
        parse_config(
            ["--mode", "mix", "--attack_enabled", "--measure_poison_gradient_flow"]
        )
