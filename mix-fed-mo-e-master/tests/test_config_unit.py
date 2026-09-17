from __future__ import annotations

import pytest

from mixfedmoe_fl.config import SUPPORTED_DATASET_NAME_TO_HF, parse_config


def test_emotion_uses_namespaced_huggingface_dataset_id() -> None:
    assert SUPPORTED_DATASET_NAME_TO_HF["emotion"] == "dair-ai/emotion"


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
    assert cfg.checkpoint_rounds == (2, 4)


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


def test_parse_config_requires_attack_for_selected_trigger(tmp_path) -> None:
    path = tmp_path / "selected_triggers.json"
    path.write_text(
        '{"target":{"layer_id":"encoder.1","expert_id":0},'
        '"best_trigger":{"trigger":"optimized"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires --attack_enabled"):
        parse_config(["--mode", "full", "--selected_triggers_path", str(path)])
