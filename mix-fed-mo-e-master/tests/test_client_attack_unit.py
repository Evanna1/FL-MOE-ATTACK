from __future__ import annotations

from mixfedmoe_fl.client import MixFedMoEClient
from mixfedmoe_fl.config import parse_config


def test_attack_activation_requires_enabled_malicious_client_and_start_round() -> None:
    cfg = parse_config(
        [
            "--mode",
            "full",
            "--num_clients",
            "2",
            "--attack_enabled",
            "--malicious_clients",
            "0",
            "--attack_start_round",
            "3",
        ]
    )
    malicious = MixFedMoEClient(0, cfg, object(), is_malicious=True)  # type: ignore[arg-type]
    benign = MixFedMoEClient(1, cfg, object(), is_malicious=False)  # type: ignore[arg-type]

    assert malicious._is_attack_active(2) is False
    assert malicious._is_attack_active(3) is True
    assert benign._is_attack_active(3) is False


def test_attack_disabled_keeps_all_clients_clean() -> None:
    cfg = parse_config(["--mode", "full", "--num_clients", "2"])
    client = MixFedMoEClient(0, cfg, object(), is_malicious=False)  # type: ignore[arg-type]
    assert client._is_attack_active(1) is False
