from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace

import torch
import torch.nn as nn

from mixfedmoe_fl.expert_profiler import (
    TrainingLFERoutingTracker,
    collect_lfe_routing_profile,
    save_local_lfe_profile,
    save_training_lfe_profile,
    should_collect_lfe_profile,
)


class TinyDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def select(self, indices):
        return TinyDataset([self.rows[index] for index in indices])


def tiny_collator(rows):
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    attention_mask = []
    for row in rows:
        padding = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [0] * padding)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


class TinyRouterContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Module()
        self.router.classifier = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.router.classifier.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
            )


class TinyProfileModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_mlp = TinyRouterContainer()
        self.decoder_mlp = TinyRouterContainer()

    def _shift_right(self, input_ids):
        return torch.cat([torch.zeros_like(input_ids[:, :1]), input_ids[:, :-1]], dim=1)

    @staticmethod
    def _features(input_ids):
        return torch.stack([input_ids.float(), 1.0 - input_ids.float()], dim=-1)

    def forward(self, input_ids, decoder_input_ids, **_kwargs):
        self.encoder_mlp.router.classifier(self._features(input_ids))
        self.decoder_mlp.router.classifier(self._features(decoder_input_ids))
        return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], 2))


def test_collect_profile_masks_padding_and_is_batch_invariant() -> None:
    dataset = TinyDataset(
        [
            {"input_ids": [1, 1, 1]},
            {"input_ids": [1]},
            {"input_ids": [0, 1]},
        ]
    )
    model = TinyProfileModel()
    handles = [
        SimpleNamespace(layer_id="encoder.1", stack_name="encoder", mlp_module=model.encoder_mlp),
        SimpleNamespace(layer_id="decoder.1", stack_name="decoder", mlp_module=model.decoder_mlp),
    ]
    profiles = []
    for batch_size in [1, 3]:
        profiles.append(
            collect_lfe_routing_profile(
                model=model,
                moe_handles=handles,
                dataset=dataset,
                collator=tiny_collator,
                batch_size=batch_size,
                max_samples=3,
                seed=7,
                device=torch.device("cpu"),
                top_k=2,
            )
        )
    for layer_id in ["encoder.1", "decoder.1"]:
        assert profiles[0]["layers"][layer_id]["counts"] == profiles[1]["layers"][layer_id]["counts"]
        assert profiles[0]["layers"][layer_id]["frequencies"] == profiles[1]["layers"][layer_id]["frequencies"]
    assert profiles[0]["layers"]["encoder.1"]["valid_tokens"] == 6
    assert sum(profiles[0]["layers"]["encoder.1"]["counts"]) == 6
    assert sum(profiles[0]["layers"]["decoder.1"]["counts"]) == 6


def test_local_summary_is_completed_after_twenty_profiles(tmp_path) -> None:
    base_profile = {
        "calibration_samples": 2,
        "calibration_indices_sha256": "fixed",
        "profile_seed": 42,
        "layers": {
            "encoder.1": {
                "stack": "encoder",
                "valid_tokens": 100,
                "counts": [40, 30, 20, 10],
                "frequencies": [0.4, 0.3, 0.2, 0.1],
                "mean_selected_probability": [0.7, 0.7, 0.7, 0.7],
                "mean_top1_top2_margin": [0.2, 0.2, 0.2, 0.2],
                "top_low_frequency_experts": [3, 2, 1],
            }
        },
    }
    for server_round in range(1, 21):
        save_local_lfe_profile(
            output_dir=str(tmp_path),
            client_id=0,
            server_round=server_round,
            profile=base_profile,
            target_trainings=20,
            top_k=3,
        )

    profile_dir = tmp_path / "lfe_profiles" / "client_0"
    with open(profile_dir / "lfe_top3_final.json", "r", encoding="utf-8") as file:
        summary = json.load(file)
    assert summary["scope"] == "malicious_client_local_only"
    assert summary["completed"] is True
    assert summary["observed_trainings"] == 20
    assert summary["layers"]["encoder.1"]["top_low_frequency_experts"] == [3, 2, 1]
    assert should_collect_lfe_profile(str(tmp_path), 0, 21, 20) is False


def test_training_tracker_profiles_batches_and_preserves_global_expert_ids(tmp_path) -> None:
    model = TinyProfileModel()
    handles = [
        SimpleNamespace(layer_id="encoder.1", stack_name="encoder", mlp_module=model.encoder_mlp),
        SimpleNamespace(layer_id="decoder.1", stack_name="decoder", mlp_module=model.decoder_mlp),
    ]
    tracker = TrainingLFERoutingTracker(
        handles,
        top_k=2,
        expert_id_maps={"encoder.1": [2, 4, 7], "decoder.1": [2, 4, 7]},
    )
    batch = tiny_collator([{"input_ids": [1, 1, 1]}, {"input_ids": [0]}])
    for _ in range(2):
        tracker.begin_batch(batch["attention_mask"])
        model(
            input_ids=batch["input_ids"],
            decoder_input_ids=model._shift_right(batch["input_ids"]),
        )
    profile = tracker.finish()

    assert profile["profile_source"] == "training_router_hooks"
    assert profile["training_batches"] == 2
    assert profile["training_examples"] == 4
    assert profile["layers"]["encoder.1"]["expert_ids"] == [2, 4, 7]
    assert sum(profile["layers"]["encoder.1"]["counts"]) == 8
    assert set(profile["layers"]["encoder.1"]["top_low_frequency_experts"]) <= {2, 4, 7}

    workbook_path = save_training_lfe_profile(
        output_dir=str(tmp_path),
        client_id=3,
        server_round=5,
        mode="drop",
        profile=profile,
    )
    with zipfile.ZipFile(workbook_path) as workbook:
        assert "xl/worksheets/sheet1.xml" in workbook.namelist()
        assert "low_frequency" in workbook.read("xl/workbook.xml").decode("utf-8")
    with open(tmp_path / "lfe_profiles" / "client_3" / "round_0005.json", encoding="utf-8") as file:
        saved = json.load(file)
    assert saved["server_round"] == 5
    assert saved["layers"]["encoder.1"]["expert_ids"] == [2, 4, 7]
