from __future__ import annotations

import csv
import json

from mixfedmoe_fl.trigger_optimization import (
    TargetExpert,
    canonical_layer_id,
    load_per_layer_lowest_experts,
    load_target_experts,
    pareto_frontier,
    rank_by_routing,
)


def test_canonical_layer_id_accepts_supported_aliases() -> None:
    assert canonical_layer_id("encoder.1") == "encoder.1"
    assert canonical_layer_id("encoder.layer.1") == "encoder.1"
    assert canonical_layer_id("decoder_block_3") == "decoder.3"


def test_load_targets_from_current_lfe_csv(tmp_path) -> None:
    path = tmp_path / "lfe_top3.csv"
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["layer_id", "rank", "expert_id", "median_frequency"])
        writer.writeheader()
        writer.writerow({"layer_id": "encoder.1", "rank": 1, "expert_id": 7, "median_frequency": 0.01})
        writer.writerow({"layer_id": "encoder.1", "rank": 2, "expert_id": 2, "median_frequency": 0.02})
    assert load_target_experts(str(path)) == [TargetExpert("encoder.1", 7), TargetExpert("encoder.1", 2)]


def test_load_targets_from_current_lfe_final_json(tmp_path) -> None:
    path = tmp_path / "lfe_top3_final.json"
    path.write_text(
        json.dumps(
            {
                "layers": {
                    "encoder.1": {"top_low_frequency_experts": [7, 2, 4]},
                    "decoder.3": {"top_low_frequency_experts": [1, 5, 0]},
                }
            }
        ),
        encoding="utf-8",
    )
    assert load_target_experts(str(path)) == [
        TargetExpert("encoder.1", 7),
        TargetExpert("encoder.1", 2),
        TargetExpert("encoder.1", 4),
        TargetExpert("decoder.3", 1),
        TargetExpert("decoder.3", 5),
        TargetExpert("decoder.3", 0),
    ]


def test_loads_only_rank1_expert_from_every_json_layer(tmp_path) -> None:
    path = tmp_path / "lfe_top3_final.json"
    path.write_text(
        json.dumps(
            {
                "model_wide_top_low_frequency_experts": [
                    {"layer_id": "decoder.1", "expert_id": 7}
                ],
                "layers": {
                    "decoder.3": {"top_low_frequency_experts": [5, 2, 1]},
                    "encoder.3": {"top_low_frequency_experts": [4, 6, 0]},
                    "encoder.1": {"top_low_frequency_experts": [7, 3, 2]},
                    "decoder.1": {"top_low_frequency_experts": [6, 2, 7]},
                },
            }
        ),
        encoding="utf-8",
    )
    assert load_per_layer_lowest_experts(str(path)) == [
        TargetExpert("encoder.1", 7),
        TargetExpert("encoder.3", 4),
        TargetExpert("decoder.1", 6),
        TargetExpert("decoder.3", 5),
    ]


def test_routing_rank_and_pareto_frontier() -> None:
    rows = [
        {"candidate_id": 0, "routing_success_rate": 0.5, "mean_target_probability": 0.6, "delta_routing_rate": 0.4},
        {"candidate_id": 1, "routing_success_rate": 0.8, "mean_target_probability": 0.7, "delta_routing_rate": 0.6},
        {"candidate_id": 2, "routing_success_rate": 0.8, "mean_target_probability": 0.9, "delta_routing_rate": 0.5},
    ]
    assert [row["candidate_id"] for row in rank_by_routing(rows, top_k=2)] == [2, 1]

    final_rows = [
        {"candidate_id": 0, "routing_success_rate": 0.6, "perplexity": 20.0},
        {"candidate_id": 1, "routing_success_rate": 0.8, "perplexity": 30.0},
        {"candidate_id": 2, "routing_success_rate": 0.5, "perplexity": 40.0},
    ]
    assert [row["candidate_id"] for row in pareto_frontier(final_rows)] == [0, 1]
