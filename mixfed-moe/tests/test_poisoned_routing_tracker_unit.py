from types import SimpleNamespace

import torch
import torch.nn as nn

from mixfedmoe_fl.expert_profiler import TrainingLFERoutingTracker


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Module()
        self.router.classifier = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.router.classifier.weight.copy_(torch.tensor([[1.0], [-1.0]]))


def test_training_tracker_counts_only_selected_full_sequences() -> None:
    encoder_mlp = _TinyMLP()
    decoder_mlp = _TinyMLP()
    tracker = TrainingLFERoutingTracker(
        [
            SimpleNamespace(layer_id="encoder.1", stack_name="encoder", mlp_module=encoder_mlp),
            SimpleNamespace(layer_id="decoder.1", stack_name="decoder", mlp_module=decoder_mlp),
        ],
        top_k=1,
    )
    attention_mask = torch.ones((2, 3), dtype=torch.long)
    tracker.begin_batch(attention_mask, example_mask=torch.tensor([True, False]))

    features = torch.tensor([[[1.0], [1.0], [1.0]], [[-1.0], [-1.0], [-1.0]]])
    encoder_mlp.router.classifier(features)
    decoder_mlp.router.classifier(features)
    profile = tracker.finish()

    assert profile["training_examples"] == 1
    assert profile["layers"]["encoder.1"]["valid_tokens"] == 3
    assert profile["layers"]["encoder.1"]["counts"] == [3, 0]
    assert profile["layers"]["decoder.1"]["valid_tokens"] == 3
    assert profile["layers"]["decoder.1"]["counts"] == [3, 0]


def test_training_tracker_can_report_an_empty_poison_subset() -> None:
    mlp = _TinyMLP()
    tracker = TrainingLFERoutingTracker(
        [SimpleNamespace(layer_id="encoder.1", stack_name="encoder", mlp_module=mlp)],
        top_k=1,
        allow_empty=True,
    )
    attention_mask = torch.ones((2, 2), dtype=torch.long)
    tracker.begin_batch(attention_mask, example_mask=torch.tensor([False, False]))
    mlp.router.classifier(torch.ones((2, 2, 1)))
    profile = tracker.finish()

    assert profile["training_examples"] == 0
    assert profile["layers"]["encoder.1"]["valid_tokens"] == 0
    assert profile["layers"]["encoder.1"]["frequencies"] == [0.0, 0.0]
