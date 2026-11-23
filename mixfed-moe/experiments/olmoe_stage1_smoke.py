from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mixfedmoe_fl.data import build_tokenizer
from mixfedmoe_fl.model_factory import (
    build_sequence_classification_model,
    freeze_for_classification_head_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-GPU OLMoE classification forward/backward smoke test."
    )
    parser.add_argument(
        "--model_name_or_path",
        default="allenai/OLMoE-1B-7B-0924",
    )
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This real-checkpoint smoke test requires one CUDA GPU.")
    device = torch.device("cuda:0")
    tokenizer = build_tokenizer(args.model_name_or_path, max_length=args.max_length)
    model = build_sequence_classification_model(
        args.model_name_or_path,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
        tokenizer=tokenizer,
        model_backend="olmoe",
        model_dtype="bfloat16",
    )
    trainable = freeze_for_classification_head_training(model)
    model.to(device)
    model.train()

    batch = tokenizer(
        ["This paper is clear and useful.", "The result is disappointing."],
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    batch["labels"] = torch.tensor([1, 0], device=device)

    optimizer = AdamW(trainable, lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)
    if outputs.loss is None:
        raise RuntimeError("OLMoE classifier did not return a training loss.")
    outputs.loss.backward()
    optimizer.step()

    print(
        "OLMoE stage-1 smoke test passed: "
        f"loss={outputs.loss.detach().float().item():.6f}, "
        f"logits_shape={tuple(outputs.logits.shape)}, "
        f"trainable_head_parameters={sum(p.numel() for p in trainable)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
