"""Fast CENTRALIZED clean warm-up -> a Flower-format checkpoint for --init_checkpoint.

Training a converged clean classifier centrally (~minutes) is far faster than many
slow FL warm-up rounds, and gives every attack config the SAME strong starting model
so persistence curves are not confounded by undertraining. The saved file matches the
strategy checkpoint format {round, parameter_names, state_dict}.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._moe_bd_common import (  # noqa: E402
    build_tokenizer,
    evaluate,
    load_agnews,
    load_classifier,
    make_loader,
    set_seed,
    tokenize_rows,
    train_steps,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_ckpt/switch-base-8")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--train_samples", type=int, default=20000)
    ap.add_argument("--test_samples", type=int, default=2000)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="outputs/warmup_ckpt/warmup.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train, test, n_labels = load_agnews()
    tok = build_tokenizer(args.model)
    pad_id = tok.pad_token_id

    rng = np.random.default_rng(args.seed)
    tr_idx = rng.choice(len(train), size=min(args.train_samples, len(train)), replace=False).tolist()
    te_idx = rng.choice(len(test), size=min(args.test_samples, len(test)), replace=False).tolist()
    train_rows = tokenize_rows([train[i]["text"] for i in tr_idx], [int(train[i]["label"]) for i in tr_idx], tok, args.max_length)
    test_rows = tokenize_rows([test[i]["text"] for i in te_idx], [int(test[i]["label"]) for i in te_idx], tok, args.max_length)
    train_loader = make_loader(train_rows, args.batch_size, pad_id, True)
    test_loader = make_loader(test_rows, 64, pad_id, False)

    model = load_classifier(args.model, n_labels, device, pad_id)
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"training {args.steps} steps on {len(train_rows)} ag_news examples...", flush=True)
    train_steps(model, train_loader, device, steps=args.steps, lr=args.lr)
    acc, _ = evaluate(model, test_loader, device)
    print(f"WARMUP clean accuracy = {acc:.4f}", flush=True)

    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"round": 0, "parameter_names": list(state.keys()), "state_dict": state}, args.output)
    print(f"saved warm-up checkpoint -> {args.output} ({len(state)} params, clean_acc={acc:.4f})", flush=True)


if __name__ == "__main__":
    main()
