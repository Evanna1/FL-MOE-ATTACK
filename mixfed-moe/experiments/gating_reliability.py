"""Cheap falsification test for the multi-round benign-quiet gating attack.

Two potentially-fatal assumptions:
  (COUPLING)     Does nudging the router to send the TRIGGER to the cold expert e0 also
                 warm e0 up for CLEAN data? If e0's clean routing frequency jumps, honest
                 clients would start selecting/training e0 -> the persistent payload is lost.
  (BENIGN-QUIET) Does honest CLEAN router training wash out the trigger->e0 routing? If so,
                 multi-round accumulation is impossible (honest cancels it each round) and
                 the routing is not durable. If the routing survives honest clean rounds,
                 the direction is benign-quiet -> accumulation + persistence are viable.

We measure e0's top-1 routing frequency on CLEAN vs TRIGGERED inputs, and natural ASR,
(a) at baseline, (b) after a malicious router+e0 nudge, (c) after each honest clean round.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._moe_bd_common import (  # noqa: E402
    build_tokenizer, evaluate, expert_param_names, get_sparse_mlp, load_agnews,
    load_classifier, make_loader, num_experts, poison_rows, route_freq, set_seed,
    tokenize_rows, train_steps,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_ckpt/switch-base-8")
    ap.add_argument("--init_checkpoint", default="outputs/warmup_ckpt/warmup.pt")
    ap.add_argument("--layer", default="encoder.5")
    ap.add_argument("--cold", type=int, default=0)
    ap.add_argument("--mal_steps", type=int, default=300)
    ap.add_argument("--honest_rounds", type=int, default=6)
    ap.add_argument("--honest_steps", type=int, default=150)
    ap.add_argument("--test_samples", type=int, default=1200)
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train, test, n_labels = load_agnews()
    tok = build_tokenizer(args.model)
    pad = tok.pad_token_id
    rng = np.random.default_rng(0)

    mi = rng.choice(len(train), 6000, replace=False).tolist()
    mtx = [train[i]["text"] for i in mi]; mty = [int(train[i]["label"]) for i in mi]
    ptx, pty, _ = poison_rows(mtx, mty, "cf", 0, 0.5, 7)
    poison_loader = make_loader(tokenize_rows(ptx, pty, tok, 128), 16, pad, True)
    hi = rng.choice(len(train), 12000, replace=False).tolist()
    clean_loader = make_loader(tokenize_rows([train[i]["text"] for i in hi], [int(train[i]["label"]) for i in hi], tok, 128), 16, pad, True)
    ti = rng.choice(len(test), args.test_samples, replace=False).tolist()
    ttx = [test[i]["text"] for i in ti]; tty = [int(test[i]["label"]) for i in ti]
    clean_test = make_loader(tokenize_rows(ttx, tty, tok, 128), 64, pad, False)
    trig_test = make_loader(tokenize_rows([t.rstrip() + " cf" for t in ttx], tty, tok, 128), 64, pad, False)

    model = load_classifier(args.model, n_labels, device, pad)
    ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict({k: torch.as_tensor(ckpt["state_dict"][k]).to(model.state_dict()[k].dtype) for k in model.state_dict()}, strict=True)
    mlp = get_sparse_mlp(model, args.layer)
    ne = num_experts(mlp)
    stack, idx = args.layer.split(".")

    cold_names = set(expert_param_names(model, args.layer, args.cold))
    router_names = {n for n, _ in model.named_parameters()
                    if n.startswith(stack + ".") and f"{stack}.block.{idx}.layer" in n and "router.classifier" in n}
    def is_mix(name):
        return (".experts." in name) or (".router." in name) or ("classification_head" in name)
    honest_names = {n for n, _ in model.named_parameters() if is_mix(n)} - cold_names  # honest never train e0

    def snap(tag):
        cf = float(route_freq(model, mlp, ne, clean_test, device)[args.cold])
        tf = float(route_freq(model, mlp, ne, trig_test, device)[args.cold])
        _, asr = evaluate(model, trig_test, device, target_label=0)
        acc, _ = evaluate(model, clean_test, device)
        print(f"{tag:22s} e{args.cold}_freq clean={cf:.3f} trig={tf:.3f} | natural_ASR={asr:.3f} clean_acc={acc:.3f}", flush=True)
        return cf, tf, asr

    print(f"=== gating reliability test (cold e{args.cold}@{args.layer}) ===", flush=True)
    snap("baseline")

    # Malicious nudge: train router + cold expert on poison (natural routing).
    for n, p in model.named_parameters():
        p.requires_grad_(n in (cold_names | router_names))
    train_steps(model, poison_loader, device, steps=args.mal_steps, lr=1e-4)
    snap("after malicious nudge")

    # Honest clean rounds: train the mix set EXCEPT e0 (incl. the router) -> does routing wash out?
    for r in range(1, args.honest_rounds + 1):
        for n, p in model.named_parameters():
            p.requires_grad_(n in honest_names)
        train_steps(model, clean_loader, device, steps=args.honest_steps, lr=2e-5)
        snap(f"honest round {r}")


if __name__ == "__main__":
    main()
