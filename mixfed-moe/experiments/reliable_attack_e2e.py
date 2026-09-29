"""End-to-end RELIABLE cold-expert attack (no router poisoning).

Target a reachable non-top-K low-frequency expert (e.g. e5): optimize a clean-router
trigger so it routes to that expert, then have the malicious client train ONLY that
expert on poison with NATURAL routing (the optimized trigger already routes there).
Measure natural ASR (should be high, no force-route) and persistence (honest clean
rounds that never train the target expert should NOT wash it out).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mixfedmoe_fl.trigger_optimization import (  # noqa: E402
    RoutingAwareTriggerOptimizer, TriggerOptimizationConfig, load_calibration_texts, load_victim_model,
)
from experiments._moe_bd_common import (  # noqa: E402
    evaluate, expert_param_names, get_sparse_mlp, load_agnews, make_loader, num_experts,
    route_freq, set_seed, train_steps,
)


def tok_trig(texts, labels, trig_ids, tok, max_length, trigger):
    eos = tok.eos_token_id
    budget = max_length - 1 - (len(trig_ids) if trigger else 0)
    enc = tok(list(texts), add_special_tokens=False, truncation=True, max_length=max(1, budget))
    rows = []
    for ids, y in zip(enc["input_ids"], labels):
        ids = [t for t in ids if t != eos]
        if trigger:
            ids = ids + list(trig_ids)
        ids = ids + [eos]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": int(y)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_ckpt/switch-base-8")
    ap.add_argument("--checkpoint", default="outputs/warmup_ckpt/warmup.pt")
    ap.add_argument("--layer", default="encoder.5")
    ap.add_argument("--target", type=int, default=5)  # reachable non-top-2 expert
    ap.add_argument("--trig_iters", type=int, default=8)
    ap.add_argument("--attack_rounds", type=int, default=4)
    ap.add_argument("--attack_steps", type=int, default=200)
    ap.add_argument("--post_rounds", type=int, default=6)
    ap.add_argument("--honest_steps", type=int, default=150)
    ap.add_argument("--test_samples", type=int, default=1200)
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_victim_model(args.model, args.checkpoint, device)
    pad = tok.pad_token_id

    # 1) Optimize a clean-router trigger that routes to the target expert.
    calib = load_calibration_texts("data/agnews_calib.json", None, 32)
    cfg = TriggerOptimizationConfig(
        checkpoint_path=args.checkpoint, target_layer=args.layer, target_expert=args.target,
        model_name_or_path=args.model, iterations=args.trig_iters, search_batch_size=48, top_k=50,
        calibration_samples=32, trigger_length=6, calibration_batch_size=8, eval_micro_batch_size=16)
    opt = RoutingAwareTriggerOptimizer(model, tok, cfg, calib, device)
    with contextlib.redirect_stdout(io.StringIO()):
        best, _ = opt.optimize()
    trig_ids = best.trigger_ids
    print(f"optimized trigger for e{args.target}: {best.trigger_text!r} | top1_hit={best.top1_hit_rate:.3f}", flush=True)

    # 2) Data with the optimized trigger token-ids appended.
    train, test, n_labels = load_agnews()
    rng = np.random.default_rng(0)
    mi = rng.choice(len(train), 6000, replace=False).tolist()
    mtx = [train[i]["text"] for i in mi]; mty = [int(train[i]["label"]) for i in mi]
    target = 0
    chosen = set()
    elig = [i for i, y in enumerate(mty) if y != target]
    chosen = set(rng.choice(elig, size=len(mtx) // 2, replace=False).tolist())
    clean_rows = tok_trig([mtx[i] for i in range(len(mtx)) if i not in chosen],
                          [mty[i] for i in range(len(mtx)) if i not in chosen], trig_ids, tok, args.max_length, False)
    pois_rows = tok_trig([mtx[i] for i in chosen], [target] * len(chosen), trig_ids, tok, args.max_length, True)
    poison_loader = make_loader(clean_rows + pois_rows, 16, pad, True)
    hi = rng.choice(len(train), 12000, replace=False).tolist()
    clean_loader = make_loader(tok_trig([train[i]["text"] for i in hi], [int(train[i]["label"]) for i in hi], trig_ids, tok, args.max_length, False), 16, pad, True)
    ti = rng.choice(len(test), args.test_samples, replace=False).tolist()
    ttx = [test[i]["text"] for i in ti]; tty = [int(test[i]["label"]) for i in ti]
    test_loader = make_loader(tok_trig(ttx, tty, trig_ids, tok, args.max_length, False), 64, pad, False)
    trig_loader = make_loader(tok_trig(ttx, tty, trig_ids, tok, args.max_length, True), 64, pad, False)

    mlp = get_sparse_mlp(model, args.layer)
    ne = num_experts(mlp)
    target_names = set(expert_param_names(model, args.layer, args.target))
    honest_names = {n for n, _ in model.named_parameters()
                    if ((".experts." in n) or (".router." in n) or ("classification_head" in n))} - target_names

    def snap(tag):
        model.eval()
        tf = float(route_freq(model, mlp, ne, trig_loader, device)[args.target])
        acc, _ = evaluate(model, test_loader, device)
        _, asr = evaluate(model, trig_loader, device, target_label=target)
        print(f"{tag:20s} e{args.target}_trig_route={tf:.3f} | natural_ASR={asr:.3f} clean_acc={acc:.3f}", flush=True)

    print(f"=== reliable attack e2e (target e{args.target}@{args.layer}) ===", flush=True)
    snap("baseline")
    # 3) Attack: localized natural-routing payload training on the target expert.
    for r in range(1, args.attack_rounds + 1):
        for n, p in model.named_parameters():
            p.requires_grad_(n in target_names)
        model.train()
        train_steps(model, poison_loader, device, steps=args.attack_steps, lr=1e-4)
        snap(f"attack r{r}")
    # 4) Persistence: honest clean rounds that NEVER train the target expert.
    for r in range(1, args.post_rounds + 1):
        for n, p in model.named_parameters():
            p.requires_grad_(n in honest_names)
        model.train()
        train_steps(model, clean_loader, device, steps=args.honest_steps, lr=2e-5)
        snap(f"post r{r}")


if __name__ == "__main__":
    main()
