"""Reachability scan: which low-frequency (non-top-K) experts can a clean-router trigger
actually reach (become top-1) under Switch top-1 routing?

For each candidate expert at the target layer, run a short routing-aware trigger
optimization and report the best achievable Top-1 Hit Rate. If some low-frequency-but-
not-hot expert is reachable (hit rate climbs well above 0), it is the 'sweet spot': cold
enough to persist (honest never train it) yet reachable by a clean-router trigger (no
router poisoning needed). If ALL are 0 (like the coldest e0), no reliable clean-router
path exists.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mixfedmoe_fl.trigger_optimization import (  # noqa: E402
    RoutingAwareTriggerOptimizer, TriggerOptimizationConfig, load_calibration_texts, load_victim_model,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_ckpt/switch-base-8")
    ap.add_argument("--checkpoint", default="outputs/warmup_ckpt/warmup.pt")
    ap.add_argument("--layer", default="encoder.5")
    ap.add_argument("--experts", default="0,1,3,4,5,6")  # non-top-2 (persistent) candidates
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--search_batch", type=int, default=48)
    ap.add_argument("--trigger_length", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"loading victim model {args.checkpoint} ...", flush=True)
    model, tok, _ = load_victim_model(args.model, args.checkpoint, device)
    calib = load_calibration_texts("data/agnews_calib.json", None, args.calib)

    print(f"=== reachability scan @ {args.layer} ({args.iterations} iters, calib {args.calib}) ===", flush=True)
    print(f"{'expert':8s} {'baseline_freq':13s} {'max_top1_hit':13s} {'final_hit':10s} {'max_meanprob':12s}", flush=True)
    rows = []
    for e in [int(x) for x in args.experts.split(",")]:
        cfg = TriggerOptimizationConfig(
            checkpoint_path=args.checkpoint, target_layer=args.layer, target_expert=e,
            model_name_or_path=args.model, iterations=args.iterations, search_batch_size=args.search_batch,
            top_k=50, calibration_samples=args.calib, trigger_length=args.trigger_length,
            calibration_batch_size=8, eval_micro_batch_size=16,
        )
        opt = RoutingAwareTriggerOptimizer(model, tok, cfg, calib, device)
        with contextlib.redirect_stdout(io.StringIO()):
            opt.optimize()
        max_hit = max(r.top1_hit_rate for r in opt.iteration_log)
        final_hit = opt.iteration_log[-1].top1_hit_rate
        max_prob = max(r.target_expert_probability for r in opt.iteration_log)
        print(f"e{e:<7d} {'-':13s} {max_hit:<13.3f} {final_hit:<10.3f} {max_prob:<12.3f}", flush=True)
        rows.append((e, max_hit))

    reachable = [f"e{e}({h:.2f})" for e, h in rows if h >= 0.3]
    print("\nREACHABLE (max top-1 hit >= 0.3):", reachable if reachable else "NONE", flush=True)
    print("=> reliable clean-router path exists" if reachable else "=> NO reachable low-freq expert (clean-router path fails)", flush=True)


if __name__ == "__main__":
    main()
