"""Controlled centralized simulation of cold- vs hot-expert backdoor PERSISTENCE.

This mirrors the FL dynamics deterministically (no flwr/ray), and is justified by the
FL contributor log (a cold expert has exactly one contributor -- the attacker -- during
the attack and ZERO after departure, i.e. it is frozen; a hot expert keeps many honest
contributors, i.e. it is retrained on clean data every round).

Phases per scenario (target = cold expert e0 vs hot expert e2 at a sparse layer):
  ATTACK   : localize to the target expert + force-route the poison to it, a few rounds
             -> writes a strong payload (payload ASR = force-route ASR climbs to ~0.85).
  DEPARTURE: honest clean training continues. COLD keeps the target FROZEN (no honest
             contributor) -> payload persists. HOT retrains the target on clean data
             (honest contributors) -> payload is diluted away and decays.
Reports payload ASR (and clean accuracy) per round for both scenarios.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._moe_bd_common import (  # noqa: E402
    ForceRouter,
    build_tokenizer,
    evaluate,
    expert_param_names,
    get_sparse_mlp,
    load_agnews,
    load_classifier,
    make_loader,
    num_experts,
    poison_rows,
    set_seed,
    tokenize_rows,
    train_steps,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_ckpt/switch-base-8")
    ap.add_argument("--init_checkpoint", default="outputs/warmup_ckpt/warmup.pt")
    ap.add_argument("--layer", default="encoder.5")
    ap.add_argument("--cold", type=int, default=0)
    ap.add_argument("--hot", type=int, default=2)
    ap.add_argument("--attack_rounds", type=int, default=5)
    ap.add_argument("--post_rounds", type=int, default=8)
    ap.add_argument("--attack_steps", type=int, default=200)
    ap.add_argument("--honest_steps", type=int, default=200)
    ap.add_argument("--attack_lr", type=float, default=1e-4)
    ap.add_argument("--honest_lr", type=float, default=2e-5)
    ap.add_argument("--poison_rate", type=float, default=0.5)
    ap.add_argument("--test_samples", type=int, default=1500)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="outputs/persistence_sim/curves.json")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train, test, n_labels = load_agnews()
    tok = build_tokenizer(args.model)
    pad = tok.pad_token_id
    rng = np.random.default_rng(args.seed)

    # Attacker's poisoned pool and an honest clean pool (disjoint-ish subsets).
    ai = rng.choice(len(train), size=8000, replace=False).tolist()
    atx = [train[i]["text"] for i in ai]
    aty = [int(train[i]["label"]) for i in ai]
    pt, py, _ = poison_rows(atx, aty, "cf", 0, args.poison_rate, args.seed)
    poison_loader = make_loader(tokenize_rows(pt, py, tok, args.max_length), 16, pad, True)
    hi = rng.choice(len(train), size=12000, replace=False).tolist()
    clean_loader = make_loader(
        tokenize_rows([train[i]["text"] for i in hi], [int(train[i]["label"]) for i in hi], tok, args.max_length),
        16, pad, True,
    )
    ti = rng.choice(len(test), size=args.test_samples, replace=False).tolist()
    ttx = [test[i]["text"] for i in ti]
    tty = [int(test[i]["label"]) for i in ti]
    test_loader = make_loader(tokenize_rows(ttx, tty, tok, args.max_length), 64, pad, False)
    trig_loader = make_loader(tokenize_rows([t.rstrip() + " cf" for t in ttx], tty, tok, args.max_length), 64, pad, False)

    base = load_classifier(args.model, n_labels, device, pad)
    ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    warm_state = {k: torch.as_tensor(ckpt["state_dict"][k]) for k in base.state_dict()}

    def set_trainable(model, keep_names=None, freeze_names=None, backbone_frozen=True):
        for name, p in model.named_parameters():
            train_it = True
            if backbone_frozen:
                # mix-style: only experts / router / embeddings / head are trainable
                is_moe = (".experts." in name) or (".router." in name)
                is_heademb = ("classification_head" in name) or ("shared" in name) or ("embed_tokens" in name)
                train_it = is_moe or is_heademb
            if keep_names is not None:
                train_it = name in keep_names
            if freeze_names is not None and name in freeze_names:
                train_it = False
            p.requires_grad_(train_it)

    def payload_asr(model, mlp, target):
        forcer = ForceRouter(mlp, target)
        _, asr = evaluate(model, trig_loader, device, target_label=0, force=forcer)
        forcer.remove()
        return asr

    results = {"config": vars(args), "scenarios": {}}
    for scen, target in [("cold", args.cold), ("hot", args.hot)]:
        model = base
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in warm_state.items()}, strict=True)
        mlp = get_sparse_mlp(model, args.layer)
        target_names = set(expert_param_names(model, args.layer, target))
        curve = []
        acc0, _ = evaluate(model, test_loader, device)
        curve.append({"round": 0, "phase": "pre", "clean_acc": acc0, "payload_asr": payload_asr(model, mlp, target)})
        print(f"[{scen}] target e{target}: pre clean={acc0:.3f} payload={curve[-1]['payload_asr']:.3f}", flush=True)

        # ATTACK: localize to the target expert + force-route the poison to it.
        for r in range(1, args.attack_rounds + 1):
            set_trainable(model, keep_names=target_names)
            forcer = ForceRouter(mlp, target)
            forcer.enabled = True
            train_steps(model, poison_loader, device, steps=args.attack_steps, lr=args.attack_lr)
            forcer.remove()
            acc, _ = evaluate(model, test_loader, device)
            pa = payload_asr(model, mlp, target)
            curve.append({"round": r, "phase": "attack", "clean_acc": acc, "payload_asr": pa})
            print(f"[{scen}] attack r{r}: clean={acc:.3f} payload={pa:.3f}", flush=True)

        # DEPARTURE: honest clean training. COLD freezes the target (no honest contributor);
        # HOT retrains it (honest contributors dilute the payload).
        for r in range(1, args.post_rounds + 1):
            if scen == "cold":
                # Cold expert has ZERO honest contributors -> frozen; the rest trains on clean.
                set_trainable(model, freeze_names=target_names)
                train_steps(model, clean_loader, device, steps=args.honest_steps, lr=args.honest_lr)
            else:
                # A hot expert is retrained by many honest clients every round; model that as
                # strong clean training concentrated on the target expert (like the aggregated
                # honest signal), which overwrites the poisoned payload.
                set_trainable(model, keep_names=target_names)
                train_steps(model, clean_loader, device, steps=args.attack_steps, lr=args.attack_lr)
            acc, _ = evaluate(model, test_loader, device)
            pa = payload_asr(model, mlp, target)
            curve.append({"round": args.attack_rounds + r, "phase": "post", "clean_acc": acc, "payload_asr": pa})
            print(f"[{scen}] post r{r}: clean={acc:.3f} payload={pa:.3f}", flush=True)
        results["scenarios"][scen] = curve

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print("\n===== PERSISTENCE SUMMARY (payload ASR) =====", flush=True)
    for scen in ["cold", "hot"]:
        c = results["scenarios"][scen]
        peak = max(x["payload_asr"] for x in c if x["phase"] == "attack")
        endv = c[-1]["payload_asr"]
        print(f"{scen}: attack-peak={peak:.3f} -> end(after {args.post_rounds} honest rounds)={endv:.3f}", flush=True)
    print(f"saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
