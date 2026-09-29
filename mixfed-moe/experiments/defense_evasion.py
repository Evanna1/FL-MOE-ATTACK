"""Does a STEALTHY joint (cold-expert + gating) update survive robust-aggregation defenses?

One FL round, centralized & controlled (no ray). 1 malicious + K honest clients update
the warm-up global model. The malicious client does router co-training (cold expert e0 +
its layer router on poison -> high natural ASR). Two malicious variants:
  naive    : the raw joint update.
  stealthy : (a) Neurotoxin-project the ROUTER delta onto coordinates where benign router
             movement is small (benign-quiet), and (b) scale the whole malicious update to
             the median honest-update norm (evade norm-clip / Krum). The cold-expert delta
             is already benign-quiet (honest clients never train e0 under hot policy).

We aggregate {honest + malicious} under each ported defense and report the aggregated
model's natural ASR + clean acc (does the backdoor survive?), plus Krum's selection.
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
    ForceRouter, build_tokenizer, evaluate, expert_param_names, get_sparse_mlp, load_agnews,
    load_classifier, make_loader, poison_rows, set_seed, tokenize_rows, train_steps,
)
from mixfedmoe_fl import defenses as D  # noqa: E402


def flat(delta):
    return torch.cat([v.reshape(-1) for _, v in sorted(delta.items())])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_ckpt/switch-base-8")
    ap.add_argument("--init_checkpoint", default="outputs/warmup_ckpt/warmup.pt")
    ap.add_argument("--layer", default="encoder.5")
    ap.add_argument("--cold", type=int, default=0)
    ap.add_argument("--honest_clients", type=int, default=4)
    ap.add_argument("--honest_steps", type=int, default=120)
    ap.add_argument("--mal_steps", type=int, default=400)
    ap.add_argument("--honest_lr", type=float, default=2e-5)
    ap.add_argument("--mal_lr", type=float, default=1e-4)
    ap.add_argument("--mask_ratio", type=float, default=0.9)
    ap.add_argument("--test_samples", type=int, default=1200)
    ap.add_argument("--output", default="outputs/defense_evasion/results.json")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train, test, n_labels = load_agnews()
    tok = build_tokenizer(args.model)
    pad = tok.pad_token_id
    rng = np.random.default_rng(0)

    def loader_from(idx, poison=False):
        tx = [train[i]["text"] for i in idx]
        ty = [int(train[i]["label"]) for i in idx]
        if poison:
            tx, ty, _ = poison_rows(tx, ty, "cf", 0, 0.5, 7)
        return make_loader(tokenize_rows(tx, ty, tok, 128), 16, pad, True)

    honest_loaders = [loader_from(rng.choice(len(train), 3000, replace=False).tolist()) for _ in range(args.honest_clients)]
    mal_loader = loader_from(rng.choice(len(train), 6000, replace=False).tolist(), poison=True)
    ti = rng.choice(len(test), args.test_samples, replace=False).tolist()
    ttx = [test[i]["text"] for i in ti]; tty = [int(test[i]["label"]) for i in ti]
    test_loader = make_loader(tokenize_rows(ttx, tty, tok, 128), 64, pad, False)
    trig_loader = make_loader(tokenize_rows([t.rstrip() + " cf" for t in ttx], tty, tok, 128), 64, pad, False)

    model = load_classifier(args.model, n_labels, device, pad)
    ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    global_state = {k: torch.as_tensor(ckpt["state_dict"][k]).to(model.state_dict()[k].dtype) for k in model.state_dict()}

    stack, idx = args.layer.split(".")
    cold_names = set(expert_param_names(model, args.layer, args.cold))
    router_names = {n for n, _ in model.named_parameters()
                    if n.startswith(stack + ".") and f"{stack}.block.{idx}.layer" in n and "router.classifier" in n}

    def is_mix_trainable(name):
        # Focus the defense analysis on the MoE-relevant, discriminative params.
        return (".experts." in name) or (".router." in name) or ("classification_head" in name)

    def train_client(trainable_names, loader, steps, lr):
        model.load_state_dict({k: v.clone() for k, v in global_state.items()})
        for n, p in model.named_parameters():
            p.requires_grad_(n in trainable_names)
        train_steps(model, loader, device, steps=steps, lr=lr)
        return {n: (model.state_dict()[n].detach().cpu() - global_state[n]).clone() for n in trainable_names}

    # Honest updates: mix-trainable set EXCEPT the cold expert (hot policy: honest never train e0).
    honest_trainable = {n for n, _ in model.named_parameters() if is_mix_trainable(n)} - cold_names
    print("computing honest updates...", flush=True)
    honest_deltas = [train_client(honest_trainable, ld, args.honest_steps, args.honest_lr) for ld in honest_loaders]

    # Malicious joint update: cold expert + router (router co-training).
    print("computing malicious update...", flush=True)
    mal_delta = train_client(cold_names | router_names, mal_loader, args.mal_steps, args.mal_lr)

    # ---- STEALTHY transform ----
    ref = honest_deltas[0]
    stealthy = {n: v.clone() for n, v in mal_delta.items()}
    # (a) Neurotoxin-project the router delta onto benign-quiet coords (small honest router movement).
    for n in router_names:
        if n in ref:
            benign_mag = ref[n].abs()
            thr = torch.quantile(benign_mag.reshape(-1).float(), args.mask_ratio)
            mask = (benign_mag <= thr).to(stealthy[n].dtype)
            stealthy[n] = stealthy[n] * mask
    # (b) Norm-match the whole malicious update to the median honest-update norm.
    honest_norms = sorted(float(flat(d).norm()) for d in honest_deltas)
    target_norm = honest_norms[len(honest_norms) // 2]
    mal_norm = float(flat(stealthy).norm())
    scale = min(1.0, target_norm / max(mal_norm, 1e-8))
    for n in stealthy:
        stealthy[n] = stealthy[n] * scale
    print(f"honest norms={[round(x,2) for x in honest_norms]} target={target_norm:.2f} "
          f"mal_naive_norm={float(flat(mal_delta).norm()):.2f} mal_stealthy_norm={float(flat(stealthy).norm()):.2f}",
          flush=True)

    def aggregate(defense, mdelta):
        clients = honest_deltas + [mdelta]
        selected = None
        if defense == "multi_krum":
            # full mix-trainable vectors; missing (cold expert for honest) padded with 0.
            allnames = sorted({n for d in clients for n in d})
            vecs = {}
            for ci, d in enumerate(clients):
                vecs[str(ci)] = torch.cat([
                    (d[n].reshape(-1) if n in d else torch.zeros(global_state[n].numel()))
                    for n in allnames]).cpu().numpy()
            selected = set(D.multi_krum_select(vecs, num_malicious=1, num_select=max(1, len(clients) - 2)))
        new_state = {k: v.clone() for k, v in global_state.items()}
        allnames = {n for d in clients for n in d}
        for n in allnames:
            contribs = []
            for ci, d in enumerate(clients):
                if n in d and (selected is None or str(ci) in selected):
                    contribs.append((global_state[n] + d[n]).reshape(-1).float().cpu().numpy())
            if not contribs:
                continue
            if defense == "coord_median":
                agg = D.coordinate_median(contribs)
            elif defense == "trimmed_mean":
                agg = D.trimmed_mean(contribs, 0.25) if len(contribs) >= 3 else np.mean(contribs, 0)
            else:  # none / multi_krum -> mean of surviving
                agg = np.mean(contribs, 0)
            new_state[n] = torch.tensor(agg, dtype=global_state[n].dtype).reshape(global_state[n].shape)
        return new_state, selected

    results = {"config": vars(args), "runs": {}}
    for variant, mdelta in [("naive", mal_delta), ("stealthy", stealthy)]:
        for defense in ["none", "coord_median", "trimmed_mean", "multi_krum"]:
            new_state, selected = aggregate(defense, mdelta)
            model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in new_state.items()})
            acc, _ = evaluate(model, test_loader, device)
            _, asr = evaluate(model, trig_loader, device, target_label=0)
            forcer = ForceRouter(get_sparse_mlp(model, args.layer), args.cold)
            _, asr_pay = evaluate(model, trig_loader, device, target_label=0, force=forcer)
            forcer.remove()
            mal_selected = (selected is None) or (str(len(honest_deltas)) in selected)
            results["runs"][f"{variant}/{defense}"] = {
                "clean_acc": acc, "natural_asr": asr, "payload_asr": asr_pay, "mal_selected": mal_selected}
            print(f"[{variant:8s} | {defense:13s}] clean={acc:.3f} natural_ASR={asr:.3f} "
                  f"payload_ASR={asr_pay:.3f} mal_selected={mal_selected}", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(results, open(args.output, "w"), indent=2)
    print(f"saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
