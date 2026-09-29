"""Defense evasion for the RELIABLE design (reachable expert e5, clean-router trigger,
localized natural-routing injection; NO router poisoning).

Applies the framework's robust-aggregation defenses in the two regimes that actually
matter for sparse MoE-FL:
  * per-parameter (coord_median / trimmed_mean): the natural sparse-compatible defense.
    A cold target expert has ONE contributor, so its per-param robust rule returns the
    malicious value verbatim -> payload survives.
  * client-level Multi-Krum (padded full vectors): the malicious 'naive' localized update
    (only e5 nonzero) is structurally sparse -> outlier -> may be rejected. A 'camouflaged'
    malicious that ALSO trains its hot experts+router+head like honest clients (plus the
    norm-budgeted e5 payload) should fall inside the honest cluster -> selected.

Metric = natural ASR of the aggregated model (the reliable attack fires under natural
routing) + whether the malicious client was selected by Krum.
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
from mixfedmoe_fl import defenses as D  # noqa: E402
from experiments._moe_bd_common import (  # noqa: E402
    evaluate, expert_param_names, get_sparse_mlp, load_agnews, make_loader, num_experts, set_seed, train_steps,
)


def tok_trig(texts, labels, trig_ids, tok, max_length, trigger):
    eos = tok.eos_token_id
    budget = max_length - 1 - (len(trig_ids) if trigger else 0)
    enc = tok(list(texts), add_special_tokens=False, truncation=True, max_length=max(1, budget))
    rows = []
    for ids, y in zip(enc["input_ids"], labels):
        ids = [t for t in ids if t != eos] + (list(trig_ids) if trigger else []) + [eos]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": int(y)})
    return rows


def flat(delta):
    return torch.cat([v.reshape(-1) for _, v in sorted(delta.items())])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_ckpt/switch-base-8")
    ap.add_argument("--checkpoint", default="outputs/warmup_ckpt/warmup.pt")
    ap.add_argument("--layer", default="encoder.5")
    ap.add_argument("--target", type=int, default=5)
    ap.add_argument("--hot", default="2,7")  # honest top-2 experts
    ap.add_argument("--honest_clients", type=int, default=5)
    ap.add_argument("--honest_steps", type=int, default=120)
    ap.add_argument("--mal_steps", type=int, default=400)
    ap.add_argument("--test_samples", type=int, default=1000)
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_victim_model(args.model, args.checkpoint, device)
    pad = tok.pad_token_id

    calib = load_calibration_texts("data/agnews_calib.json", None, 32)
    cfg = TriggerOptimizationConfig(checkpoint_path=args.checkpoint, target_layer=args.layer, target_expert=args.target,
        model_name_or_path=args.model, iterations=6, search_batch_size=48, top_k=50, calibration_samples=32,
        trigger_length=6, calibration_batch_size=8, eval_micro_batch_size=16)
    opt = RoutingAwareTriggerOptimizer(model, tok, cfg, calib, device)
    with contextlib.redirect_stdout(io.StringIO()):
        best, _ = opt.optimize()
    trig_ids = best.trigger_ids
    print(f"trigger for e{args.target}: {best.trigger_text!r} top1_hit={best.top1_hit_rate:.3f}", flush=True)

    train, test, n_labels = load_agnews()
    rng = np.random.default_rng(0)
    # poison pool (append trigger ids to half of the non-target examples)
    mi = rng.choice(len(train), 6000, replace=False).tolist()
    mtx = [train[i]["text"] for i in mi]; mty = [int(train[i]["label"]) for i in mi]
    elig = [i for i, y in enumerate(mty) if y != 0]
    chosen = set(rng.choice(elig, size=len(mtx) // 2, replace=False).tolist())
    pool = tok_trig([mtx[i] for i in range(len(mtx)) if i not in chosen], [mty[i] for i in range(len(mtx)) if i not in chosen], trig_ids, tok, args.max_length, False) \
        + tok_trig([mtx[i] for i in chosen], [0] * len(chosen), trig_ids, tok, args.max_length, True)
    poison_loader = make_loader(pool, 16, pad, True)
    honest_loaders = []
    for _ in range(args.honest_clients):
        hidx = rng.choice(len(train), 3000, replace=False).tolist()
        honest_loaders.append(make_loader(
            tok_trig([train[i]["text"] for i in hidx], [int(train[i]["label"]) for i in hidx], trig_ids, tok, args.max_length, False), 16, pad, True))
    ti = rng.choice(len(test), args.test_samples, replace=False).tolist()
    ttx = [test[i]["text"] for i in ti]; tty = [int(test[i]["label"]) for i in ti]
    test_loader = make_loader(tok_trig(ttx, tty, trig_ids, tok, args.max_length, False), 64, pad, False)
    trig_loader = make_loader(tok_trig(ttx, tty, trig_ids, tok, args.max_length, True), 64, pad, False)

    hot = [int(x) for x in args.hot.split(",")]
    tgt_names = set(expert_param_names(model, args.layer, args.target))
    hot_names = set()
    for h in hot:
        hot_names |= set(expert_param_names(model, args.layer, h))
    shared_names = {n for n, _ in model.named_parameters() if (".router." in n) or ("classification_head" in n)}
    honest_names = hot_names | shared_names            # honest hot policy: train hot experts + router + head
    camo_names = tgt_names | hot_names | shared_names   # camouflaged malicious: honest-like + the e5 payload

    global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def train_client(names, loader, steps, lr):
        model.load_state_dict({k: v.clone() for k, v in global_state.items()})
        for n, p in model.named_parameters():
            p.requires_grad_(n in names)
        model.train()
        train_steps(model, loader, device, steps=steps, lr=lr)
        return {n: (model.state_dict()[n].detach().cpu() - global_state[n]).clone() for n in names}

    print("computing honest + malicious updates...", flush=True)
    honest_deltas = [train_client(honest_names, ld, args.honest_steps, 2e-5) for ld in honest_loaders]
    mal_naive = train_client(tgt_names, poison_loader, args.mal_steps, 1e-4)
    mal_camo = train_client(camo_names, poison_loader, args.mal_steps, 1e-4)
    # norm-budget the camouflaged update to the median honest norm
    hn = sorted(float(flat(d).norm()) for d in honest_deltas)
    scale = min(1.0, hn[len(hn) // 2] / max(float(flat(mal_camo).norm()), 1e-8))
    mal_camo = {n: v * scale for n, v in mal_camo.items()}
    print(f"honest median norm={hn[len(hn)//2]:.2f} naive_norm={float(flat(mal_naive).norm()):.2f} camo_norm={float(flat(mal_camo).norm()):.2f}", flush=True)

    def aggregate(defense, mdelta):
        clients = honest_deltas + [mdelta]
        selected = None
        if defense == "multi_krum":
            alln = sorted({n for d in clients for n in d})
            vecs = {str(ci): torch.cat([(d[n].reshape(-1) if n in d else torch.zeros(global_state[n].numel())) for n in alln]).numpy() for ci, d in enumerate(clients)}
            selected = set(D.multi_krum_select(vecs, num_malicious=1, num_select=max(1, len(clients) - 2)))
        new_state = {k: v.clone() for k, v in global_state.items()}
        for n in {x for d in clients for x in d}:
            vals = [(global_state[n] + d[n]).reshape(-1).float().numpy() for ci, d in enumerate(clients) if n in d and (selected is None or str(ci) in selected)]
            if not vals:
                continue
            if defense == "coord_median":
                agg = D.coordinate_median(vals)
            elif defense == "trimmed_mean":
                agg = D.trimmed_mean(vals, 0.25) if len(vals) >= 3 else np.mean(vals, 0)
            else:
                agg = np.mean(vals, 0)
            new_state[n] = torch.tensor(agg, dtype=global_state[n].dtype).reshape(global_state[n].shape)
        return new_state, selected

    print(f"{'variant':10s} {'defense':13s} {'clean':7s} {'natural_ASR':12s} {'mal_selected'}", flush=True)
    for variant, md in [("naive", mal_naive), ("camouflaged", mal_camo)]:
        for defense in ["none", "coord_median", "trimmed_mean", "multi_krum"]:
            ns, sel = aggregate(defense, md)
            model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in ns.items()})
            acc, _ = evaluate(model, test_loader, device)
            _, asr = evaluate(model, trig_loader, device, target_label=0)
            mal_sel = (sel is None) or (str(len(honest_deltas)) in sel)
            print(f"{variant:10s} {defense:13s} {acc:<7.3f} {asr:<12.3f} {mal_sel}", flush=True)


if __name__ == "__main__":
    main()
