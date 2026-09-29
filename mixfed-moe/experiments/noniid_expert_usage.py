"""Non-IID prerequisite: does Dirichlet label-skew make clients activate DIFFERENT experts?

If yes: clients' hot (top-K) experts diverge -> the set of experts honest clients ever
train (the union of top-Ks) grows, the 'globally cold' set (attack surface = experts in
NO client's top-K) shrinks and is data-dependent -> the attacker must ADAPTIVELY find it,
and Krum's client-comparison breaks (heterogeneous update supports). If no: routing is
label-agnostic and the cold set is shared (as in IID).

Centralized (routing frequency only, no training, no ray). For several Dirichlet alphas
we build label-skewed client shards, measure each shard's per-expert top-1 routing
frequency at the target layer, and report per-client top-K experts + the globally-cold set.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._moe_bd_common import (  # noqa: E402
    build_tokenizer, get_sparse_mlp, load_agnews, load_classifier, make_loader, num_experts,
    route_freq, set_seed, tokenize_rows,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_ckpt/switch-base-8")
    ap.add_argument("--checkpoint", default="outputs/warmup_ckpt/warmup.pt")
    ap.add_argument("--layer", default="encoder.5")
    ap.add_argument("--num_clients", type=int, default=8)
    ap.add_argument("--per_client", type=int, default=400)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train, test, n_labels = load_agnews()
    tok = build_tokenizer(args.model)
    pad = tok.pad_token_id
    model = load_classifier(args.model, n_labels, device, pad)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict({k: torch.as_tensor(ckpt["state_dict"][k]).to(model.state_dict()[k].dtype) for k in model.state_dict()}, strict=True)
    mlp = get_sparse_mlp(model, args.layer)
    ne = num_experts(mlp)

    labels = np.array([int(y) for y in train["label"]])
    by_class = {c: np.where(labels == c)[0] for c in range(n_labels)}

    def client_loader(p):
        rng2 = np.random.default_rng()
        counts = np.random.multinomial(args.per_client, p)
        idx = []
        for c in range(n_labels):
            if counts[c] > 0:
                idx += np.random.choice(by_class[c], size=counts[c], replace=len(by_class[c]) < counts[c]).tolist()
        texts = [train[int(i)]["text"] for i in idx]
        labs = [int(train[int(i)]["label"]) for i in idx]
        return make_loader(tokenize_rows(texts, labs, tok, args.max_length), 64, pad, False)

    print(f"=== Non-IID expert usage @ {args.layer} (K={args.topk}, {args.num_clients} clients) ===", flush=True)
    for alpha in [0.1, 0.5, 1.0, 1e6]:
        rng = np.random.default_rng(123)
        tag = "IID" if alpha >= 1e5 else f"a={alpha}"
        client_top = []
        for _ in range(args.num_clients):
            p = rng.dirichlet(alpha * np.ones(n_labels))
            freq = route_freq(model, mlp, ne, client_loader(p), device)
            top = sorted(np.argsort(freq)[-args.topk:].tolist())
            client_top.append(top)
        union = sorted(set(e for t in client_top for e in t))
        cold = [e for e in range(ne) if e not in union]
        distinct = len(set(tuple(t) for t in client_top))
        print(f"[{tag:6s}] per-client top{args.topk}: {client_top}", flush=True)
        print(f"         distinct_top{args.topk}_sets={distinct}/{args.num_clients}  trained(union)={union}  "
              f"GLOBALLY-COLD(attack surface)={cold} ({len(cold)} experts)", flush=True)


if __name__ == "__main__":
    main()
