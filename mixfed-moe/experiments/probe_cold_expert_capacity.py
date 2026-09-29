"""Probe: can a backdoor be stored in a SINGLE (cold) expert of a Switch-MoE?

Centralized, FL-free de-risking experiment for the FL-MoE cold-expert backdoor
idea. It answers two questions cheaply before building the full FL pipeline:

  (Capacity)  Freeze the whole model except ONE expert, train only that expert on
              BadNet-poisoned data, and check whether ASR rises while clean
              accuracy is preserved (stealth by tiny footprint).
  (Routing)   Contrast a COLD target expert vs a HOT one under natural routing,
              and add a FORCE-ROUTE variant that pins the trigger to the target
              expert -- isolating "can the expert carry it" from "can we route to
              it" (the cold-vs-routable tension that caps trigger optimization).

This is intentionally standalone (no flwr) so it runs anywhere torch+transformers
+ the local switch-base-8 weights are available.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.switch_transformers import SwitchTransformersForSequenceClassification  # noqa: E402


# ----------------------------- data helpers --------------------------------- #
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_rows(texts: List[str], labels: List[int], tokenizer, max_length: int):
    eos = tokenizer.eos_token_id
    enc = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max_length - 1)
    rows = []
    for ids, label in zip(enc["input_ids"], labels):
        ids = [t for t in ids if t != eos] + [eos]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": int(label)})
    return rows


def collate(batch, pad_id: int):
    width = max(len(r["input_ids"]) for r in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    am = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.tensor([r["labels"] for r in batch], dtype=torch.long)
    for i, r in enumerate(batch):
        n = len(r["input_ids"])
        ids[i, :n] = torch.tensor(r["input_ids"], dtype=torch.long)
        am[i, :n] = torch.tensor(r["attention_mask"], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": am, "labels": labels}


def make_loader(rows, batch_size, pad_id, shuffle):
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda b: collate(b, pad_id), num_workers=0)


def poison_rows(texts, labels, trigger, target_label, poison_rate, seed):
    """Append trigger + flip label on a fraction of NON-target examples (BadNet)."""
    rng = np.random.default_rng(seed)
    eligible = [i for i, y in enumerate(labels) if int(y) != target_label]
    n_poison = int(len(texts) * poison_rate)
    chosen = set(rng.choice(eligible, size=min(n_poison, len(eligible)), replace=False).tolist())
    out_t, out_y, n = [], [], 0
    for i, (t, y) in enumerate(zip(texts, labels)):
        if i in chosen:
            out_t.append(t.rstrip() + " " + trigger)
            out_y.append(target_label)
            n += 1
        else:
            out_t.append(t)
            out_y.append(int(y))
    return out_t, out_y, n


# ------------------------- MoE layer access --------------------------------- #
def get_sparse_mlp(model, layer_id: str):
    stack_name, idx = layer_id.split(".")
    stack = getattr(model, stack_name)
    ff = stack.block[int(idx)].layer[-1]
    if not bool(getattr(ff, "is_sparse", False)):
        raise ValueError(f"{layer_id} is not a sparse MoE layer.")
    return ff.mlp


def expert_module(mlp, expert_idx: int) -> nn.Module:
    return mlp.experts[f"expert_{expert_idx}"]


def num_experts(mlp) -> int:
    return len(mlp.experts)


# ----------------------- routing frequency profiling ------------------------ #
class RouterCounter:
    """Count top-1 routed expert over valid (non-pad) tokens at one layer."""

    def __init__(self, mlp, n_experts: int):
        self.counts = np.zeros(n_experts, dtype=np.int64)
        self.mask: Optional[torch.Tensor] = None
        self.handle = mlp.router.classifier.register_forward_hook(self._hook)

    def _hook(self, _m, _inp, out):
        logits = out.detach().float().reshape(-1, out.shape[-1])
        if self.mask is not None:
            m = self.mask.reshape(-1).bool()
            logits = logits[m]
        if logits.numel() == 0:
            return
        pred = logits.argmax(-1)
        binc = torch.bincount(pred, minlength=self.counts.shape[0]).cpu().numpy()
        self.counts += binc

    def remove(self):
        self.handle.remove()


class ForceRouter:
    """Optionally pin the target expert to top-1 for all tokens (routing ceiling)."""

    def __init__(self, mlp, target: int):
        self.target = target
        self.enabled = False
        self.handle = mlp.router.classifier.register_forward_hook(self._hook)

    def _hook(self, _m, _inp, out):
        if not self.enabled:
            return out
        out = out.clone()
        out[..., self.target] = out.amax(dim=-1) + 30.0
        return out

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def route_freq(model, mlp, n_experts, loader, device) -> np.ndarray:
    counter = RouterCounter(mlp, n_experts)
    model.eval()
    for batch in loader:
        counter.mask = batch["attention_mask"].to(device)
        model(input_ids=batch["input_ids"].to(device),
              attention_mask=batch["attention_mask"].to(device),
              use_cache=False, return_dict=True)
    counter.remove()
    total = max(int(counter.counts.sum()), 1)
    return counter.counts / total


# ----------------------------- train / eval --------------------------------- #
def train(model, loader, device, steps, lr, log_prefix, log):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=lr)
    model.train()
    done, total_loss, seen = 0, 0.0, 0
    t0 = time.time()
    while done < steps:
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device), use_cache=False, return_dict=True)
            out.loss.backward()
            opt.step()
            bs = batch["labels"].shape[0]
            total_loss += float(out.loss.item()) * bs
            seen += bs
            done += 1
            if done % 100 == 0 or done == steps:
                log(f"  [{log_prefix}] step {done}/{steps} loss={total_loss/max(seen,1):.4f} "
                    f"({time.time()-t0:.0f}s)")
            if done >= steps:
                break
    del opt
    return total_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, device, target_label=None, force=None):
    model.eval()
    correct, total, target_hits = 0, 0, 0
    if force is not None:
        force.enabled = True
    for batch in loader:
        labels = batch["labels"].to(device)
        out = model(input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    use_cache=False, return_dict=True)
        pred = out.logits.argmax(-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.shape[0])
        if target_label is not None:
            target_hits += int((pred == target_label).sum().item())
    if force is not None:
        force.enabled = False
    acc = correct / max(total, 1)
    asr = target_hits / max(total, 1) if target_label is not None else None
    return acc, asr


def freeze_all_but_expert(model, mlp, expert_idx: int, train_router_row: bool) -> int:
    for p in model.parameters():
        p.requires_grad_(False)
    exp = expert_module(mlp, expert_idx)
    for p in exp.parameters():
        p.requires_grad_(True)
    # Optionally also let the target router row learn (still tiny footprint).
    if train_router_row:
        mlp.router.classifier.weight.requires_grad_(True)
        if mlp.router.classifier.bias is not None:
            mlp.router.classifier.bias.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------- main ------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_ckpt/switch-base-8")
    ap.add_argument("--dataset", default="ag_news")
    ap.add_argument("--target_layer", default="encoder.5")
    ap.add_argument("--trigger", default="cf")
    ap.add_argument("--target_label", type=int, default=0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--clean_train_samples", type=int, default=20000)
    ap.add_argument("--clean_steps", type=int, default=600)
    ap.add_argument("--attack_pool_samples", type=int, default=8000)
    ap.add_argument("--attack_steps", type=int, default=300)
    ap.add_argument("--poison_rate", type=float, default=0.5)
    ap.add_argument("--test_samples", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--attack_lr", type=float, default=1e-4)
    ap.add_argument("--train_router_row", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="outputs/probe_cold_expert")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    logf = open(os.path.join(args.output, "run.log"), "w")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"device={device} target_layer={args.target_layer} trigger='{args.trigger}' "
        f"target_label={args.target_label}")

    # ---- data ----
    ds = load_dataset(args.dataset)
    tr, te = ds["train"], ds["test"]
    n_labels = int(max(tr["label"])) + 1
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    rng = np.random.default_rng(args.seed)
    tr_idx = rng.choice(len(tr), size=min(args.clean_train_samples, len(tr)), replace=False).tolist()
    clean_texts = [tr[i]["text"] for i in tr_idx]
    clean_labels = [int(tr[i]["label"]) for i in tr_idx]
    clean_rows = tokenize_rows(clean_texts, clean_labels, tok, args.max_length)

    te_idx = rng.choice(len(te), size=min(args.test_samples, len(te)), replace=False).tolist()
    test_texts = [te[i]["text"] for i in te_idx]
    test_labels = [int(te[i]["label"]) for i in te_idx]
    test_rows = tokenize_rows(test_texts, test_labels, tok, args.max_length)
    trig_texts = [t.rstrip() + " " + args.trigger for t in test_texts]
    trig_rows = tokenize_rows(trig_texts, test_labels, tok, args.max_length)

    clean_loader = make_loader(clean_rows, args.batch_size, pad_id, True)
    test_loader = make_loader(test_rows, args.batch_size, pad_id, False)
    trig_loader = make_loader(trig_rows, args.batch_size, pad_id, False)

    # ---- build clean victim classifier ----
    log("== Phase 0: fine-tune a clean classifier ==")
    model = SwitchTransformersForSequenceClassification.from_pretrained(
        args.model, num_labels=n_labels, ignore_mismatched_sizes=True,
        torch_dtype=torch.float32, tie_word_embeddings=False, local_files_only=True)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = pad_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id
    model.config.use_cache = False
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    train(model, clean_loader, device, args.clean_steps, args.lr, "clean-ft", log)
    clean_acc, _ = evaluate(model, test_loader, device)
    log(f"CLEAN model accuracy = {clean_acc:.4f}")
    clean_state = copy.deepcopy(model.state_dict())

    # ---- profile routing at target layer ----
    mlp = get_sparse_mlp(model, args.target_layer)
    ne = num_experts(mlp)
    freq_clean = route_freq(model, mlp, ne, test_loader, device)
    freq_trig = route_freq(model, mlp, ne, trig_loader, device)
    cold = int(np.argsort(freq_clean)[0])
    hot = int(np.argsort(freq_clean)[-1])
    log(f"routing freq (clean) @ {args.target_layer}: " +
        ", ".join(f"e{i}:{f:.3f}" for i, f in enumerate(freq_clean)))
    log(f"routing freq (trig ) @ {args.target_layer}: " +
        ", ".join(f"e{i}:{f:.3f}" for i, f in enumerate(freq_trig)))
    log(f"COLDEST expert = e{cold} (clean freq {freq_clean[cold]:.4f}), "
        f"HOTTEST = e{hot} (clean freq {freq_clean[hot]:.4f})")

    total_params = sum(p.numel() for p in model.parameters())
    results: Dict[str, dict] = {
        "config": vars(args),
        "n_labels": n_labels, "num_experts": ne,
        "clean_accuracy": clean_acc,
        "route_freq_clean": freq_clean.tolist(),
        "route_freq_triggered": freq_trig.tolist(),
        "coldest_expert": cold, "hottest_expert": hot,
        "total_params": int(total_params),
        "attacks": [],
    }

    # ---- attack pool ----
    ap_idx = rng.choice(len(tr), size=min(args.attack_pool_samples, len(tr)), replace=False).tolist()
    pool_texts = [tr[i]["text"] for i in ap_idx]
    pool_labels = [int(tr[i]["label"]) for i in ap_idx]

    def run_attack(name, target_expert, force_route, train_all=False, lr=None):
        lr = args.attack_lr if lr is None else lr
        log(f"\n== Attack [{name}]: inject into "
            f"{'ALL params' if train_all else f'expert e{target_expert}'} "
            f"(force_route={force_route}, lr={lr}) ==")
        model.load_state_dict(clean_state)
        model.to(device)
        mlp_local = get_sparse_mlp(model, args.target_layer)
        if train_all:
            for p in model.parameters():
                p.requires_grad_(True)
            n_trainable = total_params
        else:
            n_trainable = freeze_all_but_expert(model, mlp_local, target_expert, args.train_router_row)
        log(f"  trainable params = {n_trainable:,} ({100*n_trainable/total_params:.4f}% of model)")
        pt, py, npois = poison_rows(pool_texts, pool_labels, args.trigger,
                                    args.target_label, args.poison_rate, args.seed)
        prows = tokenize_rows(pt, py, tok, args.max_length)
        ploader = make_loader(prows, args.batch_size, pad_id, True)
        log(f"  poisoned {npois}/{len(prows)} pool examples")
        force = ForceRouter(mlp_local, target_expert) if force_route else None
        # For force-route, also pin routing during training so the expert learns the payload.
        if force is not None:
            force.enabled = True
        train(model, ploader, device, args.attack_steps, lr, name, log)
        if force is not None:
            force.enabled = False
        acc_after, _ = evaluate(model, test_loader, device)
        _, asr = evaluate(model, trig_loader, device, target_label=args.target_label, force=force)
        # also asr WITHOUT the trigger present (clean-label leakage) for sanity
        _, base_target_rate = evaluate(model, test_loader, device, target_label=args.target_label)
        if force is not None:
            force.remove()
        log(f"  RESULT [{name}]: clean_acc {clean_acc:.4f} -> {acc_after:.4f}  |  "
            f"ASR = {asr:.4f}  |  target-rate on clean test = {base_target_rate:.4f}")
        rec = {"name": name, "target_expert": target_expert, "force_route": force_route,
               "trainable_params": int(n_trainable),
               "trainable_fraction": n_trainable / total_params,
               "clean_acc_before": clean_acc, "clean_acc_after": acc_after,
               "asr": asr, "clean_target_rate": base_target_rate, "num_poisoned": npois}
        results["attacks"].append(rec)
        return rec

    # Core comparisons.
    run_attack("cold_natural", cold, force_route=False)
    run_attack("hot_natural", hot, force_route=False)
    run_attack("cold_forceroute", cold, force_route=True)
    # Standard FL-BadNet baseline: high ASR but modifies the whole model (footprint contrast).
    run_attack("full_badnet", -1, force_route=False, train_all=True, lr=args.lr)

    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log("\n===== SUMMARY =====")
    log(f"clean_acc={clean_acc:.4f}  coldest=e{cold}({freq_clean[cold]:.3f})  hottest=e{hot}({freq_clean[hot]:.3f})")
    for r in results["attacks"]:
        log(f"  {r['name']:16s} e{r['target_expert']} "
            f"trainable={100*r['trainable_fraction']:.3f}%  "
            f"clean {r['clean_acc_before']:.3f}->{r['clean_acc_after']:.3f}  "
            f"ASR={r['asr']:.3f}")
    log(f"saved -> {os.path.join(args.output, 'results.json')}")
    logf.close()


if __name__ == "__main__":
    main()
