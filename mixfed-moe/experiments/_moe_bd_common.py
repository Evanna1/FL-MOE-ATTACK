"""Shared helpers for the centralized/FL cold-expert backdoor experiments.

Standalone (no flwr): only torch + transformers + the local switch-base-8 weights
and the repo's custom SwitchTransformers classification model are required.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.switch_transformers import SwitchTransformersForSequenceClassification  # noqa: E402


# ------------------------------- misc -------------------------------------- #
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------- data -------------------------------------- #
def load_agnews():
    from datasets import load_dataset
    ds = load_dataset("ag_news")
    n_labels = int(max(ds["train"]["label"])) + 1
    return ds["train"], ds["test"], n_labels


def build_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


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
    """BadNet: append trigger + flip label on a fraction of NON-target examples."""
    rng = np.random.default_rng(seed)
    eligible = [i for i, y in enumerate(labels) if int(y) != target_label]
    n_poison = int(len(texts) * poison_rate)
    chosen = set(rng.choice(eligible, size=min(n_poison, len(eligible)), replace=False).tolist())
    out_t, out_y, n = [], [], 0
    for i, (t, y) in enumerate(zip(texts, labels)):
        if i in chosen:
            out_t.append(t.rstrip() + " " + trigger); out_y.append(target_label); n += 1
        else:
            out_t.append(t); out_y.append(int(y))
    return out_t, out_y, n


# ------------------------------- model ------------------------------------- #
def load_classifier(model_path: str, n_labels: int, device, pad_id: int):
    model = SwitchTransformersForSequenceClassification.from_pretrained(
        model_path, num_labels=n_labels, ignore_mismatched_sizes=True,
        torch_dtype=torch.float32, tie_word_embeddings=False, local_files_only=True)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = pad_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id
    model.config.use_cache = False
    return model.to(device)


def get_sparse_mlp(model, layer_id: str):
    stack_name, idx = layer_id.split(".")
    ff = getattr(model, stack_name).block[int(idx)].layer[-1]
    if not bool(getattr(ff, "is_sparse", False)):
        raise ValueError(f"{layer_id} is not a sparse MoE layer.")
    return ff.mlp


def expert_module(mlp, expert_idx: int) -> nn.Module:
    return mlp.experts[f"expert_{expert_idx}"]


def num_experts(mlp) -> int:
    return len(mlp.experts)


def expert_param_names(model, layer_id: str, expert_idx: int) -> List[str]:
    prefix = f"{layer_id.split('.')[0]}.block.{layer_id.split('.')[1]}.layer"
    names = []
    for name, _ in model.named_parameters():
        if name.startswith(prefix) and f".experts.expert_{expert_idx}." in name:
            names.append(name)
    return names


# --------------------------- routing utilities ----------------------------- #
class RouterCounter:
    """Count top-1 routed expert over valid (non-pad) tokens at one layer."""

    def __init__(self, mlp, n_experts: int):
        self.counts = np.zeros(n_experts, dtype=np.int64)
        self.mask: Optional[torch.Tensor] = None
        self.handle = mlp.router.classifier.register_forward_hook(self._hook)

    def _hook(self, _m, _inp, out):
        logits = out.detach().float().reshape(-1, out.shape[-1])
        if self.mask is not None:
            logits = logits[self.mask.reshape(-1).bool()]
        if logits.numel() == 0:
            return
        self.counts += torch.bincount(logits.argmax(-1),
                                      minlength=self.counts.shape[0]).cpu().numpy()

    def remove(self):
        self.handle.remove()


class ForceRouter:
    """Pin the target expert to top-1 for all tokens when enabled (routing ceiling)."""

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
    return counter.counts / max(int(counter.counts.sum()), 1)


# ------------------------------ train / eval ------------------------------- #
def train_steps(model, loader, device, steps, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable params")
    opt = AdamW(params, lr=lr)
    model.train()
    done = 0
    while done < steps:
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device), use_cache=False, return_dict=True)
            out.loss.backward()
            opt.step()
            done += 1
            if done >= steps:
                break
    del opt


@torch.no_grad()
def evaluate(model, loader, device, target_label=None, force=None):
    model.eval()
    correct, total, tgt = 0, 0, 0
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
            tgt += int((pred == target_label).sum().item())
    if force is not None:
        force.enabled = False
    acc = correct / max(total, 1)
    asr = tgt / max(total, 1) if target_label is not None else None
    return acc, asr


def set_trainable(model, names: List[str]) -> int:
    wanted = set(names)
    n = 0
    for name, p in model.named_parameters():
        flag = name in wanted
        p.requires_grad_(flag)
        if flag:
            n += p.numel()
    return n
