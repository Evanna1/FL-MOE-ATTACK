from __future__ import annotations

"""Routing-gradient cosine: which OLMoE experts share an attackable direction.

For a probe input we measure, for each candidate expert ``e`` at layer ``L``, the
gradient of the "push routing toward ``e``" objective with respect to the trigger
*slot* embeddings::

    g_{L,e} = d( -log P(route to e @ L) ) / d(slot_embeddings)

``g`` is the input perturbation direction that most increases ``e``'s routing
probability.  Two experts whose gradients are aligned (high ``cos(g_i, g_j)``)
can be pushed by the *same* input change, so a single trigger is likely to
co-activate both.  Unlike router-weight cosine, every expert's gradient is taken
w.r.t. the *same* slot embeddings, so cross-layer pairs are directly comparable.

Default probe = random trigger tokens, repeated ``--init_repeats`` (K) times over
clean calibration contexts.  We report the mean cosine (and its std across
samples) so stable, context-independent co-activation directions stand out.

Model/tokenizer are read from the local server checkpoint (default
``model_ckpt/olmoe``); nothing is downloaded.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mixfedmoe_fl.attack import load_attack_checkpoint  # noqa: E402
from mixfedmoe_fl.data import build_tokenizer  # noqa: E402
from mixfedmoe_fl.igcg_trigger_optimization import (  # noqa: E402
    build_triggered_batch,
    compute_routing_objective,
    resolve_moe_layer,
    get_router,
    _stack_trigger_mask,
)
from mixfedmoe_fl.model_factory import build_sequence_classification_model  # noqa: E402
from mixfedmoe_fl.olmoe_adapter import OlmoeAdapter  # noqa: E402


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_file():
        return path.resolve()
    repository_relative = REPO_ROOT / path
    if repository_relative.is_file():
        return repository_relative.resolve()
    raise FileNotFoundError(f"File does not exist: {raw}")


def _build_model(
    model_name_or_path: str,
    checkpoint_path: Optional[str],
    tokenizer: Any,
    model_dtype: str,
    device: torch.device,
) -> torch.nn.Module:
    num_labels = 2
    state: Optional[Dict[str, torch.Tensor]] = None
    if checkpoint_path:
        state, metadata = load_attack_checkpoint(str(_resolve_path(checkpoint_path)))
        classifier = state.get("classification_head.out_proj.weight")
        if isinstance(classifier, torch.Tensor):
            num_labels = int(classifier.shape[0])
        print(f"Loaded checkpoint metadata: {metadata}", flush=True)
    model = build_sequence_classification_model(
        model_name_or_path,
        num_labels=num_labels,
        id2label={index: str(index) for index in range(num_labels)},
        label2id={str(index): index for index in range(num_labels)},
        tokenizer=tokenizer,
        model_backend="auto",
        model_dtype=model_dtype,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    if state is not None:
        incompatible = model.load_state_dict(state, strict=False)
        missing = [name for name in incompatible.missing_keys if not name.endswith("position_ids")]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch: missing={missing[:10]}, "
                f"unexpected={incompatible.unexpected_keys[:10]}"
            )
    model.config.use_cache = False
    # Reusing one forward graph across many retain_graph=True backward passes is
    # incompatible with activation checkpointing's recompute-on-backward.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval().requires_grad_(False).to(device)
    return model


def _parse_expert_sets(tokens: Optional[Sequence[str]]) -> Dict[str, List[int]]:
    if not tokens:
        return {}
    mapping: Dict[str, List[int]] = {}
    for token in tokens:
        if ":" not in token:
            raise ValueError(f"Malformed --expert_sets token (need 'layer:ids'): {token!r}")
        layer_id, raw_ids = token.split(":", 1)
        ids = [int(part) for part in raw_ids.split(",") if part.strip() != ""]
        if not ids:
            raise ValueError(f"No expert ids given for layer {layer_id!r} in --expert_sets.")
        mapping[layer_id.strip()] = ids
    return mapping


def _resolve_subset(
    layer_id: str,
    num_experts: int,
    default_experts: Optional[Sequence[int]],
    per_layer_sets: Dict[str, List[int]],
) -> Optional[List[int]]:
    chosen = per_layer_sets.get(layer_id)
    if chosen is None:
        chosen = list(default_experts) if default_experts is not None else None
    if chosen is None:
        return None
    unique = sorted(dict.fromkeys(int(e) for e in chosen))
    out_of_range = [e for e in unique if e < 0 or e >= num_experts]
    if out_of_range:
        raise ValueError(
            f"Layer {layer_id} has {num_experts} experts; subset ids out of range: {out_of_range}."
        )
    return unique


def _sampleable_token_ids(tokenizer: Any) -> List[int]:
    """Vocabulary ids usable as random trigger tokens (special ids excluded)."""
    vocab_size = int(getattr(tokenizer, "vocab_size", 0)) or len(tokenizer)
    special = set(tokenizer.all_special_ids or [])
    return [tid for tid in range(vocab_size) if tid not in special]


class _GradientProbe:
    """Hooks OLMoE routers and the input embedding to read slot gradients."""

    def __init__(
        self,
        model: nn.Module,
        layer_ids: Sequence[str],
        device: torch.device,
    ) -> None:
        self.model = model
        self.device = device
        self.handles = {layer_id: resolve_moe_layer(model, layer_id) for layer_id in layer_ids}
        self.routers = {layer_id: get_router(handle) for layer_id, handle in self.handles.items()}
        for handle in self.handles.values():
            if handle.backend != "olmoe":
                raise ValueError("This probe only supports OLMoE routers.")
        embedding = model.get_input_embeddings()
        if not isinstance(embedding, nn.Embedding):
            raise RuntimeError("OLMoE input embedding module was not found.")
        self.embedding = embedding

    def forward(self, batch) -> tuple:
        """Run one forward pass, returning per-layer router logits and the leaf."""
        router_outputs: Dict[str, List[torch.Tensor]] = {lid: [] for lid in self.routers}
        embedding_leaf: List[torch.Tensor] = []

        def make_router_hook(layer_id: str):
            def router_hook(module: nn.Module, args: Any, _output: Any) -> None:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise RuntimeError(f"OLMoE router hook at '{layer_id}' got no hidden states.")
                router_outputs[layer_id].append(
                    F.linear(args[0].float(), module.weight.float())
                )

            return router_hook

        def embedding_hook(_module: nn.Module, _args: Any, output: torch.Tensor) -> torch.Tensor:
            leaf = output.detach().requires_grad_(True)
            leaf.retain_grad()
            embedding_leaf.append(leaf)
            return leaf

        hooks = [
            router.register_forward_hook(make_router_hook(layer_id))
            for layer_id, router in self.routers.items()
        ]
        hooks.append(self.embedding.register_forward_hook(embedding_hook))
        try:
            self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                output_router_logits=False,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for hook in hooks:
                hook.remove()
        logits = {}
        for layer_id, outputs in router_outputs.items():
            if len(outputs) != 1:
                raise RuntimeError(f"Expected one router call at '{layer_id}', saw {len(outputs)}.")
            logits[layer_id] = outputs[0]
        if not embedding_leaf:
            raise RuntimeError("The embedding hook did not run.")
        return logits, embedding_leaf[0]

    def expert_loss(self, layer_id: str, logits: torch.Tensor, mask: torch.Tensor, expert: int):
        handle = self.handles[layer_id]
        aligned = _stack_trigger_mask(mask, handle.stack_name, handle.backend).to(logits.device)
        loss, _prob, _hit, _margin = compute_routing_objective(
            logits,
            aligned,
            expert,
            routing_top_k=int(handle.router_top_k),
        )
        return loss


def _collect_slot_gradients(
    probe: _GradientProbe,
    batch,
    candidates: Sequence[tuple],
) -> torch.Tensor:
    """Return ``[num_candidates, batch, slot_dim]`` per-example slot gradients."""
    logits, leaf = probe.forward(batch)
    mask = batch.trigger_mask.bool()
    batch_size = mask.shape[0]
    slots_per_row = int(mask[0].sum().item())
    embed_dim = leaf.shape[-1]
    grads = torch.zeros(len(candidates), batch_size, slots_per_row * embed_dim, dtype=torch.float32)
    for ci, (layer_id, expert) in enumerate(candidates):
        if leaf.grad is not None:
            leaf.grad = None
        loss = probe.expert_loss(layer_id, logits[layer_id], batch.trigger_mask, expert)
        loss.backward(retain_graph=True)
        grad = leaf.grad.detach()
        for row in range(batch_size):
            slot_grad = grad[row][mask[row]].reshape(-1)
            grads[ci, row] = slot_grad.to(dtype=torch.float32, device="cpu")
    del logits, leaf
    return grads


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.init_repeats <= 0 or args.trigger_length <= 0:
        raise ValueError("init_repeats and trigger_length must be > 0.")
    if args.top_pairs <= 0:
        raise ValueError("top_pairs must be > 0.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    per_layer_sets = _parse_expert_sets(args.expert_sets)
    texts = _load_calibration(args.calibration_path, args.num_contexts)

    tokenizer = build_tokenizer(args.model_name_or_path, max_length=args.max_length)
    model = _build_model(
        args.model_name_or_path, args.checkpoint_path, tokenizer, args.model_dtype, device
    )
    if not OlmoeAdapter.supports(model):
        raise ValueError("The loaded model is not an OLMoE backbone.")

    descriptors = OlmoeAdapter.collect_layers(model)
    if args.target_layer is not None:
        descriptors = [d for d in descriptors if d.layer_id == args.target_layer]
        if not descriptors:
            raise ValueError(f"Unknown target layer '{args.target_layer}'.")
    unknown = sorted(set(per_layer_sets) - {d.layer_id for d in descriptors})
    if unknown:
        raise ValueError(f"--expert_sets references unknown layers: {unknown}.")

    # Flat candidate table across the processed layers (cross-layer comparable).
    descriptor_by_layer = {d.layer_id: d for d in descriptors}
    candidates: List[tuple] = []
    if per_layer_sets:
        # Strict mode: measure ONLY the experts explicitly listed in
        # --expert_sets.  Layers and experts not named are excluded entirely.
        for layer_id, raw_ids in per_layer_sets.items():
            descriptor = descriptor_by_layer[layer_id]
            subset = _resolve_subset(layer_id, descriptor.num_experts, None, {layer_id: raw_ids})
            for expert in subset or []:
                candidates.append((layer_id, int(expert)))
    else:
        for descriptor in descriptors:
            subset = _resolve_subset(
                descriptor.layer_id, descriptor.num_experts, args.experts, {}
            )
            members = range(descriptor.num_experts) if subset is None else subset
            for expert in members:
                candidates.append((descriptor.layer_id, int(expert)))
    if len(candidates) < 2:
        raise ValueError("Need at least two candidate experts to compare.")
    layer_ids = sorted({layer_id for layer_id, _ in candidates})
    probe = _GradientProbe(model, layer_ids, device)
    token_pool = _sampleable_token_ids(tokenizer)
    scope = "expert_sets (strict)" if per_layer_sets else "experts/all"
    print(
        f"Probing {len(candidates)} experts across {len(layer_ids)} layers "
        f"[scope={scope}], "
        f"K={args.init_repeats} random inits x {len(texts)} contexts "
        f"(batch={args.batch_size}, trigger_len={args.trigger_length}).",
        flush=True,
    )

    num_candidates = len(candidates)
    cos_sum = torch.zeros(num_candidates, num_candidates, dtype=torch.float64)
    cos_sq = torch.zeros(num_candidates, num_candidates, dtype=torch.float64)
    sample_count = 0
    for repeat in range(args.init_repeats):
        trigger_ids = [rng.choice(token_pool) for _ in range(args.trigger_length)]
        for start in range(0, len(texts), args.batch_size):
            prompts = texts[start : start + args.batch_size]
            prepared = build_triggered_batch(
                tokenizer, prompts, trigger_ids, args.max_length,
                args.trigger_position, args.fixed_position,
            )
            prepared = type(prepared)(
                prepared.input_ids.to(device),
                prepared.attention_mask.to(device),
                prepared.trigger_mask.to(device),
            )
            grads = _collect_slot_gradients(probe, prepared, candidates)
            for row in range(grads.shape[1]):
                vectors = F.normalize(grads[:, row, :], p=2.0, dim=-1).double()
                cosine = vectors @ vectors.t()
                cos_sum += cosine
                cos_sq += cosine * cosine
                sample_count += 1
    if sample_count == 0:
        raise RuntimeError("No probe samples were collected.")

    mean = (cos_sum / sample_count).float()
    variance = (cos_sq / sample_count - (cos_sum / sample_count) ** 2).clamp_min(0.0)
    std = variance.sqrt().float()

    report = _build_report(candidates, mean, std, args.top_pairs)
    report.update(
        {
            "model_name_or_path": args.model_name_or_path,
            "checkpoint_path": args.checkpoint_path,
            "metric": "routing_gradient_cosine",
            "trigger_init": "random",
            "init_repeats": args.init_repeats,
            "trigger_length": args.trigger_length,
            "num_contexts": len(texts),
            "samples": sample_count,
        }
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "routing_gradient_cosine.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote gradient-similarity report to {report_path}", flush=True)
    if args.save_matrices:
        matrix_path = output_dir / "routing_gradient_cosine_matrices.npz"
        np.savez_compressed(
            matrix_path,
            candidates=np.array([f"{lid}:{e}" for lid, e in candidates]),
            mean=mean.numpy(),
            std=std.numpy(),
        )
        print(f"Wrote full cosine matrices to {matrix_path}", flush=True)


def _build_report(
    candidates: Sequence[tuple],
    mean: torch.Tensor,
    std: torch.Tensor,
    top_pairs: int,
) -> Dict[str, Any]:
    def label(idx: int) -> Dict[str, Any]:
        layer_id, expert = candidates[idx]
        return {"layer_id": layer_id, "expert": expert}

    n = len(candidates)
    triu = torch.triu_indices(n, n, offset=1)
    values = mean[triu[0], triu[1]]
    same_layer = torch.tensor(
        [candidates[int(i)][0] == candidates[int(j)][0] for i, j in zip(triu[0], triu[1])]
    )

    def rank(indices: torch.Tensor) -> List[Dict[str, Any]]:
        if indices.numel() == 0:
            return []
        sub = values[indices]
        count = min(int(top_pairs), sub.numel())
        order = torch.topk(sub, k=count).indices
        out = []
        for sel in order.tolist():
            pair = int(indices[sel].item())
            i, j = int(triu[0, pair].item()), int(triu[1, pair].item())
            out.append(
                {
                    "a": label(i),
                    "b": label(j),
                    "mean_cosine": float(mean[i, j].item()),
                    "std_cosine": float(std[i, j].item()),
                }
            )
        return out

    all_idx = torch.arange(values.numel())
    cross_idx = (~same_layer).nonzero(as_tuple=False).flatten()
    within_idx = same_layer.nonzero(as_tuple=False).flatten()

    # Nearest neighbor per expert (over all candidates, self excluded).
    masked = mean.clone()
    masked.fill_diagonal_(float("-inf"))
    neighbors = []
    for i in range(n):
        best = int(torch.argmax(masked[i]).item())
        neighbors.append(
            {
                "expert": label(i),
                "nearest": label(best),
                "mean_cosine": float(mean[i, best].item()),
                "std_cosine": float(std[i, best].item()),
                "nearest_is_same_layer": candidates[i][0] == candidates[best][0],
            }
        )

    return {
        "num_candidates": n,
        "top_pairs_overall": rank(all_idx),
        "top_pairs_within_layer": rank(within_idx),
        "top_pairs_cross_layer": rank(cross_idx),
        "nearest_neighbors": neighbors,
    }


def _load_calibration(path: str, limit: int) -> List[str]:
    calibration_path = _resolve_path(path)
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Calibration file must contain a JSON list.")
    texts = [row.get("text") for row in payload if isinstance(row, dict)]
    texts = [str(t) for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        raise ValueError("Calibration file has no usable 'text' rows.")
    return texts[:limit]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="model_ckpt/olmoe")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument(
        "--model_dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto"
    )
    parser.add_argument(
        "--calibration_path",
        default="data/calibration.json",
        help="JSON list of {'text': ...} contexts the trigger slots live in.",
    )
    parser.add_argument("--num_contexts", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--trigger_length", type=int, default=5)
    parser.add_argument(
        "--init_repeats",
        type=int,
        default=8,
        help="K: number of independent random trigger initializations to average.",
    )
    parser.add_argument(
        "--trigger_position", choices=["append", "prepend", "fixed_position"], default="append"
    )
    parser.add_argument("--fixed_position", type=int, default=-1)
    parser.add_argument("--target_layer", default=None)
    parser.add_argument(
        "--experts",
        type=int,
        nargs="+",
        default=None,
        help="Expert ids applied to every processed layer. Ignored when --expert_sets is given.",
    )
    parser.add_argument(
        "--expert_sets",
        nargs="+",
        default=None,
        help=(
            "Per-layer expert sets as 'layer_id:idx,idx,...'. When given, this is "
            "the ONLY measurement scope: layers/experts not listed are excluded."
        ),
    )
    parser.add_argument("--top_pairs", type=int, default=20)
    parser.add_argument("--output_dir", default="outputs/expert_routing_similarity")
    parser.add_argument("--save_matrices", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main()
