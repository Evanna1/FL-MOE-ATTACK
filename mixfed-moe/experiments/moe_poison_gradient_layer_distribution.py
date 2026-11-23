from __future__ import annotations

"""Where does the backdoor's poison gradient land? Per-layer / per-expert map.

We feed triggered, target-labelled samples through the model, backprop the
*classification* loss (the actual poisoning objective), and measure the gradient
L2 that lands on every expert and router. This reveals which layers/experts the
backdoor is really implemented in -- the gradient-based twin of the checkpoint
component-swap attribution.

Key facts about this OLMoE build (confirmed from olmoe_adapter regexes):
  * Experts are a *packed* tensor per layer: ``...layers.{L}.mlp.experts.gate_up_proj``
    and ``...down_proj`` with the expert axis at dim 0. A single expert's gradient
    is therefore a slice ``grad[e]`` of that packed tensor.
  * The router is ``...layers.{L}.mlp.gate.weight``.

Reported per expert/layer is the *energy share* (grad_sq / total_grad_sq), so the
shares sum to 1 and read directly as "fraction of the poison gradient". We also
report ``effective_num_layers = 1 / sum(layer_share**2)`` (and the expert twin):
an inverse-Herfindahl "how many layers/experts is the backdoor actually spread
across" -- 1.0 means one layer carries everything (Switch-like, persistent),
large means the backdoor is smeared across many layers (OLMoE's fragile regime).

Model/tokenizer/checkpoint are read locally; nothing is downloaded.

Example::

    python -m experiments.moe_poison_gradient_layer_distribution \
      --checkpoint_path outputs/.../checkpoints/round_0010.pt \
      --model_name_or_path model_ckpt/olmoe \
      --trigger "cf bb mn" --target_label 0 \
      --calibration_file mixfedmoe_fl/data/calibration.json \
      --calibration_samples 64 --batch_size 4 \
      --output_json outputs/poison_grad_distribution.json
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mixfedmoe_fl.attack import load_attack_checkpoint  # noqa: E402
from mixfedmoe_fl.data import build_tokenizer  # noqa: E402
from mixfedmoe_fl.igcg_trigger_optimization import (  # noqa: E402
    build_triggered_batch,
    load_calibration_texts,
)
from mixfedmoe_fl.model_factory import build_sequence_classification_model  # noqa: E402

# Expert packed tensors and router weight, expert axis at dim 0 for the former.
_EXPERT_RE = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
_ROUTER_RE = re.compile(r"\.layers\.(\d+)\.mlp\.gate\.weight$")


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_file():
        return path.resolve()
    candidate = REPO_ROOT / path
    if candidate.is_file():
        return candidate.resolve()
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
        missing = [n for n in incompatible.missing_keys if not n.endswith("position_ids")]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch: missing={missing[:10]}, "
                f"unexpected={incompatible.unexpected_keys[:10]}"
            )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.to(device).eval()
    return model, num_labels


def _resolve_trigger_ids(args: argparse.Namespace, tokenizer: Any) -> List[int]:
    if args.trigger_token_ids:
        ids = [int(x) for x in args.trigger_token_ids.split(",") if x.strip() != ""]
        if not ids:
            raise ValueError("--trigger_token_ids parsed to an empty list.")
        return ids
    if args.selected_triggers_path:
        payload = json.loads(_resolve_path(args.selected_triggers_path).read_text("utf-8"))
        # Tolerate a few shapes: {"best": {"trigger_ids": [...]}} / top-level / text.
        for getter in (
            lambda p: p["best"]["trigger_ids"],
            lambda p: p["trigger_ids"],
            lambda p: p["best"]["trigger"],
            lambda p: p["trigger"],
        ):
            try:
                value = getter(payload)
            except (KeyError, TypeError):
                continue
            if isinstance(value, list) and value and isinstance(value[0], int):
                return [int(x) for x in value]
            if isinstance(value, str) and value.strip():
                return list(tokenizer(value, add_special_tokens=False)["input_ids"])
        raise ValueError(
            "Could not find trigger ids/text in --selected_triggers_path; "
            "pass --trigger or --trigger_token_ids instead."
        )
    if args.trigger:
        ids = list(tokenizer(args.trigger, add_special_tokens=False)["input_ids"])
        if not ids:
            raise ValueError("--trigger tokenized to nothing.")
        return ids
    raise ValueError("Provide one of --trigger / --trigger_token_ids / --selected_triggers_path.")


def _plain_batch(tokenizer: Any, prompts: Sequence[str], max_length: int):
    enc = tokenizer(
        list(prompts),
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def _accumulate_poison_gradients(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    trigger_ids: Optional[Sequence[int]],
    target_label: int,
    *,
    batch_size: int,
    max_length: int,
    trigger_position: str,
    device: torch.device,
) -> None:
    """Zero then fill .grad on expert/router params with the (weighted) loss grad.

    ``trigger_ids=None`` runs the clean/no-trigger contrast on the same prompts.
    """
    model.zero_grad(set_to_none=True)
    total = len(prompts)
    if total == 0:
        raise ValueError("No prompts to run.")
    for start in range(0, total, batch_size):
        chunk = list(prompts[start : start + batch_size])
        if trigger_ids is None:
            input_ids, attention = _plain_batch(tokenizer, chunk, max_length)
        else:
            prepared = build_triggered_batch(
                tokenizer, chunk, trigger_ids, max_length, trigger_position, None
            )
            input_ids, attention = prepared.input_ids, prepared.attention_mask
        input_ids = input_ids.to(device)
        attention = attention.to(device)
        labels = torch.full((input_ids.shape[0],), int(target_label), device=device)
        out = model(input_ids=input_ids, attention_mask=attention, return_dict=True)
        loss = F.cross_entropy(out.logits.float(), labels)
        # Weight so the accumulated .grad equals the gradient of the mean loss.
        (loss * (len(chunk) / total)).backward()


def _collect_grad_energy(
    model: torch.nn.Module,
) -> Tuple[Dict[int, List[float]], Dict[int, float], int]:
    """Return per-(layer)->per-expert grad-sq, per-layer router grad-sq, num_experts."""
    expert_sq: Dict[int, List[float]] = {}
    router_sq: Dict[int, float] = {}
    num_experts = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        em = _EXPERT_RE.search(name)
        if em is not None:
            layer = int(em.group(1))
            grad = param.grad.detach().float()
            if grad.ndim < 1:
                raise RuntimeError(f"Unexpected scalar expert grad for {name}.")
            experts = grad.shape[0]
            num_experts = experts if num_experts == 0 else num_experts
            if experts != num_experts:
                raise RuntimeError(
                    f"Inconsistent expert count at {name}: {experts} vs {num_experts}."
                )
            per_e = grad.reshape(experts, -1).pow(2).sum(dim=1)  # [E]
            bucket = expert_sq.setdefault(layer, [0.0] * experts)
            for e in range(experts):
                bucket[e] += float(per_e[e])
            continue
        rm = _ROUTER_RE.search(name)
        if rm is not None:
            layer = int(rm.group(1))
            router_sq[layer] = router_sq.get(layer, 0.0) + float(
                param.grad.detach().float().pow(2).sum()
            )
    if not expert_sq:
        raise RuntimeError(
            "No expert gradients captured. Check that expert params require grad "
            "and that the parameter naming matches this OLMoE build."
        )
    return expert_sq, router_sq, num_experts


def _effective_count(shares: Sequence[float]) -> float:
    denom = sum(s * s for s in shares)
    return (1.0 / denom) if denom > 0 else 0.0


def _summarize(
    expert_sq: Dict[int, List[float]],
    router_sq: Dict[int, float],
    num_experts: int,
    top_experts_per_layer: int,
) -> Dict[str, Any]:
    total_expert_sq = sum(sum(v) for v in expert_sq.values())
    total_router_sq = sum(router_sq.values())
    if total_expert_sq <= 0:
        raise RuntimeError("Total expert gradient energy is zero.")

    layers_out: Dict[str, Any] = {}
    layer_shares: List[float] = []
    all_expert_shares: List[float] = []
    global_experts: List[Tuple[str, int, float, float]] = []

    for layer in sorted(expert_sq):
        bucket = expert_sq[layer]
        layer_sq = sum(bucket)
        layer_share = layer_sq / total_expert_sq
        layer_shares.append(layer_share)
        ranked = sorted(range(len(bucket)), key=lambda e: bucket[e], reverse=True)
        top = []
        for e in ranked[:top_experts_per_layer]:
            e_share_global = bucket[e] / total_expert_sq
            e_share_in_layer = (bucket[e] / layer_sq) if layer_sq > 0 else 0.0
            top.append(
                {
                    "expert": e,
                    "grad_l2": math.sqrt(bucket[e]),
                    "share_global": e_share_global,
                    "share_in_layer": e_share_in_layer,
                }
            )
            global_experts.append((f"decoder.{layer}", e, math.sqrt(bucket[e]), e_share_global))
        for e in range(len(bucket)):
            all_expert_shares.append(bucket[e] / total_expert_sq)
        layers_out[f"decoder.{layer}"] = {
            "expert_grad_l2": math.sqrt(layer_sq),
            "layer_share": layer_share,
            "router_grad_l2": math.sqrt(router_sq.get(layer, 0.0)),
            "router_share_of_all_routers": (
                router_sq.get(layer, 0.0) / total_router_sq if total_router_sq > 0 else 0.0
            ),
            "top_experts": top,
        }

    global_experts.sort(key=lambda t: t[2], reverse=True)
    sorted_layer_shares = sorted(layer_shares, reverse=True)
    return {
        "num_layers_with_experts": len(expert_sq),
        "num_experts_per_layer": num_experts,
        "total_expert_grad_l2": math.sqrt(total_expert_sq),
        "total_router_grad_l2": math.sqrt(total_router_sq),
        "router_vs_expert_energy_ratio": (
            total_router_sq / total_expert_sq if total_expert_sq > 0 else 0.0
        ),
        "concentration": {
            "effective_num_layers": _effective_count(layer_shares),
            "effective_num_experts": _effective_count(all_expert_shares),
            "top1_layer_share": sorted_layer_shares[0] if sorted_layer_shares else 0.0,
            "top3_layers_share": sum(sorted_layer_shares[:3]),
        },
        "global_top_experts": [
            {"layer": lid, "expert": e, "grad_l2": g, "share_global": s}
            for (lid, e, g, s) in global_experts[:20]
        ],
        "layers": layers_out,
    }


def _print_report(summary: Dict[str, Any], title: str) -> None:
    conc = summary["concentration"]
    print(f"\n==== {title} ====", flush=True)
    print(
        f"experts/layer={summary['num_experts_per_layer']}  "
        f"layers={summary['num_layers_with_experts']}  "
        f"router/expert energy ratio={summary['router_vs_expert_energy_ratio']:.3f}",
        flush=True,
    )
    print(
        f"BACKDOOR SPREAD -> effective layers={conc['effective_num_layers']:.2f} | "
        f"effective experts={conc['effective_num_experts']:.2f} | "
        f"top-1 layer={conc['top1_layer_share']*100:.1f}% | "
        f"top-3 layers={conc['top3_layers_share']*100:.1f}%",
        flush=True,
    )
    print("\nPer-layer expert-gradient share (desc):", flush=True)
    rows = sorted(
        summary["layers"].items(), key=lambda kv: kv[1]["layer_share"], reverse=True
    )
    for lid, info in rows:
        top = info["top_experts"][0] if info["top_experts"] else None
        top_str = (
            f"top expert #{top['expert']} ({top['share_in_layer']*100:.0f}% of layer)"
            if top
            else "-"
        )
        print(
            f"  {lid:>12}: {info['layer_share']*100:6.2f}%  "
            f"(router {info['router_share_of_all_routers']*100:5.1f}%)  {top_str}",
            flush=True,
        )
    print("\nGlobal top experts by poison-gradient L2:", flush=True)
    for item in summary["global_top_experts"][:10]:
        print(
            f"  {item['layer']:>12} expert #{item['expert']:<3} "
            f"share={item['share_global']*100:5.2f}%  l2={item['grad_l2']:.4e}",
            flush=True,
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name_or_path", default="model_ckpt/olmoe")
    p.add_argument("--checkpoint_path", default=None, help="Injected/global checkpoint to probe.")
    p.add_argument("--trigger", default=None, help="Trigger text (tokenized, no specials).")
    p.add_argument("--trigger_token_ids", default=None, help="Comma-separated trigger token ids.")
    p.add_argument("--selected_triggers_path", default=None, help="IGCG selected_triggers.json.")
    p.add_argument("--target_label", type=int, required=True)
    p.add_argument("--calibration_file", default="mixfedmoe_fl/data/calibration.json")
    p.add_argument("--calibration_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--trigger_position", choices=["append", "prepend"], default="append")
    p.add_argument("--top_experts_per_layer", type=int, default=5)
    p.add_argument(
        "--contrast",
        choices=["none", "clean"],
        default="none",
        help="'clean' also runs no-trigger baseline and reports poison-minus-clean.",
    )
    p.add_argument("--model_dtype", default="float32", help="float32 recommended for grad fidelity.")
    p.add_argument("--device", default="auto")
    p.add_argument("--output_json", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    tokenizer = build_tokenizer(args.model_name_or_path, args.max_length)
    model, num_labels = _build_model(
        args.model_name_or_path, args.checkpoint_path, tokenizer, args.model_dtype, device
    )
    if args.target_label < 0 or args.target_label >= num_labels:
        raise ValueError(f"--target_label={args.target_label} invalid for {num_labels} labels.")

    # Only experts + routers carry gradients: this is exactly the poison budget we
    # care about, and it keeps the backward light.
    tracked = 0
    for name, param in model.named_parameters():
        want = _EXPERT_RE.search(name) is not None or _ROUTER_RE.search(name) is not None
        param.requires_grad_(want)
        tracked += int(want)
    if tracked == 0:
        raise RuntimeError("No expert/router parameters matched; check the OLMoE naming.")

    prompts = load_calibration_texts(args.calibration_file, args.calibration_samples)
    if not prompts:
        raise ValueError("Calibration file produced no prompts.")
    trigger_ids = _resolve_trigger_ids(args, tokenizer)
    print(f"Trigger token ids ({len(trigger_ids)}): {trigger_ids}", flush=True)
    print(f"Poison samples: {len(prompts)} (target_label={args.target_label})", flush=True)

    _accumulate_poison_gradients(
        model, tokenizer, prompts, trigger_ids, args.target_label,
        batch_size=args.batch_size, max_length=args.max_length,
        trigger_position=args.trigger_position, device=device,
    )
    expert_sq, router_sq, num_experts = _collect_grad_energy(model)
    summary = _summarize(expert_sq, router_sq, num_experts, args.top_experts_per_layer)
    _print_report(summary, "POISON (trigger + target label)")

    output: Dict[str, Any] = {
        "config": {
            "model_name_or_path": args.model_name_or_path,
            "checkpoint_path": args.checkpoint_path,
            "target_label": args.target_label,
            "trigger_token_ids": trigger_ids,
            "calibration_samples": len(prompts),
            "trigger_position": args.trigger_position,
            "model_dtype": args.model_dtype,
        },
        "poison": summary,
    }

    if args.contrast == "clean":
        _accumulate_poison_gradients(
            model, tokenizer, prompts, None, args.target_label,
            batch_size=args.batch_size, max_length=args.max_length,
            trigger_position=args.trigger_position, device=device,
        )
        clean_expert_sq, clean_router_sq, _ = _collect_grad_energy(model)
        clean_summary = _summarize(
            clean_expert_sq, clean_router_sq, num_experts, args.top_experts_per_layer
        )
        _print_report(clean_summary, "CLEAN CONTRAST (no trigger, target label)")
        output["clean_contrast"] = clean_summary

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
