from __future__ import annotations

import csv

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, SwitchTransformersConfig

from mixfedmoe_fl.run_trigger_optimization import main as trigger_main
from mixfedmoe_fl.trigger_optimization import (
    TargetExpert,
    add_perplexity_and_score,
    discover_sparse_routers,
    evaluate_candidate_routing,
    generate_trigger_candidates,
    pareto_frontier,
    rank_by_routing,
    validate_target,
)
from models.switch_transformers import SwitchTransformersForSequenceClassification


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        logits = torch.zeros(*input_ids.shape, self.vocab_size, device=input_ids.device)
        logits.scatter_(-1, input_ids.unsqueeze(-1), 2.0)
        return type("Output", (), {"logits": logits})()


def test_ten_sample_ten_candidate_frozen_model_smoke() -> None:
    tokenizer = AutoTokenizer.from_pretrained("model_ckpt/switch-base-8", use_fast=True)
    config = SwitchTransformersConfig(
        vocab_size=len(tokenizer),
        d_model=16,
        d_ff=32,
        d_kv=8,
        num_heads=2,
        num_layers=2,
        num_decoder_layers=2,
        num_experts=8,
        encoder_sparse_step=1,
        decoder_sparse_step=1,
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        router_jitter_noise=0.0,
    )
    model = SwitchTransformersForSequenceClassification(config)
    model.eval()
    model.requires_grad_(False)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    routers = discover_sparse_routers(model)
    target = TargetExpert(sorted(routers)[0], 0)
    validate_target(target, routers)
    candidates = generate_trigger_candidates(tokenizer, candidate_size=10, trigger_length=1, seed=42)
    texts = [f"This is clean profiling sentence number {index}." for index in range(10)]
    routing_rows = evaluate_candidate_routing(
        model=model,
        tokenizer=tokenizer,
        router=routers[target.layer_id],
        target=target,
        texts=texts,
        candidates=candidates,
        max_length=32,
        batch_size=5,
        device=torch.device("cpu"),
    )
    top_rows = rank_by_routing(routing_rows, top_k=3)
    ppl_model = TinyCausalLM(len(tokenizer))
    final_rows = add_perplexity_and_score(
        rows=top_rows,
        eval_texts=texts,
        ppl_model=ppl_model,
        ppl_tokenizer=tokenizer,
        ppl_batch_size=5,
        ppl_max_length=32,
        lambda_ppl=0.1,
        device=torch.device("cpu"),
    )
    assert len(routing_rows) == 10
    assert len(final_rows) == 3
    assert pareto_frontier(final_rows)
    assert all("routing_success_rate" in row and "perplexity" in row for row in final_rows)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_independent_cli_writes_only_trigger_output(tmp_path) -> None:
    tokenizer = AutoTokenizer.from_pretrained("model_ckpt/switch-base-8", use_fast=True)
    switch_dir = tmp_path / "tiny_switch"
    ppl_dir = tmp_path / "tiny_ppl"
    checkpoint_path = tmp_path / "clean_checkpoint.pt"
    dataset_path = tmp_path / "clean_data.csv"
    profiling_path = tmp_path / "lfe_top3.csv"
    output_dir = tmp_path / "trigger_outputs"
    old_output = tmp_path / "original_fl_output.txt"
    old_output.write_text("must remain unchanged", encoding="utf-8")

    switch_config = SwitchTransformersConfig(
        vocab_size=len(tokenizer),
        d_model=16,
        d_ff=32,
        d_kv=8,
        num_heads=2,
        num_layers=2,
        num_decoder_layers=2,
        num_experts=8,
        encoder_sparse_step=1,
        decoder_sparse_step=1,
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        router_jitter_noise=0.0,
    )
    switch_model = SwitchTransformersForSequenceClassification(switch_config)
    switch_model.save_pretrained(switch_dir)
    tokenizer.save_pretrained(switch_dir)
    torch.save({"round": 1, "state_dict": switch_model.state_dict()}, checkpoint_path)

    ppl_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_positions=64,
        n_ctx=64,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    ppl_model = AutoModelForCausalLM.from_config(ppl_config)
    ppl_model.save_pretrained(ppl_dir)
    tokenizer.save_pretrained(ppl_dir)

    with open(dataset_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label"])
        writer.writeheader()
        for index in range(20):
            writer.writerow({"text": f"Clean local sentence number {index}.", "label": index % 2})
    with open(profiling_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["layer_id", "expert_id"])
        writer.writeheader()
        writer.writerow({"layer_id": "encoder.0", "expert_id": 0})

    trigger_main(
        [
            "--model_name_or_path",
            str(switch_dir),
            "--dataset_name",
            "tiny",
            "--dataset_file",
            str(dataset_path),
            "--checkpoint_path",
            str(checkpoint_path),
            "--profiling_file",
            str(profiling_path),
            "--ppl_model_name_or_path",
            str(ppl_dir),
            "--candidate_size",
            "10",
            "--trigger_length",
            "1",
            "--optimization_samples",
            "10",
            "--eval_samples",
            "10",
            "--top_k",
            "3",
            "--max_length",
            "32",
            "--ppl_max_length",
            "32",
            "--batch_size",
            "5",
            "--ppl_batch_size",
            "5",
            "--device",
            "cpu",
            "--output_dir",
            str(output_dir),
        ]
    )
    target_dir = output_dir / "tiny" / "encoder_0_expert_0"
    for filename in [
        "run_config.json",
        "trigger_candidates.csv",
        "trigger_optimization_results.csv",
        "pareto_frontier.csv",
        "selected_triggers.json",
        "trigger_optimization.log",
    ]:
        assert (target_dir / filename).is_file()
    assert old_output.read_text(encoding="utf-8") == "must remain unchanged"
