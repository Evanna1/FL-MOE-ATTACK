# MixFedMoE Implementation Plan (Final Markdown)

## 1. Summary

Implement MixFedMoE with Flower simulation using a custom strategy that inherits `flwr.server.strategy.FedAvg`, supports `full/mix/drop` client modes, enforces per-round expert coverage by default, and tracks both step-based and wall-clock-style progress (`round_time = max(client local training time)`).  
All modes (`full/mix/drop`) use full FP32 parameter sync every round from server to client.  
Add `test_samples` to truncate evaluation for faster experiments.

## 2. Final Decisions

- Use legacy Flower strategy API centered on training rounds: `configure_fit`, `aggregate_fit`.
- Run with `from flwr.simulation import run_simulation`.
- Client is treated as stateless each `fit()`: recreate model and load server parameters.
- `full`, `mix`, and `drop` always sync all model parameters from server (FP32).
- `drop` mode still performs structural prune locally (assigned experts only, gate rebuilt/pruned) after loading full weights.
- Every selected client gets exactly `K` experts per sparse layer.
- Expert coverage is mandatory by default each round and not user-exposed as a toggle.
- Per-mode execution is one command per mode.
- Add `test_samples` as a dedicated argument for eval truncation.

## 3. Repo Additions (Proposed Files)

- `mixfedmoe_fl/config.py`: argument parsing and config dataclass.
- `mixfedmoe_fl/data.py`: dataset alias resolution, tokenization, Dirichlet partitioning, test split handling.
- `mixfedmoe_fl/moe_utils.py`: MoE layer discovery, mode transforms (`full/mix/drop`), hot-profile utilities.
- `mixfedmoe_fl/params.py`: parameter key ordering and full-state conversion helpers.
- `mixfedmoe_fl/client.py`: `NumPyClient` local training implementation.
- `mixfedmoe_fl/strategy.py`: `MixFedMoEStrategy(FedAvg)` implementation.
- `mixfedmoe_fl/server.py`: strategy wiring and centralized evaluation callback.
- `mixfedmoe_fl/run_simulation.py`: experiment entrypoint calling `run_simulation`.
- `mixfedmoe_fl/logging_utils.py`: CSV/JSONL metrics/artifact writing.

## 4. CLI and Config Specification

- Required/primary args: `mode`, `model_name_or_path`, `dataset_name`, `num_clients`, `num_rounds`, `K`, `fraction_fit` (`alpha` is retained only for CLI compatibility).
- Local training args: `local_epochs`, `learning_rate`, `weight_decay`, `train_batch_size`, `eval_batch_size`, `seed`.
- Runtime args: `num_cpus_per_client`, `num_gpus_per_client`, `output_dir`.
- New eval arg: `test_samples`.
- Evaluation policy: centralized server-side evaluation only; no federated client-side evaluation (`fraction_evaluate=0.0`).
- `test_samples` behavior: if `>0`, evaluate on `min(test_samples, len(test_split))`; if `<=0`, use full test split.
- Apply deterministic shuffle with seed before truncation for reproducibility.
- Remove/avoid any `mix_full_sync` or `enforce_expert_coverage_per_round` user flags.

## 5. Data Pipeline

- Keep dataset aliases from `proof_of_concept/poc_moe.py`.
- Partition the train split evenly and randomly using `IidPartitioner`.
- Ensure each client uses its own partition id.
- Build centralized global test set on server, truncated by `test_samples` rule.

## 6. Client Mode Semantics

- `full`: all parameters FP32 and trainable.
- `mix`: full model received each round; hot experts trainable FP32; cold experts frozen BF16; shared/router trainable FP32 as required by your policy.
- `drop`: full model is received each round, then client prunes structurally to assigned experts; gate is rebuilt to assigned expert rows; maintain local-to-global expert map.
- Each `fit()` returns metrics including expert activation profile and assigned/trained expert metadata.

## 7. Strategy Behavior

- Class: `MixFedMoEStrategy(flwr.server.strategy.FedAvg)`.
- `configure_fit`: sample clients, compute per-client exact-`K` assignments per sparse layer, guarantee full expert coverage, send per-client fit config plus full FP32 model payload.
- `aggregate_fit`: weighted FedAvg for shared/router params; sparse weighted expert aggregation using only contributors assigned to each expert.
- No “missing expert fallback”: if any expert has zero contributors in a round, fail fast.
- Set `accept_failures=False` to avoid silent coverage break.
- Federated client evaluation is disabled (`configure_evaluate` returns no client eval instructions).
- Centralized eval callback runs each round and records loss/accuracy against round and cumulative time.

## 8. Expert Assignment and Coverage Algorithm

- Let `E` be experts per sparse layer and `K` experts assigned per client.
- Feasibility constraint per layer: `num_selected_clients >= ceil(E / K)`.
- If infeasible, fail fast with explicit error.
- Round 1 assignment: deterministic cyclic coverage first, then fill remaining slots to exact `K`.
- Round >1 assignment: start from activation-based preferences, then deterministic repair to restore full coverage while preserving exact `K` per client.
- Deterministic tie-breaking by client id then expert id.

## 9. Time and Metrics Recording

- In each client `fit()`, measure `local_training_time = end - start`.
- In strategy `aggregate_fit()`, compute `round_time = max(client_times)`.
- Maintain `cumulative_time += round_time`.
- Record per round: `round`, `train_loss`, `eval_loss`, `eval_accuracy`, `round_time`, `cumulative_time`, `effective_test_samples`, `num_clients_sampled`.
- Save artifacts to CSV/JSONL for plotting:
- Step-to-accuracy.
- Time-to-accuracy.
- Optional train-loss vs step/time.

## 10. Memory and Stability Requirements

- Client must cleanup model/trainer/optimizer/dataloaders after `fit()`.
- Call `gc.collect()` and `torch.cuda.empty_cache()` after local training.
- Fail fast on key mismatch, shape mismatch, invalid assignment, and coverage infeasibility.
- Keep imports clean and hardcode Switch layer naming where it simplifies reliable implementation.

## 11. Testing Plan

- Unit: coverage assignment returns exact `K` and full layer-wise coverage.
- Unit: feasibility check rejects `num_selected_clients < ceil(E/K)`.
- Unit: sparse expert aggregation only uses valid contributors.
- Unit: strict full-state load passes on each round for `mix` and `drop`.
- Unit: round-time accounting equals max client local training time.
- Unit: `test_samples` truncation semantics (`>0` capped, `<=0` full).
- Integration smoke: short runs for `full`, `mix`, `drop` on small sample counts.
- Integration: verify mandatory coverage never violated in successful runs.
- Integration: verify logs contain `effective_test_samples` and cumulative-time curve fields.

## 12. Acceptance Criteria

- Uses `run_simulation` on single machine with Flower simulation backend.
- Custom strategy inherits `flwr.server.strategy.FedAvg`.
- Supports `full/mix/drop` with required server orchestration and profile upload.
- `full/mix/drop` perform full FP32 parameter sync per round with stateless clients.
- Every round satisfies exact-`K` per client and full expert coverage globally.
- Evaluation truncation via `test_samples` works and is logged.
- Produces reproducible step/time evaluation artifacts for comparison.

## 13. Assumptions

- Switch-base-8 checkpoint is accessible in runtime environment.
- Available clients/resources are sufficient to satisfy coverage feasibility.
- `K` is global (same for all clients per round) unless explicitly extended later.
- Optimizer remains AdamW as requested.

Implement in: config/data -> client -> strategy -> runner -> tests
