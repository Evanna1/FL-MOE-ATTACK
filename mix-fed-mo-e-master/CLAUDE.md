# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MixFedMoE is a Federated Learning implementation with Mixture of Experts (MoE) models for text classification. It uses the Flower simulation framework and HuggingFace's Switch Transformers (google/switch-base-8). The project explores three client modes:
- **full**: Full-precision (FP32) training on all parameters
- **mix**: Mixed-precision - frozen BF16 for cold experts, FP32 for hot experts
- **drop**: Structural pruning - only train assigned K experts

## Common Commands

### Running Experiments

```bash
# Activate conda environment first
conda activate flwr

# Run single experiment
python -m mixfedmoe_fl.run_simulation --mode full --num_rounds 20 --num_clients 8 --K 4

# Run batch experiments (3 datasets × 3 modes = 9 experiments)
bash shells/run_9_experiments.sh --num_rounds 20 --num_clients 8 --K 4

# Run ablation experiments
bash shells/run_ablation_experiments.sh
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--mode` | Client mode: full, mix, drop |
| `--dataset_name` | Dataset: ag_news, imdb, 20news, sst2, yelp_polarity, emotion |
| `--num_clients` | Number of FL clients |
| `--num_rounds` | Federation rounds |
| `--K` | Experts assigned per client |
| `--alpha` | Deprecated compatibility option (ignored by IID partitioning) |
| `--assignment_policy` | Expert assignment: hot (activation-based) or random |
| `--test_samples` | Max test samples for evaluation (0 = full) |

### Running Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_strategy_unit.py -v

# Run tests with specific markers
pytest tests/ -v -m "not integration"

# Run integration tests (slow, requires model loading)
pytest tests/ -v -m integration
```

## Code Architecture

### Package Structure

```
mixfedmoe_fl/
├── config.py          # CLI argument parsing and MixFedMoEConfig dataclass
├── data.py            # Dataset loading, tokenization, Dirichlet partitioning
├── client.py          # MixFedMoEClient - local training with full/mix/drop modes
├── strategy.py        # MixFedMoEStrategy(FedAvg) - expert assignment & sparse aggregation
├── server.py          # Server setup, initial parameters, centralized evaluation
├── run_simulation.py # Main entry point using flwr.simulation.run_simulation
└── logging_utils.py   # CSV/JSONL metrics logging
```

### Key Design Patterns

1. **Stateless Clients**: Each `fit()` call recreates the model and loads parameters from server. This is required for Flower simulation.

2. **Expert Layer Discovery**: Uses regex patterns in `strategy.py` to identify MoE layers:
   - Expert params: `encoder.block.N.layer.M.mlp.experts.expert_N.weight`
   - Router weights: `encoder.block.N.layer.M.mlp.router.classifier.weight`

3. **Sparse Aggregation**: Strategy performs weighted FedAvg only on contributors (clients who trained specific experts), not all clients.

4. **Time Tracking**: Each client measures its own training time, strategy computes `round_time = max(client_times)` for wall-clock accuracy plots.

5. **Expert Coverage**: Every round guarantees full expert coverage (every expert assigned to at least one client). Round 1 uses cyclic assignment, subsequent rounds use activation-based with deterministic repair.

### Custom Model

`models/switch_transformers/SwitchTransformersForSequenceClassification` - HuggingFace's Switch Transformers adapted for text classification (replaces LM head with classifier).

## Environment Variables

```bash
# Offline mode for HuggingFace (use cached models/datasets)
export HF_HUB_OFFLINE="true"
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
```

## Development Notes

- Development machine: Windows 11, runs on Linux server
- Both machines have conda environment `flwr` that must be activated
- Local testing can use CPU only (set `--num_gpus_per_client 0`)
- Use small test samples (`--test_samples 1000`) for quick iteration
- Check `PLAN.md` and `AGENTS.md` for detailed implementation specifications
