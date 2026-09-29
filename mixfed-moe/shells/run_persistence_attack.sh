#!/usr/bin/env bash
set -euo pipefail

# Cold-expert backdoor PERSISTENCE experiment on the real flwr framework.
#
# Story: under memory-efficient FL-MoE (mix + hot policy + NO coverage guarantee),
# honest clients never train the cold experts. A malicious client claims the target
# cold expert (--inject_target_expert) and poisons it (BadNet). Because it is the
# sole aggregation contributor for that expert, its update is applied verbatim (no
# FedAvg dilution) and PERSISTS after the attacker leaves (--attack_end_round).
#
# Compares (per-round ASR in round_metrics.csv is the persistence curve):
#   cold_leave  : inject COLD expert,  attacker leaves at ATTACK_END
#   hot_leave   : inject a HOT expert (dilution baseline), attacker leaves
#   full_badnet : standard FL-BadNet (no localization), attacker leaves
#   cold_stay   : inject COLD expert,  attacker never leaves (upper bound)
#
# NOTE: coverage_guarantee is a separate TMC algorithm and is intentionally OFF here
# (--no_coverage_guarantee): our threat model is mainstream hot-only FL-MoE.
#
# Set COLD / HOT to the cold / hot expert indices at TARGET_LAYER (profile first).

CONDA_ENV="${CONDA_ENV:-flwr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/persistence_$(date +%Y%m%d_%H%M%S)}"
TARGET_LAYER="${TARGET_LAYER:-encoder.5}"
COLD="${COLD:-0}"
HOT="${HOT:-2}"
NUM_CLIENTS="${NUM_CLIENTS:-8}"
K="${K:-2}"
ROUNDS="${ROUNDS:-40}"
ATTACK_START="${ATTACK_START:-5}"
ATTACK_END="${ATTACK_END:-15}"
GPUS_PER_CLIENT="${GPUS_PER_CLIENT:-0.25}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

COMMON=(
  --mode mix
  --dataset_name ag_news
  --assignment_policy hot
  --no_coverage_guarantee
  --num_clients "${NUM_CLIENTS}"
  --K "${K}"
  --num_rounds "${ROUNDS}"
  --local_epochs 0.05
  --test_samples 2000
  --num_gpus_per_client "${GPUS_PER_CLIENT}"
  --attack_enabled
  --malicious_clients 0
  --poison_rate 0.5
  --target_label 0
  --trigger cf
  --attack_start_round "${ATTACK_START}"
  --lfe_target_layer "${TARGET_LAYER}"
)

run() {
  local name="$1"; shift
  local out="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out}"
  echo "==== ${name} -> ${out} ===="
  python -m mixfedmoe_fl.run_simulation "${COMMON[@]}" "$@" --output_dir "${out}"
}

mkdir -p "${OUTPUT_ROOT}"
run cold_leave   --inject_target_expert --lfe_target_expert "${COLD}" --attack_end_round "${ATTACK_END}"
run hot_leave    --inject_target_expert --lfe_target_expert "${HOT}"  --attack_end_round "${ATTACK_END}"
run full_badnet  --lfe_target_expert "${COLD}" --attack_end_round "${ATTACK_END}"
run cold_stay    --inject_target_expert --lfe_target_expert "${COLD}" --attack_end_round 0

echo "[done] persistence runs under ${OUTPUT_ROOT}"
echo "Plot ASR vs round from each */round_metrics.csv; the post-round-${ATTACK_END} tail is the persistence curve."
