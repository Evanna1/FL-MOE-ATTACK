#!/usr/bin/env bash
set -euo pipefail

# Run MixFedMoE dataset x mode experiments:
#   datasets: 20news, agnews(ag_news), emotion
#   modes:    full, mix, drop, flex
#
# Usage:
#   bash shells/run_9_experiments.sh [extra run_simulation args...]
# Example:
#   bash shells/run_9_experiments.sh --num_rounds 20 --num_clients 8 --K 4
#
# Optional env:
#   CONDA_ENV=flwr
#   OUTPUT_ROOT=outputs/mixfedmoe_batch_YYYYmmdd_HHMMSS

CONDA_ENV="${CONDA_ENV:-flwr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mixfedmoe_batch_$(date +%Y%m%d_%H%M%S)}"

export HF_HUB_OFFLINE="true"
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

mkdir -p "${OUTPUT_ROOT}"

# user-facing name : config --dataset_name value
DATASET_PAIRS=(
  "20news:20news"
  "agnews:ag_news"
  "emotion:emotion"
)
DATASET_LOCAL_EPOCHS=(
  "20news:1"
  "ag_news:0.05"
  "emotion:1"
)
# MODES=("full" "mix" "drop" "flex")
MODES=("flex")
echo "[run] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[run] HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "[run] starting ${#DATASET_PAIRS[@]}x${#MODES[@]} experiments..."

for ds_pair in "${DATASET_PAIRS[@]}"; do
  ds_label="${ds_pair%%:*}"
  ds_arg="${ds_pair##*:}"
  ds_local_epochs="1"
  for kv in "${DATASET_LOCAL_EPOCHS[@]}"; do
    key="${kv%%:*}"
    value="${kv##*:}"
    if [[ "${key}" == "${ds_arg}" ]]; then
      ds_local_epochs="${value}"
      break
    fi
  done
  for mode in "${MODES[@]}"; do
    run_name="${ds_label}_${mode}"
    out_dir="${OUTPUT_ROOT}/${run_name}"
    mkdir -p "${out_dir}"

    echo
    echo "=================================================="
    echo "[run] ${run_name}"
    echo "[run] local_epochs=${ds_local_epochs}"
    if [[ "${mode}" == "flex" ]]; then
      echo "[run] enforcing flex constraints: --K=1 --assignment_policy=hot"
    fi
    echo "=================================================="

    if [[ "${mode}" == "flex" ]]; then
      python -m mixfedmoe_fl.run_simulation \
        "$@" \
        --dataset_name "${ds_arg}" \
        --mode "${mode}" \
        --K 1 \
        --assignment_policy "hot" \
        --local_epochs "${ds_local_epochs}" \
        --output_dir "${out_dir}"
    else
      python -m mixfedmoe_fl.run_simulation \
        "$@" \
        --dataset_name "${ds_arg}" \
        --mode "${mode}" \
        --local_epochs "${ds_local_epochs}" \
        --output_dir "${out_dir}"
    fi
  done
done

echo
echo "[run] all experiments completed."
