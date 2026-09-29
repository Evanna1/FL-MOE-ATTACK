#!/usr/bin/env bash
set -euo pipefail

# K ablation (Non-IID only):
#   - datasets x K in {2, 6}, mode from K_ABLATION_MODE, policy=hot, alpha=0.5
#   - flex: datasets x K in {1} only (flex constraint), policy=hot, alpha=0.5
#
# Datasets:
#   20news, agnews(ag_news), emotion
#
# Usage:
#   bash shells/run_k_ablation.sh [extra run_simulation args...]
# Example:
#   bash shells/run_k_ablation.sh --num_rounds 20 --num_clients 8
#
# Optional env:
#   CONDA_ENV=flwr
#   OUTPUT_ROOT=outputs/mixfedmoe_k_ablation_YYYYmmdd_HHMMSS
#   K_ABLATION_MODE=drop   # can be set to mix or flex

CONDA_ENV="${CONDA_ENV:-flwr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mixfedmoe_k_ablation_$(date +%Y%m%d_%H%M%S)}"
K_ABLATION_MODE="${K_ABLATION_MODE:-drop}"
ALPHA="${ALPHA:-0.5}"  # Non-IID

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
if [[ "${K_ABLATION_MODE}" == "flex" ]]; then
  K_VALUES=(1)
else
  K_VALUES=(2 6)
fi

echo "[run] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[run] HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "[run] ALPHA=${ALPHA} (Non-IID)"
TOTAL_RUNS=$(( ${#DATASET_PAIRS[@]} * ${#K_VALUES[@]} ))
echo "[run] starting K ablation experiments (total=${TOTAL_RUNS})..."

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

  for k in "${K_VALUES[@]}"; do
    run_name="${ds_label}_${K_ABLATION_MODE}_k${k}_hot_niid"
    out_dir="${OUTPUT_ROOT}/${run_name}"
    mkdir -p "${out_dir}"

    echo
    echo "=================================================="
    echo "[run] ${run_name}"
    echo "[run] local_epochs=${ds_local_epochs}, alpha=${ALPHA}"
    echo "=================================================="

    python -m mixfedmoe_fl.run_simulation \
      "$@" \
      --dataset_name "${ds_arg}" \
      --mode "${K_ABLATION_MODE}" \
      --K "${k}" \
      --assignment_policy "hot" \
      --alpha "${ALPHA}" \
      --local_epochs "${ds_local_epochs}" \
      --output_dir "${out_dir}"
  done
done

echo
echo "[run] all K ablation experiments completed."