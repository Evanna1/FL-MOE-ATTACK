#!/usr/bin/env bash
set -euo pipefail

# Random policy ablation (both IID and Non-IID):
#   - datasets x K=4 x mode=mix x policy=random x alpha in {0.5, 100}
#
# Datasets:
#   20news, agnews(ag_news), emotion
#
# Usage:
#   bash shells/run_random_policy_ablation.sh [extra run_simulation args...]
# Example:
#   bash shells/run_random_policy_ablation.sh --num_rounds 20 --num_clients 8
#
# Optional env:
#   CONDA_ENV=flwr
#   OUTPUT_ROOT=outputs/mixfedmoe_random_policy_ablation_YYYYmmdd_HHMMSS

CONDA_ENV="${CONDA_ENV:-flwr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mixfedmoe_random_policy_ablation_$(date +%Y%m%d_%H%M%S)}"

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
K=4
ALPHA_VALUES=(0.5 100)

echo "[run] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[run] HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
TOTAL_RUNS=$(( ${#DATASET_PAIRS[@]} * ${#ALPHA_VALUES[@]} ))
echo "[run] starting random policy ablation experiments (total=${TOTAL_RUNS})..."

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

  for alpha in "${ALPHA_VALUES[@]}"; do
    if [[ "${alpha}" == "100" ]]; then
      dist_label="iid"
    else
      dist_label="niid"
    fi
    run_name="${ds_label}_mix_k${K}_random_${dist_label}"
    out_dir="${OUTPUT_ROOT}/${run_name}"
    mkdir -p "${out_dir}"

    echo
    echo "=================================================="
    echo "[run] ${run_name}"
    echo "[run] local_epochs=${ds_local_epochs}, alpha=${alpha}"
    echo "=================================================="

    python -m mixfedmoe_fl.run_simulation \
      "$@" \
      --dataset_name "${ds_arg}" \
      --mode "mix" \
      --K "${K}" \
      --assignment_policy "random" \
      --alpha "${alpha}" \
      --local_epochs "${ds_local_epochs}" \
      --output_dir "${out_dir}"
  done
done

echo
echo "[run] all random policy ablation experiments completed."