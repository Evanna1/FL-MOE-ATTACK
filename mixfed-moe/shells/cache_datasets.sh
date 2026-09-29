#!/usr/bin/env bash
set -euo pipefail

# Cache required datasets into local Hugging Face cache before running on offline nodes.
# Usage:
#   bash shells/cache_datasets.sh
# Optional env:
#   CONDA_ENV=flwr
#   HF_HOME=/path/to/hf_home
#   HF_DATASETS_CACHE=/path/to/datasets_cache

CONDA_ENV="${CONDA_ENV:-flwr}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

echo "[cache] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[cache] Caching datasets: ag_news, SetFit/20_newsgroups, emotion"

python - <<'PY'
from datasets import load_dataset

datasets_to_cache = [
    ("ag_news", "ag_news"),
    ("20news", "SetFit/20_newsgroups"),
    ("emotion", "emotion"),
]

for alias, repo in datasets_to_cache:
    print(f"[cache] downloading {alias} ({repo}) ...", flush=True)
    ds = load_dataset(repo)
    split_info = ", ".join(f"{k}={len(v)}" for k, v in ds.items())
    print(f"[cache] done {alias}: {split_info}", flush=True)

print("[cache] all datasets cached successfully.", flush=True)
PY

