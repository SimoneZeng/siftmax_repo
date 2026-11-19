#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Cached models are loaded from their local snapshot paths. Missing models are
# downloaded through the configured mirror.
: "${HF_ENDPOINT:=https://hf-mirror.com}"
export HF_ENDPOINT

# Select the physical GPU before launching, for example:
# CUDA_VISIBLE_DEVICES=1 bash scripts/generate_embeddings_gpu.sh
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

python src/common/generate_embeddings.py \
  --dataset-id lotte/science --device cuda:0 \
  --doc-maxlen 256 --query-maxlen 32 --bsize 64 \
  --reference-top-k 10 "$@"
