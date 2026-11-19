#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 NUMEXPR_NUM_THREADS=32
python src/plaid/native_plaid.py \
  --dataset-id lotte/science --doc-maxlen 256 --query-maxlen 32 \
  --nbits 4 --kmeans-niters 4 \
  --top-k-list 10 --ncells-list 1 2 3 4 5 6 7 8
