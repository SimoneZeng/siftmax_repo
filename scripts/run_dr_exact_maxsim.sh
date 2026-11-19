#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 NUMEXPR_NUM_THREADS=32
python src/plaid/dr_exact_maxsim.py \
  --dataset-id lotte/science --faiss-index-type flat \
  --doc-maxlen 256 --query-maxlen 32 --top-k-list 10 \
  --n-candidate-list 200 500 1000 1500 2000 3000 4000
