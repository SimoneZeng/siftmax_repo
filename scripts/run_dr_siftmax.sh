#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 NUMEXPR_NUM_THREADS=32
python src/siftmax/dr_siftmax.py \
  --dataset-id lotte/science --device cpu --faiss-index-type flat \
  --doc-maxlen 256 --query-maxlen 32 --rabitq-total-bits 4 \
  --centroid-objective iso_squared \
  --plaid-index-path plaid_exp/plaid_test_lotte_science/indexes/plaid_test_lotte_science_colbert-ir_colbertv2.0_docmax256_nbits4_kmeans4_full \
  --top-k-list 10 --n-candidate-list 200 500 1000 1500 2000 3000 4000 \
  --alpha-geometry-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
