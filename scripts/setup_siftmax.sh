#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RABITQ_DIR="${REPO_ROOT}/third_party/RaBitQ-Library"
RABITQ_REPO=https://github.com/VectorDB-NTU/RaBitQ-Library.git
RABITQ_REV=31cfe0ca9108c2ec4315b9a90afcdd5d457297fd

if [[ ! -d "${RABITQ_DIR}/.git" ]]; then
  git clone "${RABITQ_REPO}" "${RABITQ_DIR}"
fi
git -C "${RABITQ_DIR}" checkout --detach "${RABITQ_REV}"
ln -sfn ../../third_party/RaBitQ-Library "${REPO_ROOT}/src/siftmax/RaBitQ-Library"
cd "${REPO_ROOT}/src/siftmax"
python setup_rabitq_official.py build_ext --inplace
