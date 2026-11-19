# SiftMax Science reproduction

This repository is a focused reproduction package for the LoTTE Science
experiment in the SiftMax draft. It evaluates six methods with the following
query-time boundaries:

| Method | Excluded query preparation | First timed operation |
|---|---|---|
| PLAID | ColBERT/PLAID query encoding | PLAID `dense_search` |
| LEMUR | ColBERT query encoding and 2048D LEMUR query FDE | learned-FDE scan over W |
| DR + PLAID | MiniLM and ColBERT/PLAID query encoding | Faiss dense search |
| DR + Exact MaxSim | MiniLM and ColBERT query encoding | Faiss dense search |
| DR + SiftMax | MiniLM and ColBERT query encoding | Faiss dense search; RaBitQ query rotation remains timed search preprocessing |
| DR + LEMUR | MiniLM, ColBERT, and 2048D LEMUR query FDE generation | Faiss dense search |

The four cascaded methods share the same dense-index format and candidate protocol. All six methods use the unified CUDA-enabled `siftmax` environment, but CUDA is used only to generate FDE embeddings (and to train LEMUR when needed). ColBERT embedding generation may likewise use a GPU. Query-time retrieval and reranking for every method use the CPU.

## Fixed protocol

- LoTTE `science/dev/search`: 538 queries and 343,642 documents.
- ColBERTv2, 128 dimensions, at most 32 query vectors and 256 document vectors.
- `K=10` and 32 CPU threads. The dense stage searches all encoded queries in one Faiss batch; the four DR rerankers process the returned candidates per query. GPU use is restricted to offline embedding generation (LEMUR FDE and, optionally, ColBERT embeddings) and LEMUR training; all query search uses the CPU.
- Dense stage: normalized 384-dimensional `all-MiniLM-L6-v2` vectors and `faiss.IndexFlatIP`.
- Latency excludes all query embedding generation. Dense MiniLM, ColBERT, PLAID, and LEMUR query representations are loaded before timing; the timer covers retrieval/reranking through the returned top-10.
- Each latency is the wall time for the complete query workload divided by 538. QPS is `538 / total_wall_time_s` from the same timed run.
- PLAID and SiftMax use 4-bit residual codes by default. SiftMax uses seed 42 and sweeps the draft's candidate/alpha settings; RaBitQ bound parameters remain internal to the pinned upstream implementation.

The machine executing the draft had Xeon Gold 6526Y CPUs and 256 GB RAM. Record the retest machine; do not silently compare runs from materially different hardware.

## Layout

- `src/common`: the single shared implementation of dataset loading, metrics, dense retrieval, and ColBERT embedding generation.
- `src/plaid`: native PLAID, DR + PLAID, and DR + Exact MaxSim entry points.
- `src/siftmax`: SiftMax-specific Python/C++ code and persistent-index support.
- `src/lemur`: native LEMUR and DR + LEMUR entry points plus shared LEMUR-only integration code.
- `environments`: the unified `siftmax` Conda environment.
- `scripts`: one entry point per method plus the RaBitQ extension setup.
- `configs/science.yaml`: canonical experiment settings.
- `tools`: dataset checks and result summarization.
- `results_template/lotte_science_maxsim_recall_0.9_summary.csv`: generated final six-method table.

## Quick start

### Download the LoTTE dataset

This reproduction uses the official ColBERTv2 LoTTE release. From the
repository root, download and extract it as follows (the archive is about
3.3 GiB):

```bash
mkdir -p ../datasets
curl -fL --retry 3 \
  https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz \
  -o ../datasets/lotte.tar.gz

echo "37c0f39af23a6e3464f63395a4d04a22b91fe59c1aa64ea1773a8aff113c7ab5  ../datasets/lotte.tar.gz" \
  | sha256sum --check

tar -xzf ../datasets/lotte.tar.gz -C ../datasets
```

After extraction, the Science split must have this layout:

```text
../datasets/lotte/science/dev/
├── collection.tsv
├── questions.search.tsv
└── qas.search.jsonl
```

Verify that it is the fixed Science split used by this repository:

```bash
python tools/check_science_data.py ../datasets/lotte
```

The check should report 343,642 documents and 538 search queries. The runners
use `../datasets/lotte` by default when launched through the provided
scripts, so no environment variable is needed with the layout above. To keep
the dataset elsewhere, set `LOTTE_DATA_DIR` to the directory that contains
`science/dev`.

### Run the experiments

```bash
conda env create -f environments/siftmax.yml

conda activate siftmax
# Offline preprocessing: model encoding and the one-time exact full-corpus
# MaxSim top-10 reference use the selected GPU. Persisted tensors/indexes
# remain CPU-backed for timed retrieval.
CUDA_VISIBLE_DEVICES=0 bash scripts/generate_embeddings_gpu.sh
```

#### Download the prebuilt SiftMax centroid cache (recommended)

After generating the embeddings, download the prebuilt cache from the
[`v0.1.0` release](https://github.com/SimoneZeng/siftmax_repo/releases/tag/v0.1.0)
before starting the PLAID and exact-MaxSim experiments. The cache
contains the 16,384 base centroids, assignments for 51,727,492 document
tokens, and the refined `iso_squared` centroids used by the default SiftMax
configuration.

From the repository root, download both release assets, verify the archive,
and extract it under `indexes/`:

```bash
mkdir -p indexes

curl -fL --retry 3 \
  https://github.com/SimoneZeng/siftmax_repo/releases/download/v0.1.0/siftmax_lotte_science_b4_iso_squared_build_cache.tar.zst \
  -o indexes/siftmax_lotte_science_b4_iso_squared_build_cache.tar.zst

curl -fL --retry 3 \
  https://github.com/SimoneZeng/siftmax_repo/releases/download/v0.1.0/siftmax_lotte_science_b4_iso_squared_build_cache.tar.zst.sha256 \
  -o indexes/siftmax_lotte_science_b4_iso_squared_build_cache.tar.zst.sha256

(
  cd indexes
  sha256sum --check siftmax_lotte_science_b4_iso_squared_build_cache.tar.zst.sha256
  tar --zstd -xf siftmax_lotte_science_b4_iso_squared_build_cache.tar.zst
)
```

Once the archive is extracted, SiftMax can run immediately; native PLAID,
DR + PLAID, and DR + exact MaxSim do not need to run first:

```bash
bash scripts/setup_siftmax.sh
bash scripts/run_dr_siftmax.sh
```

The complete cache skips the time-consuming centroid clustering and assignment steps. SiftMax should report
`Loaded centroid artifact` when the cache is found. The runner still constructs
the RaBitQ packed data from the embeddings in memory; the release
cache does not contain the complete RaBitQ index.

Run the remainin baseline experiments separately when their results are needed:

```bash
bash scripts/run_plaid_native.sh
bash scripts/run_dr_plaid.sh
bash scripts/run_dr_exact_maxsim.sh

bash scripts/run_lemur_native.sh
bash scripts/run_dr_lemur.sh
```

To generate only the shared ColBERT artifacts, append `--skip-dense` to
`generate_embeddings_gpu.sh`; this includes document/query tensors and the
exact reference. Use `--skip-colbert` for only the MiniLM dense artifacts.
The ColBERT path creates and validates
`embedding_files/lotte_science/base_rankings/colbert-ir_colbertv2.0/topk_10.pkl`.
The top-10 reference is mandatory for every experiment run and is always
generated or validated together with the ColBERT artifacts. Use
`--skip-colbert` only when those artifacts and the reference already exist.

`generate_embeddings_gpu.sh` is the only ColBERT/MiniLM path that uses CUDA.
All Faiss, PLAID, SiftMax, and LEMUR index construction and query-time search
remain CPU-only. Native PLAID's official indexer performs its own internal
document encoding and cannot consume the shared padded ColBERT tensor cache;
therefore that end-to-end native PLAID index build remains entirely on CPU.

Native PLAID must run before DR + PLAID. With the complete release cache,
SiftMax loads its centroids and token assignments without reading the PLAID
index. If the cache is absent, it uses the configured PLAID index when
available and otherwise falls back to data K-means. For other datasets that do
not provide a centroid cache, we recommend running native PLAID first so that
SiftMax can build its own centroids from PLAID's centroids. This is faster than
falling back to data K-means and reduces experiment setup time. Native LEMUR
must build (or load) its learned checkpoint before DR + LEMUR. The dense index
is physically built once under `embedding_files/lotte_science/` and reused by
all four DR pipelines.

## RaBitQ-Library dependency

SiftMax compiles against the official
[`VectorDB-NTU/RaBitQ-Library`](https://github.com/VectorDB-NTU/RaBitQ-Library)
headers pinned at commit `31cfe0ca9108c2ec4315b9a90afcdd5d457297fd`.
The dependency is checked out under `third_party/RaBitQ-Library` and exposed to
the extension build through `src/siftmax/RaBitQ-Library`.

The setup script performs the pinned checkout and builds the extension:

```bash
conda activate siftmax
bash scripts/setup_siftmax.sh
```

To fetch and verify the corresponding version manually from the repository
root, run:

```bash
RABITQ_REV=31cfe0ca9108c2ec4315b9a90afcdd5d457297fd
git clone https://github.com/VectorDB-NTU/RaBitQ-Library.git \
  third_party/RaBitQ-Library
git -C third_party/RaBitQ-Library checkout --detach "${RABITQ_REV}"
test "$(git -C third_party/RaBitQ-Library rev-parse HEAD)" = "${RABITQ_REV}"
ln -sfn ../../third_party/RaBitQ-Library src/siftmax/RaBitQ-Library
```

Then compile the Python extension:

```bash
cd src/siftmax
python setup_rabitq_official.py build_ext --inplace
```

Do not replace the pinned commit with a branch tip: the C++ wrapper depends on
the API and packed-data layout of this exact revision.

Each runner records the complete query-time index size in its result CSV.
Generate the final table by selecting, for each method, the result whose
MaxSim Recall@10 is at least and closest to 0.9:

```bash
python tools/summarize_all_methods.py
```

The default output is
`results_template/lotte_science_maxsim_recall_0.9_summary.csv`. Use
`--dataset`, `--maxsim-recall`, or `--output` for another experiment.

## Result accounting

The exact full-corpus ColBERTv2 top-10 reference is stored at
`embedding_files/lotte_science/base_rankings/colbert-ir_colbertv2.0/topk_10.pkl`.
It must contain exactly 10 unique document IDs for all 538 queries. MaxSim
Recall@10 is top-10 set overlap divided by 10, MaxSim MRR@10 uses the rank of
the reference top-1 document, and MaxSim nDCG@10 treats the reference top-10 as
binary relevant documents. Relevance Recall/MRR/nDCG@10 uses
`qas.search.jsonl` and averages over queries with non-empty qrels. The
reference is evaluation data and is not counted as part of an index.

The summary tool selects, for every method, the row whose MaxSim Recall@10 is
at least and closest to the requested threshold. A tie at the same recall is
resolved in favor of higher QPS. The final table reports both MaxSim and
relevance Recall/MRR/nDCG, complete logical index size, average total retrieval
latency, available dense/rerank stage latencies, and total QPS. Latencies are
milliseconds per query; QPS and average latency use the same complete-workload
wall time.

Index size means query-time persistent files, excluding query embeddings,
reference rankings, logs, source data, and temporary build files. The logical
totals are:

- PLAID: the complete PLAID index directory;
- LEMUR: its learned checkpoint plus full-precision document IDs, embeddings,
  and masks needed for exact MaxSim;
- DR + PLAID: the dense Faiss cache plus the PLAID rerank index;
- DR + Exact MaxSim: the dense Faiss cache plus full-precision document IDs,
  embeddings, and masks;
- DR + SiftMax: the dense Faiss cache plus persistent SiftMax arrays and
  metadata;
- DR + LEMUR: the dense Faiss cache plus the LEMUR checkpoint and
  full-precision exact-MaxSim data.

The same physical dense cache is intentionally included in each DR method's
logical total. Every runner records its complete total in the raw CSV, and the
summary tool normalizes those values to `total_index_size_MB`; no separate
filesystem measurement is needed.

For an auditable reproduction, retain the raw method CSVs, top-10 rankings,
commands, hardware description, environment lock, git revisions, index-build
logs, and cache-reuse flags. Verify that selected rows use `K=10` and all 538
queries, that DR total latency equals dense plus rerank latency, and that clean
build measurements are distinguished from cached index loads.
