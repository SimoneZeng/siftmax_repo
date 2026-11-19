"""DR + PLAID retrieval pipeline.

Phase 1 uses the dense retriever to retrieve ``n_candidate`` documents for
each query. Phase 2 filters these candidates by centroid score, computes
reconstructed MaxSim over the remaining candidates, and returns the ranking.
"""

import json
import os
n_thread = 32
os.environ["OMP_NUM_THREADS"] = f"{n_thread}"
os.environ["MKL_NUM_THREADS"] = f"{n_thread}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_thread}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_thread}"

os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import glob
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm
import torch


torch.set_num_threads(n_thread)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from colbert import Searcher
from colbert.infra import ColBERTConfig, RunConfig, Run

from common.dense_retrieval import DENSE_MODEL_NAME, DenseRetriever
from common.colbert_embeddings import load_cached_query_embeddings
from common.data_load import load_dataset_by_name
from common.reference_rankings import load_reference_rankings, reference_rankings_path
from common.utils import (
    BASE_DOC_VEC_CNT,
    TOP_K_LIST,
    compute_retrieval_metrics,
    configure_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two phase retrieval: Dense candidates + official PLAID reconstructed MaxSim"
    )
    parser.add_argument("--dataset-id", type=str, default="lotte/science")
    parser.add_argument("--checkpoint", type=str, default="colbert-ir/colbertv2.0")
    parser.add_argument("--experiment-dir", type=str, default="./plaid_exp")
    parser.add_argument("--experiment", type=str, default="two_phase_plaid")
    parser.add_argument("--device", type=str, choices=["cpu"], default="cpu")
    parser.add_argument("--doc-maxlen", type=int, default=BASE_DOC_VEC_CNT)
    parser.add_argument("--query-maxlen", type=int, default=32)
    parser.add_argument("--nbits", type=int, default=4) # [NOTE] residual quantization bits, 2/4/8
    parser.add_argument("--kmeans-niters", type=int, default=4)
    parser.add_argument("--index-bsize", type=int, default=64)
    parser.add_argument("--top-k-list", type=int, nargs="+", default=TOP_K_LIST)
    parser.add_argument("--n-query-limited-load", type=int, default=None) # None, 50
    parser.add_argument("--n-doc-limited-load", type=int, default=None) # None, 10000
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--bsize", type=int, default=16)

    parser.add_argument("--dense-model-name", type=str, default=DENSE_MODEL_NAME)
    parser.add_argument("--faiss-index-type", type=str, default="flat", choices=["flat"])
    parser.add_argument("--n-candidate-list", type=int, nargs="+", default=[200, 500, 1000, 1500, 2000, 3000, 4000])
    parser.add_argument(
        "--plaid-index-path",
        type=str,
        default=None,
        help=(
            "Path to an existing complete PLAID index. If omitted, compatible indexes "
            "under <experiment-dir>/*/indexes are discovered automatically."
        ),
    )
    return parser.parse_args()


def get_dir_size(path: str) -> int:
    total_size = 0
    if not os.path.exists(path):
        return total_size
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size


def is_complete_plaid_index(index_path: str) -> bool:
    """Return whether all artifacts needed by ``Searcher`` are present."""
    metadata_path = os.path.join(index_path, "metadata.json")
    codec_paths = [
        os.path.join(index_path, "centroids.pt"),
        os.path.join(index_path, "avg_residual.pt"),
        os.path.join(index_path, "buckets.pt"),
    ]
    ivf_exists = any(
        os.path.exists(os.path.join(index_path, filename))
        for filename in ("ivf.pid.pt", "ivf.pt")
    )
    if not os.path.isfile(metadata_path) or not ivf_exists or not all(map(os.path.isfile, codec_paths)):
        return False

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            num_chunks = int(json.load(f)["num_chunks"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False

    chunk_suffixes = ("codes.pt", "residuals.pt", "metadata.json")
    for chunk_idx in range(num_chunks):
        if not all(
            os.path.isfile(os.path.join(index_path, f"{chunk_idx}.{suffix}"))
            for suffix in chunk_suffixes
        ):
            return False
        if not os.path.isfile(os.path.join(index_path, f"doclens.{chunk_idx}.json")):
            return False

    return True


def find_complete_plaid_index(
    experiment_dir: str,
    experiment_name: str,
    index_name: str,
    explicit_index_path: Optional[str] = None,
) -> Optional[str]:
    """Find a complete compatible index, including indexes made by another experiment."""
    if explicit_index_path is not None:
        explicit_index_path = os.path.abspath(os.path.expanduser(explicit_index_path))
        if not is_complete_plaid_index(explicit_index_path):
            raise FileNotFoundError(
                f"--plaid-index-path is not a complete PLAID index: {explicit_index_path}"
            )
        return explicit_index_path

    expected_path = os.path.join(experiment_dir, experiment_name, "indexes", index_name)
    if is_complete_plaid_index(expected_path):
        return os.path.abspath(expected_path)

    # Index names produced by plaid_test.py and this script have different
    # experiment prefixes, but the remainder uniquely describes the dataset,
    # checkpoint and indexing parameters.
    prefix = f"{experiment_name}_"
    compatibility_suffix = index_name[len(prefix):] if index_name.startswith(prefix) else index_name
    search_pattern = os.path.join(experiment_dir, "*", "indexes", f"*{compatibility_suffix}")
    compatible_paths = sorted(
        os.path.abspath(path)
        for path in glob.glob(search_pattern)
        if is_complete_plaid_index(path)
    )
    return compatible_paths[0] if compatible_paths else None


def make_index_name(args: argparse.Namespace) -> str:
    dataset_stub = args.dataset_id.replace("/", "_")
    checkpoint_stub = args.checkpoint.replace("/", "_")
    doc_limit = "full" if args.n_doc_limited_load is None else f"docs{args.n_doc_limited_load}"
    return (
        f"{args.experiment}_{dataset_stub}_{checkpoint_stub}"
        f"_docmax{args.doc_maxlen}_nbits{args.nbits}"
        f"_kmeans{args.kmeans_niters}_{doc_limit}"
    )


def load_base_rankings(
    checkpoint: str,
    dataset_id: str,
    max_k: int,
    use_full_data: bool,
    query_ids: Sequence[str],
    logger,
) -> Optional[Dict[str, List[str]]]:
    if not use_full_data:
        logger.info("Limited query/doc load is active; skip base rankings reference.")
        return None

    base_rankings_path = reference_rankings_path(dataset_id, checkpoint, max_k)
    if not base_rankings_path.is_file():
        raise FileNotFoundError(
            f"Exact MaxSim reference is missing: {base_rankings_path}. "
            "Run scripts/generate_embeddings_gpu.sh first."
        )

    logger.info("Loading base rankings from %s", base_rankings_path)
    return load_reference_rankings(base_rankings_path, query_ids, max_k)


class PlaidCandidateReranker:
    def __init__(
        self,
        experiment_dir: str,
        experiment_name: str,
        index_name: str,
        doc_maxlen: int,
        query_maxlen: int,
        nbits: int,
        kmeans_niters: int,
        index_bsize: int,
        doc_ids: Sequence[str],
        collection: Sequence[str],
        existing_index_path: Optional[str] = None,
    ) -> None:
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.index_name = index_name
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.nbits = nbits
        self.kmeans_niters = kmeans_niters
        self.index_bsize = index_bsize
        self.doc_ids = list(doc_ids)
        self.collection = collection
        self.existing_index_path = existing_index_path
        self.doc_id_to_pid = {doc_id: pid for pid, doc_id in enumerate(self.doc_ids)}
        self.searcher: Optional[Searcher] = None
        self.index_path: Optional[str] = None

    def load_index(self, logger) -> float:
        config = ColBERTConfig(
            doc_maxlen=self.doc_maxlen,
            query_maxlen=self.query_maxlen,
            nbits=self.nbits,
            kmeans_niters=self.kmeans_niters,
            index_bsize=self.index_bsize,
            root=self.experiment_dir,
            experiment=self.experiment_name,
        )

        run_config = RunConfig(
            nranks=1,
            experiment=self.experiment_name,
            root=self.experiment_dir,
            avoid_fork_if_possible=True,
        )

        t0 = time.perf_counter()
        with Run().context(run_config):
            self.index_path = find_complete_plaid_index(
                experiment_dir=self.experiment_dir,
                experiment_name=self.experiment_name,
                index_name=self.index_name,
                explicit_index_path=self.existing_index_path,
            )
            if self.index_path is None:
                raise FileNotFoundError(
                    "No complete compatible PLAID index was found. Run end-to-end "
                    "PLAID first, or provide one with --plaid-index-path."
                )

            # Searcher accepts an absolute index path, which allows reuse of an
            # index created under a differently named ColBERT experiment.
            logger.info("Loading compatible existing PLAID index from %s", self.index_path)
            # A reused index may record a collection path from the machine on
            # which it was built.  Use this run's already-loaded collection.
            self.searcher = Searcher(
                index=self.index_path,
                collection=self.collection,
                config=config,
            )

        return time.perf_counter() - t0

    def encode_queries(
        self,
        queries: Mapping[str, str],
        bsize: int,
        cache_dir: Optional[str] = None,
    ) -> Tuple[List[str], torch.Tensor]:
        assert self.searcher is not None
        query_ids = list(queries.keys())
        query_texts = list(queries.values())

        # The caller can still fall back to CPU encoding when preprocessing
        # artifacts have not been generated yet.
        if cache_dir is not None:
            cached = load_cached_query_embeddings(
                cache_dir, self.query_maxlen, query_ids
            )
            if cached is not None:
                return query_ids, cached

        encoded_batches = []
        for i in tqdm(range(0, len(query_texts), bsize), desc="Encoding Queries with ColBERT"):
            batch_texts = query_texts[i:i + bsize]
            encoded_batches.append(self.searcher.encode(batch_texts))
        query_tensors = torch.cat(encoded_batches, dim=0) if encoded_batches else torch.empty(0)
        return query_ids, query_tensors

    def _configure_plaid_defaults_for_k(self, k: int) -> None:
        """Apply the official ``Searcher.dense_search()`` defaults for K."""
        assert self.searcher is not None

        if k <= 10:
            if self.searcher.config.ncells is None:
                self.searcher.configure(ncells=1)
            self.searcher.configure(centroid_score_threshold=0.5)
            self.searcher.configure(ndocs=256)
        elif k <= 100:
            if self.searcher.config.ncells is None:
                self.searcher.configure(ncells=2)
            self.searcher.configure(centroid_score_threshold=0.45)
            self.searcher.configure(ndocs=1024)
        else:
            if self.searcher.config.ncells is None:
                self.searcher.configure(ncells=4)
            self.searcher.configure(centroid_score_threshold=0.4)
            self.searcher.configure(ndocs=max(k * 4, 4096))

    def rerank_candidates(
        self,
        phase1_results: Mapping[str, Sequence[str]],
        query_ids: Sequence[str],
        query_tensors: torch.Tensor,
        top_k: int,
    ) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        assert self.searcher is not None
        self._configure_plaid_defaults_for_k(top_k)

        rankings: Dict[str, List[str]] = {}
        missing_doc_count = 0
        total_candidate_count = 0

        t0 = time.perf_counter()
        with torch.inference_mode():
            # Process queries serially.
            for query_idx, qid in tqdm(
                enumerate(query_ids),
                total=len(query_ids),
                desc="Phase 2 PLAID Candidate Reranking",
            ):
                candidate_doc_ids = list(phase1_results.get(qid, []))
                total_candidate_count += len(candidate_doc_ids)

                candidate_pids = []
                for doc_id in candidate_doc_ids:
                    pid = self.doc_id_to_pid.get(doc_id)
                    if pid is None:
                        missing_doc_count += 1
                        continue
                    candidate_pids.append(pid)

                if not candidate_pids:
                    rankings[qid] = []
                    continue

                Q = query_tensors[query_idx:query_idx + 1]
                pids_tensor = torch.tensor(candidate_pids, dtype=torch.int32)
                centroids = self.searcher.ranker.codec.centroids
                Q_2d = Q.squeeze(0).to(device=centroids.device, dtype=centroids.dtype)
                centroid_scores = centroids @ Q_2d.T
                if self.searcher.ranker.use_gpu:
                    pids_tensor = pids_tensor.cuda()

                scores, scored_pids = self.searcher.ranker.score_pids(
                    self.searcher.config,
                    Q,
                    pids_tensor,
                    centroid_scores=centroid_scores,
                )

                if len(scores) == 0:
                    rankings[qid] = []
                    continue

                k_val = min(top_k, len(scores))
                top_indices = torch.topk(scores, k=k_val).indices
                top_pids = scored_pids[top_indices.detach().cpu()].detach().cpu().tolist()
                rankings[qid] = [self.doc_ids[int(pid)] for pid in top_pids]

        elapsed = time.perf_counter() - t0
        num_queries = max(len(query_ids), 1)
        return rankings, {
            "avg_search_time_ms": (elapsed / num_queries) * 1000,
            "QPS": len(query_ids) / elapsed if elapsed else 0.0,
            "missing_doc_count": float(missing_doc_count),
            "total_candidate_count": float(total_candidate_count),
        }


def main() -> None:
    args = parse_args()
    dataset_stub = args.dataset_id.replace("/", "_")
    if args.log_file is None:
        args.log_file = f"./logs/{dataset_stub}/two_phase_plaid_{time.strftime('%m%d')}.log"
    logger = configure_logger("two_phase_plaid", args.log_file)
    logger.info(f"n threads: {n_thread} \nArguments: {args}")

    results_dir = f"./results/{dataset_stub}"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(
        results_dir,
        f"two_phase_plaid_{args.faiss_index_type}_nbits{args.nbits}_centroid_{args.n_candidate_list}_{time.strftime('%m%d')}.csv",
    )

    corpus, queries, qrels = load_dataset_by_name(
        args.dataset_id,
        logger,
        args.n_query_limited_load,
        args.n_doc_limited_load,
    )

    logger.info(
        "\n%s [Phase 1]: Dense Retrieval %s\n[Initializing Dense Retriever] (%s, %s)",
        "*" * 10,
        "*" * 10,
        args.dense_model_name,
        args.faiss_index_type,
    )
    dense_retriever = DenseRetriever(args.dense_model_name, args.faiss_index_type, args.device)
    dense_retriever.index(args.dataset_id, args.n_doc_limited_load, corpus)
    logger.info(
        "[encoding time]: %.2fs, [index build time]: %.2fs, [dense vector dim]: %d",
        dense_retriever.encode_time,
        dense_retriever.index_build_time,
        dense_retriever.embedding_dim,
    )

    max_top_k = max(args.top_k_list)
    p1_query_ids, p1_query_embeddings = dense_retriever.load_query_embeddings(
        queries
    )

    logger.info(
        "\n%s [Phase 2]: Official PLAID Reconstructed MaxSim %s\n[Init PLAID index] (%s, nbits=%d)",
        "*" * 10,
        "*" * 10,
        args.checkpoint,
        args.nbits,
    )
    doc_ids = list(corpus.keys())
    index_name = make_index_name(args)
    plaid_reranker = PlaidCandidateReranker(
        experiment_dir=args.experiment_dir,
        experiment_name=args.experiment,
        index_name=index_name,
        doc_maxlen=args.doc_maxlen,
        query_maxlen=args.query_maxlen,
        nbits=args.nbits,
        kmeans_niters=args.kmeans_niters,
        index_bsize=args.index_bsize,
        doc_ids=doc_ids,
        collection=list(corpus.values()),
        existing_index_path=args.plaid_index_path,
    )
    plaid_index_time = plaid_reranker.load_index(logger)
    index_size_MB = get_dir_size(plaid_reranker.index_path or "") / (1024 ** 2)
    dense_index_size_MB = os.path.getsize(dense_retriever.index_cache_path) / (1024 ** 2)
    logger.info("[PLAID Index Load Time]: %.2fs", plaid_index_time)
    logger.info("[PLAID Index Size]: %.2f MB", index_size_MB)

    query_ids, query_tensors = plaid_reranker.encode_queries(
        queries,
        args.bsize,
        cache_dir=os.path.join(
            "embedding_files",
            dataset_stub,
            "colbert_official_data_query_emb_tensor",
        ),
    )

    base_rankings = load_base_rankings(
        checkpoint=args.checkpoint,
        dataset_id=args.dataset_id,
        max_k=max_top_k,
        use_full_data=args.n_query_limited_load is None and args.n_doc_limited_load is None,
        query_ids=list(queries.keys()),
        logger=logger,
    )

    results_rows = []
    for n_candidate in args.n_candidate_list:
        logger.info("\n--- Phase 1: dense search on [%d candidates] ---", n_candidate)
        p1_results, p1_runtime = dense_retriever.search_encoded_queries(
            p1_query_ids, p1_query_embeddings, top_k=n_candidate
        )
        logger.info(
            "[Phase 1 FAISS Search Time (per query), n_candidate=%d]: %.2fms",
            n_candidate,
            p1_runtime["query_search_ms"],
        )
        p1_metrics = compute_retrieval_metrics(p1_results, qrels, n_candidate, base_rankings)

        for top_k in args.top_k_list:
            p2_results, p2_runtime = plaid_reranker.rerank_candidates(
                p1_results,
                query_ids,
                query_tensors,
                top_k,
            )
            logger.info(
                "[Phase 2 Average Search Time (per query), top_k=%d]: %.2fms",
                top_k,
                p2_runtime["avg_search_time_ms"],
            )
            if p2_runtime["missing_doc_count"]:
                logger.warning(
                    "Skipped %.0f missing dense candidates out of %.0f total candidates.",
                    p2_runtime["missing_doc_count"],
                    p2_runtime["total_candidate_count"],
                )

            p2_metrics = compute_retrieval_metrics(p2_results, qrels, top_k, base_rankings)

            p1_faiss_search_ms = p1_runtime["query_search_ms"]
            p2_plaid_search_ms = p2_runtime["avg_search_time_ms"]
            total_search_ms = p1_faiss_search_ms + p2_plaid_search_ms
            total_latency_ms = total_search_ms
            qps_search = 1000.0 / total_search_ms if total_search_ms else 0.0

            row = {
                "method": "DR + PLAID",
                "dataset": args.dataset_id,
                "index_type": args.faiss_index_type,
                "n_candidate": n_candidate,
                "nbits": args.nbits,
                "kmeans_niters": args.kmeans_niters,
                "doc_maxlen": args.doc_maxlen,
                "query_maxlen": args.query_maxlen,
                "TOP_K": top_k,
                "index_size_MB": index_size_MB,
                "dense_index_size_MB": dense_index_size_MB,
                "total_index_size_MB": dense_index_size_MB + index_size_MB,
                "dense_encode_time_s": dense_retriever.encode_time,
                "dense_index_build_time_s": dense_retriever.index_build_time,
                "dense_index_reused": dense_retriever.index_reused,
                "plaid_index_load_time_s": plaid_index_time,
                "p1_avg_latency_ms": p1_faiss_search_ms,
                "p2_avg_latency_ms": p2_plaid_search_ms,
                "total_latency_ms": total_latency_ms,
                "total_QPS": qps_search,
                "phase2_QPS": p2_runtime["QPS"],
                "phase2_missing_doc_count": p2_runtime["missing_doc_count"],
                **{f"p1_{k}": v for k, v in p1_metrics.items()},
                **{f"p2_{k}": v for k, v in p2_metrics.items()},
            }
            results_rows.append(row)

        df = pd.DataFrame(results_rows)
        print(df)
        df.to_csv(csv_path, index=False)
        logger.info("Results saved successfully to %s", csv_path)


if __name__ == "__main__":
    main()
