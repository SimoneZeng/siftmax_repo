"""CPU-only dense retrieval followed by exhaustive exact MaxSim reranking.

Dense queries are encoded once, while FAISS search is independently timed for
every requested candidate count. During exact MaxSim reranking, queries are
processed strictly one at a time; candidates belonging to the current query
are evaluated in vectorized CPU batches.
"""

import os

# Set these before importing NumPy/PyTorch so their runtimes see the values.
CPU_THREADS = 32
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))

import argparse
import time
from typing import Dict, List, Tuple

import faiss
import pandas as pd
import torch
from tqdm import tqdm

from common.colbert_embeddings import ColbertRetriever
from common.data_load import load_dataset_by_name
from common.dense_retrieval import DENSE_MODEL_NAME, DenseRetriever
from common.utils import (
    BASE_DOC_VEC_CNT,
    TOP_K_LIST,
    compute_retrieval_metrics,
    configure_logger,
    slice_rankings,
)


torch.set_num_threads(CPU_THREADS)
faiss.omp_set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(CPU_THREADS)
except RuntimeError:
    # PyTorch only permits setting this once in a process.
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only Dense + exhaustive exact MaxSim reranking"
    )
    parser.add_argument("--dataset-id", type=str, default="lotte/science")
    parser.add_argument("--checkpoint", type=str, default="colbert-ir/colbertv2.0")
    parser.add_argument("--doc-maxlen", type=int, default=BASE_DOC_VEC_CNT)
    parser.add_argument("--query-maxlen", type=int, default=32)
    parser.add_argument("--top-k-list", type=int, nargs="+", default=TOP_K_LIST)
    parser.add_argument("--n-query-limited-load", type=int, default=None)
    parser.add_argument("--n-doc-limited-load", type=int, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument(
        "--bsize",
        type=int,
        default=128,
        help="Batch size used only while generating ColBERT embeddings.",
    )
    parser.add_argument(
        "--maxsim-candidate-batch-size",
        type=int,
        default=128,
        help=(
            "Number of candidates vectorized at a time inside one query. "
            "Queries themselves are always processed serially."
        ),
    )
    parser.add_argument("--dense-model-name", type=str, default=DENSE_MODEL_NAME)
    parser.add_argument(
        "--faiss-index-type",
        type=str,
        default="flat",
        choices=["flat"],
    )
    parser.add_argument(
        "--n-candidate-list",
        type=int,
        nargs="+",
        default=[200, 500, 1000, 1500, 2000, 3000, 4000],
    )
    args = parser.parse_args()

    if args.bsize <= 0 or args.maxsim_candidate_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if not args.top_k_list or any(k <= 0 for k in args.top_k_list):
        parser.error("--top-k-list must contain positive integers")
    if not args.n_candidate_list or any(k <= 0 for k in args.n_candidate_list):
        parser.error("--n-candidate-list must contain positive integers")
    return args


def exact_maxsim_candidates(
    phase1_results: Dict[str, List[str]],
    query_ids: List[str],
    query_tensors: torch.Tensor,
    doc_embeddings: torch.Tensor,
    doc_masks: torch.Tensor,
    doc_id_to_idx: Dict[str, int],
    top_k: int,
    candidate_batch_size: int,
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """Rerank dense candidates using exact MaxSim on CPU.

    The outer loop is deliberately serial across queries. Within one query,
    PyTorch evaluates a candidate batch with vectorized matrix multiplication;
    its CPU kernels may use ``CPU_THREADS`` worker threads.
    """
    if doc_embeddings.device.type != "cpu" or doc_masks.device.type != "cpu":
        raise ValueError("exact MaxSim inputs must remain on CPU")

    results: Dict[str, List[str]] = {}

    rerank_start = time.perf_counter()
    with torch.inference_mode():
        for query_offset, qid in enumerate(
            tqdm(query_ids, desc="Phase 2 exact MaxSim (serial queries)")
        ):
            candidate_doc_ids: List[str] = []
            candidate_indices: List[int] = []
            for doc_id in phase1_results.get(qid, []):
                doc_index = doc_id_to_idx.get(doc_id)
                if doc_index is not None:
                    candidate_doc_ids.append(doc_id)
                    candidate_indices.append(doc_index)

            if not candidate_indices:
                results[qid] = []
                continue

            query = query_tensors[query_offset].to(device="cpu")
            score_chunks: List[torch.Tensor] = []

            for start in range(0, len(candidate_indices), candidate_batch_size):
                chunk_indices = candidate_indices[start : start + candidate_batch_size]

                index_tensor = torch.tensor(chunk_indices, dtype=torch.long)
                candidate_embeddings = doc_embeddings.index_select(0, index_tensor)
                candidate_masks = doc_masks.index_select(0, index_tensor)

                # [C, Lq, D] @ [C, D, Ld] -> [C, Lq, Ld]. C is the
                # candidate dimension for one query, never a query batch.
                similarities = torch.matmul(
                    query.unsqueeze(0),
                    candidate_embeddings.to(dtype=query.dtype).transpose(-1, -2),
                )
                valid_doc_tokens = candidate_masks.reshape(
                    candidate_masks.size(0), -1
                )
                similarities.masked_fill_(
                    ~valid_doc_tokens.unsqueeze(1),
                    torch.finfo(similarities.dtype).min,
                )
                score_chunks.append(similarities.max(dim=-1).values.sum(dim=-1))

            all_scores = torch.cat(score_chunks)
            result_count = min(top_k, len(candidate_doc_ids))
            top_indices = torch.topk(all_scores, k=result_count).indices.tolist()
            results[qid] = [candidate_doc_ids[index] for index in top_indices]

    total_rerank_time = time.perf_counter() - rerank_start
    query_count = max(len(query_ids), 1)
    return results, {
        "avg_rerank_ms": 1000.0 * total_rerank_time / query_count,
    }


def main() -> None:
    args = parse_args()
    dataset_stub = args.dataset_id.replace("/", "_")
    if args.log_file is None:
        args.log_file = (
            f"./logs/{dataset_stub}/two_phase_official_pipeline_"
            f"{time.strftime('%m%d')}.log"
        )
    logger = configure_logger("two_phase_official", args.log_file)
    logger.info("Arguments: %s", args)
    logger.info(
        "CPU-only retrieval: threads=%d; MaxSim queries are serial", CPU_THREADS
    )

    results_dir = f"./results/{dataset_stub}"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(
        results_dir,
        f"two_phase_official_exact_cpu_{args.faiss_index_type}_"
        f"{args.n_candidate_list}_{time.strftime('%m%d')}.csv",
    )
    embedding_cache_dir = (
        f"./embedding_files/{dataset_stub}/colbert_official_data_query_emb_tensor/"
    )
    os.makedirs(embedding_cache_dir, exist_ok=True)

    corpus, queries, qrels = load_dataset_by_name(
        args.dataset_id,
        logger,
        args.n_query_limited_load,
        args.n_doc_limited_load,
    )

    # Phase 1: both SentenceTransformer and FAISS are explicitly CPU-backed.
    logger.info(
        "\n%s [Phase 1]: CPU Dense Retrieval %s\n"
        "[Initializing Dense Retriever] (%s, %s)",
        "*" * 10,
        "*" * 10,
        args.dense_model_name,
        args.faiss_index_type,
    )
    dense_retriever = DenseRetriever(
        args.dense_model_name, args.faiss_index_type, "cpu"
    )
    dense_retriever.index(args.dataset_id, args.n_doc_limited_load, corpus)
    logger.info(
        "[encoding time]: %.2fs, [index build time]: %.2fs, "
        "[dense vector dim]: %d",
        dense_retriever.encode_time,
        dense_retriever.index_build_time,
        dense_retriever.embedding_dim,
    )
    dense_query_ids, dense_query_embeddings = dense_retriever.load_query_embeddings(
        queries
    )

    # Phase 2: load/generate unpruned, unquantized official ColBERT vectors.
    logger.info(
        "\n%s [Phase 2]: CPU Exhaustive Exact MaxSim %s\n"
        "[Init Multi-vector Retriever] (%s)",
        "*" * 10,
        "*" * 10,
        args.checkpoint,
    )
    colbert_retriever = ColbertRetriever(
        checkpoint=args.checkpoint,
        device="cpu",
        doc_maxlen=args.doc_maxlen,
        query_maxlen=args.query_maxlen,
        bsize=args.bsize,
    )

    load_start = time.perf_counter()
    doc_ids, doc_embeddings, doc_masks = colbert_retriever.get_doc_embedding(
        corpus,
        embedding_cache_dir,
        logger,
        n_doc_limited_load=args.n_doc_limited_load,
    )
    multi_vec_load_time = time.perf_counter() - load_start

    query_ids, query_tensors = colbert_retriever.get_query_embedding(
        queries,
        embedding_cache_dir,
        logger,
        n_query_limited_load=args.n_query_limited_load,
    )

    doc_embeddings = doc_embeddings.cpu()
    doc_masks = doc_masks.cpu()
    query_tensors = query_tensors.cpu()
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    max_top_k = max(args.top_k_list)
    base_rankings = colbert_retriever.get_base_rankings(
        dataset_id=args.dataset_id,
        max_k=max_top_k,
        use_full_data=(
            args.n_query_limited_load is None
            and args.n_doc_limited_load is None
        ),
        logger=logger,
    )
    _, avg_vecs = colbert_retriever.get_index_stats()
    cache_prefix = os.path.join(
        embedding_cache_dir,
        f"colbert_official_index_{args.doc_maxlen}vecs",
    )
    full_precision_paths = [
        f"{cache_prefix}_ids.pkl",
        f"{cache_prefix}_embs.pkl",
        f"{cache_prefix}_masks.pkl",
    ]
    missing_storage = [path for path in full_precision_paths if not os.path.isfile(path)]
    if missing_storage:
        raise FileNotFoundError(
            "Exact MaxSim persistent rerank files are missing: " + ", ".join(missing_storage)
        )
    index_size = sum(os.path.getsize(path) for path in full_precision_paths) / (1024 ** 2)
    dense_index_size = os.path.getsize(dense_retriever.index_cache_path) / (1024 ** 2)
    logger.info(
        "[Official ColBERT Load/Encode Time]: %.2fs; "
        "[Index Size]: %.2f MB; [Average Vectors per Doc]: %.2f",
        multi_vec_load_time,
        index_size,
        avg_vecs,
    )

    results_rows = []
    for n_candidate in args.n_candidate_list:
        logger.info(
            "\n--- Dense top-%d -> exhaustive exact MaxSim reranking ---",
            n_candidate,
        )
        # Search separately for each candidate count so its dense latency is
        # the latency of that actual configuration, not that of max(N).
        phase1_results, phase1_runtime = dense_retriever.search_encoded_queries(
            dense_query_ids, dense_query_embeddings, top_k=n_candidate
        )
        phase1_metrics = compute_retrieval_metrics(
            phase1_results, qrels, n_candidate, base_rankings
        )

        phase2_results, phase2_runtime = exact_maxsim_candidates(
            phase1_results=phase1_results,
            query_ids=query_ids,
            query_tensors=query_tensors,
            doc_embeddings=doc_embeddings,
            doc_masks=doc_masks,
            doc_id_to_idx=doc_id_to_idx,
            top_k=max_top_k,
            candidate_batch_size=args.maxsim_candidate_batch_size,
        )
        dense_search_ms = phase1_runtime["query_search_ms"]
        maxsim_rerank_ms = phase2_runtime["avg_rerank_ms"]
        total_search_ms = dense_search_ms + maxsim_rerank_ms

        logger.info(
            "[Dense retrieval]: %.3f ms/query; [Exact MaxSim rerank]: "
            "%.3f ms/query; [Search total]: %.3f ms/query",
            dense_search_ms,
            maxsim_rerank_ms,
            total_search_ms,
        )

        for top_k in args.top_k_list:
            phase2_sliced = slice_rankings(phase2_results, top_k)
            phase2_metrics = compute_retrieval_metrics(
                phase2_sliced, qrels, top_k, base_rankings
            )
            results_rows.append(
                {
                    "method": "DR + Exact MaxSim",
                    "dataset": args.dataset_id,
                    "index_type": args.faiss_index_type,
                    "n_candidate": n_candidate,
                    "TOP_K": top_k,
                    "device": "cpu",
                    "index_size_MB": index_size,
                    "dense_index_size_MB": dense_index_size,
                    "total_index_size_MB": dense_index_size + index_size,
                    "avg_vecs_per_doc": avg_vecs,
                    "dense_index_build_time_s": dense_retriever.index_build_time,
                    "dense_index_reused": dense_retriever.index_reused,
                    "multivector_index_build_time_s": 0.0,
                    "p1_avg_latency_ms": phase1_runtime["query_search_ms"],
                    "p2_avg_latency_ms": maxsim_rerank_ms,
                    "total_latency_ms": total_search_ms,
                    "total_QPS": 1000.0 / total_search_ms
                    if total_search_ms > 0
                    else 0.0,
                    **{
                        f"p1_{key}": value
                        for key, value in phase1_metrics.items()
                    },
                    **{
                        f"p2_{key}": value
                        for key, value in phase2_metrics.items()
                    },
                }
            )

        frame = pd.DataFrame(results_rows)
        frame.to_csv(csv_path, index=False)
        logger.info("Results saved successfully to %s", csv_path)

    print(pd.DataFrame(results_rows))


if __name__ == "__main__":
    main()
