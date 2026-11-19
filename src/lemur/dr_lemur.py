from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from shared import (
    CPU_THREADS,
    create_maxsim,
    directory_size,
    document_artifact_size,
    evaluate,
    load_colbert_artifacts,
    load_qrels,
    load_reference,
    maxsim_rerank_one,
    prepare_lemur,
    query_feature,
    set_cpu_threads,
    top_indices,
    write_outputs,
)
from common.dense_retrieval import (
    DENSE_MODEL_NAME,
    DenseRetriever,
    load_serialized_dense_artifacts,
    map_dense_ids_to_colbert,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniLM Faiss Flat + LEMUR + exact MaxSim on LoTTE Science")
    parser.add_argument("--embedding-dir", type=Path, default=Path("embedding_files/lotte_science/colbert_official_data_query_emb_tensor"))
    parser.add_argument("--dense-query", type=Path, default=Path("embedding_files/lotte_science/dense_query_embeddings_all-MiniLM-L6-v2_384d_538queries.pkl"))
    parser.add_argument("--dense-index", type=Path, default=Path("embedding_files/lotte_science/dense_index_MiniLM_flat_384d_343642docs.pkl"))
    parser.add_argument("--dense-model-name", default=DENSE_MODEL_NAME)
    parser.add_argument("--qrels", type=Path, default=None)
    parser.add_argument("--exact-reference", type=Path, default=Path("embedding_files/lotte_science/base_rankings/colbert-ir_colbertv2.0/topk_10.pkl"))
    parser.add_argument("--lemur-index", type=Path, default=Path("indexes/lemur_lotte_science_2048d"))
    parser.add_argument("--lemur-repo", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=Path("results/lotte_science"))
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for LEMUR training/FDE generation only; dense search, FDE filtering, and MaxSim run on CPU",
    )
    parser.add_argument("--n-candidate-list", type=int, nargs="+", default=[200, 500, 1000, 1500, 2000, 3000, 4000])
    parser.add_argument("--filter-size-list", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--rebuild-lemur-index", action="store_true")
    parser.add_argument("--threads", type=int, default=CPU_THREADS)
    args = parser.parse_args()
    if args.qrels is None:
        data_root = Path(os.environ.get("LOTTE_DATA_DIR", "../datasets/lotte"))
        args.qrels = data_root / "science/dev/qas.search.jsonl"
    if any(value < 10 for value in args.n_candidate_list + args.filter_size_list):
        parser.error("candidate and filter sizes must be at least K=10")
    return args
def main() -> None:
    args = parse_args()
    set_cpu_threads(args.threads)
    docs, masks, doc_ids, queries, query_ids = load_colbert_artifacts(args.embedding_dir)
    dense_qids, dense_queries, dense_doc_ids, dense_index, dense_build_s = load_serialized_dense_artifacts(args.dense_query, args.dense_index)
    if query_ids != dense_qids:
        raise ValueError("MiniLM and ColBERT query IDs/order differ")
    dense_retriever = DenseRetriever(args.dense_model_name, "flat", "cpu")
    dense_retriever.doc_ids = dense_doc_ids
    dense_retriever.faiss_index = dense_index
    if len(docs) != 343642 or dense_retriever.faiss_index.ntotal != 343642:
        raise ValueError("DR + LEMUR requires the complete 343,642-document Science corpus")
    dense_to_colbert = map_dense_ids_to_colbert(dense_doc_ids, doc_ids)
    dense_id_to_colbert = {
        doc_id: int(dense_to_colbert[offset])
        for offset, doc_id in enumerate(dense_doc_ids)
    }
    qrels = load_qrels(args.qrels)
    reference = load_reference(args.exact_reference)

    lemur, MaxSim, weights, flat, counts, build_s, reused, training_included = prepare_lemur(
        args.lemur_index, args.device, args.lemur_repo, docs, masks, args.rebuild_lemur_index
    )
    maxsim, flat, counts = create_maxsim(MaxSim, docs, masks, flat, counts)
    features = [query_feature(lemur, query) for query in queries]
    features = np.ascontiguousarray(features, dtype=np.float32)

    aggregates = []
    all_rankings = {}
    query_count = len(query_ids)
    for n_candidate in args.n_candidate_list:
        dense_results, dense_runtime = dense_retriever.search_encoded_queries(
            dense_qids, dense_queries, top_k=n_candidate
        )
        dense_candidates = [
            np.asarray(
                [dense_id_to_colbert[doc_id] for doc_id in dense_results[qid]],
                dtype=np.int64,
            )
            for qid in query_ids
        ]
        dense_s = dense_runtime["query_search_time_s"]

        for filter_size in args.filter_size_list:
            if filter_size > n_candidate:
                continue
            config_key = f"n{n_candidate}_f{filter_size}"
            config_rankings = {}
            rerank_start = time.perf_counter()
            for offset, qid in enumerate(query_ids):
                candidates = dense_candidates[offset]
                learned_scores = features[offset] @ weights[candidates].T
                retained = candidates[top_indices(learned_scores, filter_size)].astype(np.int32, copy=False)

                result_indices = maxsim_rerank_one(maxsim, queries[offset], retained, 10, args.threads)
                ranking = [doc_ids[index] for index in result_indices if index >= 0]
                config_rankings[qid] = ranking
            # Query FDE generation is deliberately outside the benchmark
            # latency boundary. Measure the complete reranking loop once.
            rerank_s = time.perf_counter() - rerank_start
            total_s = dense_s + rerank_s
            metrics = evaluate(config_rankings, qrels, reference)
            rerank_bytes = directory_size(args.lemur_index) + document_artifact_size(args.embedding_dir)
            aggregates.append({
                "method": "DR + LEMUR",
                "n_candidate": n_candidate,
                "lemur_filter_size": filter_size,
                "top_k": 10,
                "queries": query_count,
                "dense_build_s": dense_build_s,
                "multivector_build_s": build_s,
                "total_build_s": dense_build_s + build_s,
                "lemur_training_included": str(training_included).lower(),
                "lemur_index_reused": str(reused).lower(),
                "dense_index_bytes": args.dense_index.stat().st_size,
                "rerank_index_bytes": rerank_bytes,
                "total_index_bytes": args.dense_index.stat().st_size + rerank_bytes,
                **{f"p3_{key}": value for key, value in metrics.items()},
                "p1_avg_latency_ms": 1000.0 * dense_s / query_count,
                "p2_avg_latency_ms": 1000.0 * rerank_s / query_count,
                "total_latency_ms": 1000.0 * total_s / query_count,
                "total_QPS": query_count / total_s,
            })
            all_rankings[config_key] = config_rankings

    write_outputs(args.result_dir, "dr_lemur", aggregates, all_rankings)
    metadata = {
        "dataset": "lotte/science/dev/search",
        "documents": len(doc_ids),
        "queries": len(query_ids),
        "pipeline": "MiniLM IndexFlatIP -> LEMUR 2048D FDE -> exact MaxSim",
        "cpu_threads": args.threads,
        "lemur_index_reused": reused,
        "lemur_training_and_fde_build_s": build_s,
        "lemur_training_included_in_multivector_build": training_included,
        "latency_boundary": "after precomputed MiniLM and LEMUR query embeddings; dense retrieval through top-10",
    }
    with (args.result_dir / "dr_lemur_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
