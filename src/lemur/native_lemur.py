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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native LEMUR on LoTTE Science")
    parser.add_argument("--embedding-dir", type=Path, default=Path("embedding_files/lotte_science/colbert_official_data_query_emb_tensor"))
    parser.add_argument("--qrels", type=Path, default=None)
    parser.add_argument("--exact-reference", type=Path, default=Path("embedding_files/lotte_science/base_rankings/colbert-ir_colbertv2.0/topk_10.pkl"))
    parser.add_argument("--lemur-index", type=Path, default=Path("indexes/lemur_lotte_science_2048d"))
    parser.add_argument("--lemur-repo", type=Path, default=None, help="Optional local checkout; otherwise use installed lemur package")
    parser.add_argument("--result-dir", type=Path, default=Path("results/lotte_science"))
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for LEMUR training/FDE generation only; query retrieval and MaxSim run on CPU",
    )
    parser.add_argument("--n-candidate-list", type=int, nargs="+", default=[200, 500, 1000, 1500, 2000, 3000, 4000])
    parser.add_argument("--rebuild-lemur-index", action="store_true")
    parser.add_argument("--threads", type=int, default=CPU_THREADS)
    args = parser.parse_args()
    if args.qrels is None:
        data_root = Path(os.environ.get("LOTTE_DATA_DIR", "../datasets/lotte"))
        args.qrels = data_root / "science/dev/qas.search.jsonl"
    if any(value < 10 for value in args.n_candidate_list):
        parser.error("all candidate counts must be at least K=10")
    return args


def main() -> None:
    args = parse_args()
    set_cpu_threads(args.threads)
    docs, masks, doc_ids, queries, query_ids = load_colbert_artifacts(args.embedding_dir)
    qrels = load_qrels(args.qrels)
    reference = load_reference(args.exact_reference)
    if len(docs) != 343642 or len(query_ids) != 538:
        raise ValueError("native LEMUR requires the complete LoTTE Science corpus and 538 queries")

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
        config_rankings = {}
        search_start = time.perf_counter()
        for offset, qid in enumerate(query_ids):
            scores = features[offset] @ weights.T
            candidates = top_indices(scores, n_candidate).astype(np.int32, copy=False)

            result_indices = maxsim_rerank_one(maxsim, queries[offset], candidates, 10, args.threads)
            ranking = [doc_ids[index] for index in result_indices if index >= 0]
            config_rankings[qid] = ranking
        # Query FDE generation is deliberately outside the benchmark latency
        # boundary. Measure candidate retrieval through top-10 as one batch.
        total_s = time.perf_counter() - search_start
        metrics = evaluate(config_rankings, qrels, reference)
        aggregates.append({
            "method": "LEMUR",
            "n_candidate": n_candidate,
            "top_k": 10,
            "queries": query_count,
            "multivector_build_s": build_s,
            "total_build_s": build_s,
            "lemur_training_included": str(training_included).lower(),
            "lemur_index_reused": str(reused).lower(),
            "rerank_index_bytes": directory_size(args.lemur_index) + document_artifact_size(args.embedding_dir),
            "total_index_bytes": directory_size(args.lemur_index) + document_artifact_size(args.embedding_dir),
            **metrics,
            "avg_query_time_ms": 1000.0 * total_s / query_count,
            "QPS": query_count / total_s,
        })
        all_rankings[str(n_candidate)] = config_rankings

    write_outputs(args.result_dir, "lemur_native", aggregates, all_rankings)
    metadata = {
        "dataset": "lotte/science/dev/search",
        "documents": len(doc_ids),
        "queries": len(query_ids),
        "colbert_dimension": 128,
        "query_max_vectors": 32,
        "document_max_vectors": 256,
        "learned_dimension": int(weights.shape[1]),
        "cpu_threads": args.threads,
        "lemur_index_reused": reused,
        "lemur_training_and_fde_build_s": build_s,
        "lemur_training_included_in_multivector_build": training_included,
        "latency_boundary": "after precomputed LEMUR query FDE; candidate retrieval through top-10",
    }
    with (args.result_dir / "lemur_native_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
