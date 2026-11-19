#!/usr/bin/env python3
"""Summarize all methods at a target MaxSim recall and write a final table."""

import argparse
import csv
import glob
import os


METHOD_ORDER = (
    "PLAID",
    "LEMUR",
    "DR + PLAID",
    "DR + Exact MaxSim",
    "DR + SiftMax",
    "DR + LEMUR",
)

OUTPUT_FIELDS = (
    "dataset",
    "method",
    "n_candidate",
    "ncells",
    "alpha",
    "b",
    "lemur_filter_size",
    "total_index_size_MB",
    "maxsim_recall_at_10",
    "maxsim_mrr_at_10",
    "maxsim_ndcg_at_10",
    "relevance_recall_at_10",
    "relevance_mrr_at_10",
    "relevance_ndcg_at_10",
    "mean_dense_ms",
    "mean_rerank_ms",
    "mean_total_retrieval_ms",
    "qps",
)


def first(row, *keys, default=""):
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    return default


def detect_method(row, path):
    method = row.get("method", "")
    if method == "plaid_test" or ("ncells" in row and "recall_maxsim@k" in row):
        return "PLAID"
    if method in METHOD_ORDER:
        return method
    if "maxsim_rerank_ms_per_query" in row:
        return "DR + Exact MaxSim"
    if "p2_plaid_search_ms" in row:
        return "DR + PLAID"
    if "multi_index_size_MB" in row:
        return "DR + SiftMax"
    raise ValueError(f"{path}: cannot determine method from CSV columns")


def normalize_row(row, path, dataset):
    method = detect_method(row, path)
    is_native = method in {"PLAID", "LEMUR"}
    metric_prefix = "" if is_native else ("p3_" if method == "DR + LEMUR" else "p2_")
    legacy_prefix = "p2_" if not is_native else ""
    metrics = {
        "maxsim_recall_at_10": first(
            row, f"{metric_prefix}recall_maxsim@k",
            f"{legacy_prefix}recall_maxsim_at_k", "maxsim_recall_at_10",
        ),
        "maxsim_mrr_at_10": first(
            row, f"{metric_prefix}MRR_maxsim@k",
            f"{legacy_prefix}mrr_maxsim_at_k", "maxsim_mrr_at_10",
        ),
        "maxsim_ndcg_at_10": first(
            row, f"{metric_prefix}nDCG_maxsim@k",
            f"{legacy_prefix}ndcg_maxsim_at_k", "maxsim_ndcg_at_10",
        ),
        "relevance_recall_at_10": first(
            row, f"{metric_prefix}recall@k",
            f"{legacy_prefix}recall_at_k", "relevance_recall_at_10",
        ),
        "relevance_mrr_at_10": first(
            row, f"{metric_prefix}MRR@k",
            f"{legacy_prefix}mrr_at_k", "relevance_mrr_at_10",
        ),
        "relevance_ndcg_at_10": first(
            row, f"{metric_prefix}nDCG@k",
            f"{legacy_prefix}ndcg_at_k", "relevance_ndcg_at_10",
        ),
    }

    missing = [key for key, value in metrics.items() if value == ""]
    if missing:
        raise ValueError(f"{path}: missing metrics: {', '.join(missing)}")

    size_mb = first(row, "total_index_size_MB", "index_size_MB")
    if not size_mb:
        size_bytes = first(row, "total_index_bytes")
        if not size_bytes:
            raise ValueError(f"{path}: missing total index size")
        size_mb = str(float(size_bytes) / (1024 ** 2))

    if is_native:
        dense_ms = ""
        rerank_ms = ""
        total_ms = first(row, "avg_query_time_ms", "mean_e2e_ms", "mean_rerank_ms")
        qps = first(row, "QPS", "qps")
    else:
        dense_ms = first(
            row, "p1_avg_latency_ms", "mean_dense_ms",
            "dense_retrieval_ms_per_query", "p1_faiss_search_ms", "p1_query_search_ms",
        )
        rerank_ms = first(
            row, "p2_avg_latency_ms", "mean_rerank_ms",
            "maxsim_rerank_ms_per_query", "p2_plaid_search_ms", "p2_query_search_ms",
        )
        total_ms = first(row, "total_latency_ms", "mean_e2e_ms", "total_search_ms")
        qps = first(row, "total_QPS", "qps", "QPS_search")

    if not total_ms or not qps:
        raise ValueError(f"{path}: missing total retrieval time or QPS")

    result = {
        "dataset": dataset,
        "method": method,
        "n_candidate": first(row, "n_candidate", "ndocs"),
        "ncells": first(row, "ncells"),
        "alpha": first(row, "alpha", "alpha_geometry"),
        "b": first(row, "b", "nbit", "nbits"),
        "lemur_filter_size": first(row, "lemur_filter_size"),
        "total_index_size_MB": size_mb,
        "mean_dense_ms": dense_ms,
        "mean_rerank_ms": rerank_ms,
        "mean_total_retrieval_ms": total_ms,
        "qps": qps,
    }
    result.update(metrics)
    return result


def select_results(rows, threshold):
    selected = []
    for method in METHOD_ORDER:
        eligible = [
            row for row in rows
            if row["method"] == method
            and float(row["maxsim_recall_at_10"]) >= threshold
        ]
        if not eligible:
            raise ValueError(
                f"{method}: no result has MaxSim Recall@10 >= {threshold}"
            )
        selected.append(min(
            eligible,
            key=lambda row: (
                float(row["maxsim_recall_at_10"]) - threshold,
                -float(row["qps"]),
            ),
        ))
    return selected


def default_output(dataset, threshold):
    dataset_slug = dataset.replace("/", "_").replace(" ", "_")
    return os.path.join(
        "results_template",
        f"{dataset_slug}_maxsim_recall_{threshold:g}_summary.csv",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="*",
        help="result CSV paths or glob patterns (defaults to results directories)",
    )
    parser.add_argument("--dataset", default="lotte_science")
    parser.add_argument("--maxsim-recall", type=float, default=0.9)
    parser.add_argument("--output", help="default includes dataset and recall threshold")
    args = parser.parse_args()
    if not 0.0 <= args.maxsim_recall <= 1.0:
        parser.error("--maxsim-recall must be between 0 and 1")

    patterns = args.inputs or (
        "results/lotte_science/*.csv",
        "results_ours/lotte_science/*.csv",
    )
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        parser.error("no result CSV files matched")

    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        if not source_rows:
            raise ValueError(f"{path}: CSV contains no result rows")
        rows.extend(normalize_row(row, path, args.dataset) for row in source_rows)

    selected = select_results(rows, args.maxsim_recall)
    output = args.output or default_output(args.dataset, args.maxsim_recall)
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    print(
        f"selected {len(selected)} method(s) at MaxSim Recall@10 >= "
        f"{args.maxsim_recall:g}; wrote {output}"
    )


if __name__ == "__main__":
    main()
