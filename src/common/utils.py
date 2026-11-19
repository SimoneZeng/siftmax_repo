from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from common.reference_rankings import (
    load_reference_rankings,
    reference_rankings_path,
)


BASE_DOC_VEC_CNT = 256
TOP_K_LIST = [10, 50, 100]

METRIC_KEYS: Sequence[str] = (
    "recall@k",
    "success@k",
    "MRR@k",
    "nDCG@k",
    "recall_maxsim@k",
    "success_maxsim@k",
    "MRR_maxsim@k",
    "nDCG_maxsim@k",
)
RUNTIME_KEYS: Sequence[str] = ("QPS", "avg_query_time_ms")


def configure_logger(
    name: str = "colbert_pruning",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def compute_retrieval_metrics(
    results: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Iterable[str]],
    top_k: int,
    reference_rankings: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, float]:
    recall = success = mrr = ndcg = 0.0
    recall_maxsim = success_maxsim = mrr_maxsim = ndcg_maxsim = 0.0
    effective_n = ref_n = 0

    for qid, retrieved in results.items():
        retrieved_k = list(retrieved[:top_k])
        relevant = set(qrels.get(qid, []))
        if relevant:
            effective_n += 1
            hits = [doc_id in relevant for doc_id in retrieved_k]
            recall += sum(hits) / len(relevant)
            success += float(any(hits))
            mrr += next(
                (1.0 / rank for rank, hit in enumerate(hits, start=1) if hit),
                0.0,
            )
            dcg = sum(hit / np.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
            idcg = sum(
                1.0 / np.log2(rank + 1)
                for rank in range(1, min(len(relevant), top_k) + 1)
            )
            ndcg += dcg / idcg if idcg else 0.0

        if reference_rankings:
            reference = list(reference_rankings.get(qid, [])[:top_k])
            if reference:
                ref_n += 1
                hits = [doc_id in reference for doc_id in retrieved_k]
                recall_maxsim += sum(hits) / len(reference)
                success_maxsim += float(any(hits))
                try:
                    mrr_maxsim += 1.0 / (retrieved_k.index(reference[0]) + 1)
                except ValueError:
                    pass
                dcg = sum(hit / np.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
                idcg = sum(
                    1.0 / np.log2(rank + 1)
                    for rank in range(1, min(len(reference), top_k) + 1)
                )
                ndcg_maxsim += dcg / idcg if idcg else 0.0

    return {
        "recall@k": recall / effective_n if effective_n else 0.0,
        "success@k": success / effective_n if effective_n else 0.0,
        "MRR@k": mrr / effective_n if effective_n else 0.0,
        "nDCG@k": ndcg / effective_n if effective_n else 0.0,
        "recall_maxsim@k": recall_maxsim / ref_n if ref_n else 0.0,
        "success_maxsim@k": success_maxsim / ref_n if ref_n else 0.0,
        "MRR_maxsim@k": mrr_maxsim / ref_n if ref_n else 0.0,
        "nDCG_maxsim@k": ndcg_maxsim / ref_n if ref_n else 0.0,
    }


def get_base_rankings(
    checkpoint: str,
    dataset_id: str,
    max_k: int,
    logger,
    query_ids: Optional[Iterable[object]] = None,
    required: bool = False,
) -> Optional[Dict[str, List[str]]]:
    path = reference_rankings_path(dataset_id, checkpoint, max_k)
    if not path.is_file():
        message = (
            f"Exact MaxSim reference is missing: {path}. "
            "Run scripts/generate_embeddings_gpu.sh first."
        )
        if required:
            raise FileNotFoundError(message)
        logger.info(message)
        return None
    logger.info("Loading base rankings from %s", path)
    if query_ids is not None:
        return load_reference_rankings(path, query_ids, max_k)
    with path.open("rb") as handle:
        return pickle.load(handle)


@dataclass
class MetricsRecorder:
    csv_path: str
    metric_keys: Sequence[str] = METRIC_KEYS
    runtime_keys: Sequence[str] = RUNTIME_KEYS
    rows: List[Dict[str, Union[float, int, str]]] = field(default_factory=list)

    def add_row(
        self,
        base_fields: Dict[str, Union[float, int, str]],
        metrics: Mapping[str, float],
        runtime_stats: Optional[Mapping[str, float]] = None,
    ) -> None:
        row = dict(base_fields)
        row.update({key: metrics[key] for key in self.metric_keys if key in metrics})
        if runtime_stats:
            row.update(
                {key: runtime_stats[key] for key in self.runtime_keys if key in runtime_stats}
            )
        self.rows.append(row)

    def save(self) -> None:
        if not self.rows:
            return
        directory = os.path.dirname(self.csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)


def slice_rankings(
    rankings: Mapping[str, Sequence[str]], top_k: int
) -> Dict[str, List[str]]:
    return {qid: list(docs[:top_k]) for qid, docs in rankings.items()}


def build_inverse_rotation_matrix(dim: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    random_matrix = rng.standard_normal((dim, dim)).astype(np.float32)
    q, r = np.linalg.qr(random_matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return torch.from_numpy(np.ascontiguousarray((q * signs).astype(np.float32)))


def ensure_rabitq_extension_support() -> None:
    import rabitq_official_ext

    required = ("encode_pack_rabitq", "rerank_topk_rabitq")
    missing = [name for name in required if not hasattr(rabitq_official_ext, name)]
    if missing:
        path = getattr(rabitq_official_ext, "__file__", "<unknown>")
        raise ImportError(
            f"rabitq_official_ext at {path} is missing {missing}; rebuild it with "
            "python setup_rabitq_official.py build_ext --inplace"
        )


__all__ = [
    "BASE_DOC_VEC_CNT",
    "TOP_K_LIST",
    "MetricsRecorder",
    "build_inverse_rotation_matrix",
    "compute_retrieval_metrics",
    "configure_logger",
    "ensure_rabitq_extension_support",
    "get_base_rankings",
    "slice_rankings",
]
