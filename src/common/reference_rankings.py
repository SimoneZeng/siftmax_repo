"""Canonical storage and validation for exact full-corpus MaxSim rankings."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def reference_rankings_path(
    dataset_id: str,
    checkpoint: str,
    top_k: int,
    embedding_root: Path | str = Path("embedding_files"),
) -> Path:
    """Return the shared path used by every experiment runner."""
    return (
        Path(embedding_root)
        / dataset_id.replace("/", "_")
        / "base_rankings"
        / checkpoint.replace("/", "_")
        / f"topk_{top_k}.pkl"
    )


def normalize_reference_rankings(
    rankings: Mapping[object, Sequence[object]],
    query_ids: Iterable[object],
    top_k: int,
) -> dict[str, list[str]]:
    """Validate a complete reference and normalize IDs to strings."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not isinstance(rankings, Mapping):
        raise TypeError(f"reference rankings must be a mapping, got {type(rankings)}")

    normalized = {
        str(qid): [str(doc_id) for doc_id in docs]
        for qid, docs in rankings.items()
    }
    expected = [str(qid) for qid in query_ids]
    expected_set = set(expected)
    actual_set = set(normalized)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)[:5]
        extra = sorted(actual_set - expected_set)[:5]
        raise ValueError(
            "reference query IDs do not match the embedding cache: "
            f"missing={missing}, extra={extra}"
        )

    for qid in expected:
        docs = normalized[qid]
        if len(docs) != top_k:
            raise ValueError(
                f"reference query {qid} has {len(docs)} documents; expected {top_k}"
            )
        if len(set(docs)) != top_k:
            raise ValueError(f"reference query {qid} contains duplicate document IDs")
    return {qid: normalized[qid] for qid in expected}


def load_reference_rankings(
    path: Path | str,
    query_ids: Iterable[object],
    top_k: int,
) -> dict[str, list[str]]:
    with Path(path).open("rb") as handle:
        rankings = pickle.load(handle)
    return normalize_reference_rankings(rankings, query_ids, top_k)


def save_reference_rankings(
    path: Path | str,
    rankings: Mapping[object, Sequence[object]],
    query_ids: Iterable[object],
    top_k: int,
) -> dict[str, list[str]]:
    """Validate and atomically persist rankings, returning normalized data."""
    normalized = normalize_reference_rankings(rankings, query_ids, top_k)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(normalized, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return normalized
