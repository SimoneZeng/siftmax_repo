"""Fixed-assignment centroid refinement for RaBitQ experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class CentroidRefineConfig:
    objective: str = "plaid"
    seed: int = 42


_DIAGNOSTIC_MAX_TOKENS = 200_000


def _sample_complete_documents(
    doc_offsets: np.ndarray,
    max_tokens: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return token indices and sampled-document offsets without splitting docs."""
    total_tokens = int(doc_offsets[-1])
    if max_tokens <= 0 or total_tokens <= max_tokens:
        return (
            np.arange(total_tokens, dtype=np.int64),
            np.asarray(doc_offsets, dtype=np.int64),
        )

    rng = np.random.default_rng(seed)
    doc_ids = rng.permutation(len(doc_offsets) - 1)
    selected = []
    token_count = 0
    for doc_id in doc_ids:
        begin = int(doc_offsets[doc_id])
        end = int(doc_offsets[doc_id + 1])
        length = end - begin
        if selected and token_count + length > max_tokens:
            continue
        selected.append((begin, end))
        token_count += length
        if token_count >= max_tokens:
            break

    selected.sort()
    token_indices = np.empty(token_count, dtype=np.int64)
    sampled_offsets = np.empty(len(selected) + 1, dtype=np.int64)
    sampled_offsets[0] = 0
    cursor = 0
    for pos, (begin, end) in enumerate(selected):
        length = end - begin
        token_indices[cursor:cursor + length] = np.arange(
            begin,
            end,
            dtype=np.int64,
        )
        cursor += length
        sampled_offsets[pos + 1] = cursor
    return token_indices, sampled_offsets


def _onebit_f_error(
    vectors: np.ndarray,
    centers: np.ndarray,
    rotation: np.ndarray,
    batch_size: int = 100_000,
) -> np.ndarray:
    """Reproduce RaBitQ's one-bit IP data-side error for diagnostics."""
    result = np.empty(len(vectors), dtype=np.float32)
    dim = vectors.shape[1]
    for begin in range(0, len(vectors), batch_size):
        end = min(begin + batch_size, len(vectors))
        residual = np.asarray(
            vectors[begin:end] - centers[begin:end],
            dtype=np.float32,
        )
        residual = np.asarray(residual @ rotation, dtype=np.float32)
        l2_sqr = np.einsum("ij,ij->i", residual, residual, dtype=np.float64)
        l2_norm = np.sqrt(l2_sqr)
        ip_resi_xucb = 0.5 * np.sum(
            np.abs(residual),
            axis=1,
            dtype=np.float64,
        )
        ratio = np.ones_like(l2_sqr)
        valid = ip_resi_xucb > 0
        ratio[valid] = (
            l2_sqr[valid] * (0.25 * dim)
            / np.square(ip_resi_xucb[valid])
        )
        angular = np.sqrt(
            np.maximum(0.0, ratio - 1.0) / max(dim - 1, 1)
        )
        result[begin:end] = np.asarray(
            l2_norm * 1.9 * angular,
            dtype=np.float32,
        )
    return result


def _cluster_statistics(
    vectors: np.ndarray,
    assignments: np.ndarray,
    n_centroids: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = torch.zeros(n_centroids, dtype=torch.float64)
    vector_sum = torch.zeros(
        (n_centroids, vectors.shape[1]),
        dtype=torch.float64,
    )
    norm_sum = torch.zeros(n_centroids, dtype=torch.float64)
    batch_size = 200_000
    for begin in range(0, len(vectors), batch_size):
        end = min(begin + batch_size, len(vectors))
        batch_vectors = torch.from_numpy(
            np.ascontiguousarray(vectors[begin:end])
        ).to(torch.float64)
        batch_assignments = torch.from_numpy(
            np.asarray(assignments[begin:end], dtype=np.int64)
        )
        counts.add_(
            torch.bincount(
                batch_assignments,
                minlength=n_centroids,
            ).to(torch.float64)
        )
        vector_sum.index_add_(0, batch_assignments, batch_vectors)
        norm_sum.index_add_(
            0,
            batch_assignments,
            torch.sum(batch_vectors * batch_vectors, dim=1),
        )
    return counts.numpy(), vector_sum.numpy(), norm_sum.numpy()


def _reconstruction_mean_update(
    original_centroids: np.ndarray,
    vectors: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    """Return the fixed-assignment Euclidean MSE optimum for each cell."""
    n_centroids = len(original_centroids)
    counts, vector_sum, _ = _cluster_statistics(
        vectors,
        assignments,
        n_centroids,
    )
    updated = np.asarray(original_centroids, dtype=np.float64).copy()
    populated = counts > 0
    updated[populated] = vector_sum[populated] / counts[populated, None]
    return np.asarray(updated, dtype=np.float32)


def _isotropic_squared_update(
    original_centroids: np.ndarray,
    vectors: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    """Minimize sum ||x-c||^2 (1+||c||^2) for fixed assignments."""
    n_centroids = len(original_centroids)
    counts, vector_sum, norm_sum = _cluster_statistics(
        vectors,
        assignments,
        n_centroids,
    )
    updated = np.asarray(original_centroids, dtype=np.float64).copy()
    populated = counts > 0
    means = np.zeros_like(updated)
    means[populated] = (
        vector_sum[populated] / counts[populated, None]
    )
    mean_norm_sq = np.einsum("ij,ij->i", means, means)
    variance = np.zeros(n_centroids, dtype=np.float64)
    variance[populated] = np.maximum(
        0.0,
        norm_sum[populated] / counts[populated]
        - mean_norm_sq[populated],
    )

    t = np.ones(n_centroids, dtype=np.float64)
    for _ in range(20):
        f = (
            2.0 * mean_norm_sq * t**3
            - 3.0 * mean_norm_sq * t**2
            + (mean_norm_sq + variance + 1.0) * t
            - 1.0
        )
        derivative = (
            6.0 * mean_norm_sq * t**2
            - 6.0 * mean_norm_sq * t
            + mean_norm_sq
            + variance
            + 1.0
        )
        step = np.divide(
            f,
            derivative,
            out=np.zeros_like(f),
            where=derivative != 0,
        )
        t = np.clip(t - step, 0.0, 1.0)
    updated[populated] = means[populated] * t[populated, None]
    return np.asarray(updated, dtype=np.float32)


def _diagnostics(
    vectors: np.ndarray,
    assignments: np.ndarray,
    centroids: np.ndarray,
    doc_offsets: np.ndarray,
    rotation: np.ndarray,
) -> Dict[str, float]:
    assigned_centers = centroids[assignments]
    residual = vectors - assigned_centers
    residual_sq = np.einsum("ij,ij->i", residual, residual)
    center_norm_sq = np.einsum(
        "ij,ij->i",
        assigned_centers,
        assigned_centers,
    )
    iso_sq = residual_sq * (1.0 + center_norm_sq)
    f_error = _onebit_f_error(vectors, assigned_centers, rotation)
    f_iso_sq = np.square(f_error) * (1.0 + center_norm_sq)
    doc_p95 = []
    for doc_id in range(len(doc_offsets) - 1):
        begin = int(doc_offsets[doc_id])
        end = int(doc_offsets[doc_id + 1])
        if end > begin:
            doc_scores = f_iso_sq[begin:end]
            doc_p95.append(float(np.quantile(doc_scores, 0.95)))
    return {
        "iso_squared_mean": float(np.mean(iso_sq)),
        "doc_f_iso_squared_p95_mean": float(np.mean(doc_p95)),
        "centroid_norm_mean": float(
            np.mean(np.linalg.norm(centroids, axis=1))
        ),
    }


def refine_centroids(
    vectors: np.ndarray,
    assignments: np.ndarray,
    doc_offsets: np.ndarray,
    centroids: np.ndarray,
    rotation: np.ndarray,
    config: CentroidRefineConfig,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """Refine centers on all vectors and return sampled diagnostics."""
    diagnostic_indices, diagnostic_doc_offsets = _sample_complete_documents(
        doc_offsets,
        _DIAGNOSTIC_MAX_TOKENS,
        config.seed,
    )
    diagnostic_vectors = np.asarray(
        vectors[diagnostic_indices],
        dtype=np.float32,
    )
    diagnostic_assignments = np.asarray(
        assignments[diagnostic_indices],
        dtype=np.int64,
    )

    before = _diagnostics(
        diagnostic_vectors,
        diagnostic_assignments,
        centroids,
        diagnostic_doc_offsets,
        rotation,
    )

    if config.objective == "plaid":
        refined = np.asarray(centroids, dtype=np.float32).copy()
    elif config.objective == "kmeans":
        refined = _reconstruction_mean_update(
            centroids,
            vectors,
            assignments,
        )
    elif config.objective == "iso_squared":
        refined = _isotropic_squared_update(
            centroids,
            vectors,
            assignments,
        )
    else:
        raise ValueError(
            f"unknown centroid objective: {config.objective}"
        )

    after = _diagnostics(
        diagnostic_vectors,
        diagnostic_assignments,
        refined,
        diagnostic_doc_offsets,
        rotation,
    )
    return refined, {
        "before": before,
        "after": after,
        "sample": {
            "training_tokens": float(len(vectors)),
            "training_documents": float(len(doc_offsets) - 1),
            "diagnostic_tokens": float(len(diagnostic_vectors)),
            "diagnostic_documents": float(len(diagnostic_doc_offsets) - 1),
        },
    }
