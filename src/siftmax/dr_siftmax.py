"""Two-stage dense retrieval and RaBitQ-bounded ColBERT reranking."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
n_thread = 32
os.environ["OMP_NUM_THREADS"] = f"{n_thread}"
os.environ["MKL_NUM_THREADS"] = f"{n_thread}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_thread}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_thread}"

import argparse
import gc
import pickle
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(n_thread)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
from tqdm import tqdm
import rabitq_official_ext
print(rabitq_official_ext.__file__)

seed = 42
from common.dense_retrieval import DENSE_MODEL_NAME, DenseRetriever
from common.data_load import load_dataset_by_name
from common.utils import (
    BASE_DOC_VEC_CNT,
    TOP_K_LIST,
    build_inverse_rotation_matrix,
    compute_retrieval_metrics,
    configure_logger,
    ensure_rabitq_extension_support,
)
from common.colbert_embeddings import ColbertRetriever
from bound_aware_centroid import CentroidRefineConfig, refine_centroids

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two phase retrieval: Dense + RaBitQ bounded candidate-only reranking")
    parser.add_argument("--dataset-id", type=str, default="lotte/science")
    parser.add_argument("--checkpoint", type=str, default="colbert-ir/colbertv2.0")

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--doc-maxlen", type=int, default=BASE_DOC_VEC_CNT)
    parser.add_argument("--query-maxlen", type=int, default=32)
    parser.add_argument("--top-k-list", type=int, nargs="+", default=TOP_K_LIST)
    parser.add_argument("--load-doc-embs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-query-embs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--bsize", type=int, default=128)
    parser.add_argument("--dense-model-name", type=str, default=DENSE_MODEL_NAME)
    parser.add_argument("--faiss-index-type", type=str, default="flat", choices=["flat"])
    parser.add_argument("--n-candidate-list", type=int, nargs="+", default=[200, 500, 1000, 1500, 2000, 3000, 4000])
    parser.add_argument("--rabitq-total-bits", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--centroid-objective", type=str, default="iso_squared",
                        choices=["plaid", "kmeans", "iso_squared"])
    
    parser.add_argument("--plaid-index-path", type=str, default=None,
                        help="Use this PLAID index when no complete centroid artifact exists.")
    parser.add_argument("--alpha-geometry", type=float, default=0.1,
                        help="Scale the query-geometry bound; 1.0 keeps safe.")
    parser.add_argument("--alpha-geometry-list", type=float, nargs="+", default=None,
                        help="Evaluate multiple values instead of --alpha-geometry.")
    parser.add_argument("--base-ranking-dir", type=str, default=None,
                        help="Load the topk_N.pkl from this directory.")
    parser.add_argument("--result-suffix", type=str, default=time.strftime("%m%d"))

    args = parser.parse_args()
    alpha_values = args.alpha_geometry_list or [args.alpha_geometry]
    if any(not 0.0 < alpha <= 1.0 for alpha in alpha_values):
        parser.error("alpha geometry values must be in (0, 1]")
    if not args.top_k_list or any(k <= 0 for k in args.top_k_list):
        parser.error("--top-k-list values must be positive")
    if not args.n_candidate_list or any(n <= 0 for n in args.n_candidate_list):
        parser.error("--n-candidate-list values must be positive")
    if min(args.n_candidate_list) < max(args.top_k_list):
        parser.error("every --n-candidate-list value must be >= max(--top-k-list)")
    args.alpha_geometry_list = alpha_values
    return args


def _floor_power_of_two(value: float) -> int:
    """Return the largest power of two not greater than a positive value."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError("value must be finite and positive")
    return 1 << int(np.floor(np.log2(value)))


def _load_centroid_artifact(
    path: str,
    target_ncentroids: int,
    n_valid: int,
    dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate a complete centroid/assignment artifact."""
    with np.load(path, allow_pickle=False) as cached:
        missing = {"centroids", "centroid_ids"}.difference(cached.files)
        if missing:
            raise ValueError(f"centroid artifact is missing {sorted(missing)}: {path}")
        centroids = np.ascontiguousarray(cached["centroids"], dtype=np.float32)
        centroid_ids = np.ascontiguousarray(cached["centroid_ids"], dtype=np.uint16)
    if centroids.shape != (target_ncentroids, dim):
        raise ValueError(f"Cached centroid shape is invalid: {centroids.shape}: {path}")
    if centroid_ids.shape != (n_valid,):
        raise ValueError(f"Cached assignment shape is invalid: {centroid_ids.shape}: {path}")
    if centroid_ids.size and int(centroid_ids.max()) >= target_ncentroids:
        raise ValueError(f"Cached centroid assignments are outside the centroid range: {path}")
    return centroids, centroid_ids


def _atomic_save_centroid_artifact(
    path: str,
    centroids: np.ndarray,
    centroid_ids: np.ndarray,
    source: str,
) -> None:
    """Atomically save a complete centroid artifact."""
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "wb") as handle:
        np.savez(
            handle,
            centroids=centroids,
            centroid_ids=centroid_ids,
            source=np.asarray(source),
        )
    os.replace(temporary_path, path)


def _sorted_plaid_code_paths(plaid_index_path: str) -> List[str]:
    """Return numeric PLAID code chunks in token order."""
    chunks = []
    for filename in os.listdir(plaid_index_path):
        if not filename.endswith(".codes.pt"):
            continue
        chunk_id = filename[: -len(".codes.pt")]
        if chunk_id.isdigit():
            chunks.append((int(chunk_id), os.path.join(plaid_index_path, filename)))
    return [path for _, path in sorted(chunks)]


def _coarsen_plaid_centroids(
    plaid_index_path: str,
    target_ncentroids: int,
    expected_plaid_ncentroids: int,
    n_valid: int,
    dim: int,
    logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """Coarsen PLAID cells and remap their token assignments exactly."""
    centroids_path = os.path.join(plaid_index_path, "centroids.pt")
    code_paths = _sorted_plaid_code_paths(plaid_index_path)
    if not os.path.isfile(centroids_path) or not code_paths:
        raise FileNotFoundError(f"incomplete PLAID centroid artifacts under: {plaid_index_path}")

    plaid_centroids = np.ascontiguousarray(
        torch.load(centroids_path, map_location="cpu").numpy(), dtype=np.float32
    )
    if plaid_centroids.shape != (expected_plaid_ncentroids, dim):
        raise ValueError(f"PLAID centroid shape is incompatible: {plaid_centroids.shape}; expected {(expected_plaid_ncentroids, dim)}: {centroids_path}")

    plaid_ids = np.empty(n_valid, dtype=np.uint16)
    offset = 0
    for code_path in code_paths:
        codes = torch.load(code_path, map_location="cpu").reshape(-1).numpy()
        end = offset + len(codes)
        if end > n_valid:
            raise ValueError(f"PLAID code chunks contain more than {n_valid} token IDs")
        if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= len(plaid_centroids)):
            raise ValueError(f"PLAID centroid IDs are outside the valid range: {code_path}")
        plaid_ids[offset:end] = codes
        offset = end
    if offset != n_valid:
        raise ValueError(f"PLAID code chunks contain {offset} token IDs, expected {n_valid}: {plaid_index_path}")

    import faiss

    logger.info("Coarsening PLAID centroids: %d -> %d cells", len(plaid_centroids), target_ncentroids)
    kmeans = faiss.Kmeans(
        dim, target_ncentroids, niter=4, nredo=1, seed=seed, verbose=True,
        min_points_per_centroid=1, max_points_per_centroid=4,
    )
    kmeans.train(plaid_centroids)
    old_to_new = np.ascontiguousarray(kmeans.index.search(plaid_centroids, 1)[1][:, 0], dtype=np.uint16)
    old_to_new_i64 = old_to_new.astype(np.int64, copy=False)

    token_counts = np.bincount(
        plaid_ids.astype(np.int64, copy=False), minlength=len(plaid_centroids)
    ).astype(np.float64, copy=False)
    weighted_sums = np.zeros((target_ncentroids, dim), dtype=np.float64)
    np.add.at(weighted_sums, old_to_new_i64, plaid_centroids.astype(np.float64) * token_counts[:, None])
    coarse_counts = np.bincount(
        old_to_new_i64, weights=token_counts, minlength=target_ncentroids,
    )
    coarse_centroids = np.ascontiguousarray(kmeans.centroids, dtype=np.float32)
    nonempty = coarse_counts > 0
    coarse_centroids[nonempty] = (weighted_sums[nonempty] / coarse_counts[nonempty, None]).astype(np.float32)
    del kmeans, weighted_sums, token_counts, coarse_counts

    coarse_ids = np.empty(n_valid, dtype=np.uint16)
    for start in range(0, n_valid, 4_000_000):
        end = min(start + 4_000_000, n_valid)
        coarse_ids[start:end] = old_to_new[plaid_ids[start:end]]
    logger.info("Remapped %d PLAID token assignments to %d cells", n_valid, target_ncentroids)
    return coarse_centroids, coarse_ids


def _get_ivf4sqrt_centroids(
    doc_vectors: np.ndarray,
    build_cache_dir: str,
    logger,
    plaid_index_path: Optional[str] = None,
    assignment_batch_size: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve centroids through artifact, PLAID coarsening, or data K-means.

    The data K-means fallback uses the same FAISS training configuration as
    before. Its trained centers are checkpointed before assignment, and exact
    nearest-center assignment is performed in bounded batches.
    """
    logger.info("\n%s [Phase 2a]: centroid index artifacts %s", "*" * 10, "*" * 10)
    n_valid = len(doc_vectors)
    if n_valid == 0:
        raise ValueError("cannot build centroid index without valid document token vectors")

    plaid_ncentroids = _floor_power_of_two(16.0 * np.sqrt(n_valid))
    target_ncentroids = plaid_ncentroids // 4
    logger.info(
        "Computed centroid counts from %d vectors: plaid=%d ours=%d",
        n_valid,
        plaid_ncentroids,
        target_ncentroids,
    )
    if target_ncentroids > np.iinfo(np.uint16).max + 1:
        raise ValueError("uint16 centroid id layout supports at most 65536 centroids")
    if target_ncentroids > n_valid:
        raise ValueError(f"cannot train {target_ncentroids} centroids from only {n_valid} data vectors")

    os.makedirs(build_cache_dir, exist_ok=True)
    default_cache_path = os.path.join(
        build_cache_dir,
        f"rabitq_data_kmeans_centers{target_ncentroids}_tokens{n_valid}_seed{seed}.npz",
    )
    centers_checkpoint_path = os.path.join(
        build_cache_dir,
        f"rabitq_data_kmeans_centers{target_ncentroids}_tokens{n_valid}_seed{seed}.centroids.npy",
    )
    started = time.perf_counter()
    if os.path.isfile(default_cache_path):
        centroids_np, centroid_ids_np = _load_centroid_artifact(
            default_cache_path,
            target_ncentroids,
            n_valid,
            doc_vectors.shape[1],
        )
        logger.info(
            "Loaded centroid artifact: centers=%d elapsed=%.3fs path=%s",
            target_ncentroids,
            time.perf_counter() - started,
            default_cache_path,
        )
    elif plaid_index_path and os.path.isdir(plaid_index_path):
        centroids_np, centroid_ids_np = _coarsen_plaid_centroids(
            os.path.abspath(os.path.expanduser(plaid_index_path)),
            target_ncentroids,
            plaid_ncentroids,
            n_valid,
            doc_vectors.shape[1],
            logger,
        )
        _atomic_save_centroid_artifact(
            default_cache_path,
            centroids_np,
            centroid_ids_np,
            source="plaid_coarsened",
        )
        logger.info("Saved PLAID-coarsened centroid artifact: %s", default_cache_path)
    else:
        if plaid_index_path:
            logger.warning("PLAID index not found; continuing to data K-means: %s", plaid_index_path)
        logger.info("Building data-vector centroid index: vectors=%d centers=%d", n_valid, target_ncentroids)
        import faiss

        if os.path.isfile(centers_checkpoint_path):
            centroids_np = np.ascontiguousarray(
                np.load(centers_checkpoint_path, allow_pickle=False), dtype=np.float32
            )
            if centroids_np.shape != (target_ncentroids, doc_vectors.shape[1]):
                raise ValueError(
                    f"Checkpointed centroid shape is invalid: {centroids_np.shape}: "
                    f"{centers_checkpoint_path}"
                )
            logger.info("Resuming assignment from trained centroids: %s", centers_checkpoint_path)
        else:
            kmeans = faiss.Kmeans(
                doc_vectors.shape[1], target_ncentroids, niter=4, nredo=1,
                seed=seed, verbose=True, min_points_per_centroid=1,
                max_points_per_centroid=256,
            )
            kmeans.train(doc_vectors)
            centroids_np = np.ascontiguousarray(kmeans.centroids, dtype=np.float32)
            np.save(centers_checkpoint_path, centroids_np, allow_pickle=False)
            logger.info("Checkpointed trained centroids before full assignment: %s", centers_checkpoint_path)
            del kmeans
            gc.collect()

        assignment_index = faiss.IndexFlatL2(doc_vectors.shape[1])
        assignment_index.add(centroids_np)
        centroid_ids_np = np.empty(n_valid, dtype=np.uint16)
        logger.info(
            "Assigning %d vectors exactly in batches of %d",
            n_valid,
            assignment_batch_size,
        )
        for start in tqdm(
            range(0, n_valid, assignment_batch_size),
            desc="Centroid assignment",
        ):
            end = min(start + assignment_batch_size, n_valid)
            _, batch_assignment = assignment_index.search(doc_vectors[start:end], 1)
            centroid_ids_np[start:end] = batch_assignment[:, 0]
        del assignment_index

        _atomic_save_centroid_artifact(
            default_cache_path,
            centroids_np,
            centroid_ids_np,
            source="data_kmeans",
        )
        logger.info("Built and saved data-vector centroid index: centers=%d elapsed=%.3fs path=%s", target_ncentroids, time.perf_counter() - started, default_cache_path)
    return centroids_np, centroid_ids_np

@dataclass
class RaBitQTokenStore:
    """Store RaBitQ-encoded document tokens and perform bounded reranking."""

    onebit_batch_data: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    ex_data: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    doc_token_offsets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    doc_32tok_batch_offsets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    centroid_ids_compact: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint16))
    doc_hit_centroids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint16))
    doc_centroid_offsets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    centroids_rot: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    centroid_norm_sq: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    inverse_rotation_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    build_cache_dir: str = ""
    dim: int = 0
    total_bits: int = 1
    alpha_geometry: float = 1.0
    centroid_objective: str = "plaid"
    plaid_index_path: Optional[str] = None

    def build(
        self,
        doc_vecs_valid: np.ndarray,
        doc_masks: np.ndarray,
        logger,
    ) -> None:
        """Build the CPU-backed RaBitQ token store from compact valid vectors."""
        ensure_rabitq_extension_support()
        if self.dim % 64 != 0:
            raise ValueError("RaBitQ FastScan/ex-code wrapper expects dim % 64 == 0")
        if doc_vecs_valid.ndim != 2 or doc_vecs_valid.shape[1] != self.dim:
            raise ValueError(f"expected compact document vectors with shape (n, {self.dim}), got {doc_vecs_valid.shape}")
        if doc_vecs_valid.dtype != np.float32 or not doc_vecs_valid.flags.c_contiguous:
            raise ValueError("compact document vectors must be C-contiguous float32")
        self.doc_token_offsets = np.empty(doc_masks.shape[0] + 1, dtype=np.int64)
        self.doc_token_offsets[0] = 0
        np.cumsum(doc_masks.sum(axis=1, dtype=np.int64), out=self.doc_token_offsets[1:])
        if int(self.doc_token_offsets[-1]) != len(doc_vecs_valid):
            raise ValueError(f"document masks contain {int(self.doc_token_offsets[-1])} valid tokens, but got {len(doc_vecs_valid)} compact vectors")
        self.inverse_rotation_matrix = np.ascontiguousarray(build_inverse_rotation_matrix(self.dim, seed).numpy(), dtype=np.float32)
        logger.info("Built global inverse rotation matrix, dim=%d, seed=%d", self.dim, seed)

        centroids_np, centroid_ids_compact = _get_ivf4sqrt_centroids(
            doc_vecs_valid,
            self.build_cache_dir,
            logger,
            plaid_index_path=self.plaid_index_path,
        )
        self.centroid_ids_compact = np.ascontiguousarray(centroid_ids_compact, dtype=np.uint16)
        if self.centroid_objective == "plaid":
            centroid_norms = np.linalg.norm(centroids_np, axis=1, keepdims=True)
            centroids_np = np.divide(
                centroids_np,
                centroid_norms,
                out=np.zeros_like(centroids_np),
                where=centroid_norms > 0,
            )
            logger.info("Applied PLAID objective: normalized %d base centroids", len(centroids_np))
        elif self.centroid_objective == "kmeans":
            logger.info("Applied k-means objective: keeping data-vector k-means centroids")
        else:
            centroid_artifact = (
                f"boundaware_centroids_{self.centroid_objective}_centers{len(centroids_np)}_"
                f"tokens{len(doc_vecs_valid)}_alltokens_seed{seed}.npy"
            )
            os.makedirs(self.build_cache_dir, exist_ok=True)
            centroid_artifact_path = os.path.join(self.build_cache_dir, centroid_artifact)
            if os.path.isfile(centroid_artifact_path):
                cached_centroids = np.load(centroid_artifact_path, allow_pickle=False)
                if cached_centroids.shape != centroids_np.shape:
                    raise ValueError(f"Cached centroid shape {cached_centroids.shape} does not match expected {centroids_np.shape}: {centroid_artifact_path}")
                centroids_np = np.ascontiguousarray(cached_centroids, dtype=np.float32)
                logger.info(f"Loaded refined centroids from {centroid_artifact_path}")
            else:
                logger.info(f"Refining base centroids with all {len(doc_vecs_valid)} document token vectors; objective={self.centroid_objective}")
                config = CentroidRefineConfig(
                    objective=self.centroid_objective,
                    seed=seed,
                )
                refined_centroids, centroid_diagnostics = refine_centroids(
                    doc_vecs_valid, self.centroid_ids_compact, self.doc_token_offsets,
                    centroids_np, self.inverse_rotation_matrix, config,
                )
                centroids_np = np.ascontiguousarray(refined_centroids, dtype=np.float32)
                np.save(centroid_artifact_path, centroids_np)
                logger.info(f"Centroid offline diagnostics before={centroid_diagnostics['before']} after={centroid_diagnostics['after']} sample={centroid_diagnostics['sample']}")
                logger.info(f"Saved bound-aware centroids to {centroid_artifact_path}")
        self.centroids_rot = np.ascontiguousarray(centroids_np @ self.inverse_rotation_matrix, dtype=np.float32)
        doc_unique_centroids = [
            np.unique(self.centroid_ids_compact[self.doc_token_offsets[doc_id]:self.doc_token_offsets[doc_id + 1]])
            for doc_id in range(doc_masks.shape[0])
        ]
        self.doc_centroid_offsets = np.empty(doc_masks.shape[0] + 1, dtype=np.int64)
        self.doc_centroid_offsets[0] = 0
        np.cumsum(
            np.fromiter((len(centroids) for centroids in doc_unique_centroids), dtype=np.int64, count=doc_masks.shape[0]),
            out=self.doc_centroid_offsets[1:],
        )
        self.doc_hit_centroids = np.ascontiguousarray(
            np.concatenate(doc_unique_centroids) if doc_unique_centroids else np.empty(0, dtype=np.uint16),
            dtype=np.uint16,
        )
        logger.info(
            "Built per-document hit centroids: avg_per_doc=%.2f",
            self.doc_hit_centroids.size / max(doc_masks.shape[0], 1),
        )
        self.centroid_norm_sq = np.ascontiguousarray(
            np.sum(self.centroids_rot * self.centroids_rot, axis=1, dtype=np.float32),
            dtype=np.float32,
        )

        n_valid = len(doc_vecs_valid)
        encode_batch_size = 200_000
        logger.info(f"Rotating {n_valid} valid document token vectors in batches")
        for start in tqdm(range(0, n_valid, encode_batch_size), desc="RaBitQ Rotation"):
            end = min(start + encode_batch_size, n_valid)
            doc_vecs_valid[start:end] = doc_vecs_valid[start:end] @ self.inverse_rotation_matrix

        logger.info("Encoding and packing valid document token vectors with RaBitQ")
        self.onebit_batch_data, self.doc_32tok_batch_offsets, self.ex_data = rabitq_official_ext.encode_pack_rabitq(
            doc_vecs_valid,
            self.centroids_rot,
            self.centroid_ids_compact,
            self.doc_token_offsets,
            self.total_bits,
            self.dim,
            "ip",
            encode_batch_size=encode_batch_size,
        )

    def rerank_topk(
        self,
        q_np: np.ndarray,
        doc_idx: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Run bounded pruning, n-bit refinement, and top-k selection in C++."""
        doc_idx_scan = np.ascontiguousarray(doc_idx, dtype=np.int64)
        sorted_positions, stats = rabitq_official_ext.rerank_topk_rabitq(
            q_np,
            doc_idx_scan,
            self.onebit_batch_data,
            self.ex_data,
            self.doc_token_offsets,
            self.doc_32tok_batch_offsets,
            self.centroids_rot,
            self.centroid_norm_sq,
            self.centroid_ids_compact,
            self.doc_hit_centroids,
            self.doc_centroid_offsets,
            self.total_bits,
            self.dim,
            top_k,
            self.alpha_geometry,
        )
        return np.ascontiguousarray(sorted_positions, dtype=np.int64), stats

    def get_index_stats(self, logger) -> float:
        """Report component sizes and return the total index size in MiB."""
        size_items = {
            "onebit_batch_data": self.onebit_batch_data.nbytes,
            "ex_data": self.ex_data.nbytes,
            "doc_offsets": self.doc_token_offsets.nbytes,
            "doc_32tok_batch_offsets": self.doc_32tok_batch_offsets.nbytes,
            "centroid_ids": self.centroid_ids_compact.nbytes,
            "doc_hit_centroids": self.doc_hit_centroids.nbytes,
            "doc_centroid_offsets": self.doc_centroid_offsets.nbytes,
            "centroids_rot": self.centroids_rot.nbytes,
            "centroid_norm_sq": self.centroid_norm_sq.nbytes,
            "inverse_rotation_matrix": self.inverse_rotation_matrix.nbytes,
        }

        for name, size_bytes in size_items.items():
            logger.info(f"[Index Size] {name}: {size_bytes / (1024 ** 2):.2f} MB")
        return sum(size_items.values()) / (1024 ** 2)


def _normalize_doc_masks(doc_masks: torch.Tensor, n_docs: int, doc_len: int) -> np.ndarray:
    """Normalize document masks to a contiguous ``(n_docs, doc_len)`` array."""
    masks = doc_masks.detach().cpu().bool()
    if masks.shape == (n_docs, doc_len, 1):
        masks = masks.squeeze(-1)
    if masks.shape != (n_docs, doc_len):
        raise ValueError(f"got doc_mask with shape {tuple(doc_masks.shape)}")
    return np.ascontiguousarray(masks.numpy().astype(bool, copy=False), dtype=bool)


def batch_rabitq_bound_candidates(
    phase1_candidate_indices: Dict[str, np.ndarray],
    query_ids: List[str],
    q_vecs_rot: np.ndarray,
    token_store: RaBitQTokenStore,
    doc_ids: Sequence[str],
    n_candidate: int,
    top_k: int,
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """Rerank all query candidates and aggregate RaBitQ runtime statistics."""
    results: Dict[str, List[str]] = {}
    total_1bit_score_time = 0.0
    total_query_lut_build_time = 0.0
    total_query_sim_precompute_time = 0.0
    total_collect_used_centroids_time = 0.0
    total_precompute_q_center_time = 0.0
    total_select_survivors_time = 0.0
    total_1bit_pruned_doc_count = 0
    total_nbit_refine_time = 0.0
    total_nbit_pair_skip_ratio = 0.0
    total_nbit_query_early_pruned_doc_count = 0
    total_nbit_doc_count = 0
    total_nbit_refined_doc_count = 0
    total_onebit_fastscan_reduction_ratio = 0.0
    total_1bit_query_skip_doc_ratio = 0.0

    search_start = time.perf_counter()
    for q_pos, qid in enumerate(tqdm(query_ids, desc="Phase 2 RaBitQ Bound Reranking")):
        q_vec_rot = q_vecs_rot[q_pos]
        cand_doc_indices = phase1_candidate_indices.get(qid)

        if cand_doc_indices is None or len(cand_doc_indices) == 0:
            results[qid] = []
            continue

        sorted_doc_idx, stats = token_store.rerank_topk(
            q_vec_rot,
            cand_doc_indices,
            top_k,
        )

        results[qid] = [doc_ids[cand_doc_indices[i]] for i in sorted_doc_idx[:top_k]]

        total_1bit_score_time += stats["1bit_score_time_s"]
        total_query_lut_build_time += stats.get("query_lut_build_time_s", 0.0)
        total_query_sim_precompute_time += stats.get("query_sim_precompute_time_s", 0.0)
        total_collect_used_centroids_time += stats.get("collect_used_centroids_time_s", 0.0)
        total_precompute_q_center_time += stats.get("precompute_q_center_time_s", 0.0)
        total_select_survivors_time += stats.get("select_survivors_time_s", 0.0)
        total_nbit_refine_time += stats["nbit_refine_time_s"]
        total_1bit_pruned_doc_count += int(stats["1bit_pruned_doc_count"])
        total_nbit_pair_skip_ratio += float(stats.get("nbit_pair_skip_ratio", 0.0))
        total_nbit_query_early_pruned_doc_count += int(stats.get("nbit_query_early_pruned_doc_count", 0))
        total_nbit_doc_count += int(stats.get("nbit_doc_count", 0))
        total_nbit_refined_doc_count += int(stats.get("nbit_refined_doc_count", 0))
        total_onebit_fastscan_reduction_ratio += float(stats.get("onebit_fastscan_reduction_ratio", 0.0))
        total_1bit_query_skip_doc_ratio += float(stats.get("1bit_query_skip_doc_ratio", 0.0))

    total_wall_time = time.perf_counter() - search_start
    num_queries = len(query_ids)
    nbit_unrefined_doc_ratio = (total_nbit_doc_count - total_nbit_refined_doc_count) / total_nbit_doc_count
    return results, {
        "avg_search_time_ms": (total_wall_time / num_queries) * 1000,
        "avg_1bit_score_ms": (total_1bit_score_time / num_queries) * 1000,
        "avg_query_lut_build_ms": (total_query_lut_build_time / num_queries) * 1000,
        "avg_query_sim_precompute_ms": (total_query_sim_precompute_time / num_queries) * 1000,
        "avg_collect_used_centroids_ms": (total_collect_used_centroids_time / num_queries) * 1000,
        "avg_precompute_q_center_ms": (total_precompute_q_center_time / num_queries) * 1000,
        "avg_select_survivors_ms": (total_select_survivors_time / num_queries) * 1000,
        "avg_nbit_refine_ms": (total_nbit_refine_time / num_queries) * 1000,
        "1bit_pruned_doc_ratio": total_1bit_pruned_doc_count / (n_candidate * num_queries),
        "nbit_pair_skip_ratio": total_nbit_pair_skip_ratio / num_queries,
        "nbit_query_early_pruned_doc_count": total_nbit_query_early_pruned_doc_count,
        "avg_nbit_doc_count": total_nbit_doc_count / num_queries,
        "avg_nbit_refined_doc_count": total_nbit_refined_doc_count / num_queries,
        "nbit_unrefined_doc_ratio": nbit_unrefined_doc_ratio,
        "onebit_fastscan_reduction_ratio": total_onebit_fastscan_reduction_ratio / num_queries,
        "1bit_query_skip_doc_ratio": total_1bit_query_skip_doc_ratio / num_queries,
    }


def build_phase1_candidate_indices(
    phase1_results: Dict[str, List[str]],
    doc_id_to_idx: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """Map phase-one document IDs to RaBitQ token-store indices."""
    phase1_candidate_indices: Dict[str, np.ndarray] = {}
    for qid, doc_id_list in phase1_results.items():
        indices = [doc_id_to_idx[did] for did in doc_id_list if did in doc_id_to_idx]
        phase1_candidate_indices[qid] = np.asarray(indices, dtype=np.int64)
    return phase1_candidate_indices


def main() -> None:
    args = parse_args()
    dataset_stub = args.dataset_id.replace("/", "_")
    if args.log_file is None:
        args.log_file = f"./logs/{dataset_stub}/two_phase_rabitq_bound_pipeline_{time.strftime('%m%d')}.log"
    logger = configure_logger("two_phase_rabitq_bound", args.log_file)
    logger.info(f"n threads: {n_thread} \nArguments: {args}")
    logger.info("RaBitQ scoring: official IP-distance score bounds with progressive extra-code refine")

    results_dir = f"./results_ours/{dataset_stub}"
    csv_path = os.path.join(
        results_dir,
        f"two_phase_rabitq_{args.rabitq_total_bits}bit_bound_c_{args.result_suffix}.csv",
    )
    os.makedirs(results_dir, exist_ok=True)
    data_query_emb_dir = f"./embedding_files/{dataset_stub}/colbert_official_data_query_emb_tensor/"
    os.makedirs(data_query_emb_dir, exist_ok=True)

    corpus, queries, qrels = load_dataset_by_name(args.dataset_id, logger)

    # Dense retrieval.
    logger.info(f"\n{'*' * 10} [Phase 1]: Dense Retrieval {'*' * 10} \n[Initializing Dense Retriever] ({args.dense_model_name}, {args.faiss_index_type})")
    dense_retriever = DenseRetriever(args.dense_model_name, args.faiss_index_type, args.device)
    dense_retriever.index(args.dataset_id, None, corpus)
    dense_index_size = os.path.getsize(dense_retriever.index_cache_path) / (1024 ** 2)
    logger.info(f"[encoding time]: {dense_retriever.encode_time:.2f}s, [index build time]: {dense_retriever.index_build_time:.2f}s, [dense vector dim]: {dense_retriever.embedding_dim}")
    logger.info(f"[Dense Index Size]: {dense_index_size:.2f} MB")

    max_top_k = max(args.top_k_list)
    dense_query_ids, dense_query_embeddings = dense_retriever.load_query_embeddings(
        queries
    )
    # MaxSim reranking.
    logger.info(f"\n{'*' * 10} [Phase 2]: Official RaBitQ bound candidate-only multi-vector reranking {'*' * 10} \n[Init ColBERT Encoder] ({args.checkpoint})")
    colbert_retriever = ColbertRetriever(
        checkpoint=args.checkpoint,
        device=args.device,
        doc_maxlen=args.doc_maxlen,
        query_maxlen=args.query_maxlen,
        bsize=args.bsize,
    )

    siftmax_build_cache_dir = os.path.join(
        "indexes",
        f"siftmax_{dataset_stub}_b{args.rabitq_total_bits}_{args.centroid_objective}_build_cache",
    )
    checkpoint_stub = args.checkpoint.replace("/", "_")
    default_plaid_index_path = os.path.join(
        "plaid_exp",
        f"plaid_test_{dataset_stub}",
        "indexes",
        (
            f"plaid_test_{dataset_stub}_{checkpoint_stub}_docmax{args.doc_maxlen}_"
            f"nbits{args.rabitq_total_bits}_kmeans4_full"
        ),
    )
    plaid_index_path = args.plaid_index_path
    if plaid_index_path is None and os.path.isdir(default_plaid_index_path):
        plaid_index_path = default_plaid_index_path
    t0 = time.perf_counter()
    doc_ids, base_embeddings, base_masks = colbert_retriever.get_doc_embedding(
        corpus,
        data_query_emb_dir,
        logger,
        use_cache=args.load_doc_embs,
    )
    multi_vec_encode_time = time.perf_counter() - t0
    logger.info("[Official ColBERT Document Encode/Load Time]: %.2fs", multi_vec_encode_time)

    query_ids, query_tensors = colbert_retriever.get_query_embedding(
        queries,
        data_query_emb_dir,
        logger,
        use_cache=args.load_query_embs,
    )
    query_vecs_np = query_tensors.detach().cpu().numpy().astype("float32", copy=False)

    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    if args.base_ranking_dir is not None:
        base_ranking_dir = os.path.abspath(args.base_ranking_dir)
        candidates = []
        if os.path.isdir(base_ranking_dir):
            for filename in os.listdir(base_ranking_dir):
                match = re.fullmatch(r"topk_(\d+)\.pkl", filename)
                if match and int(match.group(1)) >= max_top_k:
                    candidates.append((int(match.group(1)), filename))
        if not candidates:
            raise FileNotFoundError(
                f"No topk_N.pkl with N >= {max_top_k} found in base ranking "
                f"directory: {base_ranking_dir}"
            )
        _, base_rankings_filename = min(candidates)
        selected_ranking_path = os.path.join(base_ranking_dir, base_rankings_filename)
        logger.info("Loading explicitly configured base rankings from %s", selected_ranking_path)
        with open(selected_ranking_path, "rb") as handle:
            base_rankings = pickle.load(handle)
    else:
        base_rankings = colbert_retriever.get_base_rankings(
            dataset_id=args.dataset_id,
            max_k=max_top_k,
            use_full_data=True,
            logger=logger,
        )

    t0 = time.perf_counter()
    n_docs, doc_len, dim = base_embeddings.shape
    doc_masks = _normalize_doc_masks(base_masks, n_docs, doc_len)
    token_store_kwargs = {
        "dim": dim,
        "total_bits": args.rabitq_total_bits,
        "alpha_geometry": args.alpha_geometry,
        "centroid_objective": args.centroid_objective,
        "plaid_index_path": plaid_index_path,
        "build_cache_dir": siftmax_build_cache_dir,
    }
    # Compact while the cached embeddings are still float16, then release the
    # padded tensor before allocating the float32 working array. RaBitQ packed
    # data is rebuilt on every run; only centroid artifacts may be reused.
    doc_vectors_np = base_embeddings.detach().cpu().numpy().reshape(-1, dim)
    compact_source = np.ascontiguousarray(doc_vectors_np[doc_masks.reshape(-1)])
    colbert_retriever.doc_embeddings = None
    colbert_retriever.doc_masks = None
    del doc_vectors_np, base_embeddings, base_masks
    gc.collect()
    doc_vecs_valid = np.ascontiguousarray(compact_source, dtype=np.float32)
    del compact_source
    gc.collect()
    logger.info(
        "Compacted %d valid document vectors before index build; source embeddings released",
        len(doc_vecs_valid),
    )
    token_store = RaBitQTokenStore(**token_store_kwargs)
    token_store.build(doc_vecs_valid, doc_masks, logger=logger)
    del doc_vecs_valid
    rabitq_build_time = time.perf_counter() - t0
    logger.info("[Centroid Count]: %d", token_store.centroids_rot.shape[0])
    index_size = token_store.get_index_stats(logger)
    logger.info(
        "[RaBitQ build time]: %.2fs, [Index Size]: %.2f MB",
        rabitq_build_time,
        index_size,
    )

    q_vecs_rot = np.empty_like(query_vecs_np, dtype=np.float32)
    query_rotation_start = time.perf_counter()
    for query_pos in range(len(query_ids)):
        q_vecs_rot[query_pos] = query_vecs_np[query_pos] @ token_store.inverse_rotation_matrix
    query_rotation_total = time.perf_counter() - query_rotation_start
    p2_query_rotate_ms = (
        query_rotation_total * 1000.0 / len(query_ids) if query_ids else 0.0
    )
    logger.info(
        "[Average Query Rotation Time (per query)]: %.3fms",
        p2_query_rotate_ms,
    )

    del corpus, colbert_retriever
    del query_vecs_np, query_tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results_rows = []
    for n_candidate in args.n_candidate_list:
        logger.info("\n--- Phase 1: slicing dense search on [%d] candidates ---", n_candidate)
        p1_results, p1_runtime = dense_retriever.search_encoded_queries(
            dense_query_ids, dense_query_embeddings, top_k=n_candidate
        )
        p1_can_doc_idx = build_phase1_candidate_indices(p1_results, doc_id_to_idx)
        logger.info("[Phase 1 Average Dense Retrieval Latency (per query)]: %.2fms", p1_runtime["query_search_ms"])
        p1_metrics = compute_retrieval_metrics(p1_results, qrels, n_candidate, base_rankings)

        for alpha_geometry, top_k in (
            (alpha, top_k)
            for alpha in args.alpha_geometry_list
            for top_k in args.top_k_list
        ):
            token_store.alpha_geometry = alpha_geometry
            logger.info("\n--- alpha_geometry=%.1f ---", alpha_geometry)
            logger.info(f"\n--- Phase 2: get {top_k} doc ---")
            # 1-bit filtering followed by b-bit refinement inside the MaxSim reranker.
            p2_results, p2_runtime = batch_rabitq_bound_candidates(
                p1_can_doc_idx,
                query_ids,
                q_vecs_rot,
                token_store,
                doc_ids,
                n_candidate,
                top_k,
            )
            p2_metrics = compute_retrieval_metrics(p2_results, qrels, top_k, base_rankings)
            logger.info("[Phase 2 Average Search Time (per query)]: %.2fms", p2_runtime["avg_search_time_ms"])
            logger.info("[Phase 2 Avg 1-bit Score Time]: %.2fms", p2_runtime["avg_1bit_score_ms"])
            logger.info("[Phase 2 Avg Query LUT Build Time]: %.2fms", p2_runtime["avg_query_lut_build_ms"])
            logger.info("[Phase 2 Avg Collect Used Centroids Time]: %.2fms", p2_runtime["avg_collect_used_centroids_ms"])
            logger.info("[Phase 2 Avg Precompute Q-Center Time]: %.2fms", p2_runtime["avg_precompute_q_center_ms"])
            logger.info("[Phase 2 Avg N-bit Refine Time]: %.2fms", p2_runtime["avg_nbit_refine_ms"])

            # Dense query vectors are precomputed; Phase 1 times only Faiss.
            total_latency_ms = (
                p1_runtime["query_search_ms"]
                + p2_query_rotate_ms
                + p2_runtime["avg_search_time_ms"]
            )

            row = {
                "method": "DR + SiftMax",
                "dataset": args.dataset_id,
                "n_candidate": n_candidate,
                "nbit": args.rabitq_total_bits,
                "alpha_geometry": token_store.alpha_geometry,
                "centroid_count": int(token_store.centroids_rot.shape[0]),
                "multi_index_size_MB": index_size,
                "dense_index_size_MB": dense_index_size,
                "total_index_size_MB": dense_index_size + index_size,
                "TOP_K": top_k,
                "rabitq_build_time_s": rabitq_build_time,
                "dense_index_build_time_s": dense_retriever.index_build_time,
                "dense_index_reused": dense_retriever.index_reused,
                "p1_avg_latency_ms": p1_runtime["query_search_ms"],
                "p2_query_rotate_ms": p2_query_rotate_ms,
                "p2_avg_latency_ms": p2_query_rotate_ms + p2_runtime["avg_search_time_ms"],
                "total_latency_ms": total_latency_ms,
                "total_QPS": 1000.0 / total_latency_ms if total_latency_ms > 0 else float("inf"),
                "phase2_avg_1bit_score_ms": p2_runtime["avg_1bit_score_ms"],
                "avg_query_lut_build_ms": p2_runtime["avg_query_lut_build_ms"],
                "avg_query_sim_precompute_ms": p2_runtime["avg_query_sim_precompute_ms"],
                "collect_used_centroids_ms": p2_runtime["avg_collect_used_centroids_ms"],
                "precompute_q_center_ms": p2_runtime["avg_precompute_q_center_ms"],
                "select_survivors_ms": p2_runtime["avg_select_survivors_ms"],
                "phase2_avg_nbit_refine_ms": p2_runtime["avg_nbit_refine_ms"],
                "1bit_pruned_doc_ratio": p2_runtime["1bit_pruned_doc_ratio"],
                "1bit_query_skip_doc_ratio": p2_runtime["1bit_query_skip_doc_ratio"],
                "onebit_fastscan_reduction_ratio": p2_runtime["onebit_fastscan_reduction_ratio"],
                "avg_nbit_doc_count": p2_runtime["avg_nbit_doc_count"],
                "nbit_pair_skip_ratio": p2_runtime["nbit_pair_skip_ratio"],
                "nbit_query_early_pruned_doc_count": p2_runtime["nbit_query_early_pruned_doc_count"],
                "avg_nbit_refined_doc_count": p2_runtime["avg_nbit_refined_doc_count"],
                "nbit_unrefined_doc_ratio": p2_runtime["nbit_unrefined_doc_ratio"],
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
