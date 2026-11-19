from __future__ import annotations

import csv
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import faiss
import numpy as np
import torch

from common.data_load import load_lotte_qrels_file
from common.utils import compute_retrieval_metrics


CPU_THREADS = 32
LEARNED_DIM = 2048


def set_cpu_threads(threads: int = CPU_THREADS) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    faiss.omp_set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def norm_ids(values: Iterable[object]) -> List[str]:
    return [str(value.item() if isinstance(value, np.generic) else value) for value in values]


def load_colbert_artifacts(root: Path):
    prefix_doc = root / "colbert_official_index_256vecs"
    prefix_query = root / "colbert_official_query_32embs"
    docs = torch.as_tensor(load_pickle(Path(f"{prefix_doc}_embs.pkl")))
    masks = torch.as_tensor(load_pickle(Path(f"{prefix_doc}_masks.pkl")))
    doc_ids = norm_ids(load_pickle(Path(f"{prefix_doc}_ids.pkl")))
    queries = torch.as_tensor(load_pickle(Path(f"{prefix_query}_embs.pkl")))
    query_ids = norm_ids(load_pickle(Path(f"{prefix_query}_ids.pkl")))
    if docs.ndim != 3 or docs.shape[0] != len(doc_ids) or docs.shape[1:] != (256, 128):
        raise ValueError(f"unexpected Science document artifacts: docs={tuple(docs.shape)}, ids={len(doc_ids)}")
    if queries.ndim != 3 or queries.shape != (538, 32, 128) or len(query_ids) != 538:
        raise ValueError(f"unexpected Science query artifacts: queries={tuple(queries.shape)}, ids={len(query_ids)}")
    if masks.shape[0] != docs.shape[0] or masks.shape[1] != docs.shape[1]:
        raise ValueError(f"document mask mismatch: docs={tuple(docs.shape)}, masks={tuple(masks.shape)}")
    return docs, masks, doc_ids, queries, query_ids


def load_qrels(path: Path) -> Dict[str, set[str]]:
    return load_lotte_qrels_file(str(path))


def load_reference(path: Path) -> Dict[str, List[str]]:
    raw = load_pickle(path)
    if isinstance(raw, dict) and "ranked_doc_ids" in raw:
        ranked = raw["ranked_doc_ids"]
        qids = raw.get("query_ids", range(len(ranked)))
        return {str(qid): norm_ids(row) for qid, row in zip(qids, ranked)}
    if not isinstance(raw, Mapping):
        raise TypeError(f"unsupported exact-reference format: {type(raw)}")
    return {str(qid): norm_ids(row) for qid, row in raw.items()}


def flatten_documents(docs: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if masks.ndim == 3 and masks.shape[-1] == 1:
        valid = masks.squeeze(-1).bool()
    elif masks.ndim == 2:
        valid = masks.bool()
    else:
        raise ValueError(f"unsupported mask shape: {tuple(masks.shape)}")
    counts = valid.sum(1).to(torch.int32).contiguous()
    flat = docs[valid].to(torch.float32).contiguous()
    if int(counts.sum()) != len(flat):
        raise RuntimeError("flattened document/count mismatch")
    return flat, counts


def import_lemur(lemur_repo: Path | None):
    if lemur_repo is not None:
        import sys
        resolved = str(lemur_repo.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    from lemur import Lemur
    from lemur.maxsim import MaxSim
    return Lemur, MaxSim


def prepare_lemur(
    index_dir: Path,
    device: str,
    lemur_repo: Path | None,
    docs: torch.Tensor,
    masks: torch.Tensor,
    rebuild: bool,
):
    Lemur, MaxSim = import_lemur(lemur_repo)
    index_dir.mkdir(parents=True, exist_ok=True)
    mlp_path, w_path = index_dir / "mlp.pt", index_dir / "w.pt"
    cached = mlp_path.exists() and w_path.exists() and not rebuild
    training_included = rebuild or not mlp_path.exists()
    lemur = Lemur(index=index_dir, device=device)
    build_start = time.perf_counter()
    flat = counts = None
    if cached:
        lemur.load_mlp(mlp_path)
        lemur.load_w(w_path)
        build_s = 0.0
    else:
        flat, counts = flatten_documents(docs, masks)
        lemur.fit(
            train=flat,
            train_counts=counts,
            learn=flat,
            learn_counts=counts,
            epochs=10,
            train_subset_size=8192,
            learn_subset_size=100000,
            ols_sample_size=16384,
            hidden_dim=512,
            final_hidden_dim=LEARNED_DIM,
            num_layers=1,
            activation="gelu",
            batch_size=512,
            lr=3e-3,
            grad_clip=0.5,
            force_retrain=rebuild,
            verbose=True,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        build_s = time.perf_counter() - build_start
    W = lemur.W.detach().cpu().to(torch.float32).contiguous().numpy()
    if W.shape != (len(docs), LEARNED_DIM):
        raise ValueError(f"unexpected LEMUR W shape: {W.shape}")
    return lemur, MaxSim, W, flat, counts, build_s, cached, training_included


def create_maxsim(MaxSim, docs: torch.Tensor, masks: torch.Tensor, flat=None, counts=None):
    if flat is None or counts is None:
        flat, counts = flatten_documents(docs, masks)
    train = np.ascontiguousarray(flat.numpy(), dtype=np.float32)
    count_array = np.ascontiguousarray(counts.numpy(), dtype=np.int32)
    return MaxSim(train, count_array), flat, counts


def query_feature(lemur, query: torch.Tensor) -> np.ndarray:
    """Generate one LEMUR query FDE before the search timing boundary."""
    feature = lemur.compute_features(query[None].to(torch.float32))
    feature = feature.detach().cpu().to(torch.float32).contiguous().numpy()[0]
    return feature


def top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    count = min(int(count), len(scores))
    if count == len(scores):
        return np.argsort(-scores, kind="stable")
    selected = np.argpartition(-scores, count - 1)[:count]
    return selected[np.argsort(-scores[selected], kind="stable")]


def maxsim_rerank_one(maxsim, query: torch.Tensor, candidates: np.ndarray, top_k: int, threads: int):
    candidates = np.ascontiguousarray(candidates.reshape(1, -1), dtype=np.int32)
    q = np.ascontiguousarray(query.numpy(), dtype=np.float32)
    counts = np.asarray([len(query)], dtype=np.int32)
    output = np.asarray(maxsim.rerank_subset(q, counts, min(top_k, candidates.shape[1]), candidates, threads))
    return output.reshape(1, -1)[0].astype(np.int64, copy=False)


def evaluate(rankings, qrels, reference, top_k=10):
    return compute_retrieval_metrics(rankings, qrels, top_k, reference)


def write_outputs(result_dir: Path, stem: str, aggregates: List[dict], rankings: dict) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / f"{stem}_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    with (result_dir / f"{stem}_rankings.json").open("w", encoding="utf-8") as handle:
        json.dump(rankings, handle, indent=2)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def document_artifact_size(root: Path) -> int:
    prefix = root / "colbert_official_index_256vecs"
    paths = [Path(f"{prefix}_{suffix}.pkl") for suffix in ("ids", "embs", "masks")]
    return sum(path.stat().st_size for path in paths)
