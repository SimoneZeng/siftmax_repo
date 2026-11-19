"""Generate shared ColBERT, exact-reference, and dense artifacts on a GPU.

This is an offline preprocessing command. The resulting tensors, exact
full-corpus MaxSim top-k reference, and Faiss index are stored on CPU and
consumed by the CPU-only experiment runners.
"""

import os
from pathlib import Path

# Hugging Face clients read this setting during import/initialization.  Set it
# before importing ColBERT, Transformers, or SentenceTransformers.  Callers
# can still select a different endpoint through their environment.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import time

import torch

from common.colbert_embeddings import ColbertRetriever
from common.data_load import load_dataset_by_name
from common.dense_retrieval import DENSE_MODEL_NAME, DenseRetriever
from common.reference_rankings import (
    load_reference_rankings,
    reference_rankings_path,
    save_reference_rankings,
)
from common.utils import BASE_DOC_VEC_CNT, configure_logger


def _huggingface_hub_cache() -> Path:
    """Return the effective Hugging Face Hub cache without importing Hub."""
    explicit_cache = os.environ.get("HF_HUB_CACHE")
    if explicit_cache:
        return Path(explicit_cache).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    cache_home = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache_home / "huggingface" / "hub"


def find_complete_cached_model(model_id: str) -> Path | None:
    """Find a complete local snapshot for the two encoder model families."""
    model_root = _huggingface_hub_cache() / f"models--{model_id.replace('/', '--')}"
    snapshots_root = model_root / "snapshots"
    if not snapshots_root.is_dir():
        return None

    candidates = []
    main_ref = model_root / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        if revision:
            candidates.append(snapshots_root / revision)
    candidates.extend(
        sorted(
            (path for path in snapshots_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )

    required = ["config.json", "tokenizer_config.json"]
    if model_id.startswith("sentence-transformers/"):
        required.append("modules.json")
    weight_names = ("model.safetensors", "pytorch_model.bin")

    seen = set()
    for snapshot in candidates:
        try:
            resolved = snapshot.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if not all((snapshot / filename).is_file() for filename in required):
            continue
        if not any((snapshot / filename).is_file() for filename in weight_names):
            continue
        return snapshot
    return None


def resolve_model_source(model_id: str, logger) -> str:
    """Use a complete local snapshot, otherwise allow a mirror download."""
    cached_path = find_complete_cached_model(model_id)
    if cached_path is not None:
        logger.info("Using cached Hugging Face model %s from %s", model_id, cached_path)
        return str(cached_path)
    logger.info(
        "No complete local cache for %s; downloading through %s",
        model_id,
        os.environ["HF_ENDPOINT"],
    )
    return model_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="lotte/science")
    parser.add_argument("--checkpoint", default="colbert-ir/colbertv2.0")
    parser.add_argument("--dense-model-name", default=DENSE_MODEL_NAME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--doc-maxlen", type=int, default=BASE_DOC_VEC_CNT)
    parser.add_argument("--query-maxlen", type=int, default=32)
    parser.add_argument("--bsize", type=int, default=64)
    parser.add_argument("--skip-colbert", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--reference-top-k", type=int, default=10)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()
    if args.bsize <= 0 or args.reference_top_k <= 0:
        parser.error("--bsize and --reference-top-k must be positive")
    if args.skip_colbert and args.skip_dense:
        parser.error("nothing to generate: both artifact families were skipped")
    return args


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("the offline embedding generator requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")

    dataset_stub = args.dataset_id.replace("/", "_")
    log_file = args.log_file or os.path.join(
        "logs", dataset_stub, f"generate_embeddings_{time.strftime('%m%d')}.log"
    )
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logger = configure_logger("generate_embeddings", log_file)
    logger.info("Offline embedding device: %s", args.device)

    corpus, queries, _ = load_dataset_by_name(args.dataset_id, logger)

    if not args.skip_dense:
        logger.info("Generating/loading dense artifacts on %s", args.device)
        dense_model_source = resolve_model_source(args.dense_model_name, logger)
        dense = DenseRetriever(
            args.dense_model_name,
            "flat",
            device=args.device,
            search_device="cpu",
            model_source=dense_model_source,
        )
        dense.index(args.dataset_id, None, corpus)
        dense.encode_queries(queries, use_cache=True)
        logger.info("Dense CPU index: %s", dense.index_cache_path)
        logger.info("Dense query embeddings: %s", dense.query_cache_path)
        del dense
        torch.cuda.empty_cache()

    if not args.skip_colbert:
        cache_dir = os.path.join(
            "embedding_files",
            dataset_stub,
            "colbert_official_data_query_emb_tensor",
        )
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Generating/loading ColBERT artifacts on %s", args.device)
        colbert_model_source = resolve_model_source(args.checkpoint, logger)
        colbert = ColbertRetriever(
            checkpoint=colbert_model_source,
            device=args.device,
            doc_maxlen=args.doc_maxlen,
            query_maxlen=args.query_maxlen,
            bsize=args.bsize,
        )
        colbert.get_doc_embedding(corpus, cache_dir, logger, use_cache=True)
        colbert.get_query_embedding(queries, cache_dir, logger, use_cache=True)
        logger.info("ColBERT artifacts: %s", cache_dir)

        reference_path = reference_rankings_path(
            args.dataset_id,
            args.checkpoint,
            args.reference_top_k,
        )
        if reference_path.is_file():
            load_reference_rankings(
                reference_path,
                colbert.query_ids,
                args.reference_top_k,
            )
            logger.info("Validated exact MaxSim reference: %s", reference_path)
        else:
            logger.info(
                "Generating exact full-corpus MaxSim top-%d reference on %s",
                args.reference_top_k,
                args.device,
            )
            rankings, runtime = colbert.search_all(args.reference_top_k)
            save_reference_rankings(
                reference_path,
                rankings,
                colbert.query_ids,
                args.reference_top_k,
            )
            logger.info(
                "Saved exact MaxSim reference to %s (%.2fs, %.2f QPS)",
                reference_path,
                runtime["total_time_s"],
                runtime["QPS"],
            )

    logger.info("Offline GPU embedding generation completed")


if __name__ == "__main__":
    main()
