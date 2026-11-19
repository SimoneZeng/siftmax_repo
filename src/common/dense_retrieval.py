import os
import pickle
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_GPU_AVAILABLE = hasattr(faiss, "StandardGpuResources")


def load_serialized_dense_index(index_path):
    """Load the shared persisted Faiss index and its document IDs."""
    with open(index_path, "rb") as handle:
        index_payload = pickle.load(handle)
    doc_ids = [str(value.item() if isinstance(value, np.generic) else value) for value in index_payload["doc_ids"]]
    serialized = index_payload["faiss_index"]
    if isinstance(serialized, faiss.Index):
        index = serialized
    elif isinstance(serialized, np.ndarray):
        index = faiss.deserialize_index(np.asarray(serialized, dtype=np.uint8).reshape(-1))
    elif isinstance(serialized, (bytes, bytearray, memoryview)):
        index = faiss.deserialize_index(np.frombuffer(serialized, dtype=np.uint8))
    else:
        raise TypeError(f"unsupported serialized Faiss index: {type(serialized)}")
    if len(doc_ids) != index.ntotal:
        raise ValueError("dense document IDs/index artifact mismatch")
    return doc_ids, index, float(index_payload.get("index_build_time", 0.0))


def load_serialized_dense_artifacts(query_path, index_path):
    """Load the shared precomputed-query/Faiss payload without loading a model."""
    with open(query_path, "rb") as handle:
        query_payload = pickle.load(handle)
    query_ids = [str(value.item() if isinstance(value, np.generic) else value) for value in query_payload["query_ids"]]
    query_embeddings = np.ascontiguousarray(query_payload["query_embeddings"], dtype=np.float32)
    doc_ids, index, index_build_time = load_serialized_dense_index(index_path)
    if query_embeddings.ndim != 2 or query_embeddings.shape[1] != index.d:
        raise ValueError("dense query/index artifact mismatch")
    return query_ids, query_embeddings, doc_ids, index, index_build_time


def map_dense_ids_to_colbert(dense_doc_ids, colbert_doc_ids):
    """Map dense-index row IDs to rows in the shared ColBERT tensor."""
    positions = {str(doc_id): offset for offset, doc_id in enumerate(colbert_doc_ids)}
    try:
        return np.asarray([positions[str(doc_id)] for doc_id in dense_doc_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"dense document absent from ColBERT artifacts: {error.args[0]}") from error


def search_encoded_faiss_queries(
    index,
    query_ids: Sequence[str],
    query_embeddings,
    top_k: int,
):
    """Run one batch Faiss search and return indices plus amortized timing."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(query_ids):
        raise ValueError("query IDs and embeddings have incompatible shapes")

    num_queries = len(query_ids)
    if num_queries == 0:
        return np.empty((0, top_k), dtype=np.int64), {
            "query_search_time_s": 0.0,
            "query_search_ms": 0.0,
            "QPS": 0.0,
        }

    started = time.perf_counter()
    _, indices = index.search(embeddings, top_k)
    query_search_time = time.perf_counter() - started
    amortized_search_s = query_search_time / num_queries
    return indices, {
        "query_search_time_s": query_search_time,
        "query_search_ms": amortized_search_s * 1000.0,
        "QPS": num_queries / query_search_time if query_search_time else 0.0,
    }

class DenseRetriever:
    """Phrase 1 Dense Retriever using FAISS"""

    def __init__(
        self,
        model_name: str,
        index_type: str,
        device: str,
        search_device: str = "cpu",
        model_source: Optional[str] = None,
    ):
        if index_type != "flat":
            raise ValueError("DenseRetriever only supports the Faiss Flat index")
        if search_device != "cpu":
            raise ValueError("Dense retrieval search is intentionally CPU-only")
        # ``model_source`` may be a verified local Hugging Face snapshot while
        # ``model_name`` remains the stable ID used in artifact filenames.
        self.model = SentenceTransformer(model_source or model_name, device=device)
        self.model_name = model_name
        self.index_type = index_type
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            # Compatibility with older SentenceTransformers releases.
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.doc_ids = []
        self.faiss_index = None
        self.gpu_resources = None
        # ``device`` is used only by SentenceTransformer embedding generation.
        # The persisted/searchable Faiss index always remains on CPU.
        self.device = device
        self.search_device = search_device
        self.encode_time = 0.0
        self.index_build_time = 0.0
        self.index_cache_path = None
        self.query_cache_path = None
        self.dataset_id = None
        self.index_reused = False

    def _index_cache_path(self, dataset_id: str, n_docs: Optional[int], corpus: dict) -> str:
        dataset_stub = dataset_id.replace("/", "_")
        doc_count = n_docs if n_docs is not None else len(corpus)
        index_filename = (
            f"dense_index_MiniLM_{self.index_type}"
            f"_{self.embedding_dim}d_{doc_count}docs.pkl"
        )
        return os.path.join("./embedding_files", dataset_stub, index_filename)

    def _index_to_cpu(self):
        return self.faiss_index

    def _index_to_device(self, cpu_index):
        # Search latency must describe CPU Faiss, independently of the device
        # used to create document/query embeddings.
        return cpu_index

    def _query_cache_path(self, dataset_id: str, query_count: int) -> str:
        dataset_stub = dataset_id.replace("/", "_")
        model_stub = self.model_name.rstrip("/").split("/")[-1]
        filename = (
            f"dense_query_embeddings_{model_stub}_{self.embedding_dim}d_"
            f"{query_count}queries.pkl"
        )
        return os.path.join("./embedding_files", dataset_stub, filename)

    def _save_index(self, index_cache_path: str):
        os.makedirs(os.path.dirname(index_cache_path), exist_ok=True)
        cpu_index = self._index_to_cpu()
        payload = {
            "doc_ids": self.doc_ids,
            "faiss_index": faiss.serialize_index(cpu_index),
            "index_build_time": self.index_build_time,
        }
        with open(index_cache_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_index(self, index_cache_path: str):
        with open(index_cache_path, "rb") as f:
            payload = pickle.load(f)
        self.doc_ids = payload["doc_ids"]
        cpu_index = faiss.deserialize_index(payload["faiss_index"])
        self.faiss_index = self._index_to_device(cpu_index)
        self.encode_time = 0.0
        self.index_build_time = 0.0
        self.index_reused = True

    def index_building(self, corpus: dict):
        self.doc_ids = list(corpus.keys())
        texts = [f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}" for did in self.doc_ids]
        
        t0 = time.perf_counter()
        embeddings = self.model.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")
        self.encode_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        base_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index = self._index_to_device(base_index)
        self.faiss_index.add(embeddings)
        self.index_build_time = time.perf_counter() - t0

    def index(self, dataset_id: str, n_docs: Optional[int], corpus: dict):
        self.dataset_id = dataset_id
        self.index_cache_path = self._index_cache_path(dataset_id, n_docs, corpus)
        if os.path.exists(self.index_cache_path):
            print(f"Loading dense FAISS index from {self.index_cache_path}")
            self._load_index(self.index_cache_path)
            return

        self.index_building(corpus)
        self.index_reused = False
        persist_start = time.perf_counter()
        self._save_index(self.index_cache_path)
        self.index_build_time += time.perf_counter() - persist_start

    def encode_queries(
        self,
        queries: Mapping[str, str],
        batch_size: int = 512,
        use_cache: bool = True,
    ):
        """Load cached queries when available, otherwise encode them."""
        query_texts = list(queries.values())
        query_ids = list(queries.keys())

        if self.dataset_id is not None:
            self.query_cache_path = self._query_cache_path(
                self.dataset_id, len(query_ids)
            )
        if use_cache and self.query_cache_path and os.path.isfile(self.query_cache_path):
            return self.load_query_embeddings(queries)

        q_emb = self.model.encode(
            query_texts, 
            batch_size=batch_size,
            show_progress_bar=False, 
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")
        if use_cache and self.query_cache_path:
            os.makedirs(os.path.dirname(self.query_cache_path), exist_ok=True)
            with open(self.query_cache_path, "wb") as handle:
                pickle.dump(
                    {"query_ids": query_ids, "query_embeddings": q_emb},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        return query_ids, q_emb

    def load_query_embeddings(
        self,
        queries: Mapping[str, str],
    ):
        """Load and validate the shared dense query-vector artifact."""
        if self.dataset_id is None:
            raise RuntimeError("index() must be called before loading dense queries")
        query_ids = list(queries.keys())
        self.query_cache_path = self._query_cache_path(
            self.dataset_id, len(query_ids)
        )
        if not os.path.isfile(self.query_cache_path):
            raise FileNotFoundError(
                f"dense query-vector artifact not found: {self.query_cache_path}"
            )
        with open(self.query_cache_path, "rb") as handle:
            payload = pickle.load(handle)
        cached_ids = [str(qid) for qid in payload["query_ids"]]
        if cached_ids != [str(qid) for qid in query_ids]:
            raise ValueError(
                f"dense query cache IDs do not match the dataset: {self.query_cache_path}"
            )
        embeddings = np.ascontiguousarray(
            payload["query_embeddings"], dtype=np.float32
        )
        if embeddings.shape != (len(query_ids), self.embedding_dim):
            raise ValueError(
                f"invalid dense query cache shape {embeddings.shape}: "
                f"{self.query_cache_path}"
            )
        return query_ids, embeddings

    def search_encoded_queries(
        self,
        query_ids: Sequence[str],
        query_embeddings,
        top_k: int,
    ) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        """Search all pre-encoded queries in one parallel Faiss batch."""
        indices, runtime = search_encoded_faiss_queries(
            self.faiss_index,
            query_ids,
            query_embeddings,
            top_k,
        )
        results = {
            qid: [self.doc_ids[idx] for idx in indices[i] if idx != -1]
            for i, qid in enumerate(query_ids)
        }
        return results, runtime
