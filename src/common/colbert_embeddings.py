import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple
import torch
from tqdm import tqdm
import torch.nn.functional as F

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

from common.reference_rankings import load_reference_rankings, reference_rankings_path


def load_cached_query_embeddings(
    data_query_emb_dir: str,
    query_maxlen: int,
    expected_query_ids: List[str],
) -> Optional[torch.Tensor]:
    """Load shared CPU query tensors, returning ``None`` when absent."""
    base_path = os.path.join(
        data_query_emb_dir, f"colbert_official_query_{query_maxlen}embs"
    )
    id_path, emb_path = f"{base_path}_ids.pkl", f"{base_path}_embs.pkl"
    if not (os.path.isfile(id_path) and os.path.isfile(emb_path)):
        return None
    with open(id_path, "rb") as handle:
        cached_ids = [str(qid) for qid in pickle.load(handle)]
    expected_ids = [str(qid) for qid in expected_query_ids]
    if cached_ids != expected_ids:
        raise ValueError(f"ColBERT query cache IDs do not match: {id_path}")
    with open(emb_path, "rb") as handle:
        embeddings = pickle.load(handle)
    if not isinstance(embeddings, torch.Tensor) or embeddings.size(0) != len(cached_ids):
        raise ValueError(f"Invalid ColBERT query cache: {emb_path}")
    return embeddings.cpu()

@dataclass
class ColbertRetriever:
    """Wrapper around official ColBERT for embedding generation and memory-based MaxSim."""
    checkpoint: str
    device: str
    doc_maxlen: int
    query_maxlen: int
    bsize: int = 128

    def __post_init__(self) -> None:
        # Instantiate ColBERT Checkpoint
        config = ColBERTConfig(doc_maxlen=self.doc_maxlen, query_maxlen=self.query_maxlen)
        self.model = Checkpoint(self.checkpoint, colbert_config=config)
        self.model.to(self.device)
        self.model.eval()

        self.doc_ids: List[str] = []
        self.doc_embeddings: Optional[torch.Tensor] = None
        self.doc_masks: Optional[torch.Tensor] = None
        self.query_ids: List[str] = []
        self.query_tensors: Optional[torch.Tensor] = None

    def get_doc_embedding(
        self,
        corpus: Mapping[str, Mapping[str, str]],
        data_query_emb_dir: str,
        logger,
        n_doc_limited_load: Optional[int] = None,
        use_cache: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        base_path = os.path.join(data_query_emb_dir, f"colbert_official_index_{self.doc_maxlen}vecs")
        id_path, emb_path, mask_path = f"{base_path}_ids.pkl", f"{base_path}_embs.pkl", f"{base_path}_masks.pkl"

        # (1) load embeddings and masks
        if use_cache and os.path.exists(id_path) and os.path.exists(emb_path) and os.path.exists(mask_path):
            logger.info(f"Loading doc embeddings and masks from {data_query_emb_dir}")
            with open(id_path, "rb") as f: self.doc_ids = pickle.load(f)
            with open(emb_path, "rb") as f: self.doc_embeddings = pickle.load(f)
            with open(mask_path, "rb") as f: self.doc_masks = pickle.load(f)

            if n_doc_limited_load is not None:
                self.doc_ids = self.doc_ids[:n_doc_limited_load]
                self.doc_embeddings = self.doc_embeddings[:n_doc_limited_load]
                self.doc_masks = self.doc_masks[:n_doc_limited_load]
            return self.doc_ids, self.doc_embeddings, self.doc_masks

        # (2) generate
        if use_cache:
            logger.info("Generating document embeddings...")
        else:
            logger.info("Generating document embeddings directly (document cache disabled)")
        target_doc_ids = list(corpus.keys())
        if n_doc_limited_load is not None:
            target_doc_ids = target_doc_ids[:n_doc_limited_load]

        self.doc_ids = target_doc_ids
        doc_embeddings_list = []
        doc_masks_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(target_doc_ids), self.bsize), desc="Encoding Docs"):
                batch_ids = target_doc_ids[i:i + self.bsize]
                batch_docs = [corpus[d_id].get("text", "") for d_id in batch_ids]
                ids, mask = self.model.doc_tokenizer.tensorize(batch_docs)
                doc_tensor, doc_mask = self.model.doc(ids.to(self.device), mask.to(self.device), keep_dims='return_mask')

                # DocTokenizer uses padding="longest", so different batches may
                # have different token dimensions.  Pad every batch to
                # doc_maxlen before concatenating: zero embeddings are ignored
                # by the corresponding False entries in doc_mask.
                token_count = doc_tensor.size(1)
                if token_count > self.doc_maxlen:
                    raise ValueError(
                        f"Encoded document length {token_count} exceeds "
                        f"doc_maxlen={self.doc_maxlen}"
                    )
                if token_count < self.doc_maxlen:
                    pad_len = self.doc_maxlen - token_count
                    doc_tensor = F.pad(doc_tensor, (0, 0, 0, pad_len), value=0.0)
                    if doc_mask.ndim == 3 and doc_mask.size(-1) == 1:
                        doc_mask = F.pad(doc_mask, (0, 0, 0, pad_len), value=False)
                    elif doc_mask.ndim == 2:
                        doc_mask = F.pad(doc_mask, (0, pad_len), value=False)
                    else:
                        raise ValueError(
                            f"Unexpected document mask shape: {tuple(doc_mask.shape)}"
                        )

                doc_embeddings_list.append(doc_tensor.cpu())
                doc_masks_list.append(doc_mask.bool().cpu())

        self.doc_embeddings = torch.cat(doc_embeddings_list, dim=0)
        self.doc_masks = torch.cat(doc_masks_list, dim=0)

        if use_cache and n_doc_limited_load is None:
            logger.info(f"Writing caches to {data_query_emb_dir}...")
            save_start = time.time()
            with open(id_path, "wb") as f: pickle.dump(self.doc_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(emb_path, "wb") as f: pickle.dump(self.doc_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(mask_path, "wb") as f: pickle.dump(self.doc_masks, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved doc caches in %.2fs", time.time() - save_start)

        return self.doc_ids, self.doc_embeddings, self.doc_masks

    def get_query_embedding(
        self,
        queries: Mapping[str, str],
        data_query_emb_dir: str,
        logger,
        n_query_limited_load: Optional[int] = None,
        use_cache: bool = True,
    ) -> Tuple[List[str], torch.Tensor]:
        base_path = os.path.join(data_query_emb_dir, f"colbert_official_query_{self.query_maxlen}embs")
        id_path, emb_path = f"{base_path}_ids.pkl", f"{base_path}_embs.pkl"

        query_items = list(queries.items()) # [(query_id, query_text), ...]
        if n_query_limited_load is not None:
            query_items = query_items[:n_query_limited_load]

        if use_cache and os.path.exists(id_path) and os.path.exists(emb_path):
            with open(id_path, "rb") as f: self.query_ids = pickle.load(f)
            with open(emb_path, "rb") as f: self.query_tensors = pickle.load(f)
            logger.info(f"Loaded query embeddings from {data_query_emb_dir}")

            if n_query_limited_load is not None:
                self.query_ids = self.query_ids[:n_query_limited_load]
                self.query_tensors = self.query_tensors[:n_query_limited_load]
        else:
            if use_cache:
                logger.info(f"Generating query embeddings to {data_query_emb_dir}")
            else:
                logger.info("Generating query embeddings directly (query cache disabled)")

            unique_texts = list(dict.fromkeys(text for _, text in query_items))

            with torch.inference_mode():
                unique_embeddings = self.model.queryFromText(
                        unique_texts,
                        bsize=self.bsize,
                        to_cpu=True,
                    )

            self.query_ids = [qid for qid, _ in query_items]
            self.query_tensors = unique_embeddings

            if use_cache and n_query_limited_load is None:
                with open(id_path, "wb") as f: pickle.dump(self.query_ids, f)
                with open(emb_path, "wb") as f: pickle.dump(self.query_tensors, f)
                logger.info(f"Saved query embeddings to {data_query_emb_dir}")

        return self.query_ids, self.query_tensors

    def get_index_stats(self) -> Tuple[float, float]:
        if self.doc_embeddings is None:
            return 0.0, 0.0
        n_docs, avg_vecs, dim = self.doc_embeddings.shape
        bytes_per_element = self.doc_embeddings.element_size()

        total_id_bytes = n_docs * 32
        total_vector_bytes = n_docs * avg_vecs * dim * bytes_per_element
        estimated_index_size = (total_id_bytes + total_vector_bytes) / (1024 ** 2) # MB
        return estimated_index_size, avg_vecs

    def search_all(
        self,
        max_k: int = 100,
    ) -> tuple[Dict[str, List[str]], Dict[str, float]]:
        start = time.time()
        if self.doc_embeddings is None:
            empty_rankings = {qid: [] for qid in self.query_ids}
            runtime_stats = {
                "QPS": 0.0,
                "avg_query_time_ms": 0.0,
                "total_time_s": 0.0,
            }
            return empty_rankings, runtime_stats

        results = {}
        q_tensors = self.query_tensors.to(self.device)

        for i in tqdm(
            range(0, len(self.query_ids), self.bsize),
            desc="Exact full-corpus MaxSim query batches",
        ):
            q_batch = q_tensors[i:i + self.bsize] # [B_q, L_q, D]
            q_batch_unsqueeze = q_batch.unsqueeze(1) # [B_q, 1, L_q, D]

            all_scores = []
            for d_idx in range(0, len(self.doc_ids), self.bsize):
                d_batch = self.doc_embeddings[d_idx:d_idx+self.bsize].to(self.device)
                m_batch = self.doc_masks[d_idx:d_idx+self.bsize].to(self.device)

                # Compute pairwise token similarities across query and document batches:
                # [B_q, 1, L_q, D] @ [1, B_d, D, L_d] -> [B_q, B_d, L_q, L_d]
                scores = torch.matmul(q_batch_unsqueeze, d_batch.to(dtype=q_batch.dtype).permute(0, 2, 1).unsqueeze(0))
                D_padding = ~m_batch.view(1, m_batch.size(0), 1, m_batch.size(1))
                scores.masked_fill_(D_padding, -9999.0)
                sum_scores = scores.max(dim=-1).values.sum(dim=-1) # [B_q, B_d]
                all_scores.append(sum_scores)

            all_scores = torch.cat(all_scores, dim=1) # [B_q, Total_N_docs]

            k_val = min(max_k, all_scores.size(1))
            topk_indices = torch.topk(all_scores, k=k_val, dim=1).indices

            for j in range(len(q_batch)):
                results[self.query_ids[i+j]] = [self.doc_ids[idx.item()] for idx in topk_indices[j]]

        elapsed = time.time() - start
        num_queries = len(self.query_ids)
        qps = num_queries / elapsed if elapsed else 0.0
        avg_time_ms = (elapsed / max(num_queries, 1)) * 1000.0
        runtime_stats = {
            "QPS": qps,
            "avg_query_time_ms": avg_time_ms,
            "total_time_s": elapsed,
        }
        return results, runtime_stats

    def get_base_rankings(
        self,
        dataset_id: str,
        max_k: int,
        use_full_data: bool,
        logger,
    ) -> Dict[str, List[str]]:
        if not use_full_data:
            base_rankings, _ = self.search_all(max_k)
            return base_rankings

        base_rankings_path = reference_rankings_path(
            dataset_id, self.checkpoint, max_k
        )
        if not base_rankings_path.is_file():
            raise FileNotFoundError(
                f"Exact MaxSim reference is missing: {base_rankings_path}. "
                "Run scripts/generate_embeddings_gpu.sh first."
            )
        logger.info("Loading base rankings from %s", base_rankings_path)
        return load_reference_rankings(
            base_rankings_path, self.query_ids, max_k
        )
