import os

PLAID_EXPERIMENT_NAME = "plaid_test"

# Native PLAID index construction and search are both CPU-only.  GPU use is
# confined to the separate offline embedding generator.
# --- Optional: limit CPU threads for ColBERT / PyTorch / BLAS ---
n_thread = 32
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["OMP_NUM_THREADS"] = f"{n_thread}"
    os.environ["MKL_NUM_THREADS"] = f"{n_thread}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{n_thread}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{n_thread}"

import warnings
warnings.filterwarnings("ignore")
import time
import argparse
import json 
from typing import Dict, List, Sequence, Union
import torch

if __name__ == "__main__":
    torch.set_num_threads(n_thread)
    try:
        # PyTorch permits setting inter-op threads only once per process.
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

from colbert import Indexer, Searcher 
from colbert.infra import ColBERTConfig, RunConfig, Run
from colbert.data import Queries, Collection

from common.data_load import load_dataset_by_name
from common.colbert_embeddings import load_cached_query_embeddings
from common.utils import *

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test colbert_pipeline using official library")
    parser.add_argument("--dataset-id", type=str, default="lotte/science") # lotte/science, zeta-alpha-ai/NanoFiQA2018
    parser.add_argument("--checkpoint", type=str, default="colbert-ir/colbertv2.0")
    parser.add_argument("--experiment-dir", type=str, default="./plaid_exp")
    parser.add_argument("--experiment-name", type=str, default=PLAID_EXPERIMENT_NAME)
    parser.add_argument("--doc-maxlen", type=int, default=BASE_DOC_VEC_CNT)
    parser.add_argument("--query-maxlen", type=int, default=32)
    # ColBERT packs residuals into uint8 values, so nbits must divide 8.
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--kmeans-niters", type=int, default=4) # default is 4
    parser.add_argument("--top-k-list", type=int, nargs="+", default=TOP_K_LIST)
    parser.add_argument(
        "--ncells-list",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="ncells values to evaluate for every top-k (default: 1 2 3 4 5 6 7 8).",
    )
    parser.add_argument("--max-queries", type=int, default=None) # None, 50
    parser.add_argument("--max-docs", type=int, default=None) # None, 1000
    args = parser.parse_args()
    if args.ncells_list is not None and any(ncells <= 0 for ncells in args.ncells_list):
        parser.error("--ncells-list values must be positive integers")
    return args


def load_dataset_collection(dataset_id: str, logger, max_queries: int = None, max_docs: int = None):
    """Load ColBERT inputs and retain document IDs for row-to-ID mapping."""
    collection = []
    queries_dict = {}
    qrels = {}
    idx_to_docid = None

    if dataset_id.startswith("lotte"):
        # Hardcoded root based on environment
        root_dir = os.environ.get(
            "LOTTE_DATA_DIR", os.path.join("..", "datasets", "lotte")
        )
        domain = dataset_id.split("/", 1)[1]
        data_dir = os.path.join(root_dir, domain, "dev")
        
        queries_path = os.path.join(data_dir, "questions.search.tsv")
        collection_path = os.path.join(data_dir, "collection.tsv")
        qrels_path = os.path.join(data_dir, "qas.search.jsonl")

        logger.info(f"Loading queries, doc collection and qrels...")
        queries = Queries(path=queries_path)
        collection = Collection(path=collection_path)
        if qrels_path and os.path.exists(qrels_path):
            with open(qrels_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    qid = str(item['qid'])
                    if 'answer_pids' in item:
                        pids = [str(pid) for pid in item['answer_pids']]
                    else:
                        pids = []
                    qrels[qid] = pids

        # Convert Queries to dict for slicing and compatibility
        queries_dict = {str(k): v for k, v in queries.items()}
        idx_to_docid = None
    else:
        logger.info(f"Loading dataset {dataset_id} using generic loader...")
        corpus_dict, queries_dict, qrels_set_dict = load_dataset_by_name(dataset_id, logger)
        
        qrels = {k: list(v) for k, v in qrels_set_dict.items()}
        idx_to_docid = list(corpus_dict.keys())
        collection = []
        for doc_id in idx_to_docid:
            doc = corpus_dict[doc_id]
            title = doc.get("title", "")
            text = doc.get("text", "")
            if title:
                passage = f"{title} | {text}"
            else:
                passage = text
            collection.append(passage)

    # Applying limits
    if max_queries:
        logger.info(f"Limiting to {max_queries} queries...")
        queries_dict = dict(list(queries_dict.items())[:max_queries])

    if max_docs:
        logger.info(f"Limiting to {max_docs} docs...")
        # Slicing collection might convert it to a list
        collection = collection[:max_docs]
        if idx_to_docid:
            idx_to_docid = idx_to_docid[:max_docs]
    
    # Ensure qrels only contains qids present in queries
    qrels = {qid: pids for qid, pids in qrels.items() if qid in queries_dict}
    
    return collection, queries_dict, qrels, idx_to_docid

def get_dir_size(path: str): 
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

class ColbertOfficialRetriever:
    """Wrapper around official ColBERT indexer and searcher."""
    def __init__(self, checkpoint: str, experiment_dir: str, experiment_name: str, index_name: str, doc_maxlen: int, query_maxlen: int, nbits: int, kmeans_niters: int, idx_to_docid: List[str] = None):
        self.checkpoint = checkpoint
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.index_name = index_name
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.nbits = nbits
        self.kmeans_niters = kmeans_niters
        self.searcher = None
        self.index_path = None
        self.index_build_time = 0.0
        self.collection = []
        self.idx_to_docid = idx_to_docid

    def index(
        self,
        collection: Union[List[str], Collection],
        logger,
    ) -> None:
        self.collection = collection
        collection_len = len(collection)
        logger.info(f"Starting indexing with ColBERT ({collection_len} documents)...")
        
        with Run().context(
            RunConfig(
                nranks=1,
                root=self.experiment_dir,
                experiment=self.experiment_name,
            )
        ):
            config = ColBERTConfig(
                doc_maxlen=self.doc_maxlen, # [NOTE]  colbert/modeling/tokenization/doc_tokenization.py
                query_maxlen=self.query_maxlen,
                nbits=self.nbits,
                kmeans_niters=self.kmeans_niters, # default is 4
                root=self.experiment_dir,
                experiment=self.experiment_name,
            )
            indexer = Indexer(checkpoint=self.checkpoint, config=config)
            expected_path = os.path.join(
                self.experiment_dir,
                self.experiment_name,
                "indexes",
                self.index_name,
            )
            metadata_path = os.path.join(expected_path, "metadata.json")
            index_complete = os.path.isfile(metadata_path)
            build_start = time.perf_counter()
            self.index_path = indexer.index(
                name=self.index_name,
                collection=self.collection,
                # Resume an interrupted/incomplete directory instead of
                # treating directory existence as a valid reusable index.
                overwrite="reuse" if index_complete else "resume",
            )
            self.index_build_time = time.perf_counter() - build_start

    def load_searcher(self, logger) -> None:
        """Load a complete PLAID index in the current CPU-only process."""
        with Run().context(
            RunConfig(
                nranks=1,
                root=self.experiment_dir,
                experiment=self.experiment_name,
            )
        ):
            config = ColBERTConfig(
                doc_maxlen=self.doc_maxlen,
                query_maxlen=self.query_maxlen,
                nbits=self.nbits,
                kmeans_niters=self.kmeans_niters,
                root=self.experiment_dir,
                experiment=self.experiment_name,
            )
            logger.info("Initializing searcher...")
            # Searcher requires the complete PLAID index, not only embeddings.
            self.searcher = Searcher(
                index=self.index_name,
                # Reused indexes retain the collection path from the machine
                # that built them.  Always prefer the collection loaded for
                # this run so a relocated index does not follow stale metadata.
                collection=self.collection,
                config=config,
            )
            if hasattr(self.searcher, "ranker") and hasattr(self.searcher.ranker, "codec"):
                nlist = int(self.searcher.ranker.codec.centroids.size(0))
                num_embeddings = int(self.searcher.ranker.num_embeddings)
                logger.info(f"[ColBERT] IVF centroids (nlist): {nlist}. Avg vectors per cell: {num_embeddings / nlist:.2f}")

    def configure_search(self, top_k: int, ncells: int = None) -> None:
        """Apply PLAID defaults for this K, optionally overriding ncells."""
        if top_k <= 10:
            default_ncells = 1
            centroid_score_threshold = 0.5
            ndocs = 256
        elif top_k <= 100:
            default_ncells = 2
            centroid_score_threshold = 0.45
            ndocs = 1024
        else:
            default_ncells = 4
            centroid_score_threshold = 0.4
            ndocs = max(top_k * 4, 4096)

        self.searcher.configure(
            ncells=default_ncells if ncells is None else ncells,
            centroid_score_threshold=centroid_score_threshold,
            ndocs=ndocs,
        )

    def search_all_encoded(
        self,
        query_ids: Sequence[str],
        query_embeddings: torch.Tensor,
        max_k: int,
    ) -> tuple[Dict[str, List[str]], Dict[str, float]]:
        """Search pre-encoded queries and report total wall time per query."""
        ranking_results = {}

        search_start = time.perf_counter()
        for query_idx, qid in enumerate(query_ids):
            query_embedding = query_embeddings[query_idx:query_idx + 1]

            results = self.searcher.dense_search(query_embedding, k=max_k)

            # results is usually (passage_ids, passage_ranks, passage_scores).
            top_k_ids = []
            for passage_id in results[0]:
                # When utilizing Collection, passage_id (int) corresponds to the index/ID
                if self.idx_to_docid:
                    top_k_ids.append(self.idx_to_docid[passage_id])
                else:
                    top_k_ids.append(str(passage_id))
            ranking_results[qid] = top_k_ids

        dense_search_elapsed = time.perf_counter() - search_start
        num_queries = len(query_ids)
        qps = num_queries / dense_search_elapsed if dense_search_elapsed else 0.0
        avg_time_ms = (dense_search_elapsed / max(num_queries, 1)) * 1000.0
        runtime_stats = {
            "QPS": qps,
            "avg_query_time_ms": avg_time_ms,
        }
        return ranking_results, runtime_stats

def main() -> None:
    cfg = parse_args()
    cfg.log_file = f"./logs/{cfg.dataset_id.replace('/', '_')}/{cfg.experiment_name}_{time.strftime('%m%d')}.log"
    os.makedirs(os.path.dirname(cfg.log_file), exist_ok=True)
    logger = configure_logger(log_file=cfg.log_file)
    logger.info(f"\n\n\nExperiment Config: {vars(cfg)}")
    logger.info("Native PLAID index and search device: CPU (CUDA hidden)")

    logger.info("Loading dataset...")

    # corpus, queries, qrels = load_dataset_by_name(cfg.dataset_id, logger, max_queries=cfg.max_queries, max_docs=cfg.max_docs)

    collection, queries, qrels, idx_to_docid = load_dataset_collection(
        dataset_id=cfg.dataset_id,
        logger=logger,
        max_queries=cfg.max_queries,
        max_docs=cfg.max_docs
    )

    base_rankings = get_base_rankings(
        checkpoint=cfg.checkpoint,
        dataset_id=cfg.dataset_id,
        max_k=max(cfg.top_k_list),
        logger=logger,
        query_ids=queries.keys(),
        required=cfg.max_queries is None and cfg.max_docs is None,
    )

    dataset_stub = cfg.dataset_id.replace("/", "_")
    checkpoint_stub = cfg.checkpoint.replace("/", "_")
    doc_limit = "full" if cfg.max_docs is None else f"docs{cfg.max_docs}"
    experiment_name = f"{cfg.experiment_name}_{dataset_stub}"
    index_name = f"{cfg.experiment_name}_{dataset_stub}_{checkpoint_stub}_docmax{cfg.doc_maxlen}_nbits{cfg.nbits}_kmeans{cfg.kmeans_niters}_{doc_limit}"
    logger.info("ColBERT experiment=%s | index=%s", experiment_name, index_name)

    retriever = ColbertOfficialRetriever(
        checkpoint=cfg.checkpoint,
        experiment_dir=cfg.experiment_dir,
        experiment_name=experiment_name,
        index_name=index_name,
        doc_maxlen=cfg.doc_maxlen,
        query_maxlen=cfg.query_maxlen,
        nbits=cfg.nbits,
        kmeans_niters=cfg.kmeans_niters,
        idx_to_docid=idx_to_docid
    )

    # 1. Indexing
    expected_index_path = os.path.join(
        cfg.experiment_dir, experiment_name, "indexes", index_name
    )
    index_reused = os.path.isfile(os.path.join(expected_index_path, "metadata.json"))
    index_start = time.time()
    retriever.index(collection, logger)

    index_elapsed = time.time() - index_start
    logger.info(f"Index stage completed in {index_elapsed:.2f} seconds.")

    retriever.load_searcher(logger)

    index_size_MB = get_dir_size(retriever.index_path) / (1024**2)

    # 2. Searching
    logger.info("Executing searches...")
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    embedding_cache_dir = os.path.join(
        "embedding_files",
        dataset_stub,
        "colbert_official_data_query_emb_tensor",
    )
    query_embeddings = load_cached_query_embeddings(
        embedding_cache_dir, cfg.query_maxlen, query_ids
    )
    if query_embeddings is None:
        logger.warning(
            "GPU-precomputed ColBERT query embeddings were not found; encoding "
            "%d queries on CPU outside the search timing. Run "
            "scripts/generate_embeddings_gpu.sh to precompute them.",
            len(query_texts),
        )
        query_embeddings = retriever.searcher.encode(query_texts)
    else:
        logger.info("Loaded GPU-precomputed ColBERT query embeddings from %s", embedding_cache_dir)

    # 3. Evaluation
    results_dir = f"./results/{cfg.dataset_id.replace('/', '_')}"
    os.makedirs(results_dir, exist_ok=True)
    cfg.csv_path = os.path.join(results_dir, f"{cfg.experiment_name}_nbit{cfg.nbits}_{time.strftime('%m%d')}.csv")
    recorder = MetricsRecorder(cfg.csv_path)

    ncells_values = cfg.ncells_list if cfg.ncells_list is not None else [None]
    for top_k in cfg.top_k_list:
        for ncells in ncells_values:
            retriever.configure_search(top_k, ncells)
            search_config = retriever.searcher.config
            logger.info(
                "PLAID search config for top_k=%d: ncells=%d, centroid_score_threshold=%.2f, ndocs=%d",
                top_k,
                search_config.ncells,
                search_config.centroid_score_threshold,
                search_config.ndocs,
            )
            rankings, runtime_stats = retriever.search_all_encoded(
                query_ids,
                query_embeddings,
                top_k,
            )
            logger.info(
                "Dense search completed in %.2f ms/query (QPS: %.2f), excluding query encoding.",
                runtime_stats["avg_query_time_ms"],
                runtime_stats["QPS"],
            )
            metrics = compute_retrieval_metrics(
                rankings,
                qrels,
                top_k,
                reference_rankings=base_rankings,
            )
            row_data = {
                "method": "PLAID",
                "TOP_K": top_k,
                "ncells": search_config.ncells,
                "centroid_score_threshold": search_config.centroid_score_threshold,
                "ndocs": search_config.ndocs,
                "index_size_MB": index_size_MB,
                "plaid_index_build_time_s": retriever.index_build_time,
                "plaid_index_and_load_wall_time_s": index_elapsed,
                "plaid_index_reused": index_reused,
            }
            recorder.add_row(row_data, metrics, runtime_stats)

            logger.info(
                "[plaid] top_k=%d | ncells=%d | recall@k=%.4f | success@k=%.4f | MRR@k=%.4f | nDCG@k=%.4f | recall_maxsim@k=%.4f | QPS=%.2f",
                top_k, search_config.ncells, metrics["recall@k"], metrics["success@k"], metrics["MRR@k"], metrics["nDCG@k"], metrics["recall_maxsim@k"], runtime_stats["QPS"]
            )

    recorder.save()
    logger.info(f"Results saved to {cfg.csv_path}")
    logger.info("Official ColBERT pipeline finished")

if __name__ == "__main__":
    main()
