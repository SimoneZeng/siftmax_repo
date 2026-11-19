import itertools
import json
import logging
import os
from typing import Dict, Optional, Tuple


CorpusDict = Dict[str, Dict[str, str]]
QueriesDict = Dict[str, str]
QrelsDict = Dict[str, set[str]]

DEFAULT_LOTTE_ROOT = os.path.join("..", "datasets", "lotte")


def load_lotte_qrels_file(path: str) -> QrelsDict:
    """Load a LoTTE qas.search.jsonl file without materializing the corpus."""
    qrels: QrelsDict = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            qrels[str(entry["qid"])] = {
                str(pid) for pid in entry.get("answer_pids", [])
            }
    return qrels


def load_dataset_by_name(
    dataset_id: str,
    logger: Optional[logging.Logger] = None,
    max_queries: Optional[int] = None,
    max_docs: Optional[int] = None,
) -> Tuple[CorpusDict, QueriesDict, QrelsDict]:
    """Load the fixed LoTTE Science dev/search dataset used by all six methods."""
    if dataset_id != "lotte/science":
        raise ValueError(
            f"This reproduction supports only dataset_id='lotte/science', got {dataset_id!r}"
        )

    root = os.environ.get("LOTTE_DATA_DIR", DEFAULT_LOTTE_ROOT)
    split_dir = os.path.join(root, "science", "dev")
    corpus_path = os.path.join(split_dir, "collection.tsv")
    query_path = os.path.join(split_dir, "questions.search.tsv")
    qrels_path = os.path.join(split_dir, "qas.search.jsonl")

    if logger:
        logger.info("Loading LoTTE Science dev/search from %s", split_dir)

    corpus: CorpusDict = {}
    with open(corpus_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                doc_id, text = parts
                corpus[doc_id] = {"title": "", "text": text}

    queries: QueriesDict = {}
    with open(query_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                qid, text = parts
                queries[qid] = text

    qrels = load_lotte_qrels_file(qrels_path)
    if max_queries is not None:
        queries = dict(itertools.islice(queries.items(), max_queries))
        qrels = {qid: qrels[qid] for qid in queries if qid in qrels}
    if max_docs is not None:
        corpus = dict(itertools.islice(corpus.items(), max_docs))

    if logger:
        logger.info(
            "LoTTE Science loaded: %d docs | %d queries | %d qrels",
            len(corpus),
            len(queries),
            len(qrels),
        )
    return corpus, queries, qrels


__all__ = [
    "CorpusDict",
    "QueriesDict",
    "QrelsDict",
    "load_dataset_by_name",
    "load_lotte_qrels_file",
]
