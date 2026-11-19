#!/usr/bin/env python3
"""Fail fast if the supplied LoTTE Science split is not the paper split."""

import argparse
import json
import os


def count_lines(path):
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="LOTTE root containing science/dev")
    args = parser.parse_args()
    split = os.path.join(os.path.abspath(os.path.expanduser(args.root)), "science", "dev")
    query_path = os.path.join(split, "questions.search.tsv")
    corpus_path = os.path.join(split, "collection.tsv")
    qrels_path = os.path.join(split, "qas.search.jsonl")
    counts = {
        "queries": count_lines(query_path),
        "documents": count_lines(corpus_path),
        "qrel_queries": count_lines(qrels_path),
    }
    if counts["queries"] != 538 or counts["documents"] != 343642:
        raise SystemExit(f"unexpected Science split: {counts}")
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

