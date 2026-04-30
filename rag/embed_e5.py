#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="rag_eval/data")
    ap.add_argument("--model", default="intfloat/e5-large-v2")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-docs", type=int, default=1000000)
    ap.add_argument("--max-queries", type=int, default=200)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")

    if args.max_docs:
        corpus = corpus[: args.max_docs]
    if args.max_queries:
        queries = queries[: args.max_queries]

    corpus_texts = [f"passage: {row['text']}" for row in corpus]
    query_texts = [f"query: {row['question']}" for row in queries]

    model = SentenceTransformer(args.model)
    corpus_emb = model.encode(
        corpus_texts,
        batch_size=args.batch,
        normalize_embeddings=args.normalize,
        show_progress_bar=True,
    )
    query_emb = model.encode(
        query_texts,
        batch_size=args.batch,
        normalize_embeddings=args.normalize,
        show_progress_bar=True,
    )

    corpus_ids = np.array([int(row["doc_id"]) for row in corpus], dtype=np.int64)
    query_ids = np.array([int(row["qid"]) for row in queries], dtype=np.int64)

    np.save(data_dir / "corpus_emb.npy", corpus_emb)
    np.save(data_dir / "query_emb.npy", query_emb)
    np.save(data_dir / "corpus_ids.npy", corpus_ids)
    np.save(data_dir / "query_ids.npy", query_ids)

    print(f"wrote embeddings to {data_dir}")


if __name__ == "__main__":
    main()
