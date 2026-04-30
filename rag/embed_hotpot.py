#!/usr/bin/env python3
import argparse
import json
import os
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
    ap.add_argument("--out-dir", default="rag_eval/data")
    ap.add_argument("--max-docs", type=int, default=1000000)
    ap.add_argument("--max-queries", type=int, default=200)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")

    if args.max_docs:
        corpus = corpus[: args.max_docs]
    if args.max_queries:
        queries = queries[: args.max_queries]

    model = SentenceTransformer(args.model)

    corpus_texts = [f"passage: {r['text']}" for r in corpus]
    query_texts = [f"query: {r['question']}" for r in queries]

    corpus_emb = model.encode(
        corpus_texts,
        batch_size=args.batch,
        show_progress_bar=True,
        normalize_embeddings=args.normalize,
    )
    query_emb = model.encode(
        query_texts,
        batch_size=args.batch,
        show_progress_bar=True,
        normalize_embeddings=args.normalize,
    )

    np.save(out_dir / "corpus_emb.npy", corpus_emb.astype(np.float32))
    np.save(out_dir / "query_emb.npy", query_emb.astype(np.float32))
    np.save(out_dir / "corpus_ids.npy", np.array([r["doc_id"] for r in corpus], dtype=np.int64))
    np.save(out_dir / "query_ids.npy", np.array([r["qid"] for r in queries], dtype=np.int64))

    print(f"wrote embeddings to {out_dir}")


if __name__ == "__main__":
    main()
