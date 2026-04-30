#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import faiss
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="rag_eval/data")
    ap.add_argument("--out-dir", default="rag_eval/data")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=96)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--tau-max", type=int, default=6)
    ap.add_argument(
        "--ef-base",
        type=int,
        default=None,
        help="Base efSearch before tau. Defaults to k + ef_offset.",
    )
    ap.add_argument(
        "--ef-offset",
        type=int,
        default=6,
        help="If ef-base is unset, efSearch = k + ef_offset + tau.",
    )
    ap.add_argument(
        "--ef-full",
        type=int,
        default=200,
        help="efSearch for standard HNSW (upper-bound retrieval).",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_emb = np.load(data_dir / "corpus_emb.npy")
    query_emb = np.load(data_dir / "query_emb.npy")
    corpus_ids = np.load(data_dir / "corpus_ids.npy")

    dim = corpus_emb.shape[1]
    index = faiss.IndexHNSWFlat(dim, args.m)
    index.hnsw.efConstruction = args.efc
    index.add(corpus_emb)

    base = args.ef_base if args.ef_base is not None else args.k + args.ef_offset
    for tau in range(args.tau_max + 1):
        ef_search = base + tau
        index.hnsw.efSearch = ef_search
        D, I = index.search(query_emb, args.k)

        out_path = out_dir / f"retrieval_tau{tau}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for qi, idxs in enumerate(I):
                doc_ids = [int(corpus_ids[j]) for j in idxs]
                f.write(json.dumps({"qid": int(qi), "tau": tau, "doc_ids": doc_ids}) + "\n")
        print(f"wrote {out_path} (efSearch={ef_search})")

    if args.ef_full > 0:
        index.hnsw.efSearch = args.ef_full
        D, I = index.search(query_emb, args.k)
        out_path = out_dir / "retrieval_hnsw_full.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for qi, idxs in enumerate(I):
                doc_ids = [int(corpus_ids[j]) for j in idxs]
                f.write(json.dumps({"qid": int(qi), "mode": "full", "doc_ids": doc_ids}) + "\n")
        print(f"wrote {out_path} (efSearch={args.ef_full})")


if __name__ == "__main__":
    main()
