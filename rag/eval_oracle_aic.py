#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="rag_eval/data")
    ap.add_argument("--out", default="rag_eval/out/rag_oracle_aic.csv")
    ap.add_argument("--tau-max", type=int, default=6)
    ap.add_argument("--max-ctx-chars", type=int, default=20000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    corpus = {r["doc_id"]: r for r in read_jsonl(data_dir / "corpus.jsonl")}
    queries = {r["qid"]: r for r in read_jsonl(data_dir / "queries.jsonl")}

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("setting,n_queries,answer_in_context\n")

        # no-retrieval baseline
        f.write(f"no_retrieval,{len(queries)},0.0000\n")

        for tau in range(args.tau_max + 1):
            retrieval_path = data_dir / f"retrieval_tau{tau}.jsonl"
            rows = read_jsonl(retrieval_path)

            hit = 0
            total = 0
            for row in rows:
                qid = int(row["qid"])
                answer = normalize(queries[qid]["answer"])
                if not answer:
                    total += 1
                    continue

                doc_ids = row.get("doc_ids", [])
                texts = [corpus[int(did)]["text"] for did in doc_ids if int(did) in corpus]
                ctx = " ".join(texts)
                if args.max_ctx_chars > 0:
                    ctx = ctx[: args.max_ctx_chars]
                ctx = normalize(ctx)

                if answer in ctx:
                    hit += 1
                total += 1

            aic = hit / max(total, 1)
            label = f"hnsw_tau{tau}"
            f.write(f"{label},{total},{aic:.4f}\n")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
