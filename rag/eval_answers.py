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
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def f1_score(pred: str, truth: str) -> float:
    p = normalize(pred).split()
    t = normalize(truth).split()
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    common = set(p) & set(t)
    if not common:
        return 0.0
    # count overlap
    overlap = 0
    for tok in common:
        overlap += min(p.count(tok), t.count(tok))
    prec = overlap / len(p)
    rec = overlap / len(t)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="rag_eval/data/queries.jsonl")
    ap.add_argument("--answers", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    queries = {r["qid"]: r for r in read_jsonl(Path(args.queries))}
    answers = read_jsonl(Path(args.answers))

    em = 0.0
    f1 = 0.0
    n = 0
    for row in answers:
        qid = int(row["qid"])
        pred = row.get("answer", "")
        truth = queries[qid]["answer"]
        em += 1.0 if normalize(pred) == normalize(truth) else 0.0
        f1 += f1_score(pred, truth)
        n += 1

    em /= max(n, 1)
    f1 /= max(n, 1)

    out_path = Path(args.out)
    header = not out_path.exists()
    with open(out_path, "a", encoding="utf-8") as f:
        if header:
            f.write("setting,n,em,f1\n")
        f.write(f"{args.label},{n},{em:.4f},{f1:.4f}\n")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
