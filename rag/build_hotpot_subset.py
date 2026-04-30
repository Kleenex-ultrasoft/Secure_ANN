#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

from datasets import load_dataset


def maybe_copy_cache(src_root: Path, dst_root: Path, name: str) -> None:
    src = src_root / name
    dst = dst_root / name
    if dst.exists():
        return
    if not src.exists():
        raise FileNotFoundError(f"Cache not found: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def resolve_cache_candidates(primary: Path) -> list[Path]:
    candidates = [primary]
    candidates.append(Path("/home2/fahong/Experiment_12_19/Microbenchmark/bench_panther/rag_eval/.hf_datasets_cache"))
    candidates.append(Path.home() / ".cache" / "huggingface" / "datasets")
    return candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="rag_eval/data")
    ap.add_argument("--n-examples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--cache-dir", default="rag_eval/.hf_datasets_cache")
    ap.add_argument("--src-cache", default="/home2/fahong/.cache/huggingface/datasets")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    copied = False
    for candidate in resolve_cache_candidates(Path(args.src_cache)):
        if (candidate / "hotpot_qa").exists():
            maybe_copy_cache(candidate, cache_dir, "hotpot_qa")
            copied = True
            break
    if not copied:
        os.environ["HF_DATASETS_OFFLINE"] = "0"
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    else:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir)

    ds = load_dataset("hotpot_qa", "distractor", split=args.split)
    if args.n_examples < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.n_examples))

    queries = []
    for i, ex in enumerate(ds):
        queries.append({
            "qid": i,
            "question": ex["question"],
            "answer": ex["answer"],
        })

    with open(out_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for row in queries:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    meta = {
        "dataset": "hotpot_qa_distractor",
        "split": args.split,
        "n_examples": len(queries),
        "seed": args.seed,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {out_dir}/queries.jsonl")


if __name__ == "__main__":
    main()
