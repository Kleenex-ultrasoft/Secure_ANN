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


def write_corpus(rows, out_path: Path, max_docs: int | None) -> int:
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            n += 1
            if max_docs is not None and n >= max_docs:
                break
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="rag_eval/data")
    ap.add_argument("--collection", default="", help="Path to MS MARCO collection.tsv")
    ap.add_argument("--max-docs", type=int, default=1000000)
    ap.add_argument("--hf-dataset", default="ms_marco")
    ap.add_argument("--hf-config", default="v2.1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache-dir", default="rag_eval/.hf_datasets_cache")
    ap.add_argument("--src-cache", default="/home2/fahong/.cache/huggingface/datasets")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "corpus.jsonl"

    if args.collection:
        n = 0
        with open(args.collection, "r", encoding="utf-8") as f, \
                open(out_path, "w", encoding="utf-8") as out:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                doc_id, text = parts
                row = {"doc_id": int(doc_id), "text": text}
                out.write(json.dumps(row, ensure_ascii=True) + "\n")
                n += 1
                if args.max_docs and n >= args.max_docs:
                    break
        print(f"wrote {n} docs to {out_path}")
        return

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    copied = False
    for candidate in resolve_cache_candidates(Path(args.src_cache)):
        if (candidate / args.hf_dataset).exists():
            maybe_copy_cache(candidate, cache_dir, args.hf_dataset)
            copied = True
            break

    if copied:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    else:
        os.environ["HF_DATASETS_OFFLINE"] = "0"
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir)

    ds = load_dataset(args.hf_dataset, args.hf_config, split=args.split)
    rows = []
    for i, ex in enumerate(ds):
        if "passage" in ex:
            text = ex["passage"]
        elif "text" in ex:
            text = ex["text"]
        elif "contents" in ex:
            text = ex["contents"]
        else:
            continue
        doc_id = int(ex.get("doc_id", i))
        rows.append({"doc_id": doc_id, "text": text})
        if args.max_docs and len(rows) >= args.max_docs:
            break

    n = write_corpus(rows, out_path, max_docs=None)
    print(f"wrote {n} docs to {out_path}")


if __name__ == "__main__":
    main()
