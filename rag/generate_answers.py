#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_prompt(question: str, contexts: list[str] | None) -> str:
    if contexts:
        ctx = "\n\n".join(contexts)
        return (
            "You are a helpful assistant. Use the provided context to answer the question.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
        )
    return (
        "You are a helpful assistant. Answer the question.\n\n"
        f"Question: {question}\nAnswer:"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="rag_eval/data")
    ap.add_argument("--retrieval", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--max-ctx-chars", type=int, default=3000)
    args = ap.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    data_dir = Path(args.data_dir)
    corpus = {r["doc_id"]: r for r in read_jsonl(data_dir / "corpus.jsonl")}
    queries = {r["qid"]: r for r in read_jsonl(data_dir / "queries.jsonl")}

    retrieval = None
    if args.retrieval:
        retrieval = read_jsonl(Path(args.retrieval))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(device)
    model.eval()

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        if retrieval is None:
            items = [{"qid": qid, "doc_ids": []} for qid in queries.keys()]
        else:
            items = retrieval

        for item in items:
            qid = int(item["qid"])
            q = queries[qid]["question"]
            ctxs = []
            if item.get("doc_ids"):
                for did in item["doc_ids"]:
                    ctxs.append(corpus[int(did)]["text"])
            # truncate context by chars
            if ctxs:
                joined = "\n\n".join(ctxs)
                joined = joined[: args.max_ctx_chars]
                ctxs = [joined]

            prompt = build_prompt(q, ctxs if ctxs else None)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            # extract answer after last 'Answer:' if present
            ans = text.split("Answer:")[-1].strip()
            f.write(json.dumps({"qid": qid, "answer": ans}) + "\n")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
