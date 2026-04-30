# RAG evaluation (HotpotQA + MS MARCO 1M)

This folder contains the end-to-end RAG pipeline used in the paper:
- Queries: 200 multi-hop questions from HotpotQA (dev split).
- Corpus: MS MARCO passage corpus (1M docs).
- Embedding model: `intfloat/e5-large-v2`.
- Generator: `Qwen/Qwen2.5-7B-Instruct`.
- Retrieval: HNSW, Top-10, compare full (standard HNSW) vs fixed-step (`ef=k+6+tau`, `tau=1..6`).
- Metrics: Exact Match (EM) and token-level F1.

## Quick start

```bash
cd HNSecW/rag
./run_rag_eval.sh
```

The script writes:
- `data/queries.jsonl`, `data/corpus.jsonl`, embeddings, and retrieval outputs
- `out/answers_*.jsonl` and `out/rag_metrics.csv`

## Dataset sources

This package is source-only and does not include datasets or model weights.
You can provide local copies or let HuggingFace download them:

- HotpotQA: the script will use a local HF cache if present under
  `/home2/fahong/Experiment_12_19/Microbenchmark/bench_panther/rag_eval/.hf_datasets_cache`,
  otherwise it falls back to online download.
- MS MARCO: set `MS_MARCO_COLLECTION=/path/to/collection.tsv` to use a local
  MS MARCO collection; otherwise the script downloads from HF.

## Reproducibility notes

All steps are deterministic given `--seed` in `build_hotpot_subset.py` and the
same model checkpoints. The pipeline is configured for 1M documents and the
SOTA encoder/generator described in the paper.
