#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_BASE=/home2/fahong/miniconda3/bin/python
DATA_DIR="$ROOT/data"
OUT_DIR="$ROOT/out"
mkdir -p "$DATA_DIR" "$OUT_DIR"

# 1) Sample HotpotQA queries (200)
HF_DATASETS_OFFLINE=1 HF_DATASETS_CACHE="$ROOT/.hf_datasets_cache" \
  "$PY_BASE" "$ROOT/build_hotpot_subset.py" \
  --out-dir "$DATA_DIR" --n-examples 200 --seed 42 --split validation

# 2) Build MS MARCO 1M corpus (set MS_MARCO_COLLECTION if available)
if [[ -n "${MS_MARCO_COLLECTION:-}" ]]; then
  "$PY_BASE" "$ROOT/prepare_msmarco_corpus.py" \
    --out-dir "$DATA_DIR" --collection "$MS_MARCO_COLLECTION" --max-docs 1000000
else
  HF_DATASETS_OFFLINE=0 HF_DATASETS_CACHE="$ROOT/.hf_datasets_cache" \
    "$PY_BASE" "$ROOT/prepare_msmarco_corpus.py" \
    --out-dir "$DATA_DIR" --max-docs 1000000
fi

# 3) Embed corpus & queries with e5-large-v2
"$PY_BASE" "$ROOT/embed_e5.py" \
  --data-dir "$DATA_DIR" --batch 64 --max-docs 1000000 --max-queries 200 --normalize

# 4) Retrieve (standard HNSW + fixed-step via efSearch=k+6+tau)
conda run -n faiss-omp python "$ROOT/retrieve_hnsw.py" \
  --data-dir "$DATA_DIR" --out-dir "$DATA_DIR" --k 10 --m 96 --efc 200 --tau-max 6 --ef-full 200

# 5) Generate answers
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY_BASE" "$ROOT/generate_answers.py" \
  --data-dir "$DATA_DIR" \
  --out "$OUT_DIR/answers_no_retrieval.jsonl"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY_BASE" "$ROOT/generate_answers.py" \
  --data-dir "$DATA_DIR" \
  --retrieval "$DATA_DIR/retrieval_hnsw_full.jsonl" \
  --out "$OUT_DIR/answers_hnsw_full.jsonl"

for t in 1 2 3 4 5 6; do
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PY_BASE" "$ROOT/generate_answers.py" \
    --data-dir "$DATA_DIR" \
    --retrieval "$DATA_DIR/retrieval_tau${t}.jsonl" \
    --out "$OUT_DIR/answers_tau${t}.jsonl"
done

# 6) Evaluate
"$PY_BASE" "$ROOT/eval_answers.py" \
  --answers "$OUT_DIR/answers_no_retrieval.jsonl" --label "no_retrieval" --out "$OUT_DIR/rag_metrics.csv"
"$PY_BASE" "$ROOT/eval_answers.py" \
  --answers "$OUT_DIR/answers_hnsw_full.jsonl" --label "hnsw_full" --out "$OUT_DIR/rag_metrics.csv"
for t in 1 2 3 4 5 6; do
  "$PY_BASE" "$ROOT/eval_answers.py" \
    --answers "$OUT_DIR/answers_tau${t}.jsonl" --label "hnsw_tau${t}" --out "$OUT_DIR/rag_metrics.csv"
done

echo "Done. Metrics in $OUT_DIR/rag_metrics.csv"
