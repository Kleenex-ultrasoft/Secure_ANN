# SORT-3PC baseline (exact top-k)

This baseline computes exact top-k by scanning all vectors, sorting by
inner-product distance, and using ORAM to fetch the top-k vectors
(DORAM-style retrieval). It matches the HNSecW distance definition.
The sort uses MP-SPDZ's `Matrix.sort()` which under the replicated-ring
backend lowers to the replicated radix sort of Asharov et al. (the
algorithmic identity used by Three-Party-Sorting).

## Files

- `Programs/Source/sort_3pc.mpc`: MPC program (full distance + radix sort + ORAM fetch).
- `gen_sort_config.py`: build a JSON config from vectors.
- `gen_sort_inputs.py`: generate MP-SPDZ input shares.
- `bench/run_sort_3pc.sh`: end-to-end run script (compile + run).
- `parse_sort_outputs.py`: split MP-SPDZ outputs into per-party share files.

## Build and run

```bash
MP_SPDZ=/path/to/MP-SPDZ

python3 gen_sort_config.py \
  --vecs /path/to/vecs.npy \
  --out /tmp/sort_cfg.json \
  --k 10 \
  --num-queries 4 \
  --vec-bits 8

bench/run_sort_3pc.sh \
  --cfg /tmp/sort_cfg.json \
  --vecs /path/to/vecs.npy \
  --queries /path/to/queries.npy \
  --num-queries 4 \
  --output both \
  --oram optimal
```

Notes:
- `--vecs` accepts `.npy` or `.npz` (use `--layer` for `vecs_<layer>` keys).
- `--queries` is optional; if omitted, zero queries are used.
- `--output` can be `id`, `vec`, or `both`.
- `--oram` selects ORAM implementation: `optimal`, `recursive`, or `linear`.
- Use `--protocol ring|bin` and `--ring-bits` to select MP-SPDZ backend.
- Add `--reveal` to print the top-1 ID (debug only).

## Parse outputs

The run script writes `Binary-Output-P*` under `$MP_SPDZ/Player-Data`.
Use `parse_sort_outputs.py` to split the shares:

```bash
python3 parse_sort_outputs.py \
  --cfg /tmp/sort_cfg.json \
  --mp-spdz-dir "$MP_SPDZ/Player-Data" \
  --out-dir /tmp/sort_out \
  --num-queries 4 \
  --output both
```

Outputs are written to `/tmp/sort_out/p0/topk_ids.bin`,
`/tmp/sort_out/p1/topk_ids.bin`, etc. Vector outputs use
`topk_vectors.bin`.

## Small-scale correctness check

For a quick sanity check, run a tiny instance and compare with plaintext
top-k:

```bash
python3 - <<'PY'
import numpy as np
rng = np.random.default_rng(0)
vecs = rng.integers(0, 8, size=(64, 8), dtype=np.int64)
queries = rng.integers(0, 8, size=(4, 8), dtype=np.int64)
np.save("/tmp/sort_vecs.npy", vecs)
np.save("/tmp/sort_queries.npy", queries)
PY

python3 gen_sort_config.py \
  --vecs /tmp/sort_vecs.npy \
  --out /tmp/sort_cfg.json \
  --k 5 \
  --num-queries 4 \
  --vec-bits 8

bench/run_sort_3pc.sh \
  --cfg /tmp/sort_cfg.json \
  --vecs /tmp/sort_vecs.npy \
  --queries /tmp/sort_queries.npy \
  --num-queries 4 \
  --output both \
  --oram optimal

python3 parse_sort_outputs.py \
  --cfg /tmp/sort_cfg.json \
  --mp-spdz-dir "$MP_SPDZ/Player-Data" \
  --out-dir /tmp/sort_out \
  --output both

python3 verify_sort_3pc.py \
  --cfg /tmp/sort_cfg.json \
  --vecs /tmp/sort_vecs.npy \
  --queries /tmp/sort_queries.npy \
  --out-dir /tmp/sort_out \
  --output both
```

If there are distance ties inside the top-k set, the checker accepts any
valid tie ordering but emits a warning.
