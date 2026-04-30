# ABY3 (3PC) — Efficient Online Search

This directory contains the ABY3-based online search module. It aligns with
the MP-SPDZ modes (single/multi/batch), but runs on ABY3 mixed sharing and
local binary inputs to reduce per-query latency.

## Why ABY3?

ABY3 is integrated for the online query phase because:

- Mixed arithmetic/Boolean sharing speeds up comparisons and membership tests.
- A2B/B2A conversions avoid expensive bit-decomposition.
- The Boolean evaluator reduces rounds for dedup and sorting primitives.

The offline reshuffle and protocol primitives remain in MP-SPDZ as described
in the paper.

## Share format

We reuse the NPZ config and pack ABY3 shares with two-share storage:

- p0 stores (s0, s1)
- p1 stores (s1, s2)
- p2 stores (s2, s0)

Each file is written as two binaries per party, e.g. `layer0_vecs_s0.bin` and
`layer0_vecs_s1.bin`.

The share metadata (`meta.json`) includes `N_real` per layer when available.
This is used to treat out-of-range IDs as dummy rows when public indices are
used for table access.

## Protocol alignment

This implementation follows the ABY 2PC search semantics:

- Membership/dedup is done in MPC, then the selected `id_prime` is revealed.
- Neighbor/Vector/Down table access uses the revealed ID as a public index.
- In multi mode, VG/VD caches are plain (public) data paths. MPC is not used
  to fetch cached values; it is only used for membership/dedup and distance
  arithmetic. This mirrors the ABY design where cache lookup is done in the
  clear after `id_prime` is opened.

Dummy IDs are treated as any ID `>= N_real` (or the provided override), and
public accesses fall back to a dummy row.

## Protocol details

ABY3 uses replicated 3-party arithmetic sharing with Boolean shares for
comparisons. There is no Yao GC backend in ABY3. Boolean operations run in
a GMW-style evaluator (`Sh3BinaryEvaluator`), with A2B/B2A conversions.

## Build

### Prerequisites (libOTe)

ABY3 depends on libOTe and its third-party dependencies (boost, coproto,
macoro, function2). The simplest path is the upstream build helper:

```bash
cd /path/to/aby3
python3 build.py --setup
python3 build.py
```

If you prefer to build libOTe manually (advanced):

```bash
cd /path/to/aby3/libOTe
python3 build.py --setup
python3 build.py --install
```

After installation, the libOTe prefix should contain:

```
/path/to/aby3/libOTe/out/install/linux/lib/cmake/libOTe/libOTeConfig.cmake
```

### Build HNSecW target

1) Sync the ABY3 search source into your ABY3 checkout:

```bash
bash scripts/sync_aby3_examples.sh /path/to/aby3
```

2) Configure and build the target in ABY3:

```bash
bash scripts/build_aby3.sh /path/to/aby3 /path/to/aby3/build
```

To use a custom build directory at runtime, pass `--bin` to the runner or
set `ABY3_BIN`.

If you built libOTe into a custom prefix, set `LIBOTE_PREFIX`:

```bash
export LIBOTE_PREFIX=/path/to/libOTe/install/linux
bash scripts/build_aby3.sh /path/to/aby3 /path/to/aby3/build
```

## Prepare inputs

Generate the shared config (same as MP-SPDZ):

```bash
python3 3pc/mp_spdz/gen_config.py \
  --npz /path/to/plain_hnsw_inputs.npz \
  --out /tmp/hnsw_cfg.json \
  --num-queries 4
```

Pack the HNSW tables and the query shares:

```bash
python3 tools/pack_hnsw_shares.py \
  --npz /path/to/plain_hnsw_inputs.npz \
  --out-dir /tmp/hnsecw_shares_aby3 \
  --parties 3 \
  --format aby3

python3 3pc/aby3/pack_queries_aby3.py \
  --cfg /tmp/hnsw_cfg.json \
  --shares-dir /tmp/hnsecw_shares_aby3 \
  --queries /path/to/queries.npy \
  --num-queries 4
```

If `--queries` is omitted, the packer writes all-zero queries.

## Run (single/multi/batch)

```bash
ABY3_ROOT=/path/to/aby3 \
bash 3pc/aby3/bench/run_search_aby3.sh \
  --cfg /tmp/hnsw_cfg.json \
  --shares /tmp/hnsecw_shares_aby3 \
  --out /tmp/hnsecw_out_aby3 \
  --num-queries 4 \
  --mode batch \
  --output id \
  --progress
```

`--entry-count` must be `1` or `num-queries`. When it is `1`, the single entry
share is broadcast to all queries.

The runner always repacks query shares to match `--num-queries`.

## Outputs

Outputs are ABY3 shares stored under `out/p{0,1,2}`:

- `out_ids_s0.bin`, `out_ids_s1.bin`
- `out_vecs_s0.bin`, `out_vecs_s1.bin` (when `--output vec` or `both`)
