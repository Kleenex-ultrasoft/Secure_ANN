# PANTHER generalized — DORAM + secret DB + secret distance (source mod)

This folder contains a **source-level modification** of upstream
PANTHER (Li et al., USENIX Security 2025) that replaces:

| Original primitive          | Replacement                                |
|-----------------------------|--------------------------------------------|
| FHE-PIR (`SealMPir`)        | 2PC DORAM bridged to [Duoram (Vadapalli, Henry, Goldberg, USENIX Sec '23)](https://git-crysp.uwaterloo.ca/avadapal/duoram) `cpir-read/` |
| FHE-distance (`DoDistance…`)| SS-IP via `kernel::hal::matmul` on shares  |
| Plaintext server-side DB    | Secret-shared database (additive shares)   |

The substitutions are applied **in place** to PANTHER's existing C++
benchmark binary; the PANTHER k-means ANN training, IVF index layout,
batched-argmin, and garbled-circuit top-k composition are preserved
verbatim.  This folder ships the fully patched source.

## Files

```
OpenPanther/                                            # upstream PANTHER tree
  experimental/panther/benchmark/
    random_panther_server.cc                            # original (unchanged)
    random_panther_client.cc                            # original (unchanged)
    random_panther_doram.cc                             # NEW — source-modified
    random_panther_server.h                             # reused by both
  experimental/panther/protocol/                        # upstream protocol code
  experimental/panther/BUILD.bazel                      # +1 target: random_panther_doram

patches/
  random_panther_doram_substitution.patch               # unified diff vs upstream

run_oram_bench.sh                                       # MP-SPDZ DORAM cost bench
run_panther_topk_grid_tc.sh                             # upstream PANTHER topk grid
panther_estimator.py                                    # cost-analysis only (auxiliary)
```

## Build

OpenPanther is a subtree of [secretflow/spu](https://github.com/secretflow/spu).
To compile our doram-modified binary you need the full SPU repo:

```bash
git clone --depth 1 https://github.com/secretflow/spu.git
cp -r baselines/panther_generalize/OpenPanther/experimental/panther \
      spu/experimental/

cd spu
bazel build -c opt \
    //experimental/panther:random_panther_doram
```

Build prerequisites are SPU's standard set: bazel ≥ 6.0, gcc/clang with
C++20, plus the third-party deps SPU's `MODULE.bazel` pins (LLVM, abseil,
spdlog, yacl, SEAL, EMP, microsoft/SealPIR).

SIFT-100K preset baked into `random_panther_doram.cc`; SIFT-1M /
DEEP1M / Amazon presets compile via `--copt=-DTEST_SIFT|-DTEST_DEEP1M|…`.

## Run (LAN, single host, 2-party Cheetah)

Step 4 (DORAM) is bridged at runtime to upstream
[Duoram (USENIX Sec '23)](https://git-crysp.uwaterloo.ca/avadapal/duoram)
`cpir-read/cxx/spir_test{0,1}`.  Build Duoram once, then point
`DUORAM_BIN_DIR` at it before launching:

```bash
# One-time: build Duoram cpir-read
git clone https://git-crysp.uwaterloo.ca/avadapal/duoram
cd duoram/cpir-read && cargo build --release && cd cxx && make
export DUORAM_BIN_DIR=$PWD
```

End-to-end demo (SIFT-100K, real query #0 of `sift_query.npy`):

```bash
bash demo.sh
# ->  Total Latency / Total Communication (measured)
# ->  Plaintext IVF top-10 + recall@10 (panther's algorithmic
#     baseline on real SIFT-100K, identical across PIR primitives)
```

Or invoke the binary directly (Step 4 forks Duoram spir_test{rank}):

```bash
PARTIES=127.0.0.1:9530,127.0.0.1:9531
./bazel-bin/experimental/panther/random_panther_doram --rank=0 --parties=$PARTIES &
./bazel-bin/experimental/panther/random_panther_doram --rank=1 --parties=$PARTIES
```

## Substitution map (correspondence with `random_panther_server.cc`)

| Upstream PANTHER step                            | Our replacement                                                  |
|---------------------------------------------------|------------------------------------------------------------------|
| `auto encoded_db = PirData(...);`                 | (deleted — DB is shared, no PIR encoding needed)                 |
| `PrepareMpirServer/Client(...)`                   | (deleted)                                                        |
| `dis_server.RecvQuery / dis_client.GenerateQuery` | `query_s = PackAndShare1D(...)` (rank-aware seal)                |
| `dis_server.DoDistanceCmpWithH2A(...)` (Step 1)   | `kernel::hal::matmul(q_row, cluster_T)` (SS-IP via cheetah_dot)  |
| `PrepareBatchArgmin` + `BatchMinProtocol` (Step 2)| **kept verbatim** — linked from `:batch_min`                     |
| `GcTopkCluster` (Step 3)                          | **kept verbatim** — linked from `:topk`                          |
| `mpir_server.DoMultiPirAnswer / mpir_client.DoMultiPirQuery` (Step 4) | bridge to upstream Duoram `cpir-read/cxx/spir_test{0,1}` via fork+exec |
| `FixPirResultOpt(...)`                            | (deleted — DORAM returns plain shares)                           |
| Step 5 second `DoDistanceCmpWithH2A`              | `kernel::hal::matmul(q_row, retrieved_T)`                        |
| `GcEndTopk` (Step 6)                              | **kept verbatim** — linked from `:topk`                          |

`patches/random_panther_doram_substitution.patch` is the unified diff
between `random_panther_server.cc` and `random_panther_doram.cc`.

## Notes on threat model

Original PANTHER:
- Server holds the database **in plaintext**
- Client holds the query **in plaintext**
- FHE-PIR + FHE-distance protect the *query content* but assume the
  server's DB is non-secret

Our generalized version (2-party Cheetah, same as upstream PANTHER):
- **Both** parties hold additive shares of the DB and the query
  (cluster centroids, IVF directory, all members, query)
- SS inner-product (Steps 1 & 5) via SPU's `kernel::hal::matmul` over
  Cheetah ring
- Oblivious cluster fetch (Step 4) via upstream Duoram 2PC read
  (Vadapalli et al., USENIX Sec '23, `cpir-read/`), bridged into our
  binary via fork+exec at runtime
- Steps 2, 3, 6 (`BatchMinProtocol` / `GcTopkCluster` / `GcEndTopk`)
  reused unmodified from upstream PANTHER — outputs of GcTopkCluster
  reveal cluster ids to client by PANTHER's design (used to drive Step 4
  DORAM); GcEndTopk reveals final top-K ids — exact same reveal pattern
  as upstream PANTHER, no extra reveal added
- This matches the threat model of fully-MPC ANN systems (HNSecW,
  PANTHER-with-secret-DB ablations in §7 of the paper)

