# Secure ANN

Source code for our **secure approximate nearest neighbour (ANN) search**
under secret-shared MPC, accompanying the **extended version of the
HNSecW paper** (`paper.pdf` in this repo).  The package is source-only
and excludes raw datasets, logs, and build outputs.

## Repository layout

```
2pc/                  ABY (2PC) implementation: single, multi, batch query
3pc/aby3/             ABY3 (3PC) backend (faster mixed arithmetic/Boolean SS)
3pc/mp_spdz/          MP-SPDZ (3PC) backend + .mpc programs
baselines/
  panther_generalize/   PANTHER source-modified (DORAM + SS-IP + secret DB)
  sort_3pc/           Exact top-k 3PC SORT baseline (replicated radix sort)
rag/                  End-to-end RAG eval (HotpotQA + MS MARCO 1M)
scripts/              Build / sync helpers + dataset download
docs/                 Protocol notes (online metrics, plaintext HNSW, network)
```

## Implementations

|                  | 2PC (ABY)            | 3PC (MP-SPDZ + ABY3) |
| ---------------- | -------------------- | -------------------- |
| Single query     | `2pc/src/hnsecw/hnsecw_single_b2y.cpp` | `3pc/mp_spdz/Programs/Source/hnsecw_search_3pc.mpc`<br>`3pc/aby3/src/hnsecw_search_aby3.cpp` (alternative backend) |
| Multiple queries | `2pc/src/hnsecw/hnsecw_multi_b2y_dyn.cpp` | `3pc/mp_spdz/Programs/Source/hnsecw_search_3pc_multi.mpc` |
| Batch queries    | `2pc/src/hnsecw/hnsecw_batch_b2y.cpp` | `3pc/mp_spdz/Programs/Source/hnsecw_search_3pc_batch.mpc` |

The deployed search parameters from §4.1 (`ef^{(0)} = k+10`,
`Δ̄^{(0)} = ⌈log₂(N·d/M)⌉ + C`, `C = 7`) are derived automatically
by `2pc/bench/build_npz.py` and `3pc/mp_spdz/bench/gen_mpspdz_inputs.py`
from the dataset shape; pass `--k`, `--ef_base`, `--tau_base`, or
`--tau_upper` to override.  The reshuffled per-layer tables
`G_\ell`, `D_\ell`, and the dummy pool `DUM_\ell` are built offline by
`build_hnsecw_index.sh`.

## Quick start

### 0. Dependencies

Tested on Ubuntu 20.04 / 24.04 with `g++` ≥ 10, `cmake` ≥ 3.16,
Python 3.9+.

#### 0.1 System packages

```bash
sudo apt-get install -y build-essential cmake git python3 python3-pip \
  libssl-dev libboost-all-dev libgmp-dev libsodium-dev libntl-dev \
  libtool autoconf automake yasm
pip install hnswlib numpy scipy
```

#### 0.2 MPC frameworks

Each MPC backend is a separate external dependency.  Install only what
you need; the table below maps each backend to which parts of this
repository require it.

| Framework      | Required for                                                          | Build (one-time)                                                                  |
| -------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **ABY**        | `2pc/src/hnsecw/` (paper-default 2PC HNSecW)                          | `git clone https://github.com/encryptogroup/ABY && cmake -DABY_BUILD_EXE=ON ...`   |
| **MP-SPDZ**    | `2pc/mp_spdz/` (alt 2PC), `3pc/mp_spdz/` (3PC HNSecW), `baselines/sort_3pc/`, `baselines/panther_generalize/run_oram_bench.sh` | `git clone https://github.com/data61/MP-SPDZ && make -j tldr replicated-ring-party.x semi2k-party.x` |
| **OpenPanther** | `baselines/panther_generalize/` only (source-modified PANTHER)       | `git clone https://github.com/secretflow/spu` and place our `experimental/panther` into it; `bazel build` (see baseline README) |
| **ABY3** *(optional)* | `3pc/aby3/` alternative 3PC backend                            | `git clone https://github.com/ladnir/aby3 && cmake ...`                            |

Set the matching env vars before running anything:

```bash
export ABY_ROOT=/path/to/ABY                       # for 2pc/src/hnsecw
export MPSPDZ_ROOT=/path/to/MP-SPDZ                # everything MP-SPDZ-based
export SPU_ROOT=/path/to/spu                       # for panther_generalize
```

##### ABY

```bash
git clone https://github.com/encryptogroup/ABY.git "$ABY_ROOT"
cd "$ABY_ROOT"
cmake -B build -S . -DABY_BUILD_EXE=ON \
      -DCMAKE_CXX_FLAGS='-include cstdlib'        # ABY misses a stdlib include
cmake --build build -j

# Wire our 2PC HNSecW sources into ABY's example tree, then rebuild.
rsync -a 2pc/src/hnsecw "$ABY_ROOT/src/examples/"
cmake --build "$ABY_ROOT/build" -j hnsecw_cli
```

ABY has two known upstream issues we work around: [#114
"Erroneous Multiplication Results"](https://github.com/encryptogroup/ABY/issues/114)
and [#152 "IKNP base-OT thread-unsafe"](https://github.com/encryptogroup/ABY/issues/152).
The mitigations are documented in `2pc/README.md` and applied via
`patches/aby_arith_reset_clear_mvC.patch` (run-time per-query process
isolation also works around them — see `scripts/run_multi_via_single.sh`).

##### MP-SPDZ

```bash
git clone --depth 1 https://github.com/data61/MP-SPDZ.git "$MPSPDZ_ROOT"
cd "$MPSPDZ_ROOT"

# Use system g++ (clang++ is also fine but slower to set up).
# Most paper experiments only need replicated-ring-party.x and
# semi2k-party.x.  `make tldr` builds all the common deps.
make -j tldr
make -j replicated-ring-party.x        # 3PC honest-majority ring (used everywhere)
make -j semi2k-party.x                 # 2PC semi-honest ring (used by 2pc/mp_spdz)
```

If your `libsodium` / `gmpxx` come from conda, point MP-SPDZ's CONFIG.mine
at them (we ship an example in `2pc/mp_spdz/README.md`).  No special
patches needed.

##### OpenPanther (for `baselines/panther_generalize/` only)

```bash
git clone --depth 1 https://github.com/secretflow/spu.git "$SPU_ROOT"
cp -r baselines/panther_generalize/OpenPanther/experimental/panther \
      "$SPU_ROOT/experimental/"
cd "$SPU_ROOT" && bazel build -c opt \
      //experimental/panther:random_panther_doram
# bazel ≥ 6.0 required; first build ~1-2 h (LLVM, abseil, SEAL, EMP, yacl).
```

Skip this step if you only run HNSecW + sort_3pc (MP-SPDZ-based) —
neither needs SPU/bazel.

### 1. Get a small dataset

```bash
bash scripts/download_demo_dataset.sh ~/hnsecw_build/datasets
# downloads SIFT-1M (texmex.irisa.fr) and writes the first 10K vectors
# as ~/hnsecw_build/datasets/sift_base.npy so demo.sh / build_hnsecw_index.sh
# can run end-to-end.
```

Public download links for the 10 datasets reported in the paper:

| Dataset              | Public download landing page                       |
| -------------------- | -------------------------------------------------- |
| SIFT-1M / GIST-1M / BIGANN-1B | <http://corpus-texmex.irisa.fr/>          |
| Deep1B / DEEP-100M   | <http://sites.skoltech.ru/compvision/noimi/>       |
| MNIST-60K            | <http://yann.lecun.com/exdb/mnist/>                |
| Fashion-60K          | <https://github.com/zalandoresearch/fashion-mnist> |
| SPACEV-1B / SSNPP-1B | <https://big-ann-benchmarks.com/>                  |
| LAION                | <https://laion.ai/blog/laion-5b/>                  |
| MS MARCO             | <https://microsoft.github.io/msmarco/>             |

### 2. Build a reshuffled index

```bash
bash build_hnsecw_index.sh ~/hnsecw_build/datasets/sift_base.npy /tmp/hnsecw_index
# Output:
#   /tmp/hnsecw_index/plain_hnsw_inputs.npz
#   /tmp/hnsecw_index/meta.json
```

### 3. Run a query in MPC

#### 2PC (ABY)

```bash
ABY_ROOT=/path/to/ABY
rsync -a ./2pc/src/hnsecw "$ABY_ROOT/src/examples/hnsecw"
cmake -S "$ABY_ROOT" -B "$ABY_ROOT/build" -DABY_BUILD_EXE=ON
cmake --build "$ABY_ROOT/build" -j

./2pc/bench/run_single_query.sh /tmp/hnsecw_index/plain_hnsw_inputs.npz
./2pc/bench/run_multi_query.sh  /tmp/hnsecw_index/plain_hnsw_inputs.npz --num_queries 32
./2pc/bench/run_batch_query.sh  /tmp/hnsecw_index/plain_hnsw_inputs.npz --num_queries 32
```

#### 3PC (MP-SPDZ replicated-ring)

The MP-SPDZ backend uses `replicated-ring-party.x` (3-party
honest-majority, replicated additive ring sharing).  Same revealed-fetch
design as the 2PC ABY path: per-iter `id'` revealed once a layer to
plaintext-fetch graph rows + base vectors, while the per-layer
reshuffle (paper §4.3) hides the access pattern.  The per-iter sort
lowers to Asharov et al.'s replicated radix sort under the hood.

```bash
# Build MP-SPDZ replicated-ring backend (one-time)
cd $MPSPDZ_ROOT && make -j replicated-ring-party.x && cd -

# Single query
NPZ=/tmp/hnsecw_index/plain_hnsw_inputs.npz \
QUERY=/path/to/queries.npy \
MPSPDZ_ROOT=$MPSPDZ_ROOT \
  bash 3pc/mp_spdz/bench/run_search_3pc.sh
# ->  Recall@10        = 90.00%
# ->  Online latency  = 1.770 s/query

# Multi (Q queries serialized; per-query entry chains computed in plaintext)
NPZ=... QUERY=... MODE=multi NUM_Q=4 \
  bash 3pc/mp_spdz/bench/run_search_3pc.sh
# ->  Recall@10        = 95.00%  (avg over 4 queries)
# ->  Online latency  = 1.623 s/query  (avg over 4 queries)

# Batch (Q queries lockstep; sort circuit shared across queries)
NPZ=... QUERY=... MODE=batch NUM_Q=4 \
  bash 3pc/mp_spdz/bench/run_search_3pc.sh
# ->  Recall@10        = 95.00%  (avg over 4 queries)
# ->  Online latency  = 1.707 s/query  (avg over 4 queries)
```

Numbers above are SIFT-100K / M=32 / ef=32 / K=10, single LAN host.

#### Baselines

Two baselines for fair comparison; both ship with end-to-end run scripts.

```bash
# (1) PANTHER generalized — source-level modification of upstream
#     OpenPanther (Li et al., USENIX Sec 2025) that swaps:
#         FHE-PIR        → 2PC DORAM bridged to upstream Duoram cpir-read
#                          (Vadapalli, Henry, Goldberg, USENIX Sec '23)
#         FHE-distance   → SS-IP via kernel::hal::matmul on shares
#         Plaintext DB   → secret-shared DB (additive shares both ranks)
#     while preserving PANTHER's k-means ANN training + IVF index +
#     batched-argmin + GC top-k composition.
#
#     New file: baselines/panther_generalize/OpenPanther/experimental/
#               panther/benchmark/random_panther_doram.cc
#     Diff:     baselines/panther_generalize/patches/
#               random_panther_doram_substitution.patch
#
#     Build (requires SPU repo + bazel):
git clone --depth 1 https://github.com/secretflow/spu.git
cp -r baselines/panther_generalize/OpenPanther/experimental/panther \
      spu/experimental/
cd spu && bazel build -c opt \
      //experimental/panther:random_panther_doram
#     Run (LAN, single host):
./bazel-bin/.../random_panther_doram --rank=0 \
      --parties=127.0.0.1:9530,127.0.0.1:9531 &
./bazel-bin/.../random_panther_doram --rank=1 \
      --parties=127.0.0.1:9530,127.0.0.1:9531

# (2) Exact top-k 3PC SORT — distance over all N vectors + replicated
#     radix sort (Asharov et al.) + ORAM fetch of top-k.  Runs end-to-end:
MP_SPDZ=$MPSPDZ_ROOT bash baselines/sort_3pc/bench/run_sort_3pc.sh \
    --cfg /tmp/sort_cfg.json --vecs /path/to/vecs.npy \
    --queries /path/to/queries.npy --num-queries 4 --output id --reveal
```

See each subfolder's `README.md` for input-prep, parameter knobs, and
the MPC threat-model layout.

### 4. Demo (one-shot)

Two end-to-end SIFT-100K demos are shipped — both pull a real SIFT
query, run the full MPC pipeline, and print top-10 + recall@10:

```bash
# (1) HNSecW (2PC ABY) on SIFT-100K, M=16 ef=16
bash large_demo.sh
# ->  [mpc total] online latency(s) / comm(MB)
# ->  Top-1 (MPC) / MPC top-10 / Ground-truth top-10 / recall@10

# (2) panther_generalize baseline (2PC Cheetah + Duoram) on SIFT-100K, u=123
bash baselines/panther_generalize/demo.sh
# ->  Total Latency / Total Communication
# ->  Plaintext IVF top-10 (panther's algorithmic baseline) + recall@10
```

Both demos default to query #0 of `sift_query.npy` and the first
100 000 rows of `sift_base.npy`; pass `--query-idx N` to use a
different query.

## Network setup

We emulate two network conditions, matching the paper's experimental
setup (§7.1):

* **LAN**: 4000 Mbps, 1 ms RTT
* **WAN**: 320 Mbps, 50 ms RTT

`baselines/panther_generalize/throttle_ns.sh` installs the matching
`tc-netem` qdiscs.  See `docs/ONLINE_METRICS.md` for the per-system
counter mapping.


## Notes

* External MPC frameworks: [ABY](https://github.com/encryptogroup/ABY),
  [ABY3](https://github.com/ladnir/aby3), and
  [MP-SPDZ](https://github.com/data61/MP-SPDZ).
* Build / runtime issues are summarised in `docs/TROUBLESHOOTING.md`.
* If a run reports unexpectedly low recall, the most common cause is
  a mismatch between the NPZ's `ef_base` / `tau_base` and the values
  baked into the framework binary.  Inspect `meta.json` to verify.

## License

Code in this repository is released for research purposes only.
