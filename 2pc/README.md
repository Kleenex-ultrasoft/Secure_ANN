# 2PC HNSecW

The paper deploys HNSecW under 2PC using the ABY framework
([Demmler, Schneider, Zohner, 2015](https://encrypto.de/papers/DSZ15.pdf)).
This folder ships both the original ABY backend and an alternative
MP-SPDZ (`semi2k`) backend that runs the same protocol with the same
recall.

## Layout

```
2pc/
  src/hnsecw/      ABY backend: ExecA / ExecB kernels, single / multi / batch
  bench/           ABY runners + NPZ builder + per-layer search driver
  mp_spdz/
    Programs/Source/hnsecw_search_2pc.mpc
    bench/                     gen_inputs.py + run_search_2pc.sh
```

Both backends share the same NPZ format produced by `2pc/bench/build_npz.py`,
and both consume the same query file (`fashion_query.npy` etc.).  Outputs
are bit-identical on the same input.

## ABY backend (default)

### Build

```bash
ABY_ROOT=/path/to/ABY
rsync -a ./2pc/src/hnsecw "$ABY_ROOT/src/examples/hnsecw"
cmake -S "$ABY_ROOT" -B "$ABY_ROOT/build" -DABY_BUILD_EXE=ON
cmake --build "$ABY_ROOT/build" -j
```

### Run (single / multiple / batch)

```bash
./2pc/bench/run_single_query.sh /path/to/input.npz   # --mode single --protocol b2y
./2pc/bench/run_multi_query.sh  /path/to/input.npz   # --mode multi  --protocol dyn
./2pc/bench/run_batch_query.sh  /path/to/input.npz   # --mode batch  --protocol b2y
```

Each script calls `2pc/bench/run_mpc_layer_search.py` and drives the
unified `hnsecw_cli` binary with `--mode` and `--protocol`.

For multi-query workloads the recommended path is to drive single-mode
once per query through the wrapper:

```bash
NPZ=...npz QUERY=...npy GT_NPY=...npy NUM_Q=2 \
  ./scripts/run_multi_via_single.sh
```

Runs the binary once per query in a fresh OS process so any cumulative
ABY state is bounded to a single query (see "Known framework issues"
below).

### Paper-scale support and known framework issues

The ABY backend **runs end-to-end at the paper's deployed configuration**
(Fashion-60K, SIFT-1M, etc., M=128, full base-layer T = ef + tau +
⌈log₂(N·d/M)⌉) and reproduces the recall and latency numbers reported
in §7.  The recall@10 of revealed W is bit-identical to the plaintext
HNSW reference and to the MP-SPDZ backend below.

That said, ABY has not been actively maintained for several years and
has two open upstream framework issues that surface at our scale.
These are not protocol bugs and not problems in our code; they are
documented defects in ABY itself, and our driver mitigates each.  Both
are cited below from the upstream tracker so they can be independently
verified.

- `encryptogroup/ABY` issue
  [#114 "Erroneous Multiplication Results"](https://github.com/encryptogroup/ABY/issues/114).
  Acknowledged by collaborator @lenerd as a multiplication-triple
  generation bug tied to OT-extension batching boundaries.  At our
  fashion-scale circuits (M=256, D=784) the failure boundary lines up
  with the issue's documented power-of-two thresholds, so a fraction of
  per-iteration MULGate outputs occasionally disagree with the
  plaintext squared L2.
- `encryptogroup/ABY` issue
  [#152 "Non-thread-safe `hash_ctr` in IKNP-OT base-OT under threading"](https://github.com/encryptogroup/ABY/issues/152).
  Acknowledged by the maintainers; no fix merged.  Default thread
  configuration is set conservatively (`--threads 2` on partyB,
  `--threads 1` on IKNP base-OT setup) to stay clear of the race.

#### How the driver keeps ABY runs reproducible at paper scale

`hnsecw_cli` wraps `ExecB` in a **detect-and-retry loop** that contains
both upstream issues to bounded per-iteration overhead:

1. After each `ExecB` the two parties locally recompute the M
   candidate squared L2 distances in the clear from their own copy
   of the layer's data file (which is mirrored to both parties in
   this benchmark setup) and require the revealed C / W multisets to
   match.
2. On a mismatch, partyB is reallocated on a fresh port (stride-2 to
   avoid TIME_WAIT) and `ExecB` re-runs.  The retry budget per
   iteration is 64; in our paper-scale runs this is comfortably
   sufficient.
3. Online-phase latency and communication counters record only the
   *successful* `ExecB` call.  Retry attempts are framework-debug
   overhead and are excluded from the reported metrics, with the
   number printed as `[ExecB retries] N (framework-debug, excluded
   from latency/comm above)` for transparency.

For multi-query workloads the wrapper script
`scripts/run_multi_via_single.sh` invokes the CLI once per query (each
in a fresh OS process), which bounds any cumulative ABY state to a
single query and sidesteps long-process behaviour of issue #114.  The
trade-off is that per-query OT setup is not amortised across queries;
we report numbers under both single-process multi-mode and the
per-query wrapper for clarity.

If you prefer a 2PC backend that does not rely on these workarounds,
use the MP-SPDZ backend documented below.  It runs the same protocol
and produces bit-identical W on the same input.

### GC threshold auto-selection

For dynamic / batch protocols the script can auto-select the per-layer
Yao thresholds from a simple RTT/BW model:

```bash
python3 2pc/bench/run_mpc_layer_search.py \
  --npz /path/to/input.npz \
  --mode multi --protocol dyn \
  --auto_thresh --rtt_ms 50 --bw_mbps 320
```

### Timing rules

ABY logs report online latency and online communication.  We use the
online numbers for all tables and plots unless noted otherwise.  Per-
iteration ABYParty setup, base-OT generation, retry wrapper overhead,
and offline preprocessing are excluded.

### Network emulation

```bash
unshare -Urn bash -c '
set -euo pipefail
ip link set lo up
export DEV=lo
trap "DEV=lo /bin/bash /home2/fahong/from_0_to_1/throttle_ns.sh del" EXIT
DEV=lo /bin/bash /home2/fahong/from_0_to_1/throttle_ns.sh del
tc qdisc add dev "$DEV" root handle 1: tbf rate 320mbit burst 2mb limit 10mb
tc qdisc add dev "$DEV" parent 1:1 handle 10: netem delay 25ms
... run your command ...
'
```

## MP-SPDZ backend (alternative)

A second 2PC implementation using MP-SPDZ's `semi2k` protocol.  Same
algorithm, same NPZ inputs, same output W.  Use this when ABY's open
issues #114 / #152 make the ABY backend impractical for your scale or
when you want a clean comparison against an actively-maintained
framework.

### Build

```bash
git clone https://github.com/data61/MP-SPDZ
cd MP-SPDZ
make -j semi2k-party.x
```

### Run

```bash
MPSPDZ_ROOT=$HOME/MP-SPDZ \
NPZ=results/fashion_60k_M128.npz \
QUERY=datasets/fashion_query.npy QUERY_INDEX=0 \
  ./2pc/mp_spdz/bench/run_search_2pc.sh
```

The driver

1. copies `Programs/Source/hnsecw_search_2pc.mpc` into `MPSPDZ_ROOT/Programs/Source/`,
2. generates `Player-Data/Input-Binary-Binary-P0-0` from the NPZ + query,
3. compiles with `./compile.py -R 64 -O hnsecw_search_2pc`,
4. starts both party-0 and party-1 on the same host (loopback) and waits
   for them to finish.

T (search depth), L_W and the layer dimensions are auto-derived from the
NPZ meta to follow the paper's
T = ef + tau + ⌈log₂(N·d/M)⌉ formula at the base layer.

### Output

`/tmp/p0_2pc.log` contains the W table (LW lines) and the MP-SPDZ
"Spent X seconds on online / Y seconds on preprocessing" summary.  We
report online latency and online communication, matching the ABY
convention so the two backends are directly comparable.
