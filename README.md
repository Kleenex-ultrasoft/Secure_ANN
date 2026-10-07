# HNSecW

Code for the paper *HNSecW: A Logarithmic-Time MPC Protocol for Approximate
Nearest Neighbor Search* (PVLDB).  `paper.pdf` is the extended version.

The repository contains the online search protocol in two settings:

| | two parties (ABY) | three parties (MP-SPDZ, replicated sharing over Z_2^64) |
|---|---|---|
| single query (Algorithms 2 and 3) | `hnsecw_cli -X single` | `hnsecw_3pc.mpc`, `HNSW_MODE=single` |
| several queries, one after another, on one shuffled copy of the tables | `hnsecw_cli -X multi` | `hnsecw_3pc.mpc`, `HNSW_MODE=multi` |
| batch of queries on one shuffled copy (Algorithm 4) | `hnsecw_cli -X batch` | `hnsecw_3pc.mpc`, `HNSW_MODE=batch` |

```
2pc/src/hnsecw/      the ABY programs
2pc/patches/         applied by scripts/build_aby.sh
2pc/bench/           build_index.py, plain_search.py, run_search.py
3pc/Programs/Source/ the MP-SPDZ program
3pc/patches/         applied by scripts/build_mpspdz.sh
3pc/bench/           gen_inputs.py, run_search.py
scripts/             build_aby.sh, build_mpspdz.sh, netem.sh
```

Every run checks itself against a plaintext replay of the same search: the
two-party programs compare every value they open (table addresses and the
result), the three-party driver compares the results.  A run that disagrees
with the replay fails.

## Build

Ubuntu 22.04/24.04, g++ and cmake.  Python needs `numpy` and, to build an
index, `faiss-cpu`.

```bash
bash scripts/build_aby.sh        # -> build/ABY/build/bin/hnsecw_cli
bash scripts/build_mpspdz.sh     # -> build/MP-SPDZ/replicated-ring-party.x
```

`ABY_ROOT` and `MPSPDZ_ROOT` choose other locations.  Both scripts check out
a fixed commit of the framework and apply the patches in `2pc/patches` and
`3pc/patches`.

## Search an index

```bash
# 1. HNSW index with the paper's parameters (M = 128, ef = k + 10,
#    Delta = ceil(log2(N d / M)) + 7 on the base layer; ef = 1, Delta = 7 above)
python3 2pc/bench/build_index.py --dataset base.npy --out index.npz

# 2. plaintext run of the same fixed-length search: recall, no MPC
python3 2pc/bench/plain_search.py --index index.npz --queries query.npy --num 100

# 3. two parties
python3 2pc/bench/run_search.py --index index.npz --queries query.npy --num 4 \
    --mode single --bin build/ABY/build/bin/hnsecw_cli      # or multi, batch

# 4. three parties
python3 3pc/bench/run_search.py --index index.npz --queries query.npy --num 4 \
    --mode single --mpspdz build/MP-SPDZ                    # or multi, batch
```

`base.npy` is an `(N, d)` array; coordinates are quantised to 8 bits, and
`query.npy` holds uint8 vectors on the same scale.  The
drivers run all parties on the local machine, search the layers from the top,
and print recall against exact search together with latency, communication
and rounds.

The three-party program takes the tables as one input file per layer and
keeps one 64-bit share per stored bit, so it is practical for layers of up
to some ten thousand nodes on one machine.  Its driver reports two costs:
the search, and the material for the comparisons that is prepared before
the query and depends on neither the query nor the tables.

## Cost at the scale of the paper

The cost of a layer search depends only on public parameters: M, d, the
number of iterations T, ef and the address width.  It does not depend on
the table contents or on how many rows the table has.  To measure the cost
of a large data set without building it, the programs can run on a random
table with the address width of the full-size layer:

```bash
# three parties: all layers of a data set with N = 10^6, d = 128
python3 3pc/bench/run_search.py --mpspdz build/MP-SPDZ --shape 1000000,128
python3 3pc/bench/run_search.py --mpspdz build/MP-SPDZ --shape 1000000,128 --mode batch --num 4

# two parties: one layer; without HNSECW_DATA_FILE the table is synthetic
B=build/ABY/build/bin/hnsecw_cli
$B -r 0 -X single -m 128 -d 128 -c 47 -w 20 -i 20 &
$B -r 1 -X single -m 128 -d 128 -c 47 -w 20 -i 20
```

`-c` is T, `-w` is ef, `-i` the address width in bits (enough for the layer
size plus `T*M + ef + 1` dummy addresses).  Numbers obtained this way are
costs of the protocol at that size; they say nothing about recall.

### Networks

The paper uses a LAN (4000 Mbit/s, 1 ms round-trip time) and a WAN
(320 Mbit/s, 50 ms).  With all parties on one machine:

```bash
sudo bash scripts/netem.sh wan      # or lan; "off" removes the setting
```

### Reference run

M = 128, k = 10, parameters as above.  Machine: 2 cores of an Intel Xeon at
2.1 GHz shared by all parties, so compute time is pessimistic.  The machine
had no netem; the LAN and WAN columns were taken with a TCP relay between the
parties that delays each direction by 0.5 ms / 25 ms and limits it to
4000 / 320 Mbit/s.  Communication is the total over all parties.

Two parties, single query, online phase:

| shape (N, d) | T (base) | comm. | rounds | no delay | LAN | WAN |
|---|---|---|---|---|---|---|
| 10^6, 128 | 47 | 46 MB | 869 | 1.1 s | 1.7 s | 24 s |
| 10^6, 960 | 50 | 156 MB | 911 | 2.4 s | 3.1 s | 27 s |
| 10^8, 96 | 54 | 65 MB | 1133 | 1.5 s | 2.2 s | 30 s |
| 10^9, 128 | 57 | 82 MB | 1294 | 1.7 s | 2.6 s | 35 s |

ABY's setup phase (base OTs, OT extension for the multiplication triples and
the conversions, garbling) is separate: 15 s and 1.9 GB for the first row,
70 s and 10.6 GB for the second, without delay.  `hnsecw_cli` prints both
phases.

Three parties, single query, search (8 rounds per iteration, 7 on the small
upper layers):

| shape (N, d) | T (base) | comm. | rounds | no delay | LAN | WAN | prepared |
|---|---|---|---|---|---|---|---|
| 10^6, 128 | 47 | 9.0 MB | 497 | 0.23 s | 0.55 s | 12.9 s | 38 MB |
| 10^6, 960 | 50 | 10 MB | 521 | 0.32 s | 0.63 s | 13.7 s | 48 MB |
| 10^8, 96 | 54 | 16 MB | 620 | 0.40 s | 0.76 s | 16.2 s | 48 MB |
| 10^9, 128 | 57 | 17 MB | 703 | 0.52 s | 0.94 s | 18.5 s | 55 MB |

"prepared" is the material for the comparisons of one query (see
`3pc/README.md`).  It depends on neither the query nor the tables, and it
is one party's message to another: 0.2 to 0.4 s without delay, 1.2 to
1.8 s on the WAN.  The LAN adds 0.5 ms per round, which for 497 rounds is
0.25 s by itself.

With `HNSW_PREPARE=0` nothing is prepared; the first row then takes 812
rounds, 45 MB and 0.30 s without delay.

Several queries, shape (10^6, 128), per query:

| | queries | comm. | rounds | no delay | WAN | prepared |
|---|---|---|---|---|---|---|
| 2PC multi, base layer, online | 3 | 61 MB | 1004 | 1.8 s | | |
| 2PC batch, base layer, online | 4 | 81 MB | 298 | 2.8 s | | |
| 3PC multi, all layers | 4 | 30 MB | 910 | 0.56 s | 24 s | 38 MB |
| 3PC batch, all layers | 4 | 49 MB | 425 | 0.80 s | 12 s | 74 MB |
| 3PC batch, all layers | 8 | 98 MB | 237 | 1.1 s | 8.1 s | 188 MB |

A batch needs fewer rounds per query than single queries, and the more
queries it holds the fewer; it needs more communication, because every
query scores all U*M vectors fetched in an iteration.  The multi programs
need more rounds per query than a single query, because the popped address
is checked before its graph row is read, and their catch-up step grows with
the number of vectors fetched before.

## What is and is not in this repository

* **Online search only.**  The programs search tables that are assumed to be
  padded and shuffled already.  The offline reshuffle and the assignment of
  fresh table addresses are not included.  In the harness a node has the same
  address in the graph table and in the vector table.
* **Shares in the two-party harness.**  Both ABY processes load the layer
  file and derive consistent shares of every value they feed to the circuits
  from a common mask stream, so each process inputs only its own share and
  the circuits are those of a deployment with pre-shared tables.  The harness
  is for measuring and checking the protocol, not for deployment.  In the
  three-party program one party inputs the tables and they are secret-shared
  from then on.
* **Several queries on one shuffled copy.**  Section 5 of the paper keeps
  two cross-query caches that are scanned at every table access.  The `multi`
  programs reach the same goal, U queries on one shuffled copy with every
  table address opened at most once, in a cheaper way.  The visited lists are
  shared by all queries.  The addresses opened so far are public, so the
  rows behind them can be read again; a query starts by scoring every vector
  fetched so far and keeps the best as its initial candidates.  After that a
  neighbour found in the visited list is skipped, as in a single query, and
  a popped node whose graph row was opened before reads a dummy row.  As in
  a batch (Algorithm 4), a query therefore also sees the vectors that other
  queries fetched.
* **Graph table of the three-party program.**  Neighbour addresses are
  stored as one-hot digits of up to 8 bits (128 shared bits for a 20-bit
  address instead of 20, 768 for a 30-bit address), which saves several
  rounds per iteration.  `HNSW_DIGIT` sets the largest digit; `HNSW_DIGIT=1`
  stores plain bits.
* **Base-layer degree.**  `build_index.py` uses degree bound M on every
  layer, as Algorithm 2 does (`--M0` changes the base layer).
