#!/usr/bin/env python3
"""Run HNSecW search (3PC, MP-SPDZ replicated-ring) layer by layer.

Two ways to choose the layers:

  --index I.npz --queries Q.npy [--num n]
      Search an index built by 2pc/bench/build_index.py.  The entry point of
      a layer is the best node of the layer above; recall is checked against
      exact search.

  --shape N,D
      Cost of a data set of N vectors of dimension D at the paper's
      parameters (M = 128, k = 10).  The cost of a layer depends only on
      public parameters (M, D, T, ef, address width), so every layer is run
      on a random table of at most --rows rows with the address width of the
      full-size layer.

Modes:

  --mode single   independent searches, one after another (Algorithm 2);
                  every query needs its own shuffled copy of the tables
  --mode multi    the queries are searched one after another on one shuffled
                  copy, with visited lists shared by all queries
  --mode batch    the queries advance together on one shuffled copy
                  (Algorithm 4)

For every layer the script writes the party-0 input, compiles
Programs/Source/hnsecw_3pc.mpc, runs the three parties on this machine and
checks the opened results against a plaintext replay of the program.

Two costs are reported.  "search" is the search itself (timer 1 of the
program).  "prepared" is the material for the comparisons, which depends
neither on the query nor on the tables and can be produced ahead of time
(timer 2).  The input of the tables is in neither.
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_inputs  # noqa: E402

PROGRAM = Path(__file__).resolve().parent.parent / "Programs" / "Source" / "hnsecw_3pc.mpc"
TIMER = {name: re.compile(r"^Time%d = ([0-9.eE+-]+) seconds \(([0-9.eE+-]+) MB, ([0-9]+) rounds" % number, re.M)
         for name, number in (("search", 1), ("prepared", 2))}
RESULT = re.compile(r"^RESULT (\d+) \[?([-0-9, ]+)\]?", re.M)


def run_layer(mp, work, tag, graph, vecs, queries, entries, t, ef, mode, bits, port):
    """Returns (results per query, cost of the search, cost of the prepared
    material); a cost is (seconds, MB sent by all parties, rounds)."""
    n, d = vecs.shape
    m = graph.shape[1]
    q = len(queries)
    bits, expected = gen_inputs.write_inputs(
        mp / "Player-Data" / "Input-Binary-P0-0", graph, vecs, queries, entries, t, ef, mode, bits)
    env = dict(os.environ, HNSW_N=str(n), HNSW_M=str(m), HNSW_D=str(d), HNSW_T=str(t),
               HNSW_EF=str(ef), HNSW_BITS=str(bits), HNSW_Q=str(q),
               HNSW_MODE=mode)
    log = work / f"{tag}_compile.log"
    with open(log, "w") as f:
        rc = subprocess.run(["./compile.py", "-M", "-E", "ring", "-R", "64", "hnsecw_3pc"],
                            cwd=mp, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"compilation failed, see {log}")

    logs = [work / f"{tag}_p{i}.log" for i in range(3)]
    procs = []
    for i in (2, 1, 0):
        procs.append(subprocess.Popen(
            ["./replicated-ring-party.x", "-p", str(i), "-pn", str(port), "hnsecw_3pc"],
            cwd=mp, stdout=open(logs[i], "w"), stderr=subprocess.STDOUT))
        time.sleep(0.2)
    if any(p.wait() != 0 for p in procs):
        raise RuntimeError(f"a party failed, see {work}/{tag}_p*.log")

    text = [log.read_text() for log in logs]
    got = {int(a): [int(x) for x in b.split(",")] for a, b in RESULT.findall(text[0])}
    results = [got.get(i) for i in range(q)]
    if results != expected:
        raise RuntimeError(f"{tag}: the opened results differ from the plaintext replay")

    def cost(name):
        found = [TIMER[name].search(x) for x in text]
        if not all(found):
            return np.zeros(3)
        # a party counts the rounds in which it sends
        return np.array([float(found[0].group(1)), sum(float(x.group(2)) for x in found),
                         max(int(x.group(3)) for x in found)])

    search = cost("search")
    if not search[0]:
        raise RuntimeError(f"no timer output, see {work}/{tag}_p*.log")
    return results, search, cost("prepared")


def show(cost):
    return f"{cost[0]:.3f}s {cost[1]:.2f}MB rounds={int(cost[2])}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mpspdz", required=True, help="MP-SPDZ directory with replicated-ring-party.x")
    ap.add_argument("--index", help=".npz from 2pc/bench/build_index.py")
    ap.add_argument("--queries", help=".npy with uint8 query vectors")
    ap.add_argument("--num", type=int, default=1, help="number of queries")
    ap.add_argument("--shape", help="N,D: cost at this size on synthetic layers")
    ap.add_argument("--rows", type=int, default=1000, help="table rows per synthetic layer")
    ap.add_argument("--mode", default="single", choices=["single", "multi", "batch"])
    ap.add_argument("--work", default="hnsecw_3pc_work", help="directory for logs")
    ap.add_argument("--port", type=int, default=15000)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--C", type=int, default=7)
    args = ap.parse_args()
    if bool(args.index) == bool(args.shape):
        ap.error("give exactly one of --index and --shape")

    mp = Path(args.mpspdz).resolve()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    (mp / "Player-Data").mkdir(exist_ok=True)
    shutil.copy(PROGRAM, mp / "Programs" / "Source" / "hnsecw_3pc.mpc")
    mode = args.mode
    nq = args.num
    total, prepared = np.zeros(3), np.zeros(3)

    if args.shape:
        n_total, d = (int(x) for x in args.shape.split(","))
        m, k = args.M, args.k
        num_layers = max(1, math.ceil(math.log(n_total) / math.log(m)))
        rng = np.random.default_rng(1)
        print(f"N={n_total} D={d} M={m}: {num_layers} layers, synthetic tables, {nq} queries, mode={args.mode}")
        for layer in range(num_layers - 1, -1, -1):
            n_layer = max(m + 1, n_total // m ** layer)
            if layer == 0:
                ef = k + 10
                t = ef + math.ceil(math.log2(n_total * d / m)) + args.C
            else:
                ef, t = 1, 1 + args.C
            bits = gen_inputs.address_bits(n_layer, m, t, nq, mode)
            # keep the synthetic graph table below about 160 MB per share
            rows = min(n_layer, args.rows,
                       max(m + 2, 20_000_000 // (gen_inputs.graph_columns(bits) * m)))
            graph, vecs, queries = gen_inputs.synthetic_layer(rows, m, d, nq, rng)
            entries = [int(x) for x in rng.integers(0, rows, size=nq)]
            _, search, ahead = run_layer(mp, work, f"layer{layer}", graph, vecs, queries, entries,
                                         t, ef, mode, bits, args.port)
            total += search
            prepared += ahead
            print(f"  layer {layer}: N_l~{n_layer} T={t} ef={ef} address bits={bits}: "
                  f"search {show(search)} | prepared {show(ahead)}", flush=True)
    else:
        data = np.load(args.index, allow_pickle=True)
        import json
        meta = json.loads(str(data["meta_json"][0]))
        k = meta["k"]
        queries = np.load(args.queries)[:nq]
        nq = len(queries)
        entries = [meta["entry_global"]] * nq
        results = None
        for layer in range(meta["num_layers"] - 1, -1, -1):
            ids = data[f"ids_{layer}"]
            local = {int(g): i for i, g in enumerate(ids)}
            if layer > 0:
                ef, t = meta["ef_upper"], meta["ef_upper"] + meta["delta_upper"]
            else:
                ef, t = meta["ef_base"], meta["ef_base"] + meta["delta_base"]
            found, search, ahead = run_layer(
                mp, work, f"layer{layer}", data[f"neigh_{layer}"], data[f"vecs_{layer}"], queries,
                [local[e] for e in entries], t, ef, mode, 0, args.port)
            total += search
            prepared += ahead
            real = [[int(ids[a]) for a in row if a < len(ids)] for row in found]
            entries = [r[0] if r else e for r, e in zip(real, entries)]
            results = real
            print(f"  layer {layer}: N_l={len(ids)} T={t} ef={ef}: "
                  f"search {show(search)} | prepared {show(ahead)}", flush=True)
        base = data["vecs_0"].astype(np.int64)
        base_ids = data["ids_0"]
        recalls = []
        for qi in range(nq):
            diff = base - queries[qi].astype(np.int64)
            exact = base_ids[np.argsort(np.einsum("ij,ij->i", diff, diff), kind="stable")[:k]]
            recalls.append(len(set(results[qi][:k]) & set(int(x) for x in exact)) / k)
        print(f"mean recall@{k}={np.mean(recalls):.4f} min={np.min(recalls):.2f}")

    print("[Verified] every opened result matches the plaintext replay")
    print(f"per query, search:   latency={total[0] / nq:.3f}s  comm={total[1] / nq:.1f}MB (all parties)  "
          f"rounds={total[2] / nq:.0f}")
    if prepared[1]:
        print(f"per query, prepared: time={prepared[0] / nq:.3f}s  comm={prepared[1] / nq:.1f}MB (all parties)")
    if mode == "batch":
        print(f"the batch advances together: {int(total[2])} rounds for all {nq} queries")


if __name__ == "__main__":
    main()
