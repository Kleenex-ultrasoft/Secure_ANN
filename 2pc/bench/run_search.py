#!/usr/bin/env python3
"""Run HNSecW search (2PC) on an index built by build_index.py.

The search goes layer by layer, top to base.  Each layer is one run of
`hnsecw_cli` in both roles on this machine; the entry point of a layer is the
best node of the layer above.  The script adds up the online cost reported by
the programs and checks recall against exact search.

  --mode single   one run per query and layer; every query needs its own
                  shuffled copy of the tables
  --mode multi    all queries in one run per layer, one after another on one
                  shuffled copy (visited lists shared by all queries)
  --mode batch    all queries in one run per layer, in parallel on one
                  shuffled copy (Algorithm 4)
"""

import argparse
import json
import os
import re
import struct
import subprocess
import time
from pathlib import Path

import numpy as np


def write_layer_file(path: Path, vecs: np.ndarray, neigh: np.ndarray, queries: np.ndarray) -> None:
    n, d = vecs.shape
    m = neigh.shape[1]
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", n, d, m, queries.shape[0]))
        f.write(neigh.astype(np.uint32).tobytes())
        f.write(vecs.astype(np.uint32).tobytes())
        f.write(queries.astype(np.uint32).tobytes())


def id_bits(n: int, pool: int) -> int:
    # real nodes + dummy pool + 1 reserved address
    return max(1, (n + pool).bit_length())


def dummy_pool(mode: str, u: int, m: int, t: int, ef: int) -> int:
    if mode == "batch":
        return u + t * u * m + t * u + ef
    if mode == "multi":
        return u + u * t * m + u * t + ef
    return t * m + ef


TOTAL = re.compile(r"\[Total Online\] latency\(s\)=([0-9.eE+-]+) comm\(MB\)=([0-9.eE+-]+) rounds=([0-9]+)")
SETUP = re.compile(r"\[Total Setup\] latency\(s\)=([0-9.eE+-]+) comm\(MB\)=([0-9.eE+-]+)")


def run_layer(binary, env, args_common, port, log_prefix: Path):
    cmds = [[binary, "-r", str(role), "-p", str(port)] + args_common for role in (0, 1)]
    with open(f"{log_prefix}_server.log", "w") as slog, open(f"{log_prefix}_client.log", "w") as clog:
        server = subprocess.Popen(cmds[0], stdout=slog, stderr=subprocess.STDOUT, env=env)
        time.sleep(0.3)
        client = subprocess.Popen(cmds[1], stdout=clog, stderr=subprocess.STDOUT, env=env)
        rc_s, rc_c = server.wait(), client.wait()
    if rc_s != 0 or rc_c != 0:
        raise RuntimeError(f"hnsecw_cli failed (server={rc_s}, client={rc_c}); see {log_prefix}_*.log")
    text = Path(f"{log_prefix}_client.log").read_text()
    online = TOTAL.search(text)
    setup = SETUP.search(text)
    if not online or not setup:
        raise RuntimeError(f"no cost summary in {log_prefix}_client.log")
    return (float(online.group(1)), float(online.group(2)), int(online.group(3)),
            float(setup.group(1)), float(setup.group(2)))


def exact_topk(base, base_ids, q, k):
    diff = base - q.astype(np.int64)
    return base_ids[np.argsort(np.einsum("ij,ij->i", diff, diff), kind="stable")[:k]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help=".npz from build_index.py")
    ap.add_argument("--queries", required=True, help=".npy with uint8 query vectors")
    ap.add_argument("--num", type=int, default=1, help="number of queries to run")
    ap.add_argument("--mode", default="single", choices=["single", "multi", "batch"])
    ap.add_argument("--bin", required=True, help="path to hnsecw_cli")
    ap.add_argument("--work", default="hnsecw_work", help="directory for layer files and logs")
    ap.add_argument("--port", type=int, default=7766)
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    data = np.load(args.index, allow_pickle=True)
    meta = json.loads(str(data["meta_json"][0]))
    k = args.k or meta["k"]
    queries = np.load(args.queries)[: args.num]
    nq = len(queries)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    num_layers = meta["num_layers"]
    layer_files = []
    for l in range(num_layers):
        path = work / f"layer{l}.bin"
        write_layer_file(path, data[f"vecs_{l}"], data[f"neigh_{l}"], queries)
        layer_files.append(path)

    base = data["vecs_0"].astype(np.int64)
    base_ids = data["ids_0"]

    def layer_params(l):
        if l > 0:
            return meta["ef_upper"] + meta["delta_upper"], meta["ef_upper"]
        return meta["ef_base"] + meta["delta_base"], meta["ef_base"]

    # One "job" is a set of queries that goes through the layers together.
    jobs = [[qi] for qi in range(nq)] if args.mode == "single" else [list(range(nq))]
    results = {}
    tot = np.zeros(5)
    for job in jobs:
        u = len(job)
        entries = [meta["entry_global"]] * u
        cost = np.zeros(5)
        for l in range(num_layers - 1, -1, -1):
            ids = data[f"ids_{l}"]
            n_l = len(ids)
            m_l = data[f"neigh_{l}"].shape[1]
            T, ef = layer_params(l)
            local = {int(g): i for i, g in enumerate(ids)}
            tag = f"q{job[0]}_layer{l}" if args.mode == "single" else f"{args.mode}_layer{l}"
            entry_in = work / f"{tag}_entry.bin"
            entry_out = work / f"{tag}_out.bin"
            np.array([local[e] for e in entries], dtype=np.uint32).tofile(entry_in)

            env = dict(os.environ)
            env["HNSECW_DATA_FILE"] = str(layer_files[l])
            env["HNSECW_QUERY_INDEX"] = str(job[0])
            common = ["-X", args.mode, "-m", str(m_l), "-d", str(meta["d"]),
                      "-c", str(T), "-w", str(ef), "-b", str(meta["bitlen"]),
                      "-i", str(id_bits(n_l, dummy_pool(args.mode, u, m_l, T, ef))),
                      "-f", str(entry_in), "-o", str(entry_out)]
            if args.mode != "single":
                common += ["-q", str(u)]
            cost += np.array(run_layer(args.bin, env, common, args.port, work / tag))

            best = np.fromfile(entry_out, dtype=np.uint32)
            entries = [int(ids[b]) for b in best]
            if l == 0:
                topk = np.fromfile(str(entry_out) + ".topk", dtype=np.uint32)
                if args.mode == "single":
                    results[job[0]] = [int(ids[i]) for i in topk[:k]]
                else:
                    rows = topk.reshape(u, ef)
                    for j, qi in enumerate(job):
                        results[qi] = [int(ids[i]) for i in rows[j] if i != 0xFFFFFFFF][:k]
        tot += cost
        if args.mode == "single":
            qi = job[0]
            exact = exact_topk(base, base_ids, queries[qi], k)
            rec = len(set(results[qi]) & set(int(x) for x in exact)) / k
            print(f"query {qi}: recall@{k}={rec:.2f} online={cost[0]:.3f}s {cost[1]:.1f}MB "
                  f"rounds={int(cost[2])} | setup={cost[3]:.1f}s {cost[4]:.0f}MB", flush=True)

    recalls = []
    for qi in range(nq):
        exact = exact_topk(base, base_ids, queries[qi], k)
        recalls.append(len(set(results[qi]) & set(int(x) for x in exact)) / k)
    per = tot / nq
    print(f"\nmode={args.mode} queries={nq}: mean recall@{k}={np.mean(recalls):.4f} min={np.min(recalls):.2f}")
    print(f"per query: online latency={per[0]:.3f}s  online comm={per[1]:.1f}MB  "
          f"setup={per[3]:.1f}s/{per[4]:.0f}MB")
    rounds = tot[2] / nq if args.mode != "batch" else tot[2]
    print(f"rounds: {rounds:.0f} " + ("for the whole batch" if args.mode == "batch" else "per query"))


if __name__ == "__main__":
    main()
