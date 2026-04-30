"""Build the binary data file consumed by hnsecw_cli's
HNSECW_DATA_FILE env var.

Reads a Run-Mpc-Layer-Search NPZ (built by results/build_npz.py),
extracts the layer graph + vec, prepares queries, and writes a
single binary file with the schema:

    [u32 N] [u32 D] [u32 M] [u32 num_queries]
    [N * M  u32 graph]            (row-major)
    [N * D  u32 vec]              (row-major, uint8 cast to u32)
    [Q * D  u32 query]            (uint8 cast to u32)

Query source priority:
    1. --query-file PATH  (uint8 .npy of shape (Q, D), used as-is)
    2. otherwise: synthetic query derived from BASE-LAYER (vecs_0)
       statistics: mean(vecs_0) + rng.integers(-5, 5, D), seeded.
       NOTE: HNSW searches a SINGLE query across all layers, so the
       synthetic query MUST come from a layer-independent source
       (vecs_0).  Earlier versions of this script derived the query
       from the current layer's mean, which broke the recall
       guarantee at upper layers.
"""

import argparse
import json
import struct

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--num_queries", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--query-file", default=None,
                    help="Optional path to a uint8 .npy with shape (>=num_queries, D); "
                         "if set, the first num_queries rows are used as the queries "
                         "instead of the synthetic mean+uniform sample.")
    ap.add_argument("--query-index", type=int, default=None,
                    help="When set, take exactly num_queries rows starting at this "
                         "offset in --query-file (so per-query independent runs can "
                         "select query 0, 1, 2 ... without rebuilding the query file).")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    li = args.layer
    vecs = data[f"vecs_{li}"]
    neigh = data[f"neigh_{li}"].astype(np.uint32)
    N, D = vecs.shape
    M = neigh.shape[1]

    if args.query_file is not None:
        raw = np.load(args.query_file)
        if raw.ndim != 2 or raw.shape[1] != D:
            raise ValueError(
                f"query-file shape {raw.shape} not compatible with layer dim D={D}")
        offset = args.query_index if args.query_index is not None else 0
        if raw.shape[0] < offset + args.num_queries:
            raise ValueError(
                f"query-file has {raw.shape[0]} rows < offset+num_queries="
                f"{offset+args.num_queries}")
        # cast to uint8 (HNSecW data file stores all queries as uint8 -> u32).
        queries = np.clip(raw[offset: offset + args.num_queries], 0, 255).astype(np.uint8)
        query_src = f"query-file {args.query_file} offset={offset}"
    else:
        # Synthetic queries — must use base-layer (vecs_0) statistics so all
        # layers see the SAME query, otherwise multi-layer recall collapses.
        base_vecs = data["vecs_0"]
        rng = np.random.default_rng(args.seed)
        base_int = base_vecs.astype(np.int64)
        mean_vec = base_int.mean(axis=0).astype(np.int64)
        queries = []
        for _ in range(args.num_queries):
            q = np.clip(mean_vec + rng.integers(-5, 5, base_vecs.shape[1]),
                        0, 255).astype(np.uint8)
            queries.append(q)
        queries = np.asarray(queries, dtype=np.uint8)
        query_src = f"synthetic mean+uniform[-5,5] seed={args.seed} from vecs_0"

    print(f"Layer {li}: N={N} D={D} M={M} Q={args.num_queries} ({query_src})")

    with open(args.out, "wb") as f:
        f.write(struct.pack("<IIII", N, D, M, args.num_queries))
        f.write(neigh.astype(np.uint32).tobytes())
        f.write(vecs.astype(np.uint32).tobytes())
        f.write(queries.astype(np.uint32).tobytes())

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
