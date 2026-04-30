"""Build the binary input file consumed by the 2PC HNSecW programs.

Reads a Run-Mpc-Layer-Search NPZ (produced by 2pc/bench/build_npz.py),
extracts the chosen layer (default: base layer 0) and Q queries from
--query-npy, and writes a binary input file in the order:

    graph[N][M] (uint32 stored as int64)
    vec[N][D]   (uint8 cast to int64)
    query[Q][D] (uint8 cast to int64)

For Q=1 (single-query mode) only the row at --query-index is written,
matching `hnsecw_search_2pc.mpc`'s `Array(D, sint).input_from(0)`.
For Q>1 (multi / batch modes) Q consecutive rows starting at
--query-index are written, matching the `Matrix(Q, D, sint)` layout in
`hnsecw_search_2pc_multi.mpc` and `hnsecw_search_2pc_batch.mpc`.

MP-SPDZ's binary input file convention is one int64 per element,
little-endian, and the file path is `<prefix>-Binary-P0-0` when the
binary is invoked with `-IF <prefix>`.
"""
import argparse
import os
import struct
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--query-npy", required=True)
    ap.add_argument("--query-index", type=int, default=0)
    ap.add_argument("--num-queries", type=int, default=1,
                    help="Pack this many consecutive query rows starting at "
                         "--query-index (for multi / batch modes)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True,
                    help="Output path; suggested name: "
                         "Player-Data/Input-Binary-Binary-P0-0")
    ap.add_argument("--with-entries", action="store_true",
                    help="Append a Q-element 'entries' array after queries: "
                         "the per-query base-layer entry id, computed via "
                         "plaintext HNSW upper-layer descent on the NPZ's "
                         "upper layers.  Required by the multi/batch .mpc "
                         "programs; the single .mpc ignores it.")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    layer = args.layer
    vec = data[f"vecs_{layer}"].astype(np.int64)
    neigh = data[f"neigh_{layer}"].astype(np.int64)
    N, D = vec.shape
    M = neigh.shape[1]

    queries = np.load(args.query_npy)
    if queries.shape[1] != D:
        raise ValueError(
            f"query D={queries.shape[1]} mismatches layer D={D}")
    end_idx = args.query_index + args.num_queries
    if queries.shape[0] < end_idx:
        raise ValueError(
            f"query-file has {queries.shape[0]} rows, < {end_idx}")
    qs = queries[args.query_index:end_idx].astype(np.int64)

    entries = []
    if args.with_entries:
        # Plaintext HNSW upper-layer descent to produce a base-layer entry
        # for each query.  Matches paper Section 4.4 (the host-side
        # descent that the .mpc reads as input).
        import json
        meta = json.loads(str(data["meta_json"][0]))
        L = int(meta.get("L", 1))
        for q in qs:
            cur = 0
            for ll in range(L - 1, layer, -1):
                v = data[f"vecs_{ll}"].astype(np.int64)
                g = data[f"neigh_{ll}"].astype(np.int64)
                while True:
                    cand = list(g[cur]) + [cur]
                    cand = [c for c in cand if 0 <= c < v.shape[0]]
                    dists = [int(((v[c] - q) ** 2).sum()) for c in cand]
                    best = cand[int(np.argmin(dists))]
                    if best == cur:
                        break
                    cur = best
                if ll - 1 >= layer and f"down_{ll}" in data.files:
                    cur = int(data[f"down_{ll}"][cur])
            entries.append(int(cur))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        for i in range(N):
            for k in range(M):
                f.write(struct.pack("<q", int(neigh[i, k])))
        for i in range(N):
            for j in range(D):
                f.write(struct.pack("<q", int(vec[i, j])))
        for q_row in range(args.num_queries):
            for j in range(D):
                f.write(struct.pack("<q", int(qs[q_row, j])))
        if args.with_entries:
            for e in entries:
                f.write(struct.pack("<q", int(e)))

    print(f"wrote {args.out}: N={N} D={D} M={M} Q={args.num_queries} "
          f"layer={layer} (query rows {args.query_index}..{end_idx - 1})"
          + (f" entries={entries}" if entries else ""))


if __name__ == "__main__":
    main()
