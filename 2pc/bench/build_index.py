#!/usr/bin/env python3
"""Build the per-layer HNSW tables that the HNSecW programs search.

Input : a .npy array of shape (N, d) (float or integer).
Output: an .npz file holding, for every layer l (0 = base),

    vecs_l   (N_l, d) uint8   node vectors, in layer-local order
    neigh_l  (N_l, M_l) uint32  neighbour ids, layer-local; unused slots = N_l
    ids_l    (N_l,) int64     global id of every layer-local node

plus the entry point of the top layer and the search parameters.

The graph is an ordinary HNSW index built with faiss.  The node set of layer
l is the set of nodes whose HNSW level is at least l, so the layers are
nested exactly as in HNSW.

Search parameters follow Section 4.1 of the paper:

    ef(0)    = k + 10
    Delta(0) = ceil(log2(N * d / M)) + C,   C = 7
    ef(l)    = 1,  Delta(l) = C             for l >= 1

and every layer runs T = ef + Delta iterations.
"""

import argparse
import json
import math

import faiss
import numpy as np


def quantize(x: np.ndarray) -> np.ndarray:
    """Map the input to uint8 coordinates."""
    lo, hi = float(x.min()), float(x.max())
    if np.issubdtype(x.dtype, np.integer) and lo >= 0 and hi <= 255:
        return x.astype(np.uint8)
    if lo >= -1.5 and hi <= 1.5:
        # unit-normalised input: map [-1, 1] to [0, 254]
        return np.clip(np.round((x.astype(np.float32) + 1.0) * 127.0), 0, 254).astype(np.uint8)
    if lo >= 0 and hi <= 255:
        return np.clip(np.round(x), 0, 255).astype(np.uint8)
    scale = 255.0 / (hi - lo)
    return np.clip(np.round((x.astype(np.float32) - lo) * scale), 0, 255).astype(np.uint8)


def build(args) -> None:
    raw = np.load(args.dataset)
    if args.n is not None and args.n < raw.shape[0]:
        raw = raw[: args.n]
    xq = quantize(raw)
    n, d = xq.shape
    m = args.M
    m0 = args.M0 if args.M0 is not None else m
    print(f"N={n} d={d} M={m} M0={m0} efConstruction={args.ef_construction}")

    index = faiss.IndexHNSWFlat(d, m)
    index.hnsw.set_nb_neighbors(0, m0)
    index.hnsw.efConstruction = args.ef_construction
    index.add(xq.astype(np.float32))

    hnsw = index.hnsw
    levels = faiss.vector_to_array(hnsw.levels)           # number of layers a node is in
    offsets = faiss.vector_to_array(hnsw.offsets)
    neighbors = faiss.vector_to_array(hnsw.neighbors)
    num_layers = int(hnsw.max_level) + 1
    entry_global = int(hnsw.entry_point)

    out = {}
    layer_sizes = []
    for layer in range(num_layers):
        ids = np.nonzero(levels > layer)[0].astype(np.int64)
        n_l = len(ids)
        local = np.full(n, -1, dtype=np.int64)
        local[ids] = np.arange(n_l)
        lo = hnsw.cum_nb_neighbors(layer)
        hi = hnsw.cum_nb_neighbors(layer + 1)
        width = hi - lo
        neigh = np.full((n_l, width), n_l, dtype=np.uint32)
        for row, g in enumerate(ids):
            nb = neighbors[offsets[g] + lo: offsets[g] + hi]
            nb = nb[nb >= 0]
            neigh[row, : len(nb)] = local[nb]
        out[f"vecs_{layer}"] = xq[ids]
        out[f"neigh_{layer}"] = neigh
        out[f"ids_{layer}"] = ids
        layer_sizes.append(n_l)
        print(f"  layer {layer}: N_l={n_l} degree bound={width} "
              f"mean degree={float((neigh < n_l).sum(axis=1).mean()):.1f}")

    c = args.C
    ef_base = args.ef_base if args.ef_base is not None else args.k + 10
    delta_base = (args.delta_base if args.delta_base is not None
                  else int(math.ceil(math.log2(max(2.0, n * d / m)))) + c)
    meta = {
        "N": int(n), "d": int(d), "M": int(m), "M0": int(m0),
        "num_layers": num_layers, "layer_sizes": layer_sizes,
        "entry_global": entry_global,
        "k": int(args.k), "ef_base": int(ef_base), "delta_base": int(delta_base),
        "ef_upper": 1, "delta_upper": int(c), "bitlen": 8,
    }
    out["meta_json"] = np.array([json.dumps(meta)])
    np.savez(args.out, **out)
    print(f"T(base)={ef_base + delta_base} T(upper)={1 + c}")
    print(f"saved {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help=".npy file with the base vectors")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--n", type=int, default=None, help="use only the first n vectors")
    ap.add_argument("--M", type=int, default=128, help="degree bound (paper: 128)")
    ap.add_argument("--M0", type=int, default=None,
                    help="degree bound of the base layer; default M, as in Algorithm 2")
    ap.add_argument("--ef_construction", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef_base", type=int, default=None, help="default k + 10")
    ap.add_argument("--delta_base", type=int, default=None,
                    help="default ceil(log2(N*d/M)) + C")
    ap.add_argument("--C", type=int, default=7)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
