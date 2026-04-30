"""Cosine-aware NPZ builder for LAION/MSMARCO normalized float32 datasets.

Maps unit-norm float32 vectors -> uint8 by scaling each coord
from [-1, 1] -> [0, 254] (offset by 1, multiply by 127, round).
Uses HNSW under cosine space.  L2 distance on these scaled+offset
vectors approximates cosine ordering closely enough for ANN (the
fraction of order-flips is small and the paper's figure already
reports the resulting recall).

Output schema is identical to build_npz.py so run_mpc_layer_search.py
can consume the result without modification.
"""

import argparse
import json
import os
import tempfile

import hnswlib
import numpy as np
from pathlib import Path


def quantize(X_unit):
    """Map unit-norm float32 vectors in [-1, 1]^D to uint8 by
    coord-wise (x + 1) * 127, then clip to [0, 254]."""
    Xq = np.clip(np.round((X_unit + 1.0) * 127.0), 0, 254).astype(np.uint8)
    return Xq


def build(dataset_npy, out_npz, M=48, ef_construction=200,
          ef_base=16, tau_base=4, tau_upper=4):
    raw = np.load(dataset_npy).astype(np.float32)
    N, D = raw.shape
    print(f"Cosine NPZ: N={N} D={D} M={M}")

    # Renormalize to be safe; many of these are already unit-norm
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = raw / norms

    # Build HNSW under cosine in floating point (preserves true ANN ordering).
    idx = hnswlib.Index(space="cosine", dim=D)
    idx.init_index(max_elements=N, ef_construction=ef_construction, M=M)
    idx.add_items(Xn, np.arange(N), num_threads=-1)
    idx.set_ef(max(2 * (ef_base + tau_base), 50))

    Xq = quantize(Xn)  # uint8 representation for the MPC side

    # Layer sizes a la build_npz.py
    L = max(1, int(np.ceil(np.log(N) / np.log(M))))
    layer_sizes = [max(M, N // (M ** i)) for i in range(L)]
    layer_sizes[0] = N
    layer_sizes = [ls for ls in layer_sizes if ls >= 2 * M]
    L = len(layer_sizes)
    if L == 0:
        layer_sizes = [N]; L = 1
    print(f"Layer sizes: {layer_sizes}")

    rng = np.random.default_rng(0)
    layer_nodes = [None] * L
    layer_nodes[0] = np.arange(N, dtype=np.int32)
    for li in range(1, L):
        layer_nodes[li] = rng.choice(N, size=layer_sizes[li], replace=False).astype(np.int32)

    npz = {}
    for li in range(L):
        nodes = layer_nodes[li]
        Xq_li = Xq[nodes]
        # x2 over uint8 vectors
        x2 = (Xq_li.astype(np.int32) ** 2).sum(axis=1).astype(np.uint32)
        deg = 2 * M if li == 0 else M

        # build sub-HNSW on cosine for this layer's nodes only
        sub = hnswlib.Index(space="cosine", dim=D)
        sub.init_index(max_elements=len(nodes),
                       ef_construction=max(ef_construction, 2 * deg),
                       M=min(deg, max(1, len(nodes) - 1)))
        sub.add_items(Xn[nodes], np.arange(len(nodes)), num_threads=-1)
        sub.set_ef(max(deg * 2, 50))

        if len(nodes) > deg + 1:
            labels, _ = sub.knn_query(Xn[nodes], k=deg + 1)
            neigh = labels[:, 1:deg+1].astype(np.int32)
        else:
            neigh = np.tile(np.arange(len(nodes), dtype=np.int32), (len(nodes), 1))[:, :deg]
            if neigh.shape[1] < deg:
                neigh = np.pad(neigh, ((0,0),(0, deg - neigh.shape[1])), constant_values=0)

        npz[f"vecs_{li}"] = Xq_li
        npz[f"neigh_{li}"] = neigh
        npz[f"x2_{li}"] = x2
        npz[f"is_dummy_{li}"] = np.zeros(len(nodes), dtype=np.uint8)

        if li > 0:
            lower_idx_map = {int(v): i for i, v in enumerate(layer_nodes[li-1])}
            down = np.array([lower_idx_map.get(int(n), 0) for n in layer_nodes[li]], dtype=np.int32)
            npz[f"down_{li}"] = down

    meta = {
        "D": D, "d": D, "n_base": N, "L": L,
        "deg0": 2 * M, "deg0_target": 2 * M,
        "degu": M, "degU_target": M,
        "M": M, "ef_construction": ef_construction,
        "ef_base": ef_base, "tau_base": tau_base, "tau_upper": tau_upper,
        "bitlen": 8, "space": "cosine",
    }
    for li, ls in enumerate(layer_sizes):
        meta[f"layer_size_{li}"] = ls
        meta[f"layer_total_{li}"] = ls
        meta[f"layer_dummy_{li}"] = 0

    npz["meta_json"] = np.array([json.dumps(meta)], dtype=object)
    npz["entry_point_top_local"] = np.array([0], dtype=np.int32)

    np.savez(out_npz, **npz)
    print(f"Saved {out_npz}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, default=48)
    ap.add_argument("--ef_construction", type=int, default=200)
    args = ap.parse_args()
    build(args.dataset, args.out, M=args.M, ef_construction=args.ef_construction)
