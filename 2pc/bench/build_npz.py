"""Generic NPZ builder for run_mpc_layer_search.py.

Auto-detects input scale and applies the right uint8 quantization:
  - Unit-normalized in [-1, 1]  →  (x + 1) * 127 → uint8
  - Already uint8-range [0, 255] →  direct clip

A previous version of this script in `~/hnsecw_build/results/build_npz.py`
unconditionally did `np.clip(X, 0, 255).astype(np.uint8)`, which silently
truncated unit-normalized DEEP vectors (range ~[-0.55, 0.58]) to all
zeros, producing a degenerate NPZ where every plaintext distance equals
||q||^2.  The all-zero-NPZ bug surfaced when a multi U=5 sweep on DEEP
returned rank=0 for every query — caught by AZ on 2026-04-29.

Hard-fails if quantization produces fewer than 32 unique values across
the whole dataset; that's the early-warning the previous bug missed.

Output schema is identical to `build_cosine_npz.py` so
`run_mpc_layer_search.py` consumes either result without modification.
"""

import argparse
import json
import math
import sys

import hnswlib
import numpy as np
from pathlib import Path


def quantize(X):
    """Auto-detect input range and produce uint8-quantized vectors."""
    lo, hi = float(X.min()), float(X.max())
    if lo >= -1.5 and hi <= 1.5:
        print(f"  Input range [{lo:.3f}, {hi:.3f}] looks unit-normalized; "
              f"rescaling via (x + 1) * 127.")
        Xq = np.clip(np.round((X + 1.0) * 127.0), 0, 254).astype(np.uint8)
    else:
        print(f"  Input range [{lo:.3f}, {hi:.3f}] looks uint8-scale; "
              f"clipping to [0, 255].")
        Xq = np.clip(X, 0, 255).astype(np.uint8)
    n_unique = len(np.unique(Xq))
    if n_unique < 32:
        raise RuntimeError(
            f"Quantization produced only {n_unique} unique values across "
            f"N={Xq.shape[0]} D={Xq.shape[1]}.  Check input scale."
        )
    return Xq


def build(dataset_npy, out_npz, n=None, M=48, ef_construction=200,
          k=10, ef_base=None, tau_base=None, tau_upper=7, space="l2"):
    # Paper's deployed strategy (Section 4.1):
    #   ef^{(0)}      = k + 10                     (base-layer search width)
    #   Δ̄^{(0)} = ceil(log_2(N*d/M)) + C   with C=7  (base-layer slack)
    #   ef^{(\ell≥1)} = 1, Δ̄^{(\ell≥1)} = C   (upper layers)
    # Callers may override ef_base / tau_base if needed.
    if ef_base is None:
        ef_base = k + 10
    raw = np.load(dataset_npy).astype(np.float32)
    if n is not None and n < raw.shape[0]:
        raw = raw[:n]
    N, D = raw.shape
    if tau_base is None:
        # Δ̄^{(0)} = ceil(log_2(N*d/M)) + C, C = 7 from §4.1.
        tau_base = int(math.ceil(math.log2(max(2, N * D / max(1, M))))) + 7
    print(f"Building HNSW on N={N} D={D} M={M} efC={ef_construction} space={space}  "
          f"k={k} ef_base={ef_base} tau_base={tau_base} tau_upper={tau_upper}")

    if space == "cosine":
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = raw / norms
        Xq = quantize(Xn)
        idx = hnswlib.Index(space="cosine", dim=D)
        idx.init_index(max_elements=N, ef_construction=ef_construction, M=M)
        idx.add_items(Xn, np.arange(N), num_threads=-1)
    else:
        Xq = quantize(raw)
        Xq_f = Xq.astype(np.float32)
        idx = hnswlib.Index(space="l2", dim=D)
        idx.init_index(max_elements=N, ef_construction=ef_construction, M=M)
        idx.add_items(Xq_f, np.arange(N), num_threads=-1)
    idx.set_ef(max(2 * (ef_base + tau_base), 50))

    L = max(1, int(np.ceil(np.log(max(N, 2)) / np.log(max(M, 2)))))
    layer_sizes = [max(M, N // (M ** i)) for i in range(L)]
    layer_sizes[0] = N
    layer_sizes = [ls for ls in layer_sizes if ls >= 2 * M]
    if not layer_sizes:
        layer_sizes = [N]
    L = len(layer_sizes)
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
        x2 = (Xq_li.astype(np.int32) ** 2).sum(axis=1).astype(np.uint32)
        deg = 2 * M if li == 0 else M

        sub = hnswlib.Index(space=space, dim=D)
        sub.init_index(max_elements=len(nodes),
                       ef_construction=max(ef_construction, 2 * deg),
                       M=min(deg, max(1, len(nodes) - 1)))
        if space == "cosine":
            sub.add_items(Xn[nodes], np.arange(len(nodes)), num_threads=-1)
        else:
            sub.add_items(Xq[nodes].astype(np.float32), np.arange(len(nodes)), num_threads=-1)
        sub.set_ef(max(deg * 2, 50))

        labels, _dists = sub.knn_query(
            Xn[nodes] if space == "cosine" else Xq[nodes].astype(np.float32),
            k=min(deg + 1, len(nodes)),
            num_threads=-1,
        )
        neigh = np.full((len(nodes), deg), len(nodes), dtype=np.uint32)
        for i in range(len(nodes)):
            row = [int(j) for j in labels[i] if int(j) != i][:deg]
            for j, idj in enumerate(row):
                neigh[i, j] = idj

        npz[f"vecs_{li}"] = Xq_li
        npz[f"neigh_{li}"] = neigh
        npz[f"x2_{li}"] = x2

        if li > 0:
            npz[f"down_{li}"] = layer_nodes[li].astype(np.int32)

    npz["entry_point_top_local"] = np.array([0], dtype=np.int32)
    meta = {
        "n_base": int(N),
        "L": int(L),
        "D": int(D),
        "d": int(D),
        "deg0": int(2 * M),
        "deg0_target": int(2 * M),
        "degu": int(M),
        "degU_target": int(M),
        "M": int(M),
        "ef_construction": int(ef_construction),
        "ef_base": int(ef_base),
        "tau_base": int(tau_base),
        "tau_upper": int(tau_upper),
        "bitlen": 8,
    }
    for li, ls in enumerate(layer_sizes):
        meta[f"layer_size_{li}"] = int(ls)
    npz["meta_json"] = np.array([json.dumps(meta)])

    np.savez(out_npz, **npz)
    print(f"Saved {out_npz}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--M", type=int, default=48)
    ap.add_argument("--ef_construction", type=int, default=200)
    ap.add_argument("--k", type=int, default=10,
                    help="top-k requested at runtime; sets ef^{(0)}=k+10 by default")
    ap.add_argument("--ef_base", type=int, default=None,
                    help="override ef^{(0)} at base layer; default k+10")
    ap.add_argument("--tau_base", type=int, default=None,
                    help="override Δ̄^{(0)} at base layer; default ceil(log_2(N*d/M)) + 7")
    ap.add_argument("--tau_upper", type=int, default=7,
                    help="Δ̄^{(\\ell≥1)} = C; default C=7 from §4.1")
    ap.add_argument("--space", default="l2", choices=["l2", "cosine"])
    args = ap.parse_args()
    build(args.dataset, args.out, n=args.n, M=args.M,
          ef_construction=args.ef_construction, k=args.k,
          ef_base=args.ef_base, tau_base=args.tau_base,
          tau_upper=args.tau_upper, space=args.space)


if __name__ == "__main__":
    main()
