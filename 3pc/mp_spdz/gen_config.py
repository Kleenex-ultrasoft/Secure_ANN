#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


def infer_id_bits(n: int) -> int:
    return max(1, math.ceil(math.log2(max(n, 2))))


def ceil_log2(n: int) -> int:
    if n <= 1:
        return 1
    return (n - 1).bit_length()


def arith_width(bitlen: int) -> int:
    if bitlen <= 8:
        return 8
    if bitlen <= 16:
        return 16
    if bitlen <= 32:
        return 32
    return 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist-bits", type=int, default=32)
    ap.add_argument("--num-queries", type=int, default=1)
    ap.add_argument("--ef-base", type=int, default=None)
    ap.add_argument("--tau-base", type=int, default=None)
    ap.add_argument("--tau-upper", type=int, default=None)
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    meta = json.loads(str(npz["meta_json"][0]))

    L = int(meta["L"])
    D = int(meta["d"])
    vec_bits = int(npz["vecs_0"].dtype.itemsize * 8)
    x2_bits = int(npz["x2_0"].dtype.itemsize * 8) if "x2_0" in npz else 0
    dist_bits_min = 2 * vec_bits + ceil_log2(D) + 1
    dist_bits = max(int(args.dist_bits), dist_bits_min)
    vec_share_bits = arith_width(dist_bits)
    if x2_bits == 0:
        x2_bits = dist_bits
    ef_base = int(meta["ef_base"]) if args.ef_base is None else int(args.ef_base)
    tau_base = int(meta["tau_base"]) if args.tau_base is None else int(args.tau_base)
    tau_upper = int(meta["tau_upper"]) if args.tau_upper is None else int(args.tau_upper)

    layers = []
    for l in reversed(range(L)):
        neigh = npz[f"neigh_{l}"]
        N = int(neigh.shape[0])
        M = int(neigh.shape[1])
        id_bits = infer_id_bits(N)
        if f"layer_size_{l}" in meta:
            N_real = int(meta[f"layer_size_{l}"])
        else:
            N_dummy = int(meta.get(f"layer_dummy_{l}", 0))
            N_real = max(0, N - N_dummy)
        if l == 0:
            L_C = ef_base + tau_base
            L_W = ef_base
        else:
            L_C = 1 + tau_upper
            L_W = 1
        # T = number of search iterations per layer.  Older NPZ files
        # explicitly stored T_{l}; the current build_npz writes only the
        # graph + meta and we fall back to L_C, which matches the cpp
        # `T = L_C` convention in 2pc/src/hnsecw/hnsecw_single_b2y.cpp.
        if f"T_{l}" in npz.files:
            T = int(npz[f"T_{l}"][0])
        else:
            T = L_C
        layers.append({
            "idx": l,
            "N": N,
            "N_real": N_real,
            "M": M,
            "T": T,
            "L_C": L_C,
            "L_W": L_W,
            "id_bits": id_bits,
            "dist_bits": dist_bits,
            "vec_bits": vec_bits,
            "vec_share_bits": vec_share_bits,
            "x2_bits": x2_bits,
            "dummy_id": N - 1,
        })

    cfg = {
        "D": D,
        "entry_point_top": int(npz["entry_point_top_local"][0]),
        "layers": layers,
        "num_queries": int(args.num_queries),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
