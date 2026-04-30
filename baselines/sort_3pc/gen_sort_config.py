#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


def infer_id_bits(n: int) -> int:
    return max(1, math.ceil(math.log2(max(n, 2))))


def load_vectors(path: str, layer: int) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix == ".npz":
        npz = np.load(p, allow_pickle=True)
        key = f"vecs_{layer}"
        if key in npz:
            arr = npz[key]
        elif "vecs" in npz:
            arr = npz["vecs"]
        else:
            raise ValueError("NPZ missing vecs array")
    else:
        raise ValueError("Unsupported vector format (use .npy or .npz)")
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--num-queries", type=int, default=1)
    ap.add_argument("--vec-bits", type=int, default=32)
    ap.add_argument("--id-bits", type=int, default=0)
    args = ap.parse_args()

    vecs = load_vectors(args.vecs, args.layer)
    n, d = vecs.shape
    id_bits = int(args.id_bits) if args.id_bits > 0 else infer_id_bits(n)

    cfg = {
        "N": int(n),
        "D": int(d),
        "k": int(args.k),
        "num_queries": int(args.num_queries),
        "vec_bits": int(args.vec_bits),
        "id_bits": id_bits,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
