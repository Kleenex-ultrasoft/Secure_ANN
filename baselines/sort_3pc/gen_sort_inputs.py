#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def load_vectors(path: str, n: int, d: int, layer: int) -> np.ndarray:
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
    if arr.shape[1] != d:
        raise ValueError(f"vector dim mismatch: got {arr.shape[1]} expected {d}")
    if arr.shape[0] < n:
        raise ValueError(f"vector count mismatch: got {arr.shape[0]} expected >= {n}")
    return arr[:n].astype(np.int64)


def load_queries(path: str, num_queries: int, d: int) -> np.ndarray:
    if not path:
        return np.zeros((num_queries, d), dtype=np.int64)
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix == ".npz":
        npz = np.load(p, allow_pickle=True)
        if "queries" in npz:
            arr = npz["queries"]
        else:
            raise ValueError("NPZ missing queries array")
    else:
        raise ValueError("Unsupported query format (use .npy or .npz)")
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != d:
        raise ValueError(f"query dim mismatch: got {arr.shape[1]} expected {d}")
    return arr[:num_queries].astype(np.int64)


def share_array(values: np.ndarray, rng: np.random.Generator):
    vals = np.asarray(values, dtype=np.uint64)
    r0 = rng.integers(0, 1 << 64, size=vals.shape, dtype=np.uint64)
    r1 = rng.integers(0, 1 << 64, size=vals.shape, dtype=np.uint64)
    r2 = (vals - r0 - r1).astype(np.uint64)
    return r0, r1, r2


def write_values(path: Path, values, binary: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        np.asarray(values, dtype=np.uint64).tofile(path)
        return
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{int(v)}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--vecs", required=True, help="Vectors .npy/.npz")
    ap.add_argument("--layer", type=int, default=0, help="NPZ layer index")
    ap.add_argument("--queries", default="", help="Query .npy/.npz")
    ap.add_argument("--num-queries", type=int, default=1)
    ap.add_argument("--out-dir", default="Player-Data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text", action="store_true")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    n = int(cfg["N"])
    d = int(cfg["D"])
    num_queries = int(args.num_queries)

    vecs = load_vectors(args.vecs, n, d, args.layer)
    queries = load_queries(args.queries, num_queries, d)

    rng = np.random.default_rng(args.seed)
    v0, v1, v2 = share_array(vecs.reshape(-1), rng)
    q0, q1, q2 = share_array(queries.reshape(-1), rng)

    values_by_party = [
        np.concatenate([v0, q0]).tolist(),
        np.concatenate([v1, q1]).tolist(),
        np.concatenate([v2, q2]).tolist(),
    ]

    out_dir = Path(args.out_dir)
    for p in range(3):
        name = f"Input-Binary-P{p}-0" if not args.text else f"Input-P{p}-0"
        write_values(out_dir / name, values_by_party[p], binary=not args.text)

    print(f"wrote inputs to {out_dir}")


if __name__ == "__main__":
    main()
