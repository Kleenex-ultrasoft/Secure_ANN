#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def load_queries(path: str, d: int, num_queries: int) -> np.ndarray:
    if not path:
        return np.zeros((num_queries, d), dtype=np.int64)
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:num_queries].astype(np.int64)
    if p.suffix == ".npz":
        npz = np.load(p, allow_pickle=True)
        if "queries" in npz:
            arr = npz["queries"]
        else:
            raise ValueError("NPZ missing 'queries' array")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:num_queries].astype(np.int64)
    raise ValueError("Unsupported query format (use .npy or .npz)")


def dtype_for_bitlen(bitlen: int):
    if bitlen <= 8:
        return np.uint8
    if bitlen <= 16:
        return np.uint16
    if bitlen <= 32:
        return np.uint32
    return np.uint64


def share_aby3(values: np.ndarray, bitlen: int, rng: np.random.Generator):
    if bitlen > 63:
        raise ValueError("ABY3 share format requires bitlen <= 63")
    mod = 1 << bitlen
    a = rng.integers(0, mod, size=values.shape, dtype=np.uint64)
    b = rng.integers(0, mod, size=values.shape, dtype=np.uint64)
    c = (values.astype(np.uint64) - a - b) % mod
    return [
        (a, c),
        (b, a),
        (c, b),
    ]


def write_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--shares-dir", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--queries", default="")
    ap.add_argument("--num-queries", type=int, default=0)
    ap.add_argument("--vec-bits", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    shares_dir = Path(args.shares_dir)
    if not shares_dir.exists():
        raise FileNotFoundError(shares_dir)

    meta_path = shares_dir / "p0" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("share_format") != "aby3":
        raise ValueError("share_format must be aby3 in shares meta.json")

    dim = int(cfg.get("D", 0))
    if dim <= 0:
        raise ValueError("cfg D must be > 0")

    num_queries = int(args.num_queries or cfg.get("num_queries", 1))
    if num_queries <= 0:
        raise ValueError("num_queries must be > 0")

    vec_bits = int(args.vec_bits)
    if vec_bits <= 0:
        layers = meta.get("layers", [])
        if layers:
            vec_bits = int(layers[0].get("vec_bits", 0))
    if vec_bits <= 0:
        layers = cfg.get("layers", [])
        if layers:
            vec_bits = int(layers[0].get("vec_bits", 0))
    if vec_bits <= 0:
        raise ValueError("vec_bits not found in cfg or meta (use --vec-bits)")

    queries = load_queries(args.queries, dim, num_queries)
    if queries.shape[1] != dim:
        raise ValueError(f"query dim mismatch: expected {dim}, got {queries.shape[1]}")

    mod = 1 << vec_bits
    values = np.asarray(queries, dtype=np.int64) % mod
    values = values.astype(np.uint64)

    rng = np.random.default_rng(args.seed)
    shares = share_aby3(values.reshape(-1), vec_bits, rng)

    out_dir = Path(args.out_dir) if args.out_dir else shares_dir
    dtype = dtype_for_bitlen(vec_bits)
    for p in range(3):
        base = out_dir / f"p{p}"
        write_bin(base / "queries_s0.bin", shares[p][0].astype(dtype))
        write_bin(base / "queries_s1.bin", shares[p][1].astype(dtype))


if __name__ == "__main__":
    main()
