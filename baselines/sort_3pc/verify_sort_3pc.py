#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


def dtype_for_bitlen(bitlen: int):
    if bitlen <= 8:
        return np.uint8
    if bitlen <= 16:
        return np.uint16
    if bitlen <= 32:
        return np.uint32
    return np.uint64


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
    return arr.astype(np.int64)


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


def reconstruct_shares(out_dir: Path, name: str, count: int, bits: int) -> np.ndarray:
    dtype = dtype_for_bitlen(bits)
    shares = []
    for p in range(3):
        path = out_dir / f"p{p}" / name
        if not path.exists():
            raise FileNotFoundError(path)
        arr = np.fromfile(path, dtype=dtype)
        if arr.size != count:
            raise ValueError(f"{path} has {arr.size} values, expected {count}")
        shares.append(arr.astype(np.uint64))
    total = shares[0] + shares[1] + shares[2]
    if bits < 64:
        total = total % (1 << bits)
    return total.astype(np.uint64)


def validate_topk(dist: np.ndarray, actual_ids: np.ndarray, k: int) -> bool:
    if len(np.unique(actual_ids)) != len(actual_ids):
        return False
    sorted_dist = np.sort(dist)
    kth = sorted_dist[k - 1]
    strict = set(np.where(dist < kth)[0].tolist())
    ties = set(np.where(dist == kth)[0].tolist())
    actual = set(actual_ids.tolist())
    if not strict.issubset(actual):
        return False
    if not actual.issubset(strict.union(ties)):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--queries", default="")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-queries", type=int, default=0)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--output", default="id", help="id, vec, both, id+vec")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    n = int(cfg["N"])
    d = int(cfg["D"])
    num_queries = int(args.num_queries) if args.num_queries > 0 else int(cfg.get("num_queries", 1))
    k = int(args.k) if args.k > 0 else int(cfg.get("k", 1))
    id_bits = int(cfg.get("id_bits", infer_id_bits(n)))
    vec_bits = int(cfg.get("vec_bits", cfg.get("dist_bits", 32)))

    if n <= 0 or d <= 0:
        raise ValueError("cfg must provide N>0 and D>0")
    if num_queries <= 0:
        raise ValueError("num_queries must be > 0")
    if k <= 0 or k > n:
        raise ValueError("k must be in [1, N]")

    output_mode = args.output.lower()
    output_id = output_mode in ("id", "ids", "both", "id+vec", "vec+id")
    output_vec = output_mode in ("vec", "vector", "both", "id+vec", "vec+id")

    vecs = load_vectors(args.vecs, args.layer)
    if vecs.shape[0] < n or vecs.shape[1] != d:
        raise ValueError(f"vecs shape {vecs.shape} does not match cfg N={n} D={d}")
    queries = load_queries(args.queries, num_queries, d)

    out_dir = Path(args.out_dir)
    if output_id:
        ids_flat = reconstruct_shares(out_dir, "topk_ids.bin", num_queries * k, id_bits)
        ids = ids_flat.reshape(num_queries, k).astype(np.int64)
    else:
        ids = None

    if output_vec:
        vec_flat = reconstruct_shares(
            out_dir, "topk_vectors.bin", num_queries * k * d, vec_bits
        )
        vec_out = vec_flat.reshape(num_queries, k, d).astype(np.int64)
    else:
        vec_out = None

    ok = True
    for q in range(num_queries):
        dist = vecs[:n].dot(queries[q])
        order = np.lexsort((np.arange(n, dtype=np.int64), dist))
        expected_ids = order[:k]
        if output_id:
            actual_ids = ids[q]
            if not validate_topk(dist, actual_ids, k):
                ok = False
                print(f"[q={q}] top-k validation failed")
                print(f"  expected_ids={expected_ids.tolist()}")
                print(f"  actual_ids={actual_ids.tolist()}")
            else:
                if not np.array_equal(expected_ids, actual_ids):
                    print(f"[q={q}] warning: tie reordering in top-k ids")
        if output_vec:
            if ids is None:
                ok = False
                print(f"[q={q}] missing id outputs for vector verification")
            else:
                for i in range(k):
                    vid = ids[q][i]
                    if not np.array_equal(vec_out[q][i], vecs[vid]):
                        ok = False
                        print(f"[q={q}] vector mismatch at rank {i} id={vid}")
                        break

    if not ok:
        raise SystemExit(1)
    print("sort_3pc correctness check passed")


if __name__ == "__main__":
    main()
