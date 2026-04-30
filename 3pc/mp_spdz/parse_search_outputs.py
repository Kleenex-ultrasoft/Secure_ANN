#!/usr/bin/env python3
import argparse
import json
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


def mod_reduce(values: np.ndarray, bits: int) -> np.ndarray:
    vals = values.astype(np.int64).astype(np.uint64)
    if bits >= 64:
        return vals
    return vals % (1 << bits)


def write_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp-spdz-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cfg", default="")
    ap.add_argument("--num-queries", type=int, default=1)
    ap.add_argument("--dim", type=int, default=0)
    ap.add_argument("--id-bits", type=int, default=0)
    ap.add_argument("--vec-bits", type=int, default=0)
    ap.add_argument("--output", default="id", help="id, vec, both, id+vec")
    ap.add_argument("--parties", type=int, default=3, choices=[3])
    args = ap.parse_args()

    output_mode = args.output.lower()
    output_id = output_mode in ("id", "ids", "both", "id+vec", "vec+id")
    output_vec = output_mode in ("vec", "vector", "both", "id+vec", "vec+id")

    dim = args.dim
    id_bits = args.id_bits
    vec_bits = args.vec_bits
    num_queries = int(args.num_queries)

    if args.cfg:
        with open(args.cfg, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        dim = int(cfg["D"])
        top = cfg["layers"][-1]
        id_bits = int(top["id_bits"])
        vec_bits = int(top.get("vec_bits", top.get("dist_bits", 32)))
        if num_queries == 1 and int(cfg.get("num_queries", 1)) > 1:
            num_queries = int(cfg["num_queries"])

    if output_id and id_bits <= 0:
        raise ValueError("--id-bits or --cfg is required for id output")
    if output_vec and dim <= 0:
        raise ValueError("--dim or --cfg is required for vector output")
    if output_vec and vec_bits <= 0:
        raise ValueError("--vec-bits or --cfg is required for vector output")

    mp_spdz_dir = Path(args.mp_spdz_dir)
    out_dir = Path(args.out_dir)

    for p in range(args.parties):
        out_path = mp_spdz_dir / f"Binary-Output-P{p}-0"
        if not out_path.exists():
            raise FileNotFoundError(out_path)
        values = np.fromfile(out_path, dtype=np.int64)
        offset = 0

        base = out_dir / f"p{p}"
        base.mkdir(parents=True, exist_ok=True)

        if output_id:
            entry = mod_reduce(values[offset:offset + num_queries], id_bits)
            offset += num_queries
            write_bin(base / "entry_point_top_local.bin", entry.astype(dtype_for_bitlen(id_bits)))

        if output_vec:
            count = num_queries * dim
            vecs = mod_reduce(values[offset:offset + count], vec_bits)
            offset += count
            write_bin(base / "entry_vectors.bin", vecs.astype(dtype_for_bitlen(vec_bits)))

        if offset != len(values):
            raise ValueError(f"Unconsumed outputs for P{p}: {len(values) - offset} values")

    print(f"wrote search outputs to {out_dir}")


if __name__ == "__main__":
    main()
