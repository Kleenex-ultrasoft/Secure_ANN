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


def load_bin(path: Path, dtype, shape):
    arr = np.fromfile(path, dtype=dtype)
    return arr.reshape(shape)


def write_values(path: Path, values, binary: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        np.asarray(values, dtype=np.int64).tofile(path)
        return
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{int(v)}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shares-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--parties", type=int, default=3, choices=[3])
    ap.add_argument("--entry-count", type=int, default=0)
    ap.add_argument("--text", action="store_true", help="write text inputs instead of binary")
    args = ap.parse_args()

    shares_dir = Path(args.shares_dir)
    out_dir = Path(args.out_dir)

    meta_path = shares_dir / "p0" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    layers = meta["layers"]
    layers_sorted = sorted(layers, key=lambda x: x["idx"])
    d = int(meta["D"])

    entry_count = int(args.entry_count)
    if entry_count <= 0:
        entry_count = int(meta.get("num_queries", 1))
    print(
        f"[reshuffle inputs] layers={len(layers_sorted)} entry_count={entry_count} out_dir={out_dir}",
        flush=True,
    )

    for p in range(args.parties):
        print(f"[reshuffle inputs] party={p} start", flush=True)
        values = []
        base = shares_dir / f"p{p}"
        prev_id_bits = None
        for layer in layers_sorted:
            idx = layer["idx"]
            n = int(layer["N"])
            m = int(layer["M"])
            id_bits = int(layer["id_bits"])
            vec_bits = int(layer["vec_bits"])
            x2_bits = int(layer["x2_bits"])
            down_bits = int(layer.get("down_bits", prev_id_bits if prev_id_bits is not None else id_bits))
            print(
                f"[reshuffle inputs] party={p} layer={idx} n={n} m={m} d={d} id_bits={id_bits}",
                flush=True,
            )

            neigh = load_bin(base / f"layer{idx}_neigh.bin", dtype_for_bitlen(id_bits), (n, m))
            vecs = load_bin(base / f"layer{idx}_vecs.bin", dtype_for_bitlen(vec_bits), (n, d))
            x2 = load_bin(base / f"layer{idx}_x2.bin", dtype_for_bitlen(x2_bits), (n,))
            is_dummy = load_bin(base / f"layer{idx}_is_dummy.bin", np.uint8, (n,))
            values.extend(neigh.reshape(-1).tolist())
            values.extend(vecs.reshape(-1).tolist())
            values.extend(x2.reshape(-1).tolist())
            values.extend(is_dummy.reshape(-1).tolist())
            if idx > 0:
                down = load_bin(base / f"layer{idx}_down.bin", dtype_for_bitlen(down_bits), (n,))
                values.extend(down.reshape(-1).tolist())
            prev_id_bits = id_bits

        entry_path = base / "entry_point_top_local.bin"
        if not entry_path.exists():
            raise FileNotFoundError(entry_path)
        entry = np.fromfile(
            entry_path, dtype=dtype_for_bitlen(int(layers_sorted[-1]["id_bits"]))
        )
        if entry.size == 1 and entry_count > 1:
            entry = np.repeat(entry, entry_count)
        if entry.size != entry_count:
            raise ValueError(
                f"entry share size {entry.size} does not match entry_count={entry_count}"
            )
        values.extend(entry.reshape(-1).tolist())
        print(
            f"[reshuffle inputs] party={p} entry_count={entry.size}",
            flush=True,
        )

        if args.text:
            write_values(out_dir / f"Input-P{p}-0", values, binary=False)
        else:
            write_values(out_dir / f"Input-Binary-P{p}-0", values, binary=True)

        print(f"[reshuffle inputs] party={p} wrote inputs", flush=True)

    print(f"wrote inputs to {out_dir}")


if __name__ == "__main__":
    main()
