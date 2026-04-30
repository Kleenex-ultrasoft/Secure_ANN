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
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--mp-spdz-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--parties", type=int, default=3, choices=[3])
    ap.add_argument("--entry-count", type=int, default=0)
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    layers = cfg["layers"]
    layers_sorted = sorted(layers, key=lambda x: x["idx"])
    d = int(cfg["D"])
    out_dir = Path(args.out_dir)
    mp_spdz_dir = Path(args.mp_spdz_dir)

    entry_count = int(args.entry_count)
    if entry_count <= 0:
        entry_count = int(cfg.get("num_queries", 1))

    for p in range(args.parties):
        out_path = mp_spdz_dir / f"Binary-Output-P{p}-0"
        if not out_path.exists():
            raise FileNotFoundError(out_path)
        values = np.fromfile(out_path, dtype=np.int64)
        offset = 0

        base = out_dir / f"p{p}"
        base.mkdir(parents=True, exist_ok=True)

        prev_id_bits = None
        for layer in layers_sorted:
            idx = layer["idx"]
            n = int(layer["N"])
            m = int(layer["M"])
            id_bits = int(layer["id_bits"])
            vec_bits = int(layer.get("vec_bits", layer.get("dist_bits", 32)))
            x2_bits = int(layer.get("x2_bits", layer.get("dist_bits", 32)))
            down_bits = int(layer.get("down_bits", prev_id_bits if prev_id_bits is not None else id_bits))

            count_neigh = n * m
            count_vecs = n * d
            count_x2 = n
            count_dummy = n
            count_dummy_id = 1

            neigh = mod_reduce(values[offset:offset + count_neigh], id_bits)
            offset += count_neigh
            vecs = mod_reduce(values[offset:offset + count_vecs], vec_bits)
            offset += count_vecs
            x2 = mod_reduce(values[offset:offset + count_x2], x2_bits)
            offset += count_x2
            is_dummy = mod_reduce(values[offset:offset + count_dummy], 1)
            offset += count_dummy
            dummy_id = mod_reduce(values[offset:offset + count_dummy_id], id_bits)
            offset += count_dummy_id

            write_bin(base / f"layer{idx}_neigh.bin", neigh.astype(dtype_for_bitlen(id_bits)))
            write_bin(base / f"layer{idx}_vecs.bin", vecs.astype(dtype_for_bitlen(vec_bits)))
            write_bin(base / f"layer{idx}_x2.bin", x2.astype(dtype_for_bitlen(x2_bits)))
            write_bin(base / f"layer{idx}_is_dummy.bin", is_dummy.astype(np.uint8))
            write_bin(base / f"layer{idx}_dummy_id.bin", dummy_id.astype(dtype_for_bitlen(id_bits)))

            if idx > 0:
                down = mod_reduce(values[offset:offset + n], down_bits)
                offset += n
                write_bin(base / f"layer{idx}_down.bin", down.astype(dtype_for_bitlen(down_bits)))
            prev_id_bits = id_bits

        entry_bits = int(layers_sorted[-1]["id_bits"])
        entry = mod_reduce(values[offset:offset + entry_count], entry_bits)
        offset += entry_count
        write_bin(base / "entry_point_top_local.bin", entry.astype(dtype_for_bitlen(entry_bits)))

        if offset != len(values):
            raise ValueError(f"Unconsumed outputs for P{p}: {len(values) - offset} values")

        with open(base / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"layers": layers_sorted, "D": d, "num_queries": entry_count}, f, indent=2)

    print(f"wrote reshuffled shares to {out_dir}")


if __name__ == "__main__":
    main()
