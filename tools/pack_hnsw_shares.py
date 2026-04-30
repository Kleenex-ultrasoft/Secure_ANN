#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def load_meta(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return json.loads(str(data["meta_json"][0]))


def bitlen_for_max(val: int) -> int:
    if val <= 0:
        return 1
    return (val - 1).bit_length()


def dtype_for_bitlen(bitlen: int):
    if bitlen <= 8:
        return np.uint8
    if bitlen <= 16:
        return np.uint16
    if bitlen <= 32:
        return np.uint32
    return np.uint64


def share_additive(values: np.ndarray, parties: int, bitlen: int, rng: np.random.Generator):
    if bitlen > 64:
        raise ValueError("additive share format supports up to 64-bit ring")
    if bitlen == 64:
        def rand_u64(shape):
            hi = rng.integers(0, 1 << 32, size=shape, dtype=np.uint64)
            lo = rng.integers(0, 1 << 32, size=shape, dtype=np.uint64)
            return (hi << np.uint64(32)) | lo

        shares = []
        acc = np.zeros_like(values, dtype=np.uint64)
        for _ in range(parties - 1):
            share = rand_u64(values.shape)
            shares.append(share)
            acc = acc + share
        last = values.astype(np.uint64) - acc
        shares.append(last)
        return shares

    mod = 1 << bitlen
    shares = []
    acc = np.zeros_like(values, dtype=np.uint64)
    for _ in range(parties - 1):
        share = rng.integers(0, mod, size=values.shape, dtype=np.uint64)
        shares.append(share)
        acc = (acc + share) % mod
    last = (values.astype(np.uint64) - acc) % mod
    shares.append(last)
    return shares


def share_aby3(values: np.ndarray, bitlen: int, rng: np.random.Generator):
    if bitlen > 63:
        raise ValueError("ABY3 share format requires bitlen <= 63")
    mod = 1 << bitlen
    a = rng.integers(0, mod, size=values.shape, dtype=np.uint64)
    b = rng.integers(0, mod, size=values.shape, dtype=np.uint64)
    c = (values.astype(np.uint64) - a - b) % mod
    # ABY3 replicated share layout: p0=(a,c), p1=(b,a), p2=(c,b).
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
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--parties", type=int, default=2, choices=[2, 3])
    ap.add_argument("--format", default="additive", choices=["additive", "aby3"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    meta = load_meta(npz_path)
    L = int(meta["L"])
    D = int(meta["d"])

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)

    layers_cfg = []

    for l in range(L):
        n_total = int(meta[f"layer_total_{l}"])
        if f"layer_size_{l}" in meta:
            n_real = int(meta[f"layer_size_{l}"])
        else:
            n_dummy = int(meta.get(f"layer_dummy_{l}", 0))
            n_real = max(0, n_total - n_dummy)
        neigh = data[f"neigh_{l}"]
        vecs = data[f"vecs_{l}"]
        x2 = data[f"x2_{l}"]
        is_dummy = data[f"is_dummy_{l}"]
        M = int(neigh.shape[1])
        id_bits = bitlen_for_max(n_total)
        vec_bits = vecs.dtype.itemsize * 8
        x2_bits = x2.dtype.itemsize * 8
        down_bits = 0
        if l > 0:
            down_bits = bitlen_for_max(int(meta[f"layer_total_{l-1}"]))
        layers_cfg.append({
            "idx": l,
            "N": n_total,
            "N_real": n_real,
            "M": M,
            "D": D,
            "id_bits": id_bits,
            "vec_bits": vec_bits,
            "vec_share_bits": vec_bits,
            "x2_bits": x2_bits,
            "down_bits": down_bits,
        })

        down = None
        if l > 0:
            down = data[f"down_{l}"]

        if args.format == "aby3":
            if args.parties != 3:
                raise ValueError("ABY3 format requires --parties 3")
            shares_neigh = share_aby3(neigh, id_bits, rng)
            shares_vecs = share_aby3(vecs, vec_bits, rng)
            shares_x2 = share_aby3(x2, x2_bits, rng)
            shares_dummy = share_aby3(is_dummy, 1, rng)
            shares_down = None
            if down is not None:
                shares_down = share_aby3(down, down_bits, rng)

            for p in range(3):
                base = out_dir / f"p{p}"
                write_bin(
                    base / f"layer{l}_neigh_s0.bin",
                    shares_neigh[p][0].astype(dtype_for_bitlen(id_bits)),
                )
                write_bin(
                    base / f"layer{l}_neigh_s1.bin",
                    shares_neigh[p][1].astype(dtype_for_bitlen(id_bits)),
                )
                write_bin(
                    base / f"layer{l}_vecs_s0.bin",
                    shares_vecs[p][0].astype(dtype_for_bitlen(vec_bits)),
                )
                write_bin(
                    base / f"layer{l}_vecs_s1.bin",
                    shares_vecs[p][1].astype(dtype_for_bitlen(vec_bits)),
                )
                write_bin(
                    base / f"layer{l}_x2_s0.bin",
                    shares_x2[p][0].astype(dtype_for_bitlen(x2_bits)),
                )
                write_bin(
                    base / f"layer{l}_x2_s1.bin",
                    shares_x2[p][1].astype(dtype_for_bitlen(x2_bits)),
                )
                write_bin(
                    base / f"layer{l}_is_dummy_s0.bin",
                    shares_dummy[p][0].astype(np.uint8),
                )
                write_bin(
                    base / f"layer{l}_is_dummy_s1.bin",
                    shares_dummy[p][1].astype(np.uint8),
                )
                if shares_down is not None:
                    write_bin(
                        base / f"layer{l}_down_s0.bin",
                        shares_down[p][0].astype(dtype_for_bitlen(down_bits)),
                    )
                    write_bin(
                        base / f"layer{l}_down_s1.bin",
                        shares_down[p][1].astype(dtype_for_bitlen(down_bits)),
                    )
        else:
            shares_neigh = share_additive(neigh.astype(np.uint64), args.parties, id_bits, rng)
            shares_vecs = share_additive(vecs.astype(np.uint64), args.parties, vec_bits, rng)
            shares_x2 = share_additive(x2.astype(np.uint64), args.parties, x2_bits, rng)
            shares_dummy = share_additive(is_dummy.astype(np.uint64), args.parties, 1, rng)
            shares_down = None
            if down is not None:
                shares_down = share_additive(down.astype(np.uint64), args.parties, down_bits, rng)

            for p in range(args.parties):
                base = out_dir / f"p{p}"
                write_bin(base / f"layer{l}_neigh.bin", shares_neigh[p].astype(dtype_for_bitlen(id_bits)))
                write_bin(base / f"layer{l}_vecs.bin", shares_vecs[p].astype(dtype_for_bitlen(vec_bits)))
                write_bin(base / f"layer{l}_x2.bin", shares_x2[p].astype(dtype_for_bitlen(x2_bits)))
                write_bin(base / f"layer{l}_is_dummy.bin", shares_dummy[p].astype(np.uint8))
                if shares_down is not None:
                    write_bin(base / f"layer{l}_down.bin", shares_down[p].astype(dtype_for_bitlen(down_bits)))

    entry = data.get("entry_point_top_local", None)
    if entry is not None:
        entry_val = int(entry[0])
        entry_bits = bitlen_for_max(int(meta[f"layer_total_{L-1}"]))
        if args.format == "aby3":
            shares_entry = share_aby3(np.array([entry_val], dtype=np.uint64), entry_bits, rng)
            for p in range(3):
                base = out_dir / f"p{p}"
                write_bin(
                    base / "entry_point_top_local_s0.bin",
                    shares_entry[p][0].astype(dtype_for_bitlen(entry_bits)),
                )
                write_bin(
                    base / "entry_point_top_local_s1.bin",
                    shares_entry[p][1].astype(dtype_for_bitlen(entry_bits)),
                )
        else:
            shares_entry = share_additive(
                np.array([entry_val], dtype=np.uint64), args.parties, entry_bits, rng
            )
            for p in range(args.parties):
                base = out_dir / f"p{p}"
                write_bin(base / "entry_point_top_local.bin", shares_entry[p].astype(dtype_for_bitlen(entry_bits)))

    meta_out = {
        "L": L,
        "D": D,
        "layers": layers_cfg,
        "share_format": args.format,
    }
    for p in range(args.parties):
        base = out_dir / f"p{p}"
        base.mkdir(parents=True, exist_ok=True)
        with open(base / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)

    print(f"wrote shares to {out_dir}")


if __name__ == "__main__":
    main()
