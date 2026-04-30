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


def load_entry_list(path: str, num_queries: int, entry_count: int) -> np.ndarray:
    if not path:
        eff_count = entry_count if entry_count > 0 else num_queries
        return np.zeros((eff_count,), dtype=np.int64)
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix == ".npz":
        npz = np.load(p, allow_pickle=True)
        if "entry_point_top_local" in npz:
            arr = npz["entry_point_top_local"]
        elif "entries" in npz:
            arr = npz["entries"]
        else:
            raise ValueError("NPZ missing entry list (entry_point_top_local or entries)")
    else:
        size = p.stat().st_size
        if size % 8 == 0:
            dtype = np.uint64
        elif size % 4 == 0:
            dtype = np.uint32
        else:
            raise ValueError(f"entry list size {size} is not a multiple of 4 or 8 bytes")
        arr = np.fromfile(p, dtype=dtype)
    arr = np.asarray(arr, dtype=np.int64).reshape(-1)
    if entry_count <= 0:
        entry_count = num_queries
    if entry_count not in (1, num_queries):
        raise ValueError("entry_count must be 1 or num_queries")
    if arr.size == 1 and entry_count == num_queries:
        arr = np.repeat(arr, num_queries)
    if arr.size != entry_count:
        raise ValueError(f"entry list size {arr.size} does not match entry_count={entry_count}")
    return arr


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
    ap.add_argument("--npz", default="", help="Required unless --shares-dir is set")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--shares-dir", default="", help="Use existing secret-shared tables (p0/p1/p2)")
    ap.add_argument("--out-dir", default="Player-Data")
    ap.add_argument("--queries", default="", help="Path to query .npy/.npz")
    ap.add_argument("--num-queries", type=int, default=1)
    ap.add_argument("--entry-list", default="", help="Path to entry list (.bin/.npy/.npz)")
    ap.add_argument("--entry-id", type=int, default=None)
    ap.add_argument("--entry-count", type=int, default=0, help="1 or num-queries (optional)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text", action="store_true", help="write text inputs instead of binary")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    layers = cfg["layers"]
    d = int(cfg["D"])
    num_queries = int(args.num_queries)
    entry_count = int(args.entry_count)
    if entry_count <= 0:
        entry_count = num_queries
    if entry_count not in (1, num_queries):
        raise ValueError("entry_count must be 1 or num_queries")

    shares_dir = Path(args.shares_dir) if args.shares_dir else None
    shares_meta = None
    if shares_dir:
        meta_path = shares_dir / "p0" / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            shares_meta = json.load(f)

    share_layer_by_idx = {}
    if shares_meta:
        for layer in shares_meta["layers"]:
            share_layer_by_idx[layer["idx"]] = layer

    rng = np.random.default_rng(args.seed)
    values_by_party = [[] for _ in range(3)]

    if shares_dir:
        query_plain = load_queries(args.queries, d, num_queries)
        q0, q1, q2 = share_array(query_plain.reshape(-1), rng)
        query_shares = [q0, q1, q2]

        dummy_fallback = None
        entry_fallback = None
        entry_share_bits = int(layers[0]["id_bits"])

        for layer_idx, layer in enumerate(layers):
            idx = layer["idx"]
            share_layer = share_layer_by_idx.get(idx, layer)
            id_bits = int(share_layer.get("id_bits", layer["id_bits"]))
            dummy_path = shares_dir / "p0" / f"layer{idx}_dummy_id.bin"
            if not dummy_path.exists():
                dummy_plain = [int(layer.get("dummy_id", layer["N"] - 1)) for layer in layers]
                d0, d1, d2 = share_array(np.asarray(dummy_plain, dtype=np.uint64), rng)
                dummy_fallback = [d0, d1, d2]
                break

        entry_share_path = shares_dir / "p0" / "entry_point_top_local.bin"
        if not entry_share_path.exists():
            if args.entry_list:
                entry_plain = load_entry_list(args.entry_list, num_queries, entry_count)
            elif args.entry_id is not None:
                entry_plain = np.full((entry_count,), int(args.entry_id), dtype=np.int64)
            else:
                entry_plain = np.full((entry_count,), int(cfg.get("entry_point_top", 0)), dtype=np.int64)
            e0, e1, e2 = share_array(np.asarray(entry_plain, dtype=np.uint64), rng)
            entry_fallback = [e0, e1, e2]

        for p in range(3):
            base = shares_dir / f"p{p}"
            graph_vals = []
            vec_vals = []
            down_vals = []
            dummy_vals = []

            for layer_idx, layer in enumerate(layers):
                idx = layer["idx"]
                n = int(layer["N"])
                m = int(layer["M"])
                share_layer = share_layer_by_idx.get(idx, layer)
                id_bits = int(share_layer.get("id_bits", layer["id_bits"]))
                vec_bits = int(share_layer.get("vec_bits", layer.get("dist_bits", 32)))
                down_bits = int(share_layer.get(
                    "down_bits",
                    layers[layer_idx + 1]["id_bits"] if layer_idx < len(layers) - 1 else id_bits,
                ))

                neigh = load_bin(base / f"layer{idx}_neigh.bin", dtype_for_bitlen(id_bits), (n, m))
                vecs = load_bin(base / f"layer{idx}_vecs.bin", dtype_for_bitlen(vec_bits), (n, d))
                graph_vals.extend(neigh.reshape(-1).tolist())
                vec_vals.extend(vecs.reshape(-1).tolist())

                if layer_idx < len(layers) - 1:
                    down = load_bin(base / f"layer{idx}_down.bin", dtype_for_bitlen(down_bits), (n,))
                    down_vals.extend(down.reshape(-1).tolist())

                dummy_path = base / f"layer{idx}_dummy_id.bin"
                if dummy_path.exists():
                    dummy_id = np.fromfile(dummy_path, dtype=dtype_for_bitlen(id_bits))
                    dummy_vals.extend(dummy_id.reshape(-1).tolist())
                elif dummy_fallback is not None:
                    dummy_vals.append(int(dummy_fallback[p][layer_idx]))

            entry_share_path = base / "entry_point_top_local.bin"
            if entry_share_path.exists():
                entry = np.fromfile(entry_share_path, dtype=dtype_for_bitlen(entry_share_bits))
                if entry.size == 1 and entry_count == num_queries:
                    entry = np.repeat(entry, num_queries)
                if entry.size != entry_count:
                    raise ValueError(f"entry share size {entry.size} does not match entry_count={entry_count}")
                entry_vals = entry.reshape(-1).tolist()
            elif entry_fallback is not None:
                entry_vals = entry_fallback[p].reshape(-1).tolist()
            else:
                if args.entry_list:
                    entry_plain = load_entry_list(args.entry_list, num_queries, entry_count)
                elif args.entry_id is not None:
                    entry_plain = np.full((entry_count,), int(args.entry_id), dtype=np.int64)
                else:
                    entry_plain = np.full((entry_count,), int(cfg.get("entry_point_top", 0)), dtype=np.int64)
                e0, e1, e2 = share_array(np.asarray(entry_plain, dtype=np.uint64), rng)
                entry_vals = [e0, e1, e2][p].reshape(-1).tolist()

            query_vals = query_shares[p].reshape(-1).tolist()

            values_by_party[p] = graph_vals + vec_vals + down_vals + dummy_vals + entry_vals + query_vals
    else:
        if not args.npz:
            raise ValueError("--npz is required when --shares-dir is not set")
        npz = np.load(args.npz, allow_pickle=True)
        graph_vals = []
        vec_vals = []
        down_vals = []
        dummy_vals = []

        for layer in layers:
            idx = layer["idx"]
            neigh = npz[f"neigh_{idx}"]
            vecs = npz[f"vecs_{idx}"]
            graph_vals.extend(neigh.reshape(-1).tolist())
            vec_vals.extend(vecs.reshape(-1).tolist())
            dummy_vals.append(int(layer.get("dummy_id", layer["N"] - 1)))

        for layer in layers[:-1]:
            idx = layer["idx"]
            if f"down_{idx}" in npz:
                down = npz[f"down_{idx}"]
            else:
                down = np.zeros((layer["N"],), dtype=np.int64)
            down_vals.extend(down.reshape(-1).tolist())

        if args.entry_list:
            entry_plain = load_entry_list(args.entry_list, num_queries, entry_count)
        elif args.entry_id is not None:
            entry_plain = np.full((entry_count,), int(args.entry_id), dtype=np.int64)
        else:
            entry_plain = np.full((entry_count,), int(cfg.get("entry_point_top", 0)), dtype=np.int64)
        queries = load_queries(args.queries, d, num_queries)

        g0, g1, g2 = share_array(np.asarray(graph_vals, dtype=np.uint64), rng)
        v0, v1, v2 = share_array(np.asarray(vec_vals, dtype=np.uint64), rng)
        d0, d1, d2 = share_array(np.asarray(down_vals, dtype=np.uint64), rng)
        u0, u1, u2 = share_array(np.asarray(dummy_vals, dtype=np.uint64), rng)
        e0, e1, e2 = share_array(np.asarray(entry_plain, dtype=np.uint64), rng)
        q0, q1, q2 = share_array(queries.reshape(-1), rng)

        values_by_party[0] = np.concatenate([g0, v0, d0, u0, e0, q0]).tolist()
        values_by_party[1] = np.concatenate([g1, v1, d1, u1, e1, q1]).tolist()
        values_by_party[2] = np.concatenate([g2, v2, d2, u2, e2, q2]).tolist()

    out_dir = Path(args.out_dir)
    for p in range(3):
        name = f"Input-Binary-P{p}-0" if not args.text else f"Input-P{p}-0"
        write_values(out_dir / name, values_by_party[p], binary=not args.text)

    print(f"wrote inputs to {out_dir}")


if __name__ == "__main__":
    main()
