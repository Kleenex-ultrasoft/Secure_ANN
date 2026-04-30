#!/usr/bin/env python3
"""Build a cfg.json for the ABY3 hnsecw_search_aby3 binary.

The binary requires a cfg.json containing per-layer search parameters
(T, L_C, L_W) in addition to the share-format meta.json.  This script
constructs the cfg from the share's meta.json plus an HNSW (ef, tau)
specification.

Usage:
    python build_cfg_aby3.py \
        --meta /path/to/shares/p0/meta.json \
        --ef 16 --tau 4 \
        --out /path/to/cfg.json
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True,
                    help="Path to share-side meta.json (p0/meta.json)")
    ap.add_argument("--ef", type=int, default=16,
                    help="ef budget at the base layer (default 16)")
    ap.add_argument("--tau", type=int, default=4,
                    help="termination slack at the base layer (default 4)")
    ap.add_argument("--upper-tau", type=int, default=4,
                    help="termination slack for upper layers (default 4)")
    ap.add_argument("--out", required=True, help="cfg.json output path")
    args = ap.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)

    if meta.get("share_format") != "aby3":
        print(f"WARNING: meta.share_format is {meta.get('share_format')!r}, "
              f"expected 'aby3'", file=sys.stderr)

    cfg = {"D": meta["D"], "layers": []}
    for layer in meta["layers"]:
        if layer["idx"] == 0:
            T = args.ef + args.tau
            L_C = T
            L_W = args.ef
        else:
            T = 1 + args.upper_tau
            L_C = T
            L_W = 1
        cfg["layers"].append({
            "idx": layer["idx"],
            "N": layer["N"],
            "M": layer["M"],
            "T": T,
            "L_C": L_C,
            "L_W": L_W,
            "id_bits": layer["id_bits"],
            "vec_bits": layer["vec_bits"],
            "x2_bits": layer["x2_bits"],
        })

    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote cfg with {len(cfg['layers'])} layers to {args.out}")


if __name__ == "__main__":
    main()
