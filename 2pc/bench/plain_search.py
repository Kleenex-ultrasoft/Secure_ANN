#!/usr/bin/env python3
"""Plaintext reference of the fixed-T layer search (Algorithms 2 and 3).

Runs the same search the MPC programs run, on the tables written by
build_index.py, and reports recall@k against exact nearest neighbours.
Use it to choose parameters before paying for an MPC run.
"""

import argparse
import json

import numpy as np


def load_index(path):
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta_json"][0]))
    layers = []
    for l in range(meta["num_layers"]):
        layers.append({
            "vecs": data[f"vecs_{l}"].astype(np.int64),
            "neigh": data[f"neigh_{l}"],
            "ids": data[f"ids_{l}"],
        })
    return meta, layers


def search_layer(layer, q, entry, T, ef):
    """Returns the ef best (distance, local id) pairs found in T iterations."""
    vecs, neigh = layer["vecs"], layer["neigh"]
    n = vecs.shape[0]

    def dist(i):
        diff = vecs[i] - q
        return int(diff @ diff)

    visited = {entry}
    d0 = dist(entry)
    cand = [(d0, entry)]          # candidate array C
    best = [(d0, entry)]          # result array W
    for _ in range(T):
        if not cand:
            break
        cand.sort()
        _, c = cand.pop(0)
        fresh = [int(u) for u in neigh[c] if u < n and int(u) not in visited]
        fresh = list(dict.fromkeys(fresh))
        if fresh:
            visited.update(fresh)
            diff = vecs[fresh] - q
            ds = np.einsum("ij,ij->i", diff, diff)
            new = [(int(dv), u) for dv, u in zip(ds, fresh)]
            cand.extend(new)
            best.extend(new)
            best.sort()
            del best[ef:]
    return best


def search(meta, layers, q, k):
    top = meta["num_layers"] - 1
    entry_global = meta["entry_global"]
    for l in range(top, -1, -1):
        layer = layers[l]
        local = int(np.nonzero(layer["ids"] == entry_global)[0][0])
        if l > 0:
            T, ef = meta["ef_upper"] + meta["delta_upper"], meta["ef_upper"]
        else:
            T, ef = meta["ef_base"] + meta["delta_base"], meta["ef_base"]
        best = search_layer(layer, q, local, T, ef)
        entry_global = int(layer["ids"][best[0][1]])
        if l == 0:
            return [int(layer["ids"][i]) for _, i in best[:k]]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True, help=".npz from build_index.py")
    ap.add_argument("--queries", required=True, help=".npy with uint8 query vectors")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    meta, layers = load_index(args.index)
    k = args.k or meta["k"]
    queries = np.load(args.queries)[: args.num].astype(np.int64)
    base = layers[0]["vecs"]
    base_ids = layers[0]["ids"]

    recalls = []
    for q in queries:
        got = search(meta, layers, q, k)
        diff = base - q
        exact = base_ids[np.argsort(np.einsum("ij,ij->i", diff, diff), kind="stable")[:k]]
        recalls.append(len(set(got) & set(int(x) for x in exact)) / k)
    recalls = np.array(recalls)
    print(f"queries={len(recalls)} k={k} mean recall={recalls.mean():.4f} "
          f"5th percentile={np.percentile(recalls, 5):.2f} min={recalls.min():.2f}")


if __name__ == "__main__":
    main()
