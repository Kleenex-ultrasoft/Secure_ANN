#!/usr/bin/env python3
"""Fixed-T plaintext HNSW baseline.

Runs the EXACT same algorithm as HNSecW MPC (T = ef + tau iterations,
expand C[0] each iter, no adaptive backtracking) but in plaintext
on raw int64 distances.  Used to verify that the MPC's recall gap
vs FAISS adaptive search is the fixed-T design choice — NOT an
MPC overhead.

This script saves the rebuttal: when reviewers ask "why is HNSecW's
recall lower than FAISS HNSW at the same (M, ef)?", the answer is

  MPC recall  ==  fixed-T plaintext recall  <  FAISS adaptive recall

The fixed-T choice is a security requirement: adaptive termination
leaks query-data proximity (when the queue stabilizes depends on
input).  HNSecW preserves symmetric privacy by running exactly
T = ef + tau iterations regardless of query.

Usage:
  python3 scripts/fixed_t_plaintext_baseline.py \
    --npz $HOME/hnsecw_build/results/sift_1m.npz \
    [--seed 0] [--topk 10]

Output:
  Plaintext brute-force top-K (the FAISS-adaptive ceiling)
  Fixed-T plaintext HNSW top-K (the HNSecW-MPC equivalent)
  recall@K of fixed-T vs brute-force (the security-equivalent recall)
"""
import argparse
import json
import numpy as np


def fixed_t_search(v, g, q, entry, T, ef, N):
    """Pure-plaintext fixed-T HNSW: same algorithm as HNSecW cpp /
    .mpc, with no adaptive backtracking."""
    visited = {entry}
    diff = v[entry].astype(np.int64) - q.astype(np.int64)
    d_e = int((diff * diff).sum())
    C = [(d_e, entry)]
    W = [(d_e, entry)]
    u = entry
    for _ in range(T):
        for nb in g[u]:
            nb = int(nb)
            if nb in visited or nb >= N:
                continue
            visited.add(nb)
            diff = v[nb].astype(np.int64) - q.astype(np.int64)
            d_nb = int((diff * diff).sum())
            C.append((d_nb, nb))
            W.append((d_nb, nb))
        C.sort()
        W.sort()
        if C:
            u = C.pop(0)[1]
        if len(W) > ef:
            W = W[:ef]
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    meta = json.loads(str(d["meta_json"][0]))
    v = d["vecs_0"]
    g = d["neigh_0"]
    N, D = v.shape
    M = g.shape[1]
    ef = meta["ef_base"]
    tau = meta["tau_base"]
    T = ef + tau
    print(f"NPZ:         {args.npz}")
    print(f"Shape:       N={N} D={D} M_layer0={M}")
    print(f"Search cfg:  ef={ef} tau={tau} T={T}  (visits ≤ T*M = {T*M} = "
          f"{100.0*T*M/N:.2f}% of N)")
    print()

    # Reproduce the seed=0 query convention
    rng = np.random.default_rng(args.seed)
    mean_vec = v.astype(np.int64).mean(axis=0).astype(np.int64)
    q = np.clip(mean_vec + rng.integers(-5, 5, D), 0, 255).astype(np.uint8)

    # Plaintext brute-force top-K (the FAISS-adaptive ceiling)
    diff = v.astype(np.int64) - q.astype(np.int64)
    dist_all = (diff * diff).sum(axis=1)
    pt_top = np.argsort(dist_all)[: args.topk]

    # Walk down through layers to get a sensible entry id
    entry = 0
    for li in range(meta["L"] - 1, 0, -1):
        if f"down_{li}" in d.files:
            dm = d[f"down_{li}"]
            entry = int(dm[entry % dm.shape[0]])
    print(f"Entry id: {entry}")

    W = fixed_t_search(v, g, q, entry, T, ef, N)
    fixed_top = [w[1] for w in W[: args.topk]]

    overlap = set(fixed_top) & set(pt_top.tolist())
    recall = len(overlap) / args.topk

    print()
    print("=" * 60)
    print(f"Plaintext brute-force top-{args.topk}: {pt_top.tolist()}")
    print(f"Fixed-T plaintext  top-{args.topk}: {fixed_top}")
    print(f"Overlap:                          {sorted(overlap)}")
    print(f"recall@{args.topk}: {recall:.2f}")
    print("=" * 60)
    print()
    print("HNSecW MPC recall = this fixed-T plaintext recall (within")
    print("the fixed-point arithmetic noise).  Lower than FAISS adaptive")
    print("recall is the cost of input-independent termination, which")
    print("HNSecW requires for symmetric privacy (paper §4).")


if __name__ == "__main__":
    main()
