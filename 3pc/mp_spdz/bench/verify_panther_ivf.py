"""Brute-force plaintext top-K verifier for panther_ivf_doram runs.

Reads the same NPY the input generator consumed, regenerates the
query the same way (mean + per-coord uniform jitter), and prints the
plaintext top-K so the operator can eyeball it against the MPC log.

Usage:

    python3 verify_panther_ivf.py \
        --dataset_npy /tmp/p_ivf_n2000.npy \
        --N 2000 --K 10 --seed 0

The recall the MPC actually attains is bounded by IVF approximation,
so we expect agreement on top-1 (highest-density bin almost always
contains the global nearest) and partial agreement on top-K.
"""

import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npy", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = np.load(args.dataset_npy)
    if raw.shape[0] < args.N:
        raise ValueError(f"NPY only has {raw.shape[0]} rows, need {args.N}")
    X = raw[: args.N].astype(np.float32)
    X = np.clip(X, 0, 255).astype(np.int64)
    D = X.shape[1]

    rng = np.random.default_rng(args.seed)
    mean_vec = X.mean(axis=0).astype(np.int64)
    query = np.clip(mean_vec + rng.integers(-5, 5, D), 0, 255).astype(np.int64)

    diff = X - query
    dists = (diff * diff).sum(axis=1)
    order = np.argsort(dists)[: args.K]

    print(f"plaintext top-{args.K}:")
    for rank, i in enumerate(order):
        print(f"  rank={rank} id={int(i)} dist={int(dists[i])}")


if __name__ == "__main__":
    main()
