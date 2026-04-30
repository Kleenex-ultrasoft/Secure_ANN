"""Generate MP-SPDZ binary inputs for panther_ivf_doram.mpc.

Run:

    python3 gen_panther_ivf_inputs.py \
        --dataset_npy <vectors.npy> \
        --out_dir <Player-Data subdir> \
        --N 1000 --bin_size 32 --u 16 --K 10 --seed 0

Wire layout (matches the .mpc):

    centroids        T * D            sints
    bins             T * BIN_SIZE * D sints
    bin_ids          T * BIN_SIZE     sints
    query            D                sints

We only emit a 3-out-of-3 additive share where party 0 holds the
plaintext value and parties 1 and 2 hold zero.  This is enough for
benchmarking compute and communication; cryptographic random shares
would only change the byte cost of input distribution, which is
amortised away in our reported metrics.

The input vector is first integerised into the same range that
hnsw_layer_search.mpc uses (uint8, then signed int64) so that the L2
distance and the ID-packing maths line up with the existing programs.

Centroid choice is k-means.  Bin assignment is nearest-centroid.  When
a real bin is shorter than BIN_SIZE we pad with a poison vector
(coordinate value 1 << (DIST_BITS-1) - 1) and a dummy id (= N).  The
.mpc treats those slots as sortable garbage, so they never win the
final top-K.
"""

import argparse
import os
import struct
import sys

import numpy as np


# ---------------------------------------------------------------------------

def write_sint_bin(path, vals):
    """MP-SPDZ binary sint format = little-endian int64 per element."""
    arr = np.asarray(vals, dtype=np.int64)
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def kmeans_clusters(X, n_clusters, n_iter=20, seed=0):
    """Plain numpy k-means.  Avoid the sklearn import (server is conda
    minimal).  Returns (centroids[n_clusters, D], labels[N])."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    init = rng.choice(n, size=n_clusters, replace=False)
    centroids = X[init].astype(np.float64).copy()

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        # squared distances via broadcasting; chunk over rows so we
        # stay under a few GB on large N.
        chunk = 4096
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            diff = X[s:e, None, :].astype(np.float64) - centroids[None, :, :]
            d = (diff * diff).sum(axis=2)
            labels[s:e] = np.argmin(d, axis=1)
        new_c = np.zeros_like(centroids)
        counts = np.zeros(n_clusters, dtype=np.int64)
        np.add.at(new_c, labels, X.astype(np.float64))
        np.add.at(counts, labels, 1)
        empty = counts == 0
        counts[empty] = 1
        new_c /= counts[:, None]
        if empty.any():
            # reseed empty clusters with random data points
            spare = rng.choice(n, size=int(empty.sum()), replace=False)
            new_c[empty] = X[spare].astype(np.float64)
        if np.allclose(new_c, centroids, atol=1e-3):
            centroids = new_c
            break
        centroids = new_c
    return centroids, labels


def pack_bins(X, labels, T, bin_size, dummy_id, poison_val):
    """Group X rows by cluster, pad each bin to bin_size.

    Returns:
        bins      (T, bin_size, D) int64
        bin_ids   (T, bin_size)    int64    (real id for occupied
                                              slots, dummy_id for pads)
    """
    D = X.shape[1]
    bins = np.full((T, bin_size, D), poison_val, dtype=np.int64)
    bin_ids = np.full((T, bin_size), dummy_id, dtype=np.int64)
    occ = np.zeros(T, dtype=np.int64)

    for idx in range(X.shape[0]):
        c = int(labels[idx])
        if occ[c] >= bin_size:
            # overflow: drop excess members.  In practice we set
            # bin_size large enough (about 1.5 * N/T) that this is
            # rare.  If it triggers we print a notice so the user can
            # rerun with a larger bin_size.
            continue
        bins[c, occ[c]] = X[idx]
        bin_ids[c, occ[c]] = idx
        occ[c] += 1

    overflow = (occ >= bin_size).sum()
    if overflow > 0:
        sys.stderr.write(
            f"[gen_panther_ivf_inputs] {overflow}/{T} bins hit "
            f"bin_size={bin_size}; consider increasing bin_size\n")
    return bins, bin_ids


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npy", required=True,
                    help="Source vectors (.npy, float32 or float)")
    ap.add_argument("--out_dir", required=True,
                    help="Player-Data directory (will hold "
                         "Input-Binary-P{0,1,2}-0 + meta.txt)")
    ap.add_argument("--N",        type=int, required=True)
    ap.add_argument("--bin_size", type=int, default=32)
    ap.add_argument("--u",        type=int, default=16)
    ap.add_argument("--K",        type=int, default=10)
    ap.add_argument("--T",        type=int, default=0,
                    help="cluster count; default ceil(sqrt(N))")
    ap.add_argument("--id_bits",  type=int, default=20)
    ap.add_argument("--dist_bits", type=int, default=32)
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--n_iter",   type=int, default=20)
    args = ap.parse_args()

    if args.T <= 0:
        T = int(np.ceil(np.sqrt(args.N)))
    else:
        T = args.T

    if args.bin_size * T < args.N:
        # bump bin_size so .mpc precondition holds.
        bin_size = int(np.ceil(args.N / T) * 1.5) + 1
        sys.stderr.write(
            f"[gen_panther_ivf_inputs] bin_size auto-bumped to "
            f"{bin_size} (bin_size*T must be >= N)\n")
    else:
        bin_size = args.bin_size

    print(f"N={args.N} T={T} bin_size={bin_size} u={args.u} "
          f"K={args.K}")

    # ---- load vectors ---------------------------------------------------
    raw = np.load(args.dataset_npy)
    if raw.shape[0] < args.N:
        raise ValueError(f"NPY only has {raw.shape[0]} rows, need {args.N}")
    X_f = raw[: args.N].astype(np.float32)
    D = X_f.shape[1]

    # Match hnsw_layer_search.mpc integerisation: clamp to uint8 then int64.
    X = np.clip(X_f, 0, 255).astype(np.int64)

    # ---- cluster --------------------------------------------------------
    centroids_f, labels = kmeans_clusters(X, T, n_iter=args.n_iter,
                                          seed=args.seed)
    centroids = np.clip(centroids_f, 0, 255).astype(np.int64)

    # ---- bins -----------------------------------------------------------
    dummy_id = args.N
    # Use 0 for unused slots: the .mpc detects mid == dummy_id and
    # overrides the packed distance to MAX_DIST, so the actual vector
    # value is irrelevant for correctness.  Keeping it within the
    # uint8-derived range avoids any int64 wraparound surprise during
    # debugging or post-mortem inspection.
    poison_val = 0
    bins, bin_ids = pack_bins(X, labels, T, bin_size,
                              dummy_id=dummy_id, poison_val=poison_val)

    # ---- query ----------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    mean_vec = X.mean(axis=0).astype(np.int64)
    query = np.clip(mean_vec + rng.integers(-5, 5, D), 0, 255).astype(np.int64)

    # ---- write party files ---------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    centroids_flat = centroids.reshape(-1).tolist()
    bins_flat      = bins.reshape(-1).tolist()
    bin_ids_flat   = bin_ids.reshape(-1).tolist()
    query_flat     = query.reshape(-1).tolist()

    p0_vals = centroids_flat + bins_flat + bin_ids_flat + query_flat
    n_words = len(p0_vals)
    p_zero = [0] * n_words

    write_sint_bin(os.path.join(args.out_dir, "Input-Binary-P0-0"), p0_vals)
    write_sint_bin(os.path.join(args.out_dir, "Input-Binary-P1-0"), p_zero)
    write_sint_bin(os.path.join(args.out_dir, "Input-Binary-P2-0"), p_zero)

    with open(os.path.join(args.out_dir, "meta.txt"), "w") as f:
        f.write(f"N={args.N}\nD={D}\nT={T}\nBIN_SIZE={bin_size}\n"
                f"U={args.u}\nK={args.K}\n"
                f"ID_BITS={args.id_bits}\nDIST_BITS={args.dist_bits}\n"
                f"WORDS_PER_PARTY={n_words}\n")

    print(f"Wrote 3 party files, {n_words} int64 words each, to "
          f"{args.out_dir}")
    print(f"\ncompile.py args: panther_ivf_doram {args.N} {D} {T} "
          f"{bin_size} {args.u} {args.K} {args.id_bits} {args.dist_bits}")


if __name__ == "__main__":
    main()
