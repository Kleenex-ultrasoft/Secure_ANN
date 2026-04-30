"""Generate MP-SPDZ binary input files for hnsw_layer_search.mpc.

Schema per player i: a binary file with sint values in order:
  - graph_flat (N*M words): neighbor ids, row-major
  - vec_flat   (N*D words): vector coordinates, row-major
  - dummy_id   (1 word)
  - entry_id   (1 word)
  - query      (D words)

We generate shares such that s0 + s1 + s2 = plaintext (secret-shared additive).
For simplicity we split plaintext as (p, 0, 0) — not cryptographically useful but
mathematically correct for benchmarking compute + comm cost (which is what we want).
"""
import numpy as np
import hnswlib
import sys, argparse, struct, os

def write_sint_bin(path, vals, bits=64):
    """Write a binary file MP-SPDZ can read as sint.get_input_from(..., binary=True).
    MP-SPDZ binary sint format: little-endian int64 per element."""
    arr = np.array(vals, dtype=np.int64)
    with open(path, "wb") as f:
        f.write(arr.tobytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npy", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--k", type=int, default=10,
                    help="top-k requested at runtime; sets ef^{(0)}=k+10")
    ap.add_argument("--C", type=int, default=7,
                    help="constant slack from §4.1; Δ̄^{(0)}=ceil(log_2(N*d/M))+C")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X = np.load(args.dataset_npy).astype(np.float32)[:args.N]
    D = X.shape[1]
    X_u8 = np.clip(X, 0, 255).astype(np.uint8).astype(np.int64)

    # Build HNSW graph
    idx = hnswlib.Index(space="l2", dim=D)
    idx.init_index(max_elements=args.N, ef_construction=200, M=args.M)
    idx.add_items(X_u8.astype(np.float32), np.arange(args.N))
    idx.set_ef(50)

    # For each node, get M neighbors from knn_query (exclude self)
    labels, _ = idx.knn_query(X_u8.astype(np.float32), k=args.M + 1)
    neigh = labels[:, 1:args.M+1].astype(np.int64)  # (N, M)

    # Generate query: random point near dataset mean
    mean_vec = X_u8.mean(axis=0).astype(np.int64)
    query = mean_vec + rng.integers(-5, 5, D)
    query = np.clip(query, 0, 255)

    # Entry id: random node
    entry_id = int(rng.integers(0, args.N))
    dummy_id = args.N  # use N as dummy marker

    # Write share files for 3 players: (plaintext, 0, 0)
    os.makedirs(args.out_dir, exist_ok=True)
    for p in range(3):
        lines = []
        share_vals = {}
        if p == 0:
            # Graph
            g_vals = neigh.flatten().tolist()
            # Vec
            v_vals = X_u8.flatten().tolist()
            dummy_v = dummy_id
            entry_v = entry_id
            q_vals = query.tolist()
        else:
            g_vals = [0] * (args.N * args.M)
            v_vals = [0] * (args.N * D)
            dummy_v = 0
            entry_v = 0
            q_vals = [0] * D

        path = os.path.join(args.out_dir, f"Input-Binary-P{p}-0")
        all_vals = g_vals + v_vals + [dummy_v, entry_v] + q_vals
        write_sint_bin(path, all_vals)
        print(f"Wrote {path} ({len(all_vals)} int64 words)")

    # §4.1 deployed strategy: ef^{(0)}=k+10, Δ̄^{(0)}=ceil(log_2(N*d/M))+C.
    import math
    ef = args.k + 10
    delta = int(math.ceil(math.log2(max(2, args.N * D / max(1, args.M))))) + args.C
    LC = ef + delta
    LW = ef
    T = LC

    with open(os.path.join(args.out_dir, "meta.txt"), "w") as f:
        f.write(f"N={args.N}\nM={args.M}\nD={D}\nLC={LC}\nLW={LW}\nT={T}\nID_BITS=20\nDIST_BITS=32\n"
                f"# ef^{{(0)}}={ef}=k+10 (k={args.k})\n"
                f"# Delta_bar^{{(0)}}={delta}=ceil(log2(N*d/M))+C (C={args.C})\n")
    print(f"\nMP-SPDZ args: {args.N} {args.M} {D} {LC} {LW} {T} 20 32")
    print(f"  -> ef^(0)=k+10={ef}, Delta_bar^(0)={delta} (C={args.C})")

if __name__ == "__main__":
    main()
