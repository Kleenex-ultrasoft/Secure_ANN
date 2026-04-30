import argparse, json, time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import spu.utils.distributed as ppd

@jax.jit
def fetch_dyn(table, idx):
    # 1D dynamic slice length=1 => forces DynamicSlice, not Gather
    return lax.dynamic_slice(table, (idx,), (1,))[0]

def bench(label, SPU, arr_s, idx_s, reps=10):
    # warmup (compile + first run)
    _ = ppd.get(SPU(fetch_dyn)(arr_s, idx_s))

    ts = []
    for _ in range(reps):
        t0 = time.time()
        out = SPU(fetch_dyn)(arr_s, idx_s)
        _ = ppd.get(out)
        ts.append((time.time() - t0) * 1000)

    ts.sort()
    print(f"{label}: median {ts[len(ts)//2]:.3f} ms (min {ts[0]:.3f}, max {ts[-1]:.3f})")
    return ts[len(ts)//2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    conf = json.load(open(args.config))
    ppd.init(conf["nodes"], conf["devices"])
    SPU = ppd.device("SPU")
    P1  = ppd.device("P1")

    # Dimensions to test
    # Note: Microbench uses flattened array for indexing conceptually in Python, 
    # but SPU handles multi-dim DynamicSlice. 
    # Panther clustering: retrieval is picking one vector of size D from N vectors.
    # We should simulate "picking 1 vector of size D from N".
    # Or picking D scalars (if we flatten).
    # ORAM should support reading a block. 
    # In JAX/SPU, dynamic_slice on (N, D) with slice_size=(1, D) is the equivalent.
    
    # N for SIFT ~100k. D ~ 128.
    # N for DEEP10M ~500k. D ~ 96.
    # Large D: 768, 1024, 1536, 3072.
    
    Ns = [10000] # Fixed N for dimensionality study
    Ds = [128, 768, 1024, 1536, 3072]
    
    print("N,D,Secret_Index_Time_ms")
    
    for N in Ns:
        for D in Ds:
            # Create (N, D) table
            arr = np.random.randint(0, 100, size=(N, D), dtype=np.int32)
            arr_s = SPU(lambda x: x)(arr)
            
            idx = N // 2
            idx_p1 = P1(lambda: np.int32(idx))()
            idx_secret_s = SPU(lambda x: x)(idx_p1)
            
            # Define slice function for 2D
            @jax.jit
            def fetch_row(table, idx):
                # Slice 1 row: start_indices=(idx, 0), slice_sizes=(1, D)
                return lax.dynamic_slice(table, (idx, 0), (1, D))
            
            # Bench
            label = f"N={N}, D={D}"
            
            # Warmup
            _ = ppd.get(SPU(fetch_row)(arr_s, idx_secret_s))
            
            ts = []
            for _ in range(5):
                t0 = time.time()
                out = SPU(fetch_row)(arr_s, idx_secret_s)
                _ = ppd.get(out)
                ts.append((time.time() - t0) * 1000)
            
            median_time = sorted(ts)[len(ts)//2]
            print(f"{N},{D},{median_time:.2f}")

if __name__ == "__main__":
    main()

