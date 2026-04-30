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

def bench(label, SPU, arr_s, idx_s, reps=30):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--N", type=int, default=1_000_000)
    ap.add_argument("--idx", type=int, default=50_000)
    args = ap.parse_args()

    conf = json.load(open(args.config))
    ppd.init(conf["nodes"], conf["devices"])
    SPU = ppd.device("SPU")
    P1  = ppd.device("P1")

    # Big table
    arr = np.arange(args.N, dtype=np.int32)
    arr_s = SPU(lambda x: x)(arr)  # table on SPU (secret/public doesn't matter for indexing test)

    # Public index: constant created inside SPU (should be public)
    idx_public_s = SPU(lambda: jnp.int32(args.idx))()

    # Secret index: comes from a party input then moved into SPU (should be secret)
    idx_p1 = P1(lambda: np.int32(args.idx))()
    idx_secret_s = SPU(lambda x: x)(idx_p1)

    print("Run with different N (e.g., 1e5, 1e6, 2e6) to see scaling.\n")
    bench("PUBLIC index", SPU, arr_s, idx_public_s)
    bench("SECRET index", SPU, arr_s, idx_secret_s)

if __name__ == "__main__":
    main()
