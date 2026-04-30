#!/usr/bin/env bash
#
# Fetch a small SIFT subset for demo.sh / build_hnsecw_index.sh.
#
# Downloads SIFT-1M base vectors (texmex.irisa.fr mirror) and writes
# the first 10K vectors as datasets/sift_base.npy in the project root.
# That is enough to run demo.sh end-to-end in a couple of minutes.
#
# Other supported datasets (download separately):
#   - SIFT-1M / GIST-1M / BIGANN-1B   http://corpus-texmex.irisa.fr/
#   - DEEP1B / DEEP-100M              http://sites.skoltech.ru/compvision/noimi/
#   - MNIST-60K                       http://yann.lecun.com/exdb/mnist/
#   - Fashion-60K                     https://github.com/zalandoresearch/fashion-mnist
#   - SPACEV-1B / SSNPP-1B            https://big-ann-benchmarks.com/
#   - LAION                           https://laion.ai/blog/laion-5b/
#   - MS MARCO                        https://microsoft.github.io/msmarco/

set -euo pipefail

OUT_DIR="${1:-$HOME/hnsecw_build/datasets}"
mkdir -p "$OUT_DIR"

URL="http://corpus-texmex.irisa.fr/sift.tar.gz"
TGZ="$OUT_DIR/sift.tar.gz"

if [ ! -f "$OUT_DIR/sift_base.npy" ]; then
    if [ ! -f "$TGZ" ]; then
        echo "Downloading SIFT-1M from $URL ..."
        echo "  (canonical landing page: http://corpus-texmex.irisa.fr/ )"
        curl -L -o "$TGZ" "$URL"
    fi
    tar -xzf "$TGZ" -C "$OUT_DIR"
    python3 - <<EOF
import numpy as np, struct, sys
fv = open("$OUT_DIR/sift/sift_base.fvecs", "rb").read()
# .fvecs is records: (int32 dim, dim * float32 values)
buf = np.frombuffer(fv, dtype=np.uint8)
i = 0
out = []
while i < buf.size and len(out) < 10000:
    d = struct.unpack_from("<i", buf, i)[0]
    vals = np.frombuffer(buf, dtype=np.float32, count=d, offset=i + 4)
    out.append(vals.copy())
    i += 4 + 4 * d
arr = np.stack(out)
np.save("$OUT_DIR/sift_base.npy", arr)
print(f"Wrote {arr.shape} -> $OUT_DIR/sift_base.npy")
EOF
fi

echo "Demo dataset ready: $OUT_DIR/sift_base.npy"
echo "Set SIFT_NPY=$OUT_DIR/sift_base.npy and run ./demo.sh"
