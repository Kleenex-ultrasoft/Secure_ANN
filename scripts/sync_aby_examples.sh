#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 /path/to/ABY" >&2
  exit 1
fi

ABY_ROOT="$1"
SRC_DIR="$(cd "$(dirname "$0")/../2pc/src/hnsecw" && pwd)"
DEST_DIR="$ABY_ROOT/src/examples/hnsecw"
mkdir -p "$DEST_DIR"
rsync -a --delete "$SRC_DIR/" "$DEST_DIR/"
