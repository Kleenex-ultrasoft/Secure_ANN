#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 /path/to/aby3" >&2
  exit 1
fi

ABY3_ROOT="$1"
SRC_FILE="$(cd "$(dirname "$0")/../3pc/aby3/src" && pwd)/hnsecw_search_aby3.cpp"
DEST_DIR="$ABY3_ROOT/frontend"
DEST_FILE="$DEST_DIR/hnsecw_search_aby3.cpp"

mkdir -p "$DEST_DIR"
cp "$SRC_FILE" "$DEST_FILE"

CMAKE_FILE="$DEST_DIR/CMakeLists.txt"
python3 - "$CMAKE_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

list_line = 'list(FILTER SRC_FRONTEND EXCLUDE REGEX "hnsecw_search_aby3.cpp")'
if list_line not in text:
    marker = "add_executable(frontend"
    if marker in text:
        text = text.replace(
            marker,
            list_line + "\n\n" + marker,
            1,
        )

if "add_executable(hnsecw_search_aby3" not in text:
    text = text.rstrip() + "\n\nadd_executable(hnsecw_search_aby3 ${CMAKE_SOURCE_DIR}/frontend/hnsecw_search_aby3.cpp)\n"
    text += "target_link_libraries(hnsecw_search_aby3 com-psi)\n"
    text += "target_link_libraries(hnsecw_search_aby3 aby3-ML)\n"
    text += "target_link_libraries(hnsecw_search_aby3 com-psi_Tests)\n"
    text += "target_link_libraries(hnsecw_search_aby3 aby3_Tests)\n"
    text += "target_link_libraries(hnsecw_search_aby3 oc::tests_cryptoTools)\n"

path.write_text(text, encoding="utf-8")
PY
