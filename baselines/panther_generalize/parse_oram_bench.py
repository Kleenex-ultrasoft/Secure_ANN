#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def parse_timer_seconds(text: str) -> float:
    patterns = [
        r"Timer\s+1\s*:\s*([0-9.]+)\s*s",
        r"timer\s+1\s*:\s*([0-9.]+)\s*s",
        r"Timer\s+1\s*:\s*([0-9.]+)\s*ms",
        r"timer\s+1\s*:\s*([0-9.]+)\s*ms",
        r"Time\s*=\s*([0-9.]+)\s*s",
        r"Time\s*=\s*([0-9.]+)\s*seconds",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if not m:
            continue
        val = float(m.group(1))
        if "ms" in pat:
            return val / 1000.0
        return val
    return 0.0


def parse_comm_bytes(text: str) -> float:
    total = 0.0
    # Prefer explicit sent/received bytes.
    for key in ("sent", "received"):
        m = re.search(rf"(?i){key}[^0-9]*([0-9]+)\s*bytes", text)
        if m:
            total += float(m.group(1))
    if total > 0:
        return total

    # Fall back to MB patterns.
    mb_patterns = [
        r"(?i)communication[^0-9]*([0-9.]+)\s*MB",
        r"(?i)comm[^0-9]*([0-9.]+)\s*MB",
        r"(?i)sent[^0-9]*([0-9.]+)\s*MB",
        r"(?i)received[^0-9]*([0-9.]+)\s*MB",
    ]
    for pat in mb_patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1)) * 1024.0 * 1024.0
    return 0.0


def parse_bench_line(text: str):
    m = re.search(r"ORAM_BENCH\s+N=(\d+)\s+entry_words=(\d+)\s+accesses=(\d+)\s+impl=([a-z0-9_]+)", text)
    if not m:
        return None
    return {
        "N": int(m.group(1)),
        "entry_words": int(m.group(2)),
        "accesses": int(m.group(3)),
        "impl": m.group(4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="replicated")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    rows = []

    for log0 in sorted(log_dir.glob("*oram_N*_acc*_0")):
        text0 = log0.read_text(errors="replace")
        meta = parse_bench_line(text0)
        if meta is None:
            continue
        prefix = log0.name[:-1]
        total_comm = 0.0
        for party in range(3):
            path = log_dir / f"{prefix}{party}"
            if not path.exists():
                continue
            total_comm += parse_comm_bytes(path.read_text(errors="replace"))
        time_s = parse_timer_seconds(text0)
        accesses = max(1, meta["accesses"])
        rows.append({
            "backend": args.backend,
            "impl": meta["impl"],
            "N": meta["N"],
            "entry_words": meta["entry_words"],
            "time_per_access_s": time_s / accesses if time_s > 0 else 0.0,
            "comm_per_access_mb": (total_comm / (1024.0 * 1024.0)) / accesses if total_comm > 0 else 0.0,
        })

    if not rows:
        raise SystemExit("no oram bench logs found")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "impl",
                "N",
                "entry_words",
                "time_per_access_s",
                "comm_per_access_mb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
