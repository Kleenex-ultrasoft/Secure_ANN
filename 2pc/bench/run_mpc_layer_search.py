#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def load_meta(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return json.loads(str(data["meta_json"][0]))


def calc_id_bitlen(n_real: int, m: int, t: int) -> int:
    needed = n_real + (m * t) + 1
    return max(1, (needed - 1).bit_length())


def calc_id_bitlen_batch(n_real: int, m: int, t: int, q: int) -> int:
    # Dummy pool includes per-iteration C0 dummies (q) plus neighbor dummies (q * m).
    dummy_pool = t * q + t * q * m
    needed = n_real + dummy_pool + 1
    return max(1, (needed - 1).bit_length())


def ceil_log2(n: int) -> int:
    if n <= 1:
        return 0
    return (n - 1).bit_length()


def bool_cost_seconds(eq_len: int, id_bits: int, rtt_s: float, bw_Bps: float) -> float:
    if eq_len <= 0:
        return 0.0
    rounds = ceil_log2(id_bits) + ceil_log2(eq_len) + 1
    return rounds * rtt_s


def yao_cost_seconds(eq_len: int, id_bits: int, rtt_s: float, bw_Bps: float) -> float:
    if eq_len <= 0:
        return 0.0
    ands = eq_len * max(id_bits - 1, 1) + max(eq_len - 1, 0)
    comm_B = ands * 32.0
    return rtt_s + (comm_B / bw_Bps)


def find_yao_threshold(max_len: int, id_bits: int, rtt_s: float, bw_Bps: float) -> int:
    if max_len <= 0:
        return 0

    def yao_leq_bool(eq_len: int) -> bool:
        return yao_cost_seconds(eq_len, id_bits, rtt_s, bw_Bps) <= bool_cost_seconds(
            eq_len, id_bits, rtt_s, bw_Bps
        )

    if not yao_leq_bool(1):
        return 0
    if yao_leq_bool(max_len):
        return max_len

    lo, hi = 1, max_len
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if yao_leq_bool(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def find_dedup_threshold(
    max_vis: int, cand_len: int, id_bits: int, rtt_s: float, bw_Bps: float
) -> int:
    if max_vis <= 0:
        return 0

    def yao_leq_bool(vis_len: int) -> bool:
        eq_len = vis_len + max(cand_len - 1, 0)
        return yao_cost_seconds(eq_len, id_bits, rtt_s, bw_Bps) <= bool_cost_seconds(
            eq_len, id_bits, rtt_s, bw_Bps
        )

    if not yao_leq_bool(1):
        return 0
    if yao_leq_bool(max_vis):
        return max_vis

    lo, hi = 1, max_vis
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if yao_leq_bool(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def parse_total_online(text: str) -> tuple[float, float]:
    patterns = [
        r"\[Total Online\] latency\(s\)=([0-9.]+) comm\(MB\)=([0-9.]+)",
        r"\[mpc total\] online latency\(s\)=([0-9.]+) comm\(MB\)=([0-9.]+)",
        r"\[mpc total\] latency\(s\)=([0-9.]+) comm\(MB\)=([0-9.]+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            lat_s, comm_mb = matches[-1]
            return float(lat_s), float(comm_mb)
    raise RuntimeError("missing online summary in output")


def read_dummy_id(path: Path, id_bitlen: int) -> int:
    if id_bitlen <= 8:
        dtype = np.uint8
    elif id_bitlen <= 16:
        dtype = np.uint16
    elif id_bitlen <= 32:
        dtype = np.uint32
    else:
        dtype = np.uint64
    arr = np.fromfile(path, dtype=dtype, count=1)
    if arr.size != 1:
        raise RuntimeError(f"invalid dummy_id file: {path}")
    return int(arr[0])

def read_entry_list(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise RuntimeError(f"entry file size is not a multiple of 4 bytes: {path}")
    if not data:
        return []
    return list(np.frombuffer(data, dtype=np.uint32))


def write_entry_list(path: Path, entries: list[int]) -> None:
    arr = np.array(entries, dtype=np.uint32)
    arr.tofile(path)


def calc_real_limit_single(id_bitlen: int, m: int, t: int) -> int:
    u = 1 << id_bitlen
    dummy_pool = t * m
    return max(1, u - 1 - dummy_pool)


def calc_real_limit_batch(id_bitlen: int, m: int, t: int, q: int) -> int:
    u = 1 << id_bitlen
    k = q * m
    dummy_pool = t * q + t * k
    return max(1, u - 1 - dummy_pool)


def run_layer(
    layer_idx: int,
    aby_bin: Path,
    addr: str,
    port: int,
    m: int,
    d: int,
    lc: int,
    lw: int,
    bitlen: int,
    id_bitlen: int,
    mode: str,
    protocol: str,
    num_queries: int | None,
    yao_eq_thresh: int | None,
    vg_yao_thresh: int | None,
    vd_yao_thresh: int | None,
    yao_dedup_thresh: int | None,
    dedup_algo: str | None,
    force_dedup_yao: int | None,
    dummy_id: int | None,
    entry_file: Path | None,
    entry_out: Path | None,
    debug_tag: int,
    threads: int,
    sleep_s: float,
    log_dir: Path,
    dry_run: bool,
) -> tuple[float, float]:
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / f"layer{layer_idx}_server.log"
    client_log = log_dir / f"layer{layer_idx}_client.log"

    base_cmd = [
        str(aby_bin),
        "-m", str(m),
        "-d", str(d),
        "-c", str(lc),
        "-w", str(lw),
        "-b", str(bitlen),
        "-i", str(id_bitlen),
        "-X", mode,
        "-P", protocol,
        "-g", str(debug_tag),
        "-a", addr,
        "-p", str(port),
        "-t", str(threads),
    ]
    if num_queries is not None:
        base_cmd += ["-q", str(num_queries)]
    if yao_eq_thresh is not None:
        base_cmd += ["-y", str(yao_eq_thresh)]
    if vg_yao_thresh is not None:
        base_cmd += ["-G", str(vg_yao_thresh)]
    if vd_yao_thresh is not None:
        base_cmd += ["-V", str(vd_yao_thresh)]
    if yao_dedup_thresh is not None:
        base_cmd += ["-Y", str(yao_dedup_thresh)]
    if dedup_algo is not None:
        base_cmd += ["-A", str(dedup_algo)]
    if force_dedup_yao is not None:
        base_cmd += ["-F", str(force_dedup_yao)]
    if dummy_id is not None:
        base_cmd += ["-U", str(dummy_id)]
    if entry_file is not None:
        base_cmd += ["-f", str(entry_file)]
    if entry_out is not None:
        base_cmd += ["-o", str(entry_out)]
    server_cmd = base_cmd + ["-r", "0"]
    client_cmd = base_cmd + ["-r", "1"]

    if dry_run:
        print(f"[dry-run] layer {layer_idx} server: {' '.join(server_cmd)}")
        print(f"[dry-run] layer {layer_idx} client: {' '.join(client_cmd)}")
        return 0.0, 0.0

    # Inherit env, but override HNSECW_DATA_FILE per layer if the
    # caller pre-built per-layer data files.  Convention:
    # `HNSECW_DATA_DIR/<layer_idx>.bin`.  When unset, the cpp falls
    # back to its synthetic-microbench mode (legacy behaviour).
    run_env = dict(os.environ)
    data_dir = os.environ.get("HNSECW_DATA_DIR", "")
    if data_dir:
        per_layer = Path(data_dir) / f"layer{layer_idx}.bin"
        if per_layer.exists():
            run_env["HNSECW_DATA_FILE"] = str(per_layer)
        else:
            run_env.pop("HNSECW_DATA_FILE", None)

    with server_log.open("w") as s_log, client_log.open("w") as c_log:
        s_proc = subprocess.Popen(server_cmd, stdout=s_log, stderr=subprocess.STDOUT, env=run_env)
        time.sleep(sleep_s)
        c_proc = subprocess.Popen(client_cmd, stdout=c_log, stderr=subprocess.STDOUT, env=run_env)
        s_rc = s_proc.wait()
        c_rc = c_proc.wait()

    if s_rc != 0 or c_rc != 0:
        raise RuntimeError(f"layer {layer_idx} failed: server={s_rc} client={c_rc}")

    s_out = server_log.read_text()
    c_out = client_log.read_text()
    try:
        s_lat, s_comm = parse_total_online(s_out)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc} (server log: {server_log})") from exc
    try:
        c_lat, c_comm = parse_total_online(c_out)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc} (client log: {client_log})") from exc
    return max(s_lat, c_lat), max(s_comm, c_comm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="NPZ file from build_hnsw_plain_inputs.py")
    repo_root = Path(__file__).resolve().parents[3]
    default_bin = repo_root / "ABY" / "build" / "bin" / "hnsecw_cli"
    ap.add_argument("--aby_bin", default=str(default_bin))
    ap.add_argument("--addr", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7766, help="base port (each layer uses +2)")
    ap.add_argument("--bitlen", type=int, default=8)
    ap.add_argument("--id_bitlen", type=int, default=0, help="0=auto per layer")
    ap.add_argument("--debug_tag", type=int, default=0)
    ap.add_argument("--mode", default="single", choices=["single", "multi", "batch"])
    ap.add_argument("--protocol", default="b2y", choices=["b2y", "b2a", "dyn"])
    ap.add_argument("--auto_thresh", action="store_true",
                    help="Auto-select GC thresholds from RTT/BW and layer scale")
    ap.add_argument("--ef-override", type=int, default=None,
                    help="Override ef_base from NPZ meta (useful for ablation)")
    ap.add_argument("--tau-override", type=int, default=None,
                    help="Override tau_base from NPZ meta")
    ap.add_argument("--no-depth-extra", action="store_true",
                    help="Drop the ceil(log2(N*d/M)) term at the base layer "
                         "and use lc = ef + tau directly.  The paper formula "
                         "is the default; this flag is an escape hatch for "
                         "configurations where the resulting T exceeds what "
                         "the underlying ABY framework can sustain.")
    ap.add_argument("--rtt_ms", type=float, default=50.0,
                    help="RTT in ms for auto threshold selection")
    ap.add_argument("--bw_mbps", type=float, default=320.0,
                    help="Bandwidth in Mbps for auto threshold selection")
    ap.add_argument("--yao_eq_thresh", type=int, default=None,
                    help="Use Yao EQ when visited_len <= threshold (multi/dyn)")
    ap.add_argument("--num_queries", type=int, default=None,
                    help="Number of queries Q (multi/batch)")
    ap.add_argument("--entry_id", type=int, default=None,
                    help="Override entry ID (local ID at top layer, applies to all queries)")
    ap.add_argument("--entry_file", default=None,
                    help="Binary entry ID list (uint32), one per query, for top layer")
    ap.add_argument("--entry_out", default=None,
                    help="Write final per-query entry IDs after layer 0 (uint32 list)")
    ap.add_argument("--vg_yao_thresh", type=int, default=None,
                    help="Use Yao EQ for graph cache when hist_len_g <= threshold (multi/dyn)")
    ap.add_argument("--vd_yao_thresh", type=int, default=None,
                    help="Use Yao EQ for vector cache when hist_len_v <= threshold (multi/dyn)")
    ap.add_argument("--yao_dedup_thresh", type=int, default=None,
                    help="Use Yao dedup threshold (batch mode)")
    ap.add_argument("--dedup_algo", default="bitonic",
                    choices=["bitonic", "radix"],
                    help="Batch dedup algorithm (bitonic or radix)")
    ap.add_argument("--dedup_force_yao", type=int, default=0,
                    help="Force Yao for de-dup (0/1, batch mode)")
    ap.add_argument("--dummy_id", type=int, default=None,
                    help="Dummy ID override (applied to all layers)")
    ap.add_argument("--dummy_dir", default=None,
                    help="Directory with layerX_dummy_id.bin outputs")
    ap.add_argument("--max_layer", type=int, default=-1, help="-1=all layers")
    ap.add_argument("--sleep", type=float, default=0.25, help="delay before client starts")
    ap.add_argument("--log_dir", default="mpc_logs")
    ap.add_argument("--threads", type=int, default=2,
                    help="ABY worker threads.  Defaults to 2 because the IKNP "
                         "OT extension at higher thread counts (>= 4) "
                         "produces a non-trivial fraction of incorrect "
                         "Beaver triples; the cpp retry wrapper still handles "
                         "the residual rate but its overhead grows quickly.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    npz_path = Path(args.npz).resolve()
    if not npz_path.exists():
        raise RuntimeError(f"missing npz: {npz_path}")

    aby_bin = Path(args.aby_bin).resolve()
    if not aby_bin.exists():
        raise RuntimeError(f"missing ABY binary: {aby_bin}")

    npz_data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(npz_data["meta_json"][0]))
    d = int(meta["d"])
    deg0 = int(meta["deg0_target"])
    degu = int(meta["degU_target"])
    ef_base = int(meta["ef_base"])
    tau_base = int(meta["tau_base"])
    if args.ef_override is not None:
        ef_base = args.ef_override
        print(f"[run_mpc] ef_base overridden to {ef_base}")
    if args.tau_override is not None:
        tau_base = args.tau_override
        print(f"[run_mpc] tau_base overridden to {tau_base}")
    tau_upper = int(meta["tau_upper"])
    L = int(meta["L"])

    max_layer = args.max_layer if args.max_layer >= 0 else (L - 1)
    if max_layer >= L:
        raise RuntimeError(f"max_layer {max_layer} exceeds L={L}")

    log_dir = Path(args.log_dir).resolve()

    total_lat_s = 0.0
    total_comm_mb = 0.0
    rtt_s = args.rtt_ms / 1000.0
    bw_Bps = (args.bw_mbps * 1_000_000.0) / 8.0

    q = args.num_queries or 1

    def layer_info(layer_idx: int) -> dict:
        n_l = int(meta.get(f"layer_size_{layer_idx}", meta["n_base"]))
        if layer_idx == 0:
            m_slot = deg0
            ef_l = ef_base
            tau_l = tau_base
            # Paper, Sec. 4: base-layer search depth is O(log(N * d / M)),
            # while non-base layers are O(1).  We add the graph-depth term
            # to the per-layer iteration budget so that the search has
            # enough rounds to converge on dense base-layer graphs.
            if args.no_depth_extra:
                depth_extra = 0
            else:
                depth_extra = max(1, int(math.ceil(math.log2(max(2.0, n_l * d / max(1, m_slot))))))
        else:
            m_slot = degu
            ef_l = 1
            tau_l = tau_upper
            depth_extra = 0
        lc = ef_l + tau_l + depth_extra
        lw = ef_l
        m = min(m_slot, max(1, n_l - 1))
        if args.id_bitlen:
            id_bitlen = args.id_bitlen
        elif args.num_queries and args.mode == "batch":
            id_bitlen = calc_id_bitlen_batch(n_l, m, lc, q)
        else:
            id_bitlen = calc_id_bitlen(n_l, m, lc)
        return {
            "m_slot": m_slot,
            "m": m,
            "ef": ef_l,
            "tau": tau_l,
            "lc": lc,
            "lw": lw,
            "n_l": n_l,
            "id_bitlen": id_bitlen,
        }

    def layer_dummy_base(layer_idx: int, info: dict) -> int:
        if args.mode == "batch":
            real_limit = calc_real_limit_batch(info["id_bitlen"], info["m"], info["lc"], q)
        else:
            real_limit = calc_real_limit_single(info["id_bitlen"], info["m"], info["lc"])
        return real_limit

    provided_entry = args.entry_file is not None or args.entry_id is not None
    entry_ids: list[int] = []
    if args.entry_file is not None:
        entry_ids = read_entry_list(Path(args.entry_file))
        if entry_ids and len(entry_ids) != q:
            raise RuntimeError("entry_file count does not match num_queries")
    elif args.entry_id is not None:
        entry_ids = [args.entry_id] * q
    else:
        if "entry_point_top_local" not in npz_data:
            raise RuntimeError("missing entry_point_top_local in npz")
        entry_ids = [int(npz_data["entry_point_top_local"][0])] * q
    if not entry_ids:
        raise RuntimeError("entry_ids are empty")

    # Pre-loop down-mapping when --max_layer < L-1.  When an upstream entry
    # falls outside the layer's id range we used to fall back to
    # layer_dummy_base (a sentinel >= N), which the cpp then treats as a
    # dummy entry — wrecking the seed for the real-data search.  Snap to
    # a known-real layer-(l-1) id instead (down_map[0]) so the cpp always
    # starts from a valid graph node.
    if not provided_entry and max_layer < (L - 1):
        for l in range(L - 1, max_layer, -1):
            down_key = f"down_{l}"
            if down_key not in npz_data:
                raise RuntimeError(f"missing {down_key} in npz")
            down_map = npz_data[down_key]
            max_idx = int(down_map.shape[0])
            real_fallback = int(down_map[0]) if max_idx > 0 else 0
            entry_ids = [
                int(down_map[e]) if 0 <= e < max_idx else real_fallback
                for e in entry_ids
            ]

    entry_dir = log_dir / "entry_chain"
    entry_dir.mkdir(parents=True, exist_ok=True)

    for l in range(max_layer, -1, -1):
        info = layer_info(l)
        m_slot = info["m_slot"]
        m = info["m"]
        ef_l = info["ef"]
        tau_l = info["tau"]
        lc = info["lc"]
        lw = info["lw"]
        n_l = info["n_l"]
        id_bitlen = info["id_bitlen"]
        port = args.port + (2 * l)
        max_vis = lc * m * q

        yao_eq_thresh = args.yao_eq_thresh
        vg_yao_thresh = args.vg_yao_thresh
        vd_yao_thresh = args.vd_yao_thresh
        yao_dedup_thresh = args.yao_dedup_thresh
        dedup_algo = args.dedup_algo if args.mode == "batch" else None
        force_dedup_yao = args.dedup_force_yao if args.mode == "batch" else None
        dummy_id = args.dummy_id

        if args.dummy_dir is not None:
            dummy_path = Path(args.dummy_dir) / f"layer{l}_dummy_id.bin"
            if not dummy_path.exists():
                raise RuntimeError(f"missing dummy_id file: {dummy_path}")
            dummy_id = read_dummy_id(dummy_path, id_bitlen)

        auto_thresh = args.auto_thresh or (
            (args.protocol == "dyn" and (
                args.yao_eq_thresh is None
                or args.vg_yao_thresh is None
                or args.vd_yao_thresh is None
            ))
            or (args.mode == "batch" and args.yao_dedup_thresh is None)
        )

        if auto_thresh:
            if args.protocol == "dyn":
                if yao_eq_thresh is None:
                    yao_eq_thresh = find_yao_threshold(max_vis, id_bitlen, rtt_s, bw_Bps)
                if vg_yao_thresh is None:
                    max_vg = q * lc
                    vg_yao_thresh = find_yao_threshold(max_vg, id_bitlen, rtt_s, bw_Bps)
                if vd_yao_thresh is None:
                    max_vd = q * lc * m
                    vd_yao_thresh = find_yao_threshold(max_vd, id_bitlen, rtt_s, bw_Bps)
            if args.mode == "batch" and yao_dedup_thresh is None:
                cand_c0 = q
                cand_neigh = q * m
                thr_c0 = find_dedup_threshold(max_vis, cand_c0, id_bitlen, rtt_s, bw_Bps)
                thr_neigh = find_dedup_threshold(max_vis, cand_neigh, id_bitlen, rtt_s, bw_Bps)
                yao_dedup_thresh = min(thr_c0, thr_neigh)

        if m != m_slot:
            m_note = f"{m} (cap from {m_slot})"
        else:
            m_note = str(m)
        print(
            f"[layer {l}] n={n_l} M={m_note} ef={ef_l} tau={tau_l} "
            f"L_C={lc} L_W={lw} bitlen={args.bitlen} id_bitlen={id_bitlen} port={port}"
        )
        if auto_thresh:
            print(
                f"[layer {l}] auto-thresh eq={yao_eq_thresh} vg={vg_yao_thresh} "
                f"vd={vd_yao_thresh} dedup={yao_dedup_thresh}"
            )
        if args.mode == "batch":
            print(
                f"[layer {l}] dedup algo={dedup_algo} force_yao={force_dedup_yao}"
            )

        entry_file = entry_dir / f"layer{l}_entry.bin"
        entry_out = entry_dir / f"layer{l}_entry_out.bin"
        write_entry_list(entry_file, entry_ids)

        entry_out_ids: list[int] = []

        if n_l <= 1:
            print(f"[layer {l}] singleton layer -> skip MPC search")
            if log_dir.exists():
                for stale in log_dir.glob(f"layer{l}_*.log"):
                    stale.unlink()
            entry_out_ids = list(entry_ids)
        else:
            lat_s, comm_mb = run_layer(
                layer_idx=l,
                aby_bin=aby_bin,
                addr=args.addr,
                port=port,
                m=m,
                d=d,
                lc=lc,
                lw=lw,
                bitlen=args.bitlen,
                id_bitlen=id_bitlen,
                mode=args.mode,
                protocol=args.protocol,
                num_queries=args.num_queries,
                yao_eq_thresh=yao_eq_thresh,
                vg_yao_thresh=vg_yao_thresh,
                vd_yao_thresh=vd_yao_thresh,
                yao_dedup_thresh=yao_dedup_thresh,
                dedup_algo=dedup_algo,
                force_dedup_yao=force_dedup_yao,
                dummy_id=dummy_id,
                entry_file=entry_file,
                entry_out=entry_out,
                debug_tag=args.debug_tag,
                threads=args.threads,
                sleep_s=args.sleep,
                log_dir=log_dir,
                dry_run=args.dry_run,
            )
            total_lat_s += lat_s
            total_comm_mb += comm_mb
            if not args.dry_run:
                print(f"[layer {l}] online latency(s)={lat_s:.6f} comm(MB)={comm_mb:.6f}")
            if args.dry_run:
                entry_out_ids = list(entry_ids)
            else:
                if entry_out.exists():
                    entry_out_ids = read_entry_list(entry_out)
                    if entry_out_ids and len(entry_out_ids) != q:
                        raise RuntimeError("entry_out count does not match num_queries")
                if not entry_out_ids:
                    entry_out_ids = list(entry_ids)

        if l > 0:
            down_key = f"down_{l}"
            if down_key not in npz_data:
                raise RuntimeError(f"missing {down_key} in npz")
            down_map = npz_data[down_key]
            max_idx = int(down_map.shape[0])
            # When the upper layer's entry_out lands outside [0, N_l),
            # snap to a real layer-(l-1) id.  Diversify per query
            # (down_map[(qi * stride) % max_idx]) so all-dummy upper
            # layers don't collapse the lower layers' search to a
            # single seed and produce identical degenerate results
            # for every query.
            stride = max(1, max_idx // max(1, len(entry_out_ids)))
            entry_ids = []
            for qi, e in enumerate(entry_out_ids):
                if 0 <= e < max_idx:
                    entry_ids.append(int(down_map[e]))
                else:
                    fb_idx = (qi * stride) % max_idx
                    entry_ids.append(int(down_map[fb_idx]))
        else:
            entry_ids = entry_out_ids

    if args.entry_out is not None:
        write_entry_list(Path(args.entry_out), entry_ids)

    if not args.dry_run:
        print(f"\n[mpc total] online latency(s)={total_lat_s:.6f} comm(MB)={total_comm_mb:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
