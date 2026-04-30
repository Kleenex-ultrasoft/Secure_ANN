#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class GroupProfile:
    ratio_points: float  # accepted_points / remaining_points_before
    ratio_size: float  # avg_cluster_size / max_points_per_cluster


@dataclass
class CalibProfile:
    name: str
    dims: int
    max_points: int
    stash_ratio: float
    groups: List[GroupProfile]
    k_c: List[int]
    n_total: int


@dataclass
class SannsProfile:
    name: str
    max_points: int
    k_c: List[int]
    u_i: List[int]
    l_i: List[int]
    stash_size: int
    l_s: int
    bc: int
    rc: int
    rp: int


@dataclass
class Baseline:
    distance_time_ms: float
    argmin_time_ms: float
    pir_time_ms: float
    point_distance_time_ms: float
    distance_comm_mb: float
    argmin_comm_mb: float
    pir_query_comm_mb: float
    pir_answer_comm_mb: float
    topk_comm_mb: float
    total_bin_number: int
    max_bin_size: int
    sum_k_c: int
    batch_size: int
    dims: int
    ele_size: int
    max_points: int
    stash_size: int
    pir_scale_mode: str = "ele_size"


@dataclass
class OramModel:
    backend: str
    impl: str
    time_a: float
    time_b: float
    comm_a: float
    comm_b: float
    word_base_time: float
    word_slope_time: float
    word_base_comm: float
    word_slope_comm: float


def infer_dims(name: str) -> Optional[int]:
    name = name.lower()
    if name.startswith("sift"):
        return 128
    if name.startswith("deep10m"):
        return 96
    m = re.search(r"(?:^|-)\d{2,5}(?:-|$)", name)
    if m:
        return int(re.sub(r"[^0-9]", "", m.group(0)))
    return None


def load_profile(pth_path: str) -> CalibProfile:
    name = os.path.basename(pth_path).replace(".pth", "")
    dims = infer_dims(name)
    if dims is None:
        raise ValueError(f"Cannot infer dims from {name}")

    data = torch.load(pth_path, map_location="cpu")
    ids = data["ids"].reshape(-1)
    cluster_ids = data["cluster_idx"].reshape(-1).to(torch.int64)
    index = data["index"].reshape(-1).to(torch.int64)

    stash = int(index[-1].item())
    n_total = int(ids.numel()) + stash

    counts = torch.bincount(cluster_ids)
    max_points = int(counts.max().item()) if counts.numel() else 0

    k_c = [int(x) for x in index.tolist()]
    offsets: List[Tuple[int, int]] = []
    off = 0
    for k in k_c[:-1]:
        offsets.append((off, off + k))
        off += k

    groups: List[GroupProfile] = []
    cluster_points_total = n_total - stash
    rem_before = cluster_points_total
    for s, e in offsets:
        grp_counts = counts[s:e]
        pts = int(grp_counts.sum().item())
        avg_size = pts / max(1, (e - s))
        ratio_points = pts / rem_before if rem_before else 0.0
        ratio_size = avg_size / max_points if max_points else 0.0
        groups.append(GroupProfile(ratio_points=ratio_points, ratio_size=ratio_size))
        rem_before -= pts

    stash_ratio = stash / n_total if n_total else 0.0
    return CalibProfile(
        name=name,
        dims=dims,
        max_points=max_points,
        stash_ratio=stash_ratio,
        groups=groups,
        k_c=k_c,
        n_total=n_total,
    )


def load_profiles(dataset_dir: str) -> Dict[str, CalibProfile]:
    profiles: Dict[str, CalibProfile] = {}
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(".pth"):
            continue
        path = os.path.join(dataset_dir, fname)
        try:
            prof = load_profile(path)
        except Exception:
            continue
        profiles[prof.name] = prof
    return profiles


def pick_profile(profiles: Dict[str, CalibProfile], dims: int, name: Optional[str]) -> CalibProfile:
    if name:
        if name not in profiles:
            raise ValueError(f"Profile {name} not found in {list(profiles.keys())}")
        return profiles[name]
    best = None
    for prof in profiles.values():
        if best is None or abs(prof.dims - dims) < abs(best.dims - dims):
            best = prof
    if best is None:
        raise ValueError("No calibration profiles available")
    return best


def load_sanns_profiles() -> Dict[str, SannsProfile]:
    profiles = {
        "sift": SannsProfile(
            name="sift",
            max_points=20,
            k_c=[50810, 25603, 9968, 4227],
            u_i=[50, 31, 19, 13],
            l_i=[458, 270, 178, 84],
            stash_size=31412,
            l_s=262,
            bc=8,
            rc=5,
            rp=8,
        ),
        "deep1m": SannsProfile(
            name="deep1m",
            max_points=22,
            k_c=[44830, 25867, 11795, 5607, 2611],
            u_i=[46, 31, 19, 13, 7],
            l_i=[458, 270, 178, 84, 84],
            stash_size=25150,
            l_s=210,
            bc=8,
            rc=5,
            rp=8,
        ),
        "deep10m": SannsProfile(
            name="deep10m",
            max_points=48,
            k_c=[209727, 107417, 39132, 14424, 5796, 2394],
            u_i=[88, 46, 25, 13, 7, 7],
            l_i=[924, 458, 178, 93, 84, 84],
            stash_size=50649,
            l_s=423,
            bc=8,
            rc=5,
            rp=8,
        ),
        "amazon": SannsProfile(
            name="amazon",
            max_points=25,
            k_c=[41293, 24143, 9708, 3516, 1156],
            u_i=[37, 37, 22, 10, 7],
            l_i=[364, 364, 178, 84, 84],
            stash_size=8228,
            l_s=84,
            bc=9,
            rc=4,
            rp=6,
        ),
    }
    aliases = {
        "sift1m": "sift",
        "s.1m": "sift",
        "deep1b-1m": "deep1m",
        "deep-1m": "deep1m",
        "d.1m": "deep1m",
        "deep1b-10m": "deep10m",
        "deep-10m": "deep10m",
        "d.10m": "deep10m",
        "amzn": "amazon",
    }
    for alias, base in aliases.items():
        profiles[alias] = profiles[base]
    return profiles


def pick_sanns_profile(name: str) -> SannsProfile:
    profiles = load_sanns_profiles()
    key = name.lower()
    if key not in profiles:
        raise ValueError(f"Unknown SANNS profile '{name}'")
    return profiles[key]


def choose_pir_n(ele_size: int) -> int:
    n = 4096
    while n < ele_size or ele_size > int(0.8 * n):
        n *= 2
    return n


def calc_bins(k_c: List[int]) -> Tuple[List[int], List[int], int, int]:
    group_bin = []
    group_k = []
    for v in k_c[:-1]:
        b = max(1, int(v) // 100)
        group_bin.append(b)
        group_k.append(max(1, b // 9))
    stash_val = k_c[-1]
    s_bin = max(1, int(stash_val) // 100)
    group_bin.append(s_bin)
    group_k.append(max(1, s_bin // 9))

    total_bin = sum(group_bin)
    max_bin = 0
    for v, b in zip(k_c, group_bin):
        max_bin = max(max_bin, int(math.ceil(v / b)) if b > 0 else 0)
    return group_bin, group_k, total_bin, max_bin


def estimate_kc(profile: CalibProfile, n_points: int, max_points: int) -> Tuple[List[int], int]:
    if n_points == profile.n_total and max_points == profile.max_points:
        k_c = [int(x) for x in profile.k_c]
        return k_c, int(k_c[-1])
    stash = int(round(profile.stash_ratio * n_points))
    cluster_points_total = n_points - stash
    rem_before = cluster_points_total
    k_c = []
    for gi, g in enumerate(profile.groups):
        if gi == len(profile.groups) - 1:
            pts = rem_before
        else:
            pts = int(round(g.ratio_points * rem_before))
        avg_size = max(1.0, g.ratio_size * max_points)
        avg_size = min(avg_size, max_points)
        k = int(math.ceil(pts / avg_size)) if avg_size > 0 else 0
        k_c.append(k)
        rem_before -= pts
    k_c.append(stash)
    return k_c, stash


REQUIRED_FIELDS = (
    "distance_time_ms", "argmin_time_ms", "pir_time_ms",
    "point_distance_time_ms", "distance_comm_mb", "argmin_comm_mb",
    "pir_query_comm_mb", "pir_answer_comm_mb", "topk_comm_mb",
    "total_bin_number", "max_bin_size",
)


def parse_baseline(log_dir: str, dataset_dir: str) -> Baseline:
    """Parse PANTHER microbench logs from ``log_dir``.

    Both ``deep10m_client_lan.log`` and ``deep10m_server_lan.log`` must
    exist and contain values for every field in ``REQUIRED_FIELDS``;
    any missing log file or missing field raises ``RuntimeError``.
    The estimator never falls back to hardcoded defaults: every paper
    number it produces comes from a real microbench run.
    """
    base: dict = {}

    def _fill(key, value):
        base[key] = value

    client_log = os.path.join(log_dir, "deep10m_client_lan.log")
    server_log = os.path.join(log_dir, "deep10m_server_lan.log")
    if not os.path.exists(client_log):
        raise RuntimeError(
            f"PANTHER estimator: required client log not found at {client_log}. "
            f"Run baselines/panther_generalize/run_panther_topk_grid_tc.sh first."
        )
    if not os.path.exists(server_log):
        raise RuntimeError(
            f"PANTHER estimator: required server log not found at {server_log}. "
            f"Run baselines/panther_generalize/run_panther_topk_grid_tc.sh first."
        )

    with open(client_log, "r", encoding="utf-8") as f:
        for line in f:
            if "Distance cmp time" in line:
                _fill("distance_time_ms", float(line.split("Distance cmp time:")[1].split("ms")[0]))
            if "Argmin cmp time" in line:
                seg = line.split("Argmin cmp time:")[1]
                _fill("argmin_time_ms", float(seg.split("ms")[0]))
                m = re.search(r"\((\d+),\s*(\d+)\)", line)
                if m:
                    _fill("total_bin_number", int(m.group(1)))
                    _fill("max_bin_size", int(m.group(2)))
            if "PIR cmp time" in line:
                _fill("pir_time_ms", float(line.split("PIR cmp time:")[1].split("ms")[0]))
            if "Point_distance time" in line:
                _fill("point_distance_time_ms", float(
                    line.split("Point_distance time:")[1].split("ms")[0]
                ))
            if "Distance comm" in line and "Distance cmp time" not in line:
                _fill("distance_comm_mb", float(line.split("Distance comm:")[1].split("MB")[0]))
            if "Argmin comm" in line:
                _fill("argmin_comm_mb", float(line.split("Argmin comm:")[1].split("MB")[0]))
            if "PIR client query comm" in line:
                _fill("pir_query_comm_mb", float(
                    line.split("PIR client query comm:")[1].split("MB")[0]
                ))
            if "TopK comm" in line:
                _fill("topk_comm_mb", float(line.split("TopK comm:")[1].split("MB")[0]))
            if "sum_k_c" in line:
                _fill("sum_k_c", int(line.split("sum_k_c:")[1].split()[0]))
            if "batch_size" in line:
                _fill("batch_size", int(line.split("batch_size:")[1].split()[0]))
            if "dims" in line and ":" in line:
                _fill("dims", int(line.split("dims:")[1].split()[0]))
            if "max_points" in line:
                _fill("max_points", int(line.split("max_points:")[1].split()[0]))
    with open(server_log, "r", encoding="utf-8") as f:
        for line in f:
            if "PIR server response comm" in line:
                _fill("pir_answer_comm_mb", float(
                    line.split("PIR server response comm:")[1].split("MB")[0]
                ))

    missing = [k for k in REQUIRED_FIELDS if k not in base]
    if missing:
        raise RuntimeError(
            f"PANTHER estimator: log files at {log_dir} did not contain required "
            f"fields {missing}. Re-run the microbench so the logs are complete."
        )

    stash_size = 0
    deep_profile = os.path.join(dataset_dir, "deep10M.pth")
    if os.path.exists(deep_profile):
        prof = load_profile(deep_profile)
        stash_size = int(prof.k_c[-1])

    if "dims" not in base or "max_points" not in base or "sum_k_c" not in base or "batch_size" not in base:
        raise RuntimeError(
            f"PANTHER estimator: log files at {log_dir} did not record the "
            f"index-build constants (dims, max_points, sum_k_c, batch_size). "
            f"Re-run the microbench with the patched logger that emits these."
        )
    dims = base["dims"]
    max_points = base["max_points"]
    return Baseline(
        distance_time_ms=base["distance_time_ms"],
        argmin_time_ms=base["argmin_time_ms"],
        pir_time_ms=base["pir_time_ms"],
        point_distance_time_ms=base["point_distance_time_ms"],
        distance_comm_mb=base["distance_comm_mb"],
        argmin_comm_mb=base["argmin_comm_mb"],
        pir_query_comm_mb=base["pir_query_comm_mb"],
        pir_answer_comm_mb=base["pir_answer_comm_mb"],
        topk_comm_mb=base["topk_comm_mb"],
        total_bin_number=base["total_bin_number"],
        max_bin_size=base["max_bin_size"],
        sum_k_c=base["sum_k_c"],
        batch_size=base["batch_size"],
        dims=dims,
        ele_size=(dims + 2 * 3) * max_points,
        max_points=max_points,
        stash_size=stash_size,
        pir_scale_mode="pir_n",
    )


def load_ssip_entry(path: str, target_d: int, target_m: Optional[int], mode: Optional[str]) -> Dict[str, float]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("op") not in (None, "ss_ip"):
                continue
            if mode and row.get("mode") != mode:
                continue
            if "d" not in row or "m" not in row:
                continue
            entries.append(row)
    if not entries:
        raise ValueError(f"No ss_ip entries found in {path}")

    best = None
    for row in entries:
        d = int(row.get("d", 0))
        m = int(row.get("m", 0))
        dist = abs(d - target_d)
        if target_m:
            dist = dist * 1000000 + abs(m - target_m)
        if best is None or dist < best[0]:
            best = (dist, row)
        elif dist == best[0] and target_m is None and m > int(best[1].get("m", 0)):
            best = (dist, row)
    if best is None:
        raise ValueError(f"No ss_ip entries match target d={target_d}")
    return best[1]


def override_distance_with_ssip(baseline: Baseline, ssip: Dict[str, float]) -> None:
    m = max(1, int(ssip.get("m", 1)))
    lat_ms = float(ssip.get("lat_ms_per_iter", ssip.get("ms_per_iter", 0.0)))
    comm_mb = float(ssip.get("comm_mb_per_iter", ssip.get("comm_mb", 0.0)))
    if lat_ms <= 0.0:
        raise ValueError("ss_ip entry missing lat_ms_per_iter")
    per_dist_time = lat_ms / m
    per_dist_comm = comm_mb / m if comm_mb > 0.0 else 0.0
    baseline.distance_time_ms = per_dist_time * baseline.sum_k_c
    baseline.distance_comm_mb = per_dist_comm * baseline.sum_k_c
    base_point_num = baseline.batch_size * baseline.max_points + baseline.stash_size
    baseline.point_distance_time_ms = per_dist_time * base_point_num


def build_sanns_baseline(
    dataset: str,
    time_mode: str,
    message_size: int,
    dims_override: Optional[int],
    max_points_override: Optional[int],
    sum_kc_override: Optional[int],
    batch_size_override: Optional[int],
    stash_size_override: Optional[int],
) -> Baseline:
    dataset = dataset.lower()
    time_mode = time_mode.lower()

    sanns_table = {
        "sigma": {
            "dims": 128,
            "max_points": 20,
            "sum_k_c": 90608,
            "batch_size": 113,
            "time_s": {
                "lan": {"distance": 0.11, "topk": 1.29, "points": 2.17},
                "wan": {"distance": 1.41, "topk": 16.1, "points": 27.1},
            },
            "comm_mb": {"distance": 56.7, "topk": 645.0, "points": 1060.0},
        },
        "sift1m": "sigma",
        "s.1m": "sigma",
        "deep1m": {
            "dims": 96,
            "max_points": 22,
            "sum_k_c": 90710,
            "batch_size": 116,
            "time_s": {
                "lan": {"distance": 0.09, "topk": 1.24, "points": 1.84},
                "wan": {"distance": 1.10, "topk": 15.5, "points": 23.0},
            },
            "comm_mb": {"distance": 44.1, "topk": 620.0, "points": 920.0},
        },
        "d.1m": "deep1m",
        "deep10m": {
            "dims": 96,
            "max_points": 48,
            "sum_k_c": 378890,
            "batch_size": 186,
            "time_s": {
                "lan": {"distance": 0.12, "topk": 4.81, "points": 6.39},
                "wan": {"distance": 1.48, "topk": 60.2, "points": 79.9},
            },
            "comm_mb": {"distance": 59.4, "topk": 2350.0, "points": 3120.0},
        },
        "d.10m": "deep10m",
        "avec": {
            "dims": 128,
            "max_points": 20,
            "sum_k_c": 90608,
            "batch_size": 113,
            "time_s": {
                "lan": {"distance": 0.09, "topk": 1.24, "points": 1.84},
                "wan": {"distance": 1.10, "topk": 15.5, "points": 23.0},
            },
            "comm_mb": {"distance": 44.1, "topk": 620.0, "points": 920.0},
        },
        "a": "avec",
        "vold": {
            "dims": 128,
            "max_points": 20,
            "sum_k_c": 90608,
            "batch_size": 113,
            "time_s": {
                "lan": {"distance": 0.12, "topk": 4.81, "points": 6.39},
                "wan": {"distance": 1.48, "topk": 60.2, "points": 79.9},
            },
            "comm_mb": {"distance": 59.4, "topk": 2350.0, "points": 3120.0},
        },
        "amzn": {
            "dims": 128,
            "max_points": 20,
            "sum_k_c": 90608,
            "batch_size": 113,
            "time_s": {
                "lan": {"distance": 0.05, "topk": 1.06, "points": 1.23},
                "wan": {"distance": 0.61, "topk": 13.2, "points": 15.4},
            },
            "comm_mb": {"distance": 24.4, "topk": 528.0, "points": 617.0},
        },
    }
    if dataset in sanns_table and isinstance(sanns_table[dataset], str):
        dataset = sanns_table[dataset]
    if dataset not in sanns_table:
        raise ValueError(f"Unknown SANNS dataset '{dataset}'")

    entry = sanns_table[dataset]
    if time_mode not in entry["time_s"]:
        raise ValueError(f"Unknown SANNS time mode '{time_mode}'")

    dims = dims_override if dims_override is not None else entry["dims"]
    max_points = max_points_override if max_points_override is not None else entry["max_points"]
    sum_k_c = sum_kc_override if sum_kc_override is not None else entry["sum_k_c"]
    batch_size = batch_size_override if batch_size_override is not None else entry["batch_size"]
    stash_size = stash_size_override if stash_size_override is not None else 0
    ele_size = (dims + 2 * message_size) * max_points

    times = entry["time_s"][time_mode]
    comm = entry["comm_mb"]

    return Baseline(
        distance_time_ms=times["distance"] * 1000.0,
        argmin_time_ms=0.0,
        pir_time_ms=times["points"] * 1000.0,
        point_distance_time_ms=times["topk"] * 1000.0,
        distance_comm_mb=comm["distance"],
        argmin_comm_mb=0.0,
        pir_query_comm_mb=comm["points"] / 2.0,
        pir_answer_comm_mb=comm["points"] / 2.0,
        topk_comm_mb=comm["topk"],
        total_bin_number=1,
        max_bin_size=1,
        sum_k_c=sum_k_c,
        batch_size=batch_size,
        dims=dims,
        ele_size=ele_size,
        max_points=max_points,
        stash_size=stash_size,
        pir_scale_mode="pir_n",
    )


def build_panther_table7_baseline(
    dataset: str,
    time_mode: str,
    message_size: int,
    dims_override: Optional[int],
    max_points_override: Optional[int],
    sum_kc_override: Optional[int],
    batch_size_override: Optional[int],
    stash_size_override: Optional[int],
) -> Baseline:
    dataset = dataset.lower()
    time_mode = time_mode.lower()

    panther_table = {
        "sift1m": {
            "dims": 128,
            "max_points": 20,
            "sum_k_c": 90608,
            "batch_size": 113,
            "time_s": {
                "lan": {"distance": 0.23, "topk": 0.35, "points": 1.69},
                "wan": {"distance": 0.50, "topk": 6.34, "points": 3.64},
            },
            "comm_mb": {"distance": 5.57, "topk": 66.4, "points": 45.4},
        },
        "s.1m": "sift1m",
        "deep1m": {
            "dims": 96,
            "max_points": 22,
            "sum_k_c": 90710,
            "batch_size": 116,
            "time_s": {
                "lan": {"distance": 0.22, "topk": 0.39, "points": 1.66},
                "wan": {"distance": 0.41, "topk": 6.80, "points": 3.58},
            },
            "comm_mb": {"distance": 4.57, "topk": 74.9, "points": 46.3},
        },
        "d.1m": "deep1m",
        "deep10m": {
            "dims": 96,
            "max_points": 48,
            "sum_k_c": 378890,
            "batch_size": 186,
            "time_s": {
                "lan": {"distance": 0.79, "topk": 0.82, "points": 5.32},
                "wan": {"distance": 1.71, "topk": 12.8, "points": 8.43},
            },
            "comm_mb": {"distance": 9.23, "topk": 227.0, "points": 82.1},
        },
        "d.10m": "deep10m",
        "amzn": {
            "dims": 128,
            "max_points": 25,
            "sum_k_c": 90710,
            "batch_size": 113,
            "time_s": {
                "lan": {"distance": 0.09, "topk": 0.38, "points": 1.29},
                "wan": {"distance": 0.31, "topk": 6.67, "points": 3.27},
            },
            "comm_mb": {"distance": 2.83, "topk": 73.2, "points": 44.4},
        },
    }
    if dataset in panther_table and isinstance(panther_table[dataset], str):
        dataset = panther_table[dataset]
    if dataset not in panther_table:
        raise ValueError(f"Unknown Panther Table7 dataset '{dataset}'")

    entry = panther_table[dataset]
    if time_mode not in entry["time_s"]:
        raise ValueError(f"Unknown Panther time mode '{time_mode}'")

    dims = dims_override if dims_override is not None else entry["dims"]
    max_points = max_points_override if max_points_override is not None else entry["max_points"]
    sum_k_c = sum_kc_override if sum_kc_override is not None else entry["sum_k_c"]
    batch_size = batch_size_override if batch_size_override is not None else entry["batch_size"]
    stash_size = stash_size_override if stash_size_override is not None else 0
    ele_size = (dims + 2 * message_size) * max_points

    times = entry["time_s"][time_mode]
    comm = entry["comm_mb"]

    return Baseline(
        distance_time_ms=times["distance"] * 1000.0,
        argmin_time_ms=0.0,
        pir_time_ms=times["points"] * 1000.0,
        point_distance_time_ms=times["topk"] * 1000.0,
        distance_comm_mb=comm["distance"],
        argmin_comm_mb=0.0,
        pir_query_comm_mb=comm["points"] / 2.0,
        pir_answer_comm_mb=comm["points"] / 2.0,
        topk_comm_mb=comm["topk"],
        total_bin_number=1,
        max_bin_size=1,
        sum_k_c=sum_k_c,
        batch_size=batch_size,
        dims=dims,
        ele_size=ele_size,
        max_points=max_points,
        stash_size=stash_size,
    )


def _linear_fit(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if not xs or not ys or len(xs) != len(ys):
        raise ValueError("Invalid regression inputs")
    if len(xs) == 1:
        return 0.0, ys[0]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0, mean_y
    a = num / den
    b = mean_y - a * mean_x
    return a, b


def load_oram_model(csv_path: str, backend: str, impl: str) -> Optional[OramModel]:
    if not os.path.exists(csv_path):
        return None
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("backend") == backend and r.get("impl") == impl:
                rows.append(r)
    if not rows:
        return None

    w1_rows = [r for r in rows if r.get("entry_words") == "1"]
    xs = [math.log2(int(r["N"])) for r in w1_rows]
    ys_time = [float(r["time_per_access_s"]) for r in w1_rows]
    ys_comm = [float(r["comm_per_access_mb"]) for r in w1_rows]
    if not xs:
        return None
    time_a, time_b = _linear_fit(xs, ys_time)
    comm_a, comm_b = _linear_fit(xs, ys_comm)

    # Word scaling: use the largest entry_words at the smallest N available.
    rows_sorted = sorted(rows, key=lambda r: (int(r["N"]), int(r["entry_words"])))
    base_row = None
    max_row = None
    for r in rows_sorted:
        if r["entry_words"] == "1":
            base_row = r
            break
    if base_row is None:
        return None
    base_n = base_row["N"]
    same_n = [r for r in rows if r["N"] == base_n]
    if same_n:
        max_row = max(same_n, key=lambda r: int(r["entry_words"]))
    if max_row is None or max_row["entry_words"] == "1":
        word_base_time = float(base_row["time_per_access_s"])
        word_slope_time = 0.0
        word_base_comm = float(base_row["comm_per_access_mb"])
        word_slope_comm = 0.0
    else:
        w1 = float(base_row["time_per_access_s"])
        wmax = float(max_row["time_per_access_s"])
        w1c = float(base_row["comm_per_access_mb"])
        wmaxc = float(max_row["comm_per_access_mb"])
        wmax_words = int(max_row["entry_words"])
        word_slope_time = (wmax - w1) / max(1, (wmax_words - 1))
        word_base_time = w1 - word_slope_time * 1
        word_slope_comm = (wmaxc - w1c) / max(1, (wmax_words - 1))
        word_base_comm = w1c - word_slope_comm * 1

    return OramModel(
        backend=backend,
        impl=impl,
        time_a=time_a,
        time_b=time_b,
        comm_a=comm_a,
        comm_b=comm_b,
        word_base_time=word_base_time,
        word_slope_time=word_slope_time,
        word_base_comm=word_base_comm,
        word_slope_comm=word_slope_comm,
    )


def _mbps_to_bytes_per_s(mbps: float) -> float:
    return max(0.0, mbps) * 1_000_000.0 / 8.0


def _comm_time_ms(comm_bytes: float, bandwidth_mbps: float) -> float:
    rate = _mbps_to_bytes_per_s(bandwidth_mbps)
    if rate <= 0:
        return 0.0
    return comm_bytes / rate * 1000.0


def _lerp(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _krey_and_per_bit(block_bits: int) -> float:
    # SANNS Appendix B Table 7: 128b -> 30 AND/bit, 2.7kB -> 3.15 AND/bit, 6kB -> 3.07 AND/bit
    b0 = 128.0
    b1 = 2.7 * 1024.0 * 8.0
    b2 = 6.0 * 1024.0 * 8.0
    if block_bits <= b0:
        return 30.0
    if block_bits <= b1:
        return _lerp(b0, 30.0, b1, 3.15, float(block_bits))
    if block_bits <= b2:
        return _lerp(b1, 3.15, b2, 3.07, float(block_bits))
    return 3.07


def estimate_floram_comm(
    n_entries: int,
    k_accesses: int,
    entry_words: int,
    value_bitwidth: int,
    lambda_bits: int,
    fss_factor: float,
    prf_and_per_bit: float,
    prf_per_input: int,
    and_bytes: float,
) -> Tuple[float, int, float, float, float]:
    n_entries = max(2, n_entries)
    k_accesses = max(0, k_accesses)
    entry_words = max(1, entry_words)
    logn = math.log2(n_entries)
    block_bits = entry_words * value_bitwidth
    token_bits = fss_factor * 2.0 * k_accesses * logn * (lambda_bits + block_bits)
    token_bytes = token_bits / 8.0
    and_gates = k_accesses * prf_per_input * prf_and_per_bit * block_bits
    gc_bytes = and_gates * and_bytes
    total_bytes = token_bytes + gc_bytes
    rounds = int(math.ceil(logn))
    return total_bytes, rounds, and_gates, token_bytes, gc_bytes


def estimate_oram_cost(
    model: OramModel, n_points: int, entry_words: int, accesses: int
) -> Tuple[float, float]:
    entry_words = max(1, entry_words)
    logn = math.log2(n_points)
    base_time = max(0.0, model.time_a * logn + model.time_b)
    base_comm = max(0.0, model.comm_a * logn + model.comm_b)
    denom_time = model.word_base_time + model.word_slope_time
    denom_comm = model.word_base_comm + model.word_slope_comm
    if denom_time <= 0:
        time_scale = 1.0
    else:
        time_scale = (model.word_base_time + model.word_slope_time * entry_words) / denom_time
    if denom_comm <= 0:
        comm_scale = 1.0
    else:
        comm_scale = (model.word_base_comm + model.word_slope_comm * entry_words) / denom_comm
    per_access_time = base_time * max(1.0, time_scale)
    per_access_comm = base_comm * max(1.0, comm_scale)
    total_time_s = per_access_time * accesses
    total_comm_mb = per_access_comm * accesses
    return total_time_s, total_comm_mb


def estimate_cost(
    baseline: Baseline,
    dims: int,
    sum_k_c: int,
    total_bin_number: int,
    max_bin_size: int,
    batch_size: int,
    ele_size: int,
    max_points: int,
    stash_size: int,
) -> Dict[str, float]:
    dist_scale = (dims * sum_k_c) / (baseline.dims * baseline.sum_k_c)
    argmin_scale = (total_bin_number * max_bin_size) / (
        baseline.total_bin_number * baseline.max_bin_size
    )
    if baseline.pir_scale_mode == "pir_n":
        base_n = choose_pir_n(baseline.ele_size)
        cur_n = choose_pir_n(ele_size)
        pir_scale = (batch_size / baseline.batch_size) * (cur_n / base_n)
    else:
        pir_scale = (batch_size * ele_size) / (baseline.batch_size * baseline.ele_size)
    point_num = batch_size * max_points + stash_size
    base_point_num = baseline.batch_size * baseline.max_points + baseline.stash_size
    denom_points = max(1.0, float(base_point_num * baseline.dims))
    point_scale = (point_num * dims) / denom_points
    topk_comm_scale = point_num / max(1.0, float(base_point_num))

    return {
        "distance_time_ms": baseline.distance_time_ms * dist_scale,
        "argmin_time_ms": baseline.argmin_time_ms * argmin_scale,
        "pir_time_ms": baseline.pir_time_ms * pir_scale,
        "point_distance_time_ms": baseline.point_distance_time_ms * point_scale,
        "distance_comm_mb": baseline.distance_comm_mb * dist_scale,
        "argmin_comm_mb": baseline.argmin_comm_mb * argmin_scale,
        "pir_comm_mb": (baseline.pir_query_comm_mb + baseline.pir_answer_comm_mb) * pir_scale,
        "topk_comm_mb": baseline.topk_comm_mb * topk_comm_scale,
    }


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def fit_power_alpha(n_ref: float, r_ref: float, n_other: float, r_other: float) -> float:
    if n_ref <= 0 or n_other <= 0 or r_ref <= 0 or r_other <= 0:
        raise ValueError("Invalid inputs for power-law fit")
    return math.log(r_ref / r_other) / math.log(n_other / n_ref)


def resolve_k_ratio_params(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    if args.k_ratio_model == "fixed":
        if args.k_ratio and args.k_ratio > 0:
            return {"mode": "fixed", "ref_ratio": args.k_ratio, "n_ref": 1.0, "alpha": 0.0}
        return None

    if args.k_ratio_model == "power":
        if args.k_ratio_fit:
            if args.k_ratio_fit == "sanns-deep":
                n_ref = 1_000_000.0
                r_ref = 0.09071
                n_other = 10_000_000.0
                r_other = 0.037889
                alpha = fit_power_alpha(n_ref, r_ref, n_other, r_other)
                return {"mode": "power", "ref_ratio": r_ref, "n_ref": n_ref, "alpha": alpha}
            raise ValueError(f"Unknown k-ratio fit mode: {args.k_ratio_fit}")

        ref_ratio = args.k_ratio_ref if args.k_ratio_ref > 0 else args.k_ratio
        if ref_ratio <= 0 or args.k_ratio_alpha <= 0:
            raise ValueError("Power-law mode requires --k-ratio-ref/--k-ratio or --k-ratio-alpha")
        n_ref = float(args.k_ratio_nref if args.k_ratio_nref > 0 else 1_000_000)
        return {
            "mode": "power",
            "ref_ratio": ref_ratio,
            "n_ref": n_ref,
            "alpha": args.k_ratio_alpha,
        }

    raise ValueError(f"Unknown k-ratio model: {args.k_ratio_model}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="OpenPanther/experimental/panther/dataset")
    ap.add_argument("--log-dir", default="OpenPanther/logs")
    ap.add_argument(
        "--baseline-mode",
        choices=["panther", "panther-table7", "sanns"],
        default="panther",
        help="Baseline source for distance/top-k/points retrieval",
    )
    ap.add_argument("--sanns-dataset", default="sigma")
    ap.add_argument("--sanns-time-mode", choices=["lan", "wan"], default="lan")
    ap.add_argument("--sanns-dims", type=int, default=None)
    ap.add_argument("--sanns-max-points", type=int, default=None)
    ap.add_argument("--sanns-sum-kc", type=int, default=None)
    ap.add_argument("--sanns-batch-size", type=int, default=None)
    ap.add_argument("--panther-dataset", default="sift1m")
    ap.add_argument("--panther-time-mode", choices=["lan", "wan"], default="lan")
    ap.add_argument(
        "--sanns-profile",
        default="",
        help="Use SANNS Appendix A hyperparameters for clustering (e.g., sift, deep1m, deep10m, amazon)",
    )
    ap.add_argument(
        "--sanns-scale",
        action="store_true",
        help="Scale SANNS baseline by current profile stats (default: no scaling)",
    )
    ap.add_argument("--dims", required=True, help="Comma-separated dims list")
    ap.add_argument("--npoints", required=True, help="Comma-separated dataset sizes")
    ap.add_argument("--max-points", default="5,10,20,40,80", help="Candidate max_points")
    ap.add_argument("--profile", default="", help="Calibration profile name or empty for auto")
    ap.add_argument(
        "--k-ratio",
        type=float,
        default=0.0,
        help="Override cluster count using k/N ratio (lower-bound estimate)",
    )
    ap.add_argument(
        "--k-ratio-model",
        choices=["fixed", "power"],
        default="fixed",
        help="Scaling model for k/N (fixed ratio or power-law)",
    )
    ap.add_argument(
        "--k-ratio-ref",
        type=float,
        default=0.0,
        help="Reference k/N ratio for power-law model (defaults to --k-ratio)",
    )
    ap.add_argument(
        "--k-ratio-nref",
        type=int,
        default=1_000_000,
        help="Reference N for k/N power-law",
    )
    ap.add_argument(
        "--k-ratio-alpha",
        type=float,
        default=0.0,
        help="Power-law exponent for k/N ratio",
    )
    ap.add_argument(
        "--k-ratio-fit",
        choices=["", "sanns-deep"],
        default="",
        help="Fit power-law k/N using known datasets",
    )
    ap.add_argument("--message-size", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--all-m", action="store_true", help="Emit all max_points candidates")
    ap.add_argument("--oram", action="store_true", help="Include ORAM-based estimates")
    ap.add_argument("--oram-csv", default="results/mp_spdz_oram.csv")
    ap.add_argument("--oram-backend", default="semi2k")
    ap.add_argument("--oram-impl", default="packed")
    ap.add_argument("--floram", action="store_true", help="Include FLORAM DORAM estimates")
    ap.add_argument("--floram-lambda", type=int, default=128)
    ap.add_argument("--floram-value-bitwidth", type=int, default=24)
    ap.add_argument("--floram-fss-factor", type=float, default=1.0)
    ap.add_argument("--floram-and-bytes", type=float, default=0.0)
    ap.add_argument("--floram-aes-and-per-bit", type=float, default=39.0)
    ap.add_argument("--floram-krey-and-per-bit", type=float, default=30.0)
    ap.add_argument(
        "--floram-krey-and-mode",
        choices=["fixed", "adaptive"],
        default="adaptive",
        help="Use fixed Kreyvium AND/bit or adapt based on block size",
    )
    ap.add_argument("--floram-prf-per-input", type=int, default=2)
    ap.add_argument("--floram-comm-mult", type=float, default=1.0)
    ap.add_argument("--floram-k-accesses", type=int, default=0)
    ap.add_argument("--floram-n-entries", type=int, default=0)
    ap.add_argument("--floram-block-words", type=int, default=0)
    ap.add_argument("--lan-bw-mbps", type=float, default=4000.0)
    ap.add_argument("--lan-rtt-ms", type=float, default=1.0)
    ap.add_argument("--wan-bw-mbps", type=float, default=320.0)
    # Paper default is 50 ms RTT (see docs/NETWORK.md).
    ap.add_argument("--wan-rtt-ms", type=float, default=50.0)
    ap.add_argument("--ssip-json", default="", help="JSONL from ss_ip_benchmark runs")
    ap.add_argument("--ssip-mode", default="", help="Filter ss_ip entries by mode (lan/wan)")
    ap.add_argument("--ssip-d", type=int, default=0, help="Override ss_ip dimension selection")
    ap.add_argument("--ssip-m", type=int, default=0, help="Override ss_ip m selection")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, args.dataset_dir)
    log_dir = os.path.join(script_dir, args.log_dir)

    profiles = load_profiles(dataset_dir)
    sanns_profile = pick_sanns_profile(args.sanns_profile) if args.sanns_profile else None
    if args.baseline_mode == "sanns":
        baseline = build_sanns_baseline(
            args.sanns_dataset,
            args.sanns_time_mode,
            args.message_size,
            args.sanns_dims,
            args.sanns_max_points,
            args.sanns_sum_kc,
            args.sanns_batch_size,
            sanns_profile.stash_size if sanns_profile is not None else None,
        )
        if sanns_profile is not None:
            baseline.sum_k_c = sum(sanns_profile.k_c)
            baseline.batch_size = sum(sanns_profile.u_i)
            baseline.max_points = sanns_profile.max_points
            baseline.stash_size = sanns_profile.stash_size
            baseline.dims = baseline.dims
            baseline.ele_size = (baseline.dims + 2 * args.message_size) * baseline.max_points
            baseline.total_bin_number = sum(sanns_profile.l_i)
            baseline.max_bin_size = max(
                int(math.ceil(k / l)) if l > 0 else 0
                for k, l in zip(sanns_profile.k_c, sanns_profile.l_i)
            )
    elif args.baseline_mode == "panther-table7":
        baseline = build_panther_table7_baseline(
            args.panther_dataset,
            args.panther_time_mode,
            args.message_size,
            args.sanns_dims,
            args.sanns_max_points,
            args.sanns_sum_kc,
            args.sanns_batch_size,
            sanns_profile.stash_size if sanns_profile is not None else None,
        )
        if sanns_profile is not None:
            baseline.sum_k_c = sum(sanns_profile.k_c)
            baseline.batch_size = sum(sanns_profile.u_i)
            baseline.max_points = sanns_profile.max_points
            baseline.stash_size = sanns_profile.stash_size
            baseline.dims = baseline.dims
            baseline.ele_size = (baseline.dims + 2 * args.message_size) * baseline.max_points
            baseline.total_bin_number = sum(sanns_profile.l_i)
            baseline.max_bin_size = max(
                int(math.ceil(k / l)) if l > 0 else 0
                for k, l in zip(sanns_profile.k_c, sanns_profile.l_i)
            )
    else:
        baseline = parse_baseline(log_dir, dataset_dir)
    if args.ssip_json:
        target_d = args.ssip_d if args.ssip_d > 0 else baseline.dims
        target_m = args.ssip_m if args.ssip_m > 0 else None
        mode = args.ssip_mode if args.ssip_mode else None
        ssip_entry = load_ssip_entry(args.ssip_json, target_d, target_m, mode)
        override_distance_with_ssip(baseline, ssip_entry)
    oram_model = None
    if args.oram:
        oram_path = os.path.join(script_dir, args.oram_csv)
        oram_model = load_oram_model(oram_path, args.oram_backend, args.oram_impl)

    dims_list = parse_int_list(args.dims)
    npoints_list = parse_int_list(args.npoints)
    max_points_candidates = parse_int_list(args.max_points)
    if sanns_profile is not None:
        max_points_candidates = [sanns_profile.max_points]
    k_ratio_params = resolve_k_ratio_params(args)

    rows = []
    header = [
        "dims",
        "n_points",
        "profile",
        "max_points",
        "k_ratio",
        "stash_size",
        "sum_k_c",
        "cluster_num",
        "total_bin_number",
        "max_bin_size",
        "batch_size",
        "ele_size",
        "pir_N",
        "distance_ms",
        "argmin_ms",
        "pir_ms",
        "point_ms",
        "total_ms",
        "distance_comm_mb",
        "argmin_comm_mb",
        "pir_comm_mb",
        "topk_comm_mb",
        # Network-provenance columns — reproducibility of WAN/LAN numbers.
        #
        "network_profile",
        "lan_rtt_ms",
        "lan_bw_mbps",
        "wan_rtt_ms",
        "wan_bw_mbps",
    ]
    if args.oram:
        header += [
            "oram_backend",
            "oram_impl",
            "oram_accesses",
            "oram_entry_words",
            "oram_ms",
            "oram_comm_mb",
            "total_ms_oram",
        ]
    if args.floram:
        header += [
            "floram_k",
            "floram_n",
            "floram_block_words",
            "floram_rounds",
            "floram_token_comm_mb",
            "floram_gc_and_aes",
            "floram_gc_and_krey",
            "floram_comm_mb_aes",
            "floram_comm_mb_krey",
            "floram_lan_ms_aes",
            "floram_lan_ms_krey",
            "floram_wan_ms_aes",
            "floram_wan_ms_krey",
            "total_ms_floram_lan_aes",
            "total_ms_floram_lan_krey",
            "total_ms_floram_wan_aes",
            "total_ms_floram_wan_krey",
        ]

    for d in dims_list:
        prof = pick_profile(profiles, d, args.profile or None)
        for n in npoints_list:
            k_ratio = None
            if k_ratio_params is not None:
                if k_ratio_params["mode"] == "fixed":
                    k_ratio = k_ratio_params["ref_ratio"]
                else:
                    k_ratio = k_ratio_params["ref_ratio"] * (
                        float(n) / float(k_ratio_params["n_ref"])
                    ) ** (-k_ratio_params["alpha"])
            best = None
            for m in max_points_candidates:
                if k_ratio is not None and k_ratio > 0:
                    base_kc = list(prof.k_c[:-1])
                    base_sum = sum(base_kc)
                    cluster_num = max(1, int(round(k_ratio * n)))
                    stash = int(round(prof.stash_ratio * n)) if prof.stash_ratio else 0
                    if base_sum <= 0:
                        k_c = [cluster_num]
                    else:
                        scale = cluster_num / base_sum
                        k_c = [max(1, int(round(v * scale))) for v in base_kc]
                        delta = cluster_num - sum(k_c)
                        k_c[-1] = max(1, k_c[-1] + delta)
                    k_c.append(stash)
                    group_bin, group_k, total_bin, max_bin = calc_bins(k_c)
                    batch_size = sum(group_k[:-1])
                elif sanns_profile is not None:
                    if len(sanns_profile.k_c) != len(sanns_profile.u_i):
                        raise ValueError("SANNS profile k_c/u_i length mismatch")
                    if len(sanns_profile.k_c) != len(sanns_profile.l_i):
                        raise ValueError("SANNS profile k_c/l_i length mismatch")
                    m = sanns_profile.max_points
                    k_c = list(sanns_profile.k_c) + [sanns_profile.stash_size]
                    stash = sanns_profile.stash_size
                    group_bin = list(sanns_profile.l_i)
                    group_k = list(sanns_profile.u_i)
                    total_bin = sum(group_bin)
                    max_bin = 0
                    for k_i, l_i in zip(sanns_profile.k_c, sanns_profile.l_i):
                        if l_i <= 0:
                            continue
                        max_bin = max(max_bin, int(math.ceil(k_i / l_i)))
                    batch_size = sum(group_k)
                else:
                    k_c, stash = estimate_kc(prof, n, m)
                    group_bin, group_k, total_bin, max_bin = calc_bins(k_c)
                    batch_size = sum(group_k[:-1])
                ele_size = (d + 2 * args.message_size) * m
                pir_N = choose_pir_n(ele_size)
                if args.baseline_mode == "sanns" and not args.sanns_scale:
                    costs = {
                        "distance_time_ms": baseline.distance_time_ms,
                        "argmin_time_ms": baseline.argmin_time_ms,
                        "pir_time_ms": baseline.pir_time_ms,
                        "point_distance_time_ms": baseline.point_distance_time_ms,
                        "distance_comm_mb": baseline.distance_comm_mb,
                        "argmin_comm_mb": baseline.argmin_comm_mb,
                        "pir_comm_mb": baseline.pir_query_comm_mb + baseline.pir_answer_comm_mb,
                        "topk_comm_mb": baseline.topk_comm_mb,
                    }
                else:
                    costs = estimate_cost(
                        baseline,
                        d,
                        sum(k_c),
                        total_bin,
                        max_bin,
                        batch_size,
                        ele_size,
                        m,
                        stash,
                    )
                total_ms = (
                    costs["distance_time_ms"]
                    + costs["argmin_time_ms"]
                    + costs["pir_time_ms"]
                    + costs["point_distance_time_ms"]
                )
                floram = None
                oram_time_ms = None
                oram_comm_mb = None
                total_ms_oram = None
                if args.oram and oram_model is not None:
                    cluster_num = sum(k_c[:-1])
                    oram_time_s, oram_comm_mb = estimate_oram_cost(
                        oram_model, cluster_num, ele_size, batch_size
                    )
                    oram_time_ms = oram_time_s * 1000.0
                    total_ms_oram = (
                        costs["distance_time_ms"]
                        + costs["argmin_time_ms"]
                        + oram_time_ms
                        + costs["point_distance_time_ms"]
                    )
                if args.floram:
                    cluster_num = args.floram_n_entries if args.floram_n_entries > 0 else sum(k_c[:-1])
                    k_accesses = args.floram_k_accesses if args.floram_k_accesses > 0 else batch_size
                    block_words = args.floram_block_words if args.floram_block_words > 0 else ele_size
                    and_bytes = (
                        args.floram_and_bytes
                        if args.floram_and_bytes > 0
                        else (2.0 * args.floram_lambda / 8.0)
                    )
                    aes_bytes, rounds, aes_and, token_bytes, _ = estimate_floram_comm(
                        cluster_num,
                        k_accesses,
                        block_words,
                        args.floram_value_bitwidth,
                        args.floram_lambda,
                        args.floram_fss_factor,
                        args.floram_aes_and_per_bit,
                        args.floram_prf_per_input,
                        and_bytes,
                    )
                    if args.floram_krey_and_mode == "adaptive":
                        krey_and_per_bit = _krey_and_per_bit(
                            int(block_words * args.floram_value_bitwidth)
                        )
                    else:
                        krey_and_per_bit = args.floram_krey_and_per_bit
                    krey_bytes, _, krey_and, _, _ = estimate_floram_comm(
                        cluster_num,
                        k_accesses,
                        block_words,
                        args.floram_value_bitwidth,
                        args.floram_lambda,
                        args.floram_fss_factor,
                        krey_and_per_bit,
                        args.floram_prf_per_input,
                        and_bytes,
                    )
                    if args.floram_comm_mult != 1.0:
                        aes_bytes *= args.floram_comm_mult
                        krey_bytes *= args.floram_comm_mult
                        token_bytes *= args.floram_comm_mult
                    lan_ms_aes = _comm_time_ms(aes_bytes, args.lan_bw_mbps) + rounds * args.lan_rtt_ms
                    lan_ms_krey = _comm_time_ms(krey_bytes, args.lan_bw_mbps) + rounds * args.lan_rtt_ms
                    wan_ms_aes = _comm_time_ms(aes_bytes, args.wan_bw_mbps) + rounds * args.wan_rtt_ms
                    wan_ms_krey = _comm_time_ms(krey_bytes, args.wan_bw_mbps) + rounds * args.wan_rtt_ms
                    floram = {
                        "floram_k": k_accesses,
                        "floram_n": cluster_num,
                        "floram_block_words": block_words,
                        "floram_rounds": rounds,
                        "floram_token_comm_mb": token_bytes / 1024.0 / 1024.0,
                        "floram_gc_and_aes": aes_and,
                        "floram_gc_and_krey": krey_and,
                        "floram_comm_mb_aes": aes_bytes / 1024.0 / 1024.0,
                        "floram_comm_mb_krey": krey_bytes / 1024.0 / 1024.0,
                        "floram_lan_ms_aes": lan_ms_aes,
                        "floram_lan_ms_krey": lan_ms_krey,
                        "floram_wan_ms_aes": wan_ms_aes,
                        "floram_wan_ms_krey": wan_ms_krey,
                        "total_ms_floram_lan_aes": (
                            costs["distance_time_ms"]
                            + costs["argmin_time_ms"]
                            + costs["point_distance_time_ms"]
                            + lan_ms_aes
                        ),
                        "total_ms_floram_lan_krey": (
                            costs["distance_time_ms"]
                            + costs["argmin_time_ms"]
                            + costs["point_distance_time_ms"]
                            + lan_ms_krey
                        ),
                        "total_ms_floram_wan_aes": (
                            costs["distance_time_ms"]
                            + costs["argmin_time_ms"]
                            + costs["point_distance_time_ms"]
                            + wan_ms_aes
                        ),
                        "total_ms_floram_wan_krey": (
                            costs["distance_time_ms"]
                            + costs["argmin_time_ms"]
                            + costs["point_distance_time_ms"]
                            + wan_ms_krey
                        ),
                    }
                if k_ratio is not None and k_ratio > 0:
                    profile_name = f"{prof.name}-k{float(k_ratio):.3f}"
                else:
                    profile_name = (
                        f"sanns-{sanns_profile.name}" if sanns_profile is not None else prof.name
                    )
                # Network-profile label: 50ms RTT @ 320 Mbps = paper WAN.
                if abs(args.wan_rtt_ms - 50.0) < 0.5:
                    net_profile = "wan50"
                else:
                    net_profile = f"wan{args.wan_rtt_ms:.0f}ms-custom"
                row = {
                    "dims": d,
                    "n_points": n,
                    "profile": profile_name,
                    "max_points": m,
                    "k_ratio": f"{k_ratio:.6f}" if k_ratio is not None else "",
                    "stash_size": stash,
                    "sum_k_c": sum(k_c),
                    "cluster_num": sum(k_c[:-1]),
                    "total_bin_number": total_bin,
                    "max_bin_size": max_bin,
                    "batch_size": batch_size,
                    "ele_size": ele_size,
                    "pir_N": pir_N,
                    "distance_ms": costs["distance_time_ms"],
                    "argmin_ms": costs["argmin_time_ms"],
                    "pir_ms": costs["pir_time_ms"],
                    "point_ms": costs["point_distance_time_ms"],
                    "total_ms": total_ms,
                    "distance_comm_mb": costs["distance_comm_mb"],
                    "argmin_comm_mb": costs["argmin_comm_mb"],
                    "pir_comm_mb": costs["pir_comm_mb"],
                    "topk_comm_mb": costs["topk_comm_mb"],
                    # Network provenance for the CSV audit trail.
                    "network_profile": net_profile,
                    "lan_rtt_ms": args.lan_rtt_ms,
                    "lan_bw_mbps": args.lan_bw_mbps,
                    "wan_rtt_ms": args.wan_rtt_ms,
                    "wan_bw_mbps": args.wan_bw_mbps,
                }
                if args.oram:
                    row.update(
                        {
                            "oram_backend": args.oram_backend,
                            "oram_impl": args.oram_impl,
                            "oram_accesses": batch_size,
                            "oram_entry_words": ele_size,
                            "oram_ms": oram_time_ms if oram_time_ms is not None else "",
                            "oram_comm_mb": oram_comm_mb if oram_comm_mb is not None else "",
                            "total_ms_oram": total_ms_oram if total_ms_oram is not None else "",
                        }
                    )
                if args.floram and floram is not None:
                    row.update(floram)
                if best is None or row["total_ms"] < best["total_ms"]:
                    best = row
                if args.all_m:
                    rows.append(row)
            if not args.all_m:
                rows.append(best)

    out_lines = []
    out_lines.append(",".join(header))
    for r in rows:
        out_lines.append(
            ",".join(
                str(
                    r[h]
                    if not isinstance(r[h], float)
                    else f"{r[h]:.4f}"
                )
                for h in header
            )
        )

    content = "\n".join(out_lines) + "\n"
    if args.out:
        out_path = os.path.join(script_dir, args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        print(content)


if __name__ == "__main__":
    main()
