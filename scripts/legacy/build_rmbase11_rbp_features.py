#!/usr/bin/env python
"""Build RBP overlap/count/distance features for RMBase11 sites."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Interval = Tuple[int, int]


def canonical_chrom(chrom: str) -> str:
    chrom = str(chrom or "").strip()
    if not chrom:
        return chrom
    if chrom.startswith("chr"):
        return chrom
    upper = chrom.upper()
    if upper in {"X", "Y", "M", "MT"} or upper.isdigit():
        return "chrM" if upper in {"M", "MT"} else f"chr{upper}"
    return chrom


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def iter_bed(path: Path) -> Iterable[Tuple[str, int, int]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:  # type: ignore[arg-type]
        for line in handle:
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            chrom = canonical_chrom(fields[0])
            start = parse_int(fields[1], -1)
            end = parse_int(fields[2], -1)
            if chrom and start >= 0 and end > start:
                yield chrom, start, end


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


class IntervalIndex:
    def __init__(self, intervals: Sequence[Interval]):
        self.intervals = list(intervals)
        self.starts = [item[0] for item in self.intervals]
        self.ends = sorted(item[1] for item in self.intervals)

    def count_overlaps(self, start: int, end: int) -> int:
        if end <= start:
            return 0
        start_before_end = bisect.bisect_left(self.starts, end)
        end_at_or_before_start = bisect.bisect_right(self.ends, start)
        return max(0, start_before_end - end_at_or_before_start)

    def nearest_distance(self, start: int, end: int, cap: int) -> int:
        if self.count_overlaps(start, end) > 0:
            return 0
        idx = bisect.bisect_left(self.starts, end)
        best = cap
        if idx < len(self.intervals):
            best = min(best, max(0, self.intervals[idx][0] - end))
        if idx > 0:
            best = min(best, max(0, start - self.intervals[idx - 1][1]))
        return min(best, cap)


def load_manifest(path: Path, targets: Optional[Sequence[str]]) -> List[Dict[str, str]]:
    rows = read_tsv(path)
    target_filter = {target.strip() for target in targets or [] if target.strip()}
    selected = []
    for row in rows:
        if target_filter and row.get("target") not in target_filter:
            continue
        peak_path = Path(row.get("download_path", ""))
        if not peak_path.exists() or peak_path.stat().st_size == 0:
            continue
        selected.append(row)
    return selected


def load_peak_indices(manifest_rows: Sequence[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, IntervalIndex]], Dict[str, Dict[str, int]]]:
    raw: Dict[str, Dict[str, List[Interval]]] = defaultdict(lambda: defaultdict(list))
    file_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"files": 0, "raw_peaks": 0, "merged_peaks": 0})
    for row in manifest_rows:
        target = str(row["target"])
        peak_path = Path(row["download_path"])
        file_counts[target]["files"] += 1
        for chrom, start, end in iter_bed(peak_path):
            raw[target][chrom].append((start, end))
            file_counts[target]["raw_peaks"] += 1

    indices: Dict[str, Dict[str, IntervalIndex]] = {}
    for target, chrom_map in raw.items():
        indices[target] = {}
        for chrom, intervals in chrom_map.items():
            merged = merge_intervals(intervals)
            file_counts[target]["merged_peaks"] += len(merged)
            indices[target][chrom] = IntervalIndex(merged)
    return indices, file_counts


def make_feature_names(targets: Sequence[str], windows: Sequence[int]) -> List[str]:
    fields: List[str] = []
    for window in windows:
        fields.extend([f"rbp_any_overlap_{window}", f"rbp_unique_targets_{window}", f"rbp_merged_peak_count_{window}"])
    for target in targets:
        safe = target.replace("-", "_")
        fields.extend(
            [
                f"rbp_{safe}_overlap_100",
                f"rbp_{safe}_overlap_200",
                f"rbp_{safe}_nearest_log10_distance",
            ]
        )
    return fields


def site_center(row: Dict[str, str]) -> int:
    start = parse_int(row.get("start", 0))
    end = parse_int(row.get("end", start + 1))
    if end <= start:
        end = start + 1
    return (start + end) // 2


def build_features_for_site(
    row: Dict[str, str],
    indices: Dict[str, Dict[str, IntervalIndex]],
    targets: Sequence[str],
    windows: Sequence[int],
    distance_cap: int,
) -> Dict[str, float]:
    chrom = canonical_chrom(row.get("chr", ""))
    center = site_center(row)
    features: Dict[str, float] = {}

    overlap_by_target_window: Dict[Tuple[str, int], int] = {}
    for window in windows:
        q_start = max(0, center - window)
        q_end = center + window + 1
        unique = 0
        peak_count = 0
        for target in targets:
            idx = indices.get(target, {}).get(chrom)
            count = idx.count_overlaps(q_start, q_end) if idx is not None else 0
            overlap_by_target_window[(target, window)] = count
            if count > 0:
                unique += 1
                peak_count += count
        features[f"rbp_any_overlap_{window}"] = 1.0 if unique > 0 else 0.0
        features[f"rbp_unique_targets_{window}"] = float(unique)
        features[f"rbp_merged_peak_count_{window}"] = float(peak_count)

    point_start = max(0, center)
    point_end = point_start + 1
    for target in targets:
        safe = target.replace("-", "_")
        overlap_100 = overlap_by_target_window.get((target, 100), 0)
        overlap_200 = overlap_by_target_window.get((target, 200), 0)
        idx = indices.get(target, {}).get(chrom)
        distance = idx.nearest_distance(point_start, point_end, distance_cap) if idx is not None else distance_cap
        features[f"rbp_{safe}_overlap_100"] = 1.0 if overlap_100 > 0 else 0.0
        features[f"rbp_{safe}_overlap_200"] = 1.0 if overlap_200 > 0 else 0.0
        features[f"rbp_{safe}_nearest_log10_distance"] = math.log10(float(distance + 1))
    return features


def write_tsv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build RBP peak overlap features for RMBase11 sites.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_mmseqs80_strict_splits.tsv")
    parser.add_argument("--manifest", default="rmbase11_full_work/rbp/encode_eclip_file_manifest.tsv")
    parser.add_argument("--out", default="rmbase11_full_work/rbp/rmbase11_rbp_features.tsv")
    parser.add_argument("--windows", default="50,100,200")
    parser.add_argument("--targets", default="")
    parser.add_argument("--distance-cap", type=int, default=1000000)
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args(argv)

    windows = [int(item) for item in args.windows.split(",") if item.strip()]
    requested_targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    manifest_rows = load_manifest(Path(args.manifest), requested_targets or None)
    if not manifest_rows:
        raise SystemExit(f"No usable peak files found in manifest: {args.manifest}")
    targets = sorted({row["target"] for row in manifest_rows})
    print(f"[RBP] loading {len(manifest_rows)} peak files across {len(targets)} targets", flush=True)
    indices, file_counts = load_peak_indices(manifest_rows)
    feature_names = make_feature_names(targets, windows)

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coverage = {name: 0.0 for name in feature_names if name.endswith("_overlap_100") or name.endswith("_overlap_200")}
    total = 0
    with dataset_path.open("r", encoding="utf-8", newline="") as in_handle, out_path.open("w", encoding="utf-8", newline="") as out_handle:
        reader = csv.DictReader(in_handle, delimiter="\t")
        writer = csv.DictWriter(out_handle, fieldnames=["site_id"] + feature_names, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            feats = build_features_for_site(row, indices, targets, windows, args.distance_cap)
            out_row: Dict[str, object] = {"site_id": row["site_id"]}
            out_row.update({name: f"{feats[name]:.6g}" for name in feature_names})
            writer.writerow(out_row)
            for key in coverage:
                coverage[key] += feats.get(key, 0.0)
            total += 1
            if args.progress_every and total % args.progress_every == 0:
                print(f"[RBP] processed {total:,} sites", flush=True)

    coverage_rows = []
    for key, value in sorted(coverage.items()):
        coverage_rows.append({"feature": key, "overlap_count": int(value), "overlap_fraction": value / max(total, 1)})
    coverage_path = out_path.with_name("rbp_feature_coverage.tsv")
    write_tsv(coverage_path, ["feature", "overlap_count", "overlap_fraction"], coverage_rows)
    summary = {
        "dataset": str(dataset_path),
        "manifest": args.manifest,
        "output": str(out_path),
        "sites": total,
        "targets": targets,
        "windows": windows,
        "distance_cap": args.distance_cap,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "peak_file_counts": file_counts,
        "coverage_table": str(coverage_path),
    }
    summary_path = out_path.with_name("rmbase11_rbp_feature_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
