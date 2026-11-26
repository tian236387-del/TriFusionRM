#!/usr/bin/env python
"""Select RBP peak files whose binding sites overlap candidate RNA-mod sites.

This script is intended as a practical bridge between a comprehensive RBP peak
source, such as ENCODE eCLIP, and the dense RBP positional tensor used by
``build_rbp_positional_features.py``.  It ranks RBP targets by the number of
RMBase sites that overlap any peak within a centered genomic window, then writes
a filtered manifest containing the top targets.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Interval = Tuple[int, int]
SiteInterval = Tuple[int, int, int]


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
        if value in {"", None}:
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def resolve_peak_path(manifest: Path, path_text: str) -> Optional[Path]:
    normalized = str(path_text or "").replace("\\", "/")
    candidates = [
        Path(path_text),
        Path(normalized),
        manifest.parent / path_text,
        manifest.parent / normalized,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_bed_intervals(path: Path) -> Iterable[Tuple[str, int, int]]:
    with open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
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
    merged = [tuple(sorted(intervals)[0])]
    for start, end in sorted(intervals)[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def site_center(row: Mapping[str, str]) -> int:
    start = parse_int(row.get("start", 0))
    end = parse_int(row.get("end", start + 1))
    if end <= start:
        end = start + 1
    return (start + end) // 2


def load_site_windows(
    dataset: Path,
    window: int,
    split_column: str = "",
    split_value: str = "",
) -> Tuple[Dict[str, Dict[str, object]], int]:
    by_chrom: Dict[str, List[SiteInterval]] = defaultdict(list)
    total = 0
    with dataset.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if split_column and split_column not in (reader.fieldnames or []):
            raise ValueError(f"Missing split column {split_column!r} in {dataset}")
        for row in reader:
            if split_column and row.get(split_column, "") != split_value:
                continue
            chrom = canonical_chrom(row.get("chr", ""))
            if not chrom:
                continue
            center = site_center(row)
            by_chrom[chrom].append((max(0, center - window), center + window + 1, total))
            total += 1

    indexed: Dict[str, Dict[str, object]] = {}
    for chrom, intervals in by_chrom.items():
        intervals.sort(key=lambda item: item[0])
        starts = [item[0] for item in intervals]
        max_len = max((end - start for start, end, _ in intervals), default=1)
        indexed[chrom] = {"intervals": intervals, "starts": starts, "max_len": max_len}
    return indexed, total


def load_target_peaks(manifest: Path, rows: Sequence[Mapping[str, str]]) -> Tuple[Dict[str, List[Interval]], int]:
    raw: Dict[str, List[Interval]] = defaultdict(list)
    raw_count = 0
    for row in rows:
        peak_path = resolve_peak_path(manifest, str(row.get("download_path", "")))
        if peak_path is None:
            continue
        for chrom, start, end in iter_bed_intervals(peak_path):
            raw[chrom].append((start, end))
            raw_count += 1
    merged = {chrom: merge_intervals(intervals) for chrom, intervals in raw.items()}
    return merged, raw_count


def count_overlapping_sites(target_peaks: Mapping[str, Sequence[Interval]], site_index: Mapping[str, Mapping[str, object]], site_count: int) -> int:
    seen = bytearray(site_count)
    for chrom, peaks in target_peaks.items():
        site_data = site_index.get(chrom)
        if site_data is None:
            continue
        intervals = site_data["intervals"]
        starts = site_data["starts"]
        max_len = int(site_data["max_len"])
        assert isinstance(intervals, list) and isinstance(starts, list)
        for peak_start, peak_end in peaks:
            right = bisect.bisect_left(starts, peak_end)
            lower_scan_bound = peak_start - max_len
            for j in range(right - 1, -1, -1):
                site_start, site_end, site_idx = intervals[j]
                if site_start < lower_scan_bound:
                    break
                if site_end > peak_start:
                    seen[site_idx] = 1
    return int(sum(seen))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select top RBP targets by overlap with RNA modification sites.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv")
    parser.add_argument("--manifest", default="rmbase11_full_work/rbp_encode_eclip_all/encode_eclip_file_manifest.tsv")
    parser.add_argument("--out-manifest", default="rmbase11_full_work/rbp_encode_eclip_all/encode_eclip_top30_manifest.tsv")
    parser.add_argument("--out-coverage", default="rmbase11_full_work/rbp_encode_eclip_all/encode_eclip_target_overlap_200.tsv")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--min-overlap-sites", type=int, default=1)
    parser.add_argument("--split-column", default="")
    parser.add_argument("--split-value", default="")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = Path(args.dataset)
    manifest = Path(args.manifest)
    rows = read_tsv(manifest)
    if not rows:
        raise SystemExit(f"Empty manifest: {manifest}")

    by_target: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        target = row.get("target", "").strip()
        if target:
            by_target[target].append(row)

    site_index, site_count = load_site_windows(dataset, args.window, args.split_column, args.split_value)
    coverage_rows: List[Dict[str, object]] = []
    for idx, target in enumerate(sorted(by_target), start=1):
        target_rows = by_target[target]
        peaks, raw_count = load_target_peaks(manifest, target_rows)
        merged_count = sum(len(items) for items in peaks.values())
        overlap_count = count_overlapping_sites(peaks, site_index, site_count)
        coverage_rows.append(
            {
                "target": target,
                "files": len(target_rows),
                "raw_peaks": raw_count,
                "merged_peaks": merged_count,
                "overlapping_sites": overlap_count,
                "overlap_fraction": overlap_count / max(site_count, 1),
            }
        )
        if idx % 25 == 0:
            print(f"processed_targets={idx} current={target}", flush=True)

    coverage_rows.sort(key=lambda row: (int(row["overlapping_sites"]), int(row["merged_peaks"])), reverse=True)
    selected_targets = [
        str(row["target"])
        for row in coverage_rows
        if int(row["overlapping_sites"]) >= args.min_overlap_sites
    ][: args.top_n]
    selected_set = set(selected_targets)
    selected_rows = [row for row in rows if row.get("target", "").strip() in selected_set]

    if rows:
        write_tsv(Path(args.out_manifest), list(rows[0].keys()), selected_rows)
    write_tsv(
        Path(args.out_coverage),
        ["target", "files", "raw_peaks", "merged_peaks", "overlapping_sites", "overlap_fraction"],
        coverage_rows,
    )
    summary = {
        "dataset": str(dataset),
        "manifest": str(manifest),
        "out_manifest": args.out_manifest,
        "out_coverage": args.out_coverage,
        "site_count": site_count,
        "split_column": args.split_column,
        "split_value": args.split_value,
        "window": args.window,
        "top_n": args.top_n,
        "selected_targets": selected_targets,
    }
    Path(args.out_manifest).with_suffix(Path(args.out_manifest).suffix + ".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
