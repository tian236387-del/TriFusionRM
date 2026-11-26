#!/usr/bin/env python
"""Select RBP peak files by class-specific overlap with RNA modification sites.

This complements the global RBP selector.  The global selector can be dominated
by large classes such as m6A/m5C, while this script ranks RBP targets within
each requested modification class and writes the union of the selected targets.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
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


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


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


def class_label(row: Mapping[str, str]) -> str:
    for key in ("mod_type", "label", "class", "modification"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def load_site_windows(dataset: Path, window: int) -> Tuple[Dict[str, Dict[str, object]], List[str], Dict[str, int]]:
    by_chrom: Dict[str, List[SiteInterval]] = defaultdict(list)
    labels: List[str] = []
    class_counts: Dict[str, int] = defaultdict(int)
    with dataset.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for idx, row in enumerate(reader):
            chrom = canonical_chrom(row.get("chr", ""))
            label = class_label(row)
            labels.append(label)
            if label:
                class_counts[label] += 1
            if not chrom:
                continue
            center = site_center(row)
            by_chrom[chrom].append((max(0, center - window), center + window + 1, idx))

    indexed: Dict[str, Dict[str, object]] = {}
    for chrom, intervals in by_chrom.items():
        intervals.sort(key=lambda item: item[0])
        starts = [item[0] for item in intervals]
        max_len = max((end - start for start, end, _ in intervals), default=1)
        indexed[chrom] = {"intervals": intervals, "starts": starts, "max_len": max_len}
    return indexed, labels, dict(class_counts)


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


def count_overlaps_by_class(
    target_peaks: Mapping[str, Sequence[Interval]],
    site_index: Mapping[str, Mapping[str, object]],
    labels: Sequence[str],
) -> Tuple[int, Dict[str, int]]:
    seen = bytearray(len(labels))
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

    class_hits: Dict[str, int] = defaultdict(int)
    total_hits = 0
    for idx, flag in enumerate(seen):
        if not flag:
            continue
        total_hits += 1
        label = labels[idx]
        if label:
            class_hits[label] += 1
    return total_hits, dict(class_hits)


def load_targets_from_manifest(path: Path) -> List[str]:
    if not path.exists():
        return []
    return sorted({row.get("target", "").strip() for row in read_tsv(path) if row.get("target", "").strip()})


def score_row(row: Mapping[str, object], rank_by: str) -> Tuple[float, int]:
    overlap = int(row["class_overlapping_sites"])
    enrichment = float(row["enrichment"])
    fraction = float(row["class_overlap_fraction"])
    if rank_by == "overlap":
        score = float(overlap)
    elif rank_by == "fraction":
        score = fraction
    elif rank_by == "enrichment":
        score = enrichment
    else:
        score = math.log1p(overlap) * enrichment
    return score, overlap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select class-specific top RBP targets by overlap with RNA modification classes.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv")
    parser.add_argument("--manifest", default="rmbase11_full_work/rbp_encode_eclip_all/encode_eclip_file_manifest.tsv")
    parser.add_argument("--base-manifest", default="", help="Optional existing manifest whose targets are always retained.")
    parser.add_argument("--out-manifest", default="rmbase11_full_work/rbp_encode_eclip_all/encode_eclip_class_specific_manifest.tsv")
    parser.add_argument("--out-coverage", default="rmbase11_full_work/rbp_encode_eclip_all/encode_eclip_class_specific_overlap.tsv")
    parser.add_argument("--classes", default="Gm,m1A,m7G", help="Comma-separated classes to optimize; use empty for all classes.")
    parser.add_argument("--top-k-per-class", type=int, default=10)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--min-class-overlap-sites", type=int, default=1)
    parser.add_argument("--rank-by", choices=["enrichment_log_count", "enrichment", "fraction", "overlap"], default="enrichment_log_count")
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

    site_index, labels, class_counts = load_site_windows(dataset, args.window)
    site_count = len(labels)
    requested_classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    if not requested_classes:
        requested_classes = sorted(class_counts)

    coverage_rows: List[Dict[str, object]] = []
    for idx, target in enumerate(sorted(by_target), start=1):
        target_rows = by_target[target]
        peaks, raw_count = load_target_peaks(manifest, target_rows)
        merged_count = sum(len(items) for items in peaks.values())
        total_hits, class_hits = count_overlaps_by_class(peaks, site_index, labels)
        total_fraction = total_hits / max(site_count, 1)
        for mod_type in requested_classes:
            class_count = class_counts.get(mod_type, 0)
            class_overlap = class_hits.get(mod_type, 0)
            class_fraction = class_overlap / max(class_count, 1)
            enrichment = class_fraction / max(total_fraction, 1e-12)
            row = {
                "target": target,
                "mod_type": mod_type,
                "files": len(target_rows),
                "raw_peaks": raw_count,
                "merged_peaks": merged_count,
                "site_count": site_count,
                "total_overlapping_sites": total_hits,
                "total_overlap_fraction": total_fraction,
                "class_support": class_count,
                "class_overlapping_sites": class_overlap,
                "class_overlap_fraction": class_fraction,
                "enrichment": enrichment,
            }
            row["score"] = score_row(row, args.rank_by)[0]
            coverage_rows.append(row)
        if idx % 25 == 0:
            print(f"processed_targets={idx} current={target}", flush=True)

    selected_targets = set(load_targets_from_manifest(Path(args.base_manifest)) if args.base_manifest else [])
    selected_by_class: Dict[str, List[str]] = {}
    for mod_type in requested_classes:
        class_rows = [
            row
            for row in coverage_rows
            if row["mod_type"] == mod_type and int(row["class_overlapping_sites"]) >= args.min_class_overlap_sites
        ]
        class_rows.sort(key=lambda row: (score_row(row, args.rank_by)[0], score_row(row, args.rank_by)[1], int(row["merged_peaks"])), reverse=True)
        top_targets = [str(row["target"]) for row in class_rows[: args.top_k_per_class]]
        selected_by_class[mod_type] = top_targets
        selected_targets.update(top_targets)

    coverage_rows.sort(
        key=lambda row: (str(row["mod_type"]), -float(row["score"]), -int(row["class_overlapping_sites"]), str(row["target"]))
    )
    selected_rows = [row for row in rows if row.get("target", "").strip() in selected_targets]

    if rows:
        write_tsv(Path(args.out_manifest), list(rows[0].keys()), selected_rows)
    write_tsv(
        Path(args.out_coverage),
        [
            "mod_type",
            "target",
            "files",
            "raw_peaks",
            "merged_peaks",
            "site_count",
            "total_overlapping_sites",
            "total_overlap_fraction",
            "class_support",
            "class_overlapping_sites",
            "class_overlap_fraction",
            "enrichment",
            "score",
        ],
        coverage_rows,
    )
    summary = {
        "dataset": str(dataset),
        "manifest": str(manifest),
        "base_manifest": args.base_manifest,
        "out_manifest": args.out_manifest,
        "out_coverage": args.out_coverage,
        "site_count": site_count,
        "window": args.window,
        "classes": requested_classes,
        "top_k_per_class": args.top_k_per_class,
        "rank_by": args.rank_by,
        "selected_targets": sorted(selected_targets),
        "selected_target_count": len(selected_targets),
        "selected_by_class": selected_by_class,
    }
    Path(args.out_manifest).with_suffix(Path(args.out_manifest).suffix + ".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
