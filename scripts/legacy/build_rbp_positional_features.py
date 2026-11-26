#!/usr/bin/env python
"""Build RBP-by-distance-bin positional features for RMBase11 sites.

The output is a site-level TSV plus a JSON sidecar.  Each candidate site is
represented by an RBP x distance-bin x feature tensor flattened into columns.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


DEFAULT_BINS = [(-200, -100), (-100, -50), (-50, 0), (0, 50), (50, 100), (100, 200)]
FEATURES = [
    "presence",
    "peak_count_log1p",
    "max_signal_log1p",
    "mean_signal_log1p",
    "summit_closeness_200",
    "overlap_fraction",
    "strand_match_fraction",
]


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_int(value: object, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def nonnegative_signal(value: object) -> float:
    """Return a log-safe peak signal.

    ENCODE narrowPeak files can use negative sentinel values in score-like
    columns.  For this feature builder, signal is an intensity/count proxy, so
    negative or non-finite values are treated as absent signal.
    """

    signal = parse_float(value, 0.0)
    if not math.isfinite(signal) or signal < 0.0:
        return 0.0
    return signal


def bin_label(left: int, right: int) -> str:
    def fmt(x: int) -> str:
        return f"m{abs(x)}" if x < 0 else str(x)

    return f"{fmt(left)}_{fmt(right)}"


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_manifest(manifest: Path) -> List[Dict[str, str]]:
    rows = read_tsv(manifest)
    if not rows:
        raise ValueError(f"Empty RBP manifest: {manifest}")
    return rows


def resolve_peak_path(manifest: Path, path_text: str) -> Optional[Path]:
    """Resolve manifest paths written on either Windows or Linux."""
    normalized = path_text.replace("\\", "/")
    candidates = [
        Path(path_text),
        Path(normalized),
        manifest.parent / path_text,
        manifest.parent / normalized,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_peaks(manifest: Path) -> Tuple[Dict[str, Dict[str, Dict[str, object]]], List[str]]:
    peaks: Dict[str, Dict[str, List[Tuple[int, int, float, int, str]]]] = defaultdict(lambda: defaultdict(list))
    targets: List[str] = []
    for row in load_manifest(manifest):
        target = row.get("target", "").strip()
        path_text = row.get("download_path", "").strip()
        if not target or not path_text:
            continue
        path = resolve_peak_path(manifest, path_text)
        if path is None:
            print(f"skip missing peak file: {path_text}")
            continue
        if target not in targets:
            targets.append(target)
        with open_text(path) as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    continue
                chrom = fields[0]
                start = parse_int(fields[1])
                end = parse_int(fields[2])
                if end <= start:
                    continue
                strand = fields[5] if len(fields) > 5 else "."
                signal = nonnegative_signal(fields[6] if len(fields) > 6 else fields[4])
                peak_offset = parse_int(fields[9], -1) if len(fields) > 9 else -1
                summit = start + peak_offset if peak_offset >= 0 else (start + end) // 2
                peaks[target][chrom].append((start, end, signal, summit, strand))

    indexed: Dict[str, Dict[str, Dict[str, object]]] = {}
    for target, by_chrom in peaks.items():
        indexed[target] = {}
        for chrom, values in by_chrom.items():
            values.sort(key=lambda item: item[0])
            starts = [item[0] for item in values]
            max_len = max((end - start for start, end, *_ in values), default=0)
            indexed[target][chrom] = {"peaks": values, "starts": starts, "max_len": max_len}
    return indexed, targets


def extract_bin_features(
    index: Mapping[str, object],
    center: int,
    site_strand: str,
    rel_left: int,
    rel_right: int,
) -> List[float]:
    peaks = index["peaks"]
    starts = index["starts"]
    max_len = int(index["max_len"])
    assert isinstance(peaks, list) and isinstance(starts, list)
    left = center + rel_left
    right = center + rel_right
    idx = bisect.bisect_left(starts, right)
    count = 0
    signals: List[float] = []
    min_summit_distance = 10**9
    overlap_total = 0
    strand_match = 0
    lower_scan_bound = left - max(1, max_len)
    for j in range(idx - 1, -1, -1):
        start, end, signal, summit, peak_strand = peaks[j]
        if start < lower_scan_bound:
            break
        if end <= left or start >= right:
            continue
        count += 1
        signals.append(nonnegative_signal(signal))
        min_summit_distance = min(min_summit_distance, abs(int(summit) - center))
        overlap_total += max(0, min(end, right) - max(start, left))
        if site_strand in {"+", "-"} and peak_strand == site_strand:
            strand_match += 1
    if count == 0:
        return [0.0] * len(FEATURES)
    max_signal = max(0.0, max(signals))
    mean_signal = max(0.0, sum(signals) / len(signals))
    width = max(1, right - left)
    return [
        1.0,
        math.log1p(count),
        math.log1p(max_signal),
        math.log1p(mean_signal),
        max(0.0, 1.0 - min_summit_distance / 200.0),
        min(1.0, overlap_total / width),
        strand_match / count,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RBP positional feature tensor TSV.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_mmseqs80_strict_splits.tsv")
    parser.add_argument("--manifest", default="rmbase11_full_work/rbp/encode_eclip_file_manifest.tsv")
    parser.add_argument("--out", default="rmbase11_full_work/rbp/rmbase11_rbp_positional_features.tsv")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = Path(args.dataset)
    manifest = Path(args.manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    peak_index, targets = load_peaks(manifest)
    rows = read_tsv(dataset)
    if args.max_rows:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_rows]

    bins = DEFAULT_BINS
    columns = []
    for target in targets:
        for left, right in bins:
            label = bin_label(left, right)
            for feature in FEATURES:
                columns.append(f"rbp_pos__{target}__{label}__{feature}")

    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["site_id"] + columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            chrom = row.get("chr", "")
            start = parse_int(row.get("start", 0))
            end = parse_int(row.get("end", start + 1))
            center = (start + end) // 2
            site_strand = row.get("strand", ".")
            out_row: Dict[str, object] = {"site_id": row["site_id"]}
            for target in targets:
                chrom_index = peak_index.get(target, {}).get(chrom)
                for left, right in bins:
                    label = bin_label(left, right)
                    values = (
                        extract_bin_features(chrom_index, center, site_strand, left, right)
                        if chrom_index is not None
                        else [0.0] * len(FEATURES)
                    )
                    for feature, value in zip(FEATURES, values):
                        out_row[f"rbp_pos__{target}__{label}__{feature}"] = value
            writer.writerow(out_row)
            if idx % 100000 == 0:
                print(f"processed_sites={idx}", flush=True)

    metadata = {
        "dataset": str(dataset),
        "manifest": str(manifest),
        "output": str(out),
        "site_count": len(rows),
        "rbps": targets,
        "bins": [{"label": bin_label(left, right), "left": left, "right": right} for left, right in bins],
        "features": FEATURES,
        "shape": [len(targets), len(bins), len(FEATURES)],
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix(out.suffix + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
