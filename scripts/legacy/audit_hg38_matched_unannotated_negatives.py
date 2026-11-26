#!/usr/bin/env python
"""Strict independent audit for matched hg38 unannotated candidates."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SPLIT_COLUMNS = [
    "split_cluster201",
    "split_gene_holdout",
    "split_transcript_holdout",
    "split_study_holdout",
    "split_lockbox",
    "is_frozen_lockbox",
]


def gc_fraction(sequence: str) -> float:
    return (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)


def standardized_mean_difference(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return float("nan")
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    left_var = statistics.variance(left) if len(left) > 1 else 0.0
    right_var = statistics.variance(right) if len(right) > 1 else 0.0
    pooled = math.sqrt((left_var + right_var) / 2.0)
    return (left_mean - right_mean) / pooled if pooled > 0 else 0.0


def nearest_distance(values: Sequence[int], position: int) -> int:
    index = bisect.bisect_left(values, position)
    candidates = []
    if index < len(values):
        candidates.append(abs(values[index] - position))
    if index:
        candidates.append(abs(values[index - 1] - position))
    return min(candidates) if candidates else 10**12


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(rows[0]) if rows else ["item", "status", "detail"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    positives: dict[str, dict[str, str]] = {}
    positive_gc: dict[str, list[float]] = defaultdict(list)
    positive_counts: Counter[str] = Counter()
    with args.positives.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            positives[row["site_id"]] = row
            positive_counts[row["mod_type"]] += 1
            positive_gc[row["mod_type"]].append(gc_fraction(row["sequence_201nt"]))
    known_path = args.known_modifications or args.positives
    known: dict[str, list[int]] = defaultdict(list)
    with known_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = {"chr", "start"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Known-modification table is missing fields: {sorted(missing)}")
        for row in reader:
            known[row["chr"]].append(int(row["start"]))
    known = {chrom: sorted(set(values)) for chrom, values in known.items()}

    negative_counts: Counter[str] = Counter()
    match_levels: Counter[str] = Counter()
    negative_gc: dict[str, list[float]] = defaultdict(list)
    paired_gc_differences: dict[str, list[float]] = defaultdict(list)
    invalid_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_coordinates: set[tuple[str, int, str]] = set()
    total = 0
    with args.negatives.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "site_id", "matched_positive_site_id", "matched_mod_type", "match_level", "chr",
            "start", "strand", "center_base", "sequence_41nt", "sequence_101nt", "sequence_201nt",
            *SPLIT_COLUMNS,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Negative table is missing fields: {sorted(missing)}")
        for row in reader:
            total += 1
            site_id = row["site_id"]
            pair_id = row["matched_positive_site_id"]
            positive = positives.get(pair_id)
            if site_id in seen_ids:
                invalid_counts["duplicate_site_id"] += 1
            seen_ids.add(site_id)
            coordinate = (row["chr"], int(row["start"]), row["strand"])
            if coordinate in seen_coordinates:
                invalid_counts["duplicate_coordinate_strand"] += 1
            seen_coordinates.add(coordinate)
            if positive is None:
                invalid_counts["missing_matched_positive"] += 1
                continue
            mod_type = positive["mod_type"]
            negative_counts[mod_type] += 1
            match_levels[row["match_level"]] += 1
            if row["matched_mod_type"] != mod_type:
                invalid_counts["matched_mod_type_mismatch"] += 1
            for column in SPLIT_COLUMNS:
                if row[column] != positive[column]:
                    invalid_counts[f"pair_split_mismatch:{column}"] += 1
            sequence_201 = row["sequence_201nt"]
            sequence_101 = row["sequence_101nt"]
            sequence_41 = row["sequence_41nt"]
            if len(sequence_201) != 201 or len(sequence_101) != 101 or len(sequence_41) != 41:
                invalid_counts["window_length"] += 1
            if len(sequence_201) == 201:
                if sequence_201[50:151] != sequence_101 or sequence_201[80:121] != sequence_41:
                    invalid_counts["window_nesting"] += 1
                if sequence_201[100] != positive["center_base"].replace("U", "T"):
                    invalid_counts["center_base_mismatch"] += 1
                if set(sequence_201) - set("ACGT"):
                    invalid_counts["non_acgt_sequence"] += 1
            distance = nearest_distance(known.get(row["chr"], []), int(row["start"]))
            if distance <= args.exclusion_radius:
                invalid_counts["known_modification_proximity"] += 1
                if len(examples["known_modification_proximity"]) < 10:
                    examples["known_modification_proximity"].append(site_id)
            neg_gc = gc_fraction(sequence_201)
            pos_gc = gc_fraction(positive["sequence_201nt"])
            negative_gc[mod_type].append(neg_gc)
            paired_gc_differences[mod_type].append(abs(pos_gc - neg_gc))

    per_class = []
    for mod_type in sorted(positive_counts):
        requested = positive_counts[mod_type]
        matched = negative_counts[mod_type]
        per_class.append(
            {
                "mod_type": mod_type,
                "positive_rows": requested,
                "matched_negative_rows": matched,
                "match_rate": matched / max(requested, 1),
                "positive_gc_mean": statistics.fmean(positive_gc[mod_type]) if positive_gc[mod_type] else float("nan"),
                "negative_gc_mean": statistics.fmean(negative_gc[mod_type]) if negative_gc[mod_type] else float("nan"),
                "gc_standardized_mean_difference": standardized_mean_difference(
                    positive_gc[mod_type], negative_gc[mod_type]
                ),
                "paired_gc_absolute_difference_mean": statistics.fmean(paired_gc_differences[mod_type])
                if paired_gc_differences[mod_type]
                else float("nan"),
            }
        )
    minimum_match_rate = min((float(row["match_rate"]) for row in per_class), default=0.0)
    maximum_gc_smd = max(
        (abs(float(row["gc_standardized_mean_difference"])) for row in per_class if math.isfinite(float(row["gc_standardized_mean_difference"]))),
        default=float("inf"),
    )
    core_invalid = sum(invalid_counts.values())
    checks = [
        {
            "item": "negative_rows_present",
            "status": "PASS" if total > 0 else "FAIL",
            "detail": f"rows={total}",
        },
        {
            "item": "no_coordinate_or_sequence_contamination",
            "status": "PASS" if core_invalid == 0 else "FAIL",
            "detail": json.dumps(dict(invalid_counts), sort_keys=True),
        },
        {
            "item": "per_class_match_rate",
            "status": "PASS" if minimum_match_rate >= args.minimum_match_rate else "FAIL",
            "detail": f"minimum={minimum_match_rate:.6f} required={args.minimum_match_rate:.6f}",
        },
        {
            "item": "gc_matching",
            "status": "PASS" if maximum_gc_smd <= args.maximum_gc_smd else "FAIL",
            "detail": f"maximum_abs_smd={maximum_gc_smd:.6f} allowed={args.maximum_gc_smd:.6f}",
        },
    ]
    failed = [row for row in checks if row["status"] == "FAIL"]
    summary = {
        "overall_status": "PASS" if not failed else "FAIL",
        "positive_rows": len(positives),
        "negative_rows": total,
        "known_modification_dataset": str(known_path),
        "known_modification_coordinates": sum(len(values) for values in known.values()),
        "unique_negative_ids": len(seen_ids),
        "unique_negative_coordinate_strands": len(seen_coordinates),
        "exclusion_radius_nt": args.exclusion_radius,
        "minimum_class_match_rate": minimum_match_rate,
        "maximum_absolute_gc_smd": maximum_gc_smd,
        "match_levels": dict(match_levels),
        "invalid_counts": dict(invalid_counts),
        "invalid_examples": dict(examples),
        "checks": checks,
        "failed_items": [row["item"] for row in failed],
        "negative_label_caveat": (
            "Absence from RMBase is not experimental proof of no modification. The benchmark is a "
            "positive-versus-matched-unannotated candidate task and must be reported as such."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "negative_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_tsv(args.output / "negative_audit_checks.tsv", checks)
    write_tsv(args.output / "negative_matching_by_class.tsv", per_class)
    print(json.dumps(summary, indent=2))
    if failed and args.strict:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument(
        "--known-modifications",
        type=Path,
        help="Complete hg38 RMBase coordinate table used for the exclusion audit.",
    )
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusion-radius", type=int, default=25)
    parser.add_argument("--minimum-match-rate", type=float, default=0.90)
    parser.add_argument("--maximum-gc-smd", type=float, default=0.15)
    parser.add_argument("--strict", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
