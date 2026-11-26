#!/usr/bin/env python
"""Create a train-only per-class cap while preserving a fixed held-out split.

This script is intended for imbalance/interference controls.  It never changes
valid/test rows and never recomputes clusters or split assignments.  Only rows
of the selected class in the selected training split are subsampled using a
deterministic SHA-256 ranking of site_id and seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fields or not rows:
        raise ValueError(f"Empty or malformed TSV: {path}")
    return rows, fields


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rank_key(row: dict[str, str], seed: int) -> str:
    site_id = row.get("site_id", "")
    return hashlib.sha256(f"{seed}\t{site_id}".encode("utf-8")).hexdigest()


def parse_nonnegative_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def build(args: argparse.Namespace) -> dict[str, object]:
    rows, fields = read_tsv(Path(args.input))
    if args.split_column not in fields:
        raise ValueError(f"Missing split column {args.split_column!r} in {args.input}")
    for required in (args.class_column, "site_id"):
        if required not in fields:
            raise ValueError(f"Missing required column {required!r} in {args.input}")

    train_rows = [row for row in rows if row.get(args.split_column) == args.train_label]
    reference_count = sum(
        1 for row in train_rows if row.get(args.class_column) == args.reference_class
    )
    if args.cap_count is not None:
        cap_count = parse_nonnegative_int(args.cap_count, "--cap-count")
        cap_basis = "explicit"
    else:
        if args.ratio is None:
            raise ValueError("Provide either --cap-count or --ratio")
        cap_count = int(round(reference_count * float(args.ratio)))
        cap_basis = f"reference_train_count({args.reference_class})*{args.ratio}"

    target_train = [
        row for row in train_rows if row.get(args.class_column) == args.target_class
    ]
    selected_ids: set[str]
    if cap_count >= len(target_train):
        selected_ids = {row["site_id"] for row in target_train}
    else:
        selected_ids = {
            row["site_id"]
            for row in sorted(target_train, key=lambda row: rank_key(row, args.seed))[:cap_count]
        }

    output_rows: list[dict[str, str]] = []
    removed_rows = 0
    for row in rows:
        if row.get(args.split_column) == args.train_label and row.get(args.class_column) == args.target_class:
            if row.get("site_id") not in selected_ids:
                removed_rows += 1
                continue
        output_rows.append(row)

    before_counts = Counter((row.get(args.split_column, ""), row.get(args.class_column, "")) for row in rows)
    after_counts = Counter((row.get(args.split_column, ""), row.get(args.class_column, "")) for row in output_rows)
    summary = {
        "input": str(Path(args.input)),
        "output": str(Path(args.output)),
        "split_column": args.split_column,
        "train_label": args.train_label,
        "class_column": args.class_column,
        "target_class": args.target_class,
        "reference_class": args.reference_class,
        "reference_train_count": reference_count,
        "requested_ratio": args.ratio,
        "cap_count": cap_count,
        "cap_basis": cap_basis,
        "target_train_before": len(target_train),
        "target_train_after": len(selected_ids),
        "removed_target_train_rows": removed_rows,
        "total_rows_before": len(rows),
        "total_rows_after": len(output_rows),
        "valid_test_unchanged": all(
            before_counts[(split, cls)] == after_counts[(split, cls)]
            for split in ("valid", "test")
            for cls in {row.get(args.class_column, "") for row in rows}
        ),
        "seed": args.seed,
        "selection": "lowest SHA-256(seed + site_id) ranking within target training class",
        "counts_before": {f"{split}:{cls}": count for (split, cls), count in sorted(before_counts.items())},
        "counts_after": {f"{split}:{cls}": count for (split, cls), count in sorted(after_counts.items())},
    }
    write_tsv(Path(args.output), output_rows, fields)
    summary_path = Path(args.summary_out) if args.summary_out else Path(str(args.output) + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--split-column", default="split_cdhit80_cluster")
    parser.add_argument("--train-label", default="train")
    parser.add_argument("--class-column", default="mod_type")
    parser.add_argument("--target-class", required=True)
    parser.add_argument("--reference-class", default="m1A")
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--cap-count", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    build(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
