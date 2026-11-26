#!/usr/bin/env python
"""Create sensitivity metrics excluding very small RMBase 11-class test classes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


SUMMARY_FIELDS = [
    "variant",
    "included_classes",
    "excluded_classes",
    "included_class_count",
    "macro_AUPR_ovr_included",
    "macro_F1_included",
    "mean_recall_included",
    "mean_precision_included",
]


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def f(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values: List[float]) -> float:
    values = [value for value in values if value == value]
    return sum(values) / max(len(values), 1)


def run(args: argparse.Namespace) -> None:
    rows = read_tsv(Path(args.per_class))
    excluded = set(args.exclude.split(","))
    by_variant: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    out_rows: List[Dict[str, object]] = []
    for variant, group in by_variant.items():
        included = [row for row in group if row["mod_type"] not in excluded]
        out_rows.append(
            {
                "variant": variant,
                "included_classes": ",".join(row["mod_type"] for row in included),
                "excluded_classes": ",".join(sorted(excluded)),
                "included_class_count": len(included),
                "macro_AUPR_ovr_included": mean([f(row["AUPR_ovr"]) for row in included]),
                "macro_F1_included": mean([f(row["F1"]) for row in included]),
                "mean_recall_included": mean([f(row["Recall"]) for row in included]),
                "mean_precision_included": mean([f(row["Precision"]) for row in included]),
            }
        )
    out_rows.sort(key=lambda row: float(row["macro_AUPR_ovr_included"]), reverse=True)
    write_tsv(Path(args.out), out_rows, SUMMARY_FIELDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize RMBase11 sensitivity excluding low-support classes.")
    parser.add_argument("--per-class", default="rmbase11_work/metrics/rmbase11_per_class_metrics.tsv")
    parser.add_argument("--out", default="rmbase11_work/metrics/rmbase11_sensitivity_excluding_D_m5U.tsv")
    parser.add_argument("--exclude", default="D,m5U")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
