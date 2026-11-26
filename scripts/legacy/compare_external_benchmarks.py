#!/usr/bin/env python
"""Combine StructDualRM and external benchmark metrics into manuscript tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


FIELDS = [
    "model_dir",
    "variant",
    "test_macro_AUPR",
    "test_macro_MCC",
    "test_average_F1",
    "test_m1A_AUC",
    "test_m1A_AUPR",
    "test_m1A_F1",
    "test_m1A_MCC",
    "test_m1A_Recall",
    "test_m1A_Specificity",
    "test_m6A_AUC",
    "test_m6A_AUPR",
    "test_m6A_F1",
    "test_m6A_MCC",
    "test_m6A_Recall",
    "test_m6A_Specificity",
    "train_rows",
    "valid_rows",
    "test_rows",
]


def summarize_external_metrics(metrics_path: Path) -> Dict[str, object]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = metrics["test"]
    row_counts = metrics.get("row_counts", {})
    return {
        "model_dir": metrics.get("model_name") or metrics_path.parent.name,
        "variant": metrics.get("variant", metrics_path.parent.name),
        "test_macro_AUPR": test["macro"]["macro_AUPR"],
        "test_macro_MCC": test["macro"]["macro_MCC"],
        "test_average_F1": test["macro"]["average_F1"],
        "test_m1A_AUC": test["m1A"]["AUC"],
        "test_m1A_AUPR": test["m1A"]["AUPR"],
        "test_m1A_F1": test["m1A"]["F1"],
        "test_m1A_MCC": test["m1A"]["MCC"],
        "test_m1A_Recall": test["m1A"]["Recall"],
        "test_m1A_Specificity": test["m1A"]["Specificity"],
        "test_m6A_AUC": test["m6A"]["AUC"],
        "test_m6A_AUPR": test["m6A"]["AUPR"],
        "test_m6A_F1": test["m6A"]["F1"],
        "test_m6A_MCC": test["m6A"]["MCC"],
        "test_m6A_Recall": test["m6A"]["Recall"],
        "test_m6A_Specificity": test["m6A"]["Specificity"],
        "train_rows": row_counts.get("train", ""),
        "valid_rows": row_counts.get("valid", ""),
        "test_rows": row_counts.get("test", ""),
    }


def collect_metrics(paths: Iterable[Path]) -> List[Dict[str, object]]:
    rows = []
    for path in paths:
        metrics_path = path / "metrics.json" if path.is_dir() else path
        if metrics_path.exists():
            rows.append(summarize_external_metrics(metrics_path))
    return rows


def write_table(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare external benchmark metrics.")
    parser.add_argument("--metrics", nargs="+", default=[
        "12_benchmarks/kmer_logreg",
        "12_benchmarks/kmer_extratrees",
        "12_benchmarks/multirm_official_101nt",
        "12_benchmarks/multirm_style",
        "08_models/M0_transformer_only",
        "08_models/M4_structdualrm_full",
    ])
    parser.add_argument("--out", default="09_results/metrics/external_benchmark_comparison.tsv")
    parser.add_argument("--table-out", default="11_tables/Table3_external_benchmark.tsv")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = collect_metrics([Path(item) for item in args.metrics])
    write_table(rows, Path(args.out))
    write_table(rows, Path(args.table_out))
    print(f"wrote {len(rows)} rows to {args.out} and {args.table_out}")


if __name__ == "__main__":
    main()
