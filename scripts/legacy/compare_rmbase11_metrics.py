#!/usr/bin/env python
"""Summarize RMBase hg38 11-class model and baseline metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
SUMMARY_FIELDS = [
    "variant",
    "model_dir",
    "train_rows",
    "valid_rows",
    "test_rows",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_F1",
    "test_weighted_F1",
    "test_MCC",
    "test_macro_AUC_ovr",
    "test_macro_AUPR_ovr",
    "valid_macro_AUPR_ovr",
]
PER_CLASS_FIELDS = [
    "variant",
    "mod_type",
    "support",
    "AUC_ovr",
    "AUPR_ovr",
    "F1",
    "Precision",
    "Recall",
    "MCC_ovr",
]


def write_tsv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_metrics(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_metric_file(path: Path) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    metrics = read_metrics(path)
    test = metrics["test"]
    valid = metrics.get("validation", {})
    rows = metrics.get("row_counts", {})
    variant = str(metrics.get("variant", path.parent.name))
    summary = {
        "variant": variant,
        "model_dir": str(path.parent),
        "train_rows": rows.get("train", ""),
        "valid_rows": rows.get("valid", ""),
        "test_rows": rows.get("test", ""),
        "test_accuracy": test.get("accuracy", ""),
        "test_balanced_accuracy": test.get("balanced_accuracy", ""),
        "test_macro_F1": test.get("macro_F1", ""),
        "test_weighted_F1": test.get("weighted_F1", ""),
        "test_MCC": test.get("MCC", ""),
        "test_macro_AUC_ovr": test.get("macro_AUC_ovr", ""),
        "test_macro_AUPR_ovr": test.get("macro_AUPR_ovr", ""),
        "valid_macro_AUPR_ovr": valid.get("macro_AUPR_ovr", ""),
    }
    per_class = []
    for mod_type in CLASS_NAMES:
        item = test.get("per_class", {}).get(mod_type, {})
        row = {"variant": variant, "mod_type": mod_type}
        row.update(item)
        per_class.append(row)
    return summary, per_class


def run(args: argparse.Namespace) -> Dict[str, object]:
    root = Path(args.models_dir)
    metric_paths = sorted(root.glob("*/metrics.json"))
    summaries: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []
    for metric_path in metric_paths:
        if metric_path.parent.name.startswith("_smoke"):
            continue
        summary, per_class = summarize_metric_file(metric_path)
        summaries.append(summary)
        per_class_rows.extend(per_class)

    def score(row: Dict[str, object]) -> float:
        try:
            return float(row["test_macro_AUPR_ovr"])
        except (TypeError, ValueError):
            return -1.0

    summaries.sort(key=score, reverse=True)
    out_dir = Path(args.out_dir)
    write_tsv(out_dir / "rmbase11_model_comparison.tsv", summaries, SUMMARY_FIELDS)
    write_tsv(out_dir / "rmbase11_per_class_metrics.tsv", per_class_rows, PER_CLASS_FIELDS)
    payload = {
        "models_dir": args.models_dir,
        "out_dir": args.out_dir,
        "model_count": len(summaries),
        "best_by_test_macro_AUPR_ovr": summaries[0] if summaries else None,
        "summary_tsv": str(out_dir / "rmbase11_model_comparison.tsv"),
        "per_class_tsv": str(out_dir / "rmbase11_per_class_metrics.tsv"),
    }
    (out_dir / "rmbase11_metric_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare RMBase 11-class metrics.")
    parser.add_argument("--models-dir", default="rmbase11_work/models")
    parser.add_argument("--out-dir", default="rmbase11_work/metrics")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
