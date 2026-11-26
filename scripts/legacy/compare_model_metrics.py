#!/usr/bin/env python
"""Collect M0-M4 metrics into comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence


MODEL_ORDER = [
    "M0_transformer_only",
    "M1_transformer_global_structure",
    "M2_transformer_gnn",
    "M3_structdualrm_gate",
    "M4_structdualrm_full",
]


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


def summarize_model_metrics(models_root: Path) -> List[Dict[str, object]]:
    rows = []
    for model_name in MODEL_ORDER:
        metrics_path = models_root / model_name / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = metrics["test"]
        counts = metrics.get("row_counts", {})
        rows.append(
            {
                "model_dir": model_name,
                "variant": metrics.get("variant", ""),
                "test_macro_AUPR": test["macro"]["macro_AUPR"],
                "test_macro_MCC": test["macro"]["macro_MCC"],
                "test_average_F1": test["macro"]["average_F1"],
                "test_m1A_AUC": test["m1A"]["AUC"],
                "test_m1A_AUPR": test["m1A"]["AUPR"],
                "test_m1A_F1": test["m1A"]["F1"],
                "test_m1A_MCC": test["m1A"]["MCC"],
                "test_m1A_Recall": test["m1A"].get("Recall", ""),
                "test_m1A_Specificity": test["m1A"].get("Specificity", ""),
                "test_m6A_AUC": test["m6A"]["AUC"],
                "test_m6A_AUPR": test["m6A"]["AUPR"],
                "test_m6A_F1": test["m6A"]["F1"],
                "test_m6A_MCC": test["m6A"]["MCC"],
                "test_m6A_Recall": test["m6A"].get("Recall", ""),
                "test_m6A_Specificity": test["m6A"].get("Specificity", ""),
                "train_rows": counts.get("train", ""),
                "valid_rows": counts.get("valid", ""),
                "test_rows": counts.get("test", ""),
            }
        )
    return rows


def write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare StructDualRM model metrics")
    parser.add_argument("--models-root", default="08_models")
    parser.add_argument("--out", default="09_results/metrics/model_comparison.tsv")
    parser.add_argument("--table-out", default="11_tables/Table2_model_performance.tsv")
    args = parser.parse_args(argv)
    rows = summarize_model_metrics(Path(args.models_root))
    write_tsv(Path(args.out), rows)
    write_tsv(Path(args.table_out), rows)
    print(f"wrote {len(rows)} model rows to {args.out} and {args.table_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
