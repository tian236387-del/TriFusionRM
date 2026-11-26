#!/usr/bin/env python
"""Aggregate audited hg38 detection experiments without test-set model selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


CONTROL_PATTERNS = (
    "center_base",
    "hash_kmer",
    "control",
    "shuffled",
    "no_pair_edges",
)


def finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def neural_metadata(model_dir: Path) -> dict[str, object]:
    path = model_dir / "metrics.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    efficiency = payload.get("efficiency", {})
    if not isinstance(efficiency, Mapping):
        efficiency = {}
    return {
        "best_epoch": payload.get("best_epoch", ""),
        "parameter_count": efficiency.get("parameter_count", ""),
        "training_seconds": efficiency.get("training_seconds", ""),
        "inference_samples_per_second": efficiency.get("samples_per_second", ""),
    }


def run(args: argparse.Namespace) -> None:
    metrics_paths = sorted(args.models_dir.glob("*/audited/detection_metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(f"No audited detection metrics under {args.models_dir}")

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for path in metrics_paths:
        model_dir = path.parent.parent
        model = model_dir.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        valid = payload["validation"]
        test = payload["test"]
        valid_macro = payload["macro_original_modification_validation"]
        test_macro = payload["macro_original_modification_test"]
        eligible = not any(pattern in model.lower() for pattern in CONTROL_PATTERNS)
        row: dict[str, object] = {
            "model": model,
            "eligible_for_final_selection": int(eligible),
            "validation_overall_AUPR": valid["AUPR"],
            "validation_macro_all_AUPR": valid_macro["macro_all_AUPR"],
            "validation_macro_stable_AUPR": valid_macro["macro_stable_AUPR"],
            "validation_macro_stable_F1": valid_macro["macro_stable_F1"],
            "validation_macro_stable_MCC": valid_macro["macro_stable_MCC"],
            "test_overall_AUPR": test["AUPR"],
            "test_overall_AUC": test["AUC"],
            "test_overall_F1": test["F1"],
            "test_overall_MCC": test["MCC"],
            "test_macro_all_AUPR": test_macro["macro_all_AUPR"],
            "test_macro_stable_AUPR": test_macro["macro_stable_AUPR"],
            "test_macro_stable_AUC": test_macro["macro_stable_AUC"],
            "test_macro_stable_F1": test_macro["macro_stable_F1"],
            "test_macro_stable_MCC": test_macro["macro_stable_MCC"],
            **neural_metadata(model_dir),
        }
        summary_rows.append(row)
        for split_name, key in (
            ("validation", "per_original_modification_validation"),
            ("test", "per_original_modification_test"),
        ):
            for values in payload[key]:
                detail_rows.append({"model": model, "split": split_name, **values})

    summary_rows.sort(
        key=lambda row: (
            -int(row["eligible_for_final_selection"]),
            -finite(row["validation_macro_stable_AUPR"]),
            str(row["model"]),
        )
    )
    eligible_rows = [row for row in summary_rows if int(row["eligible_for_final_selection"])]
    selected = eligible_rows[0] if eligible_rows else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = list(summary_rows[0])
    detail_fields = [
        "model", "split", "mod_type", "rows", "positive_rows", "negative_rows",
        "positive_prevalence", "threshold", "AUPR", "AUC", "F1", "Precision",
        "Recall", "Specificity", "Accuracy", "Balanced_accuracy", "MCC", "Brier", "ECE",
    ]
    write_tsv(args.output_dir / "detection_model_summary.tsv", summary_rows, summary_fields)
    write_tsv(args.output_dir / "detection_per_modification_comparison.tsv", detail_rows, detail_fields)
    selection = {
        "selection_rule": "highest validation macro-stable AUPR; test metrics excluded from selection",
        "minimum_positive_rows": 20,
        "excluded_control_patterns": list(CONTROL_PATTERNS),
        "selected_model": selected["model"] if selected else None,
        "selected_validation_macro_stable_AUPR": (
            selected["validation_macro_stable_AUPR"] if selected else None
        ),
        "models_seen": len(summary_rows),
        "eligible_models_seen": len(eligible_rows),
    }
    (args.output_dir / "detection_validation_selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("rmbase12_work/detection_strict_20260814/models"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rmbase12_work/detection_strict_20260814/summary"),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
