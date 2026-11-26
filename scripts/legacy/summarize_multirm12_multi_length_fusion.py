#!/usr/bin/env python
"""Summarize class-specific multi-length fusion against single-length runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping


CLASS_KEYS = ["support", "AUPR_ovr", "AUC_ovr", "F1", "Precision", "Recall", "MCC_ovr"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_tsv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or key not in reader.fieldnames:
            return {}
        return {row[key]: row for row in reader}


def number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def fmt(value: object) -> str:
    parsed = number(value)
    return "" if parsed is None else f"{parsed:.6f}"


def metric(per_class: Mapping[str, object], mod_type: str, key: str) -> object:
    values = per_class.get(mod_type, {})
    return values.get(key, "") if isinstance(values, Mapping) else ""


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", default="multi_length_fusion")
    parser.add_argument("--class-names", required=True)
    parser.add_argument(
        "--single-length-reference",
        default="rmbase12_work/metrics/multirm12_sequence_length_ablation_cdhit80/multirm12_sequence_length_per_class.tsv",
    )
    parser.add_argument(
        "--baseline-reference",
        default="rmbase12_work/metrics/multirm12_cdhit80_baseline_vs_structdualrm/cdhit80_baseline_vs_structdualrm_per_class.tsv",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    metrics = load_json(model_dir / "metrics.json")
    test = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    per_class = test.get("per_class", {}) if isinstance(test, Mapping) else {}
    length_labels = [str(label) for label in metrics.get("length_gate_labels", [])] if isinstance(metrics, dict) else []
    true_length_gate = metrics.get("test_true_class_length_gate_mean", []) if isinstance(metrics, dict) else []

    single_rows = load_tsv_by_key(Path(args.single_length_reference), "mod_type")
    baseline_rows = load_tsv_by_key(Path(args.baseline_reference), "mod_type")
    classes = [name.strip() for name in args.class_names.split(",") if name.strip()]

    rows: list[dict[str, object]] = []
    for class_idx, mod_type in enumerate(classes):
        support = number(metric(per_class, mod_type, "support"))
        model_aupr = number(metric(per_class, mod_type, "AUPR_ovr"))
        single_row = single_rows.get(mod_type, {})
        baseline_row = baseline_rows.get(mod_type, {})
        best_single_aupr = number(single_row.get("best_AUPR_ovr"))
        baseline_aupr = number(
            baseline_row.get("reference_best_baseline_AUPR")
            or baseline_row.get("best_baseline_AUPR")
            or baseline_row.get("reference_best_baseline_aupr")
        )
        row: dict[str, object] = {
            "mod_type": mod_type,
            "evaluation_status": "no_test_support" if not support else "evaluable",
            "support": fmt(support),
            "model_name": args.model_name,
        }
        for key in CLASS_KEYS[1:]:
            row[f"{args.model_name}_{key}"] = fmt(metric(per_class, mod_type, key))
        row["best_single_length"] = single_row.get("best_length", single_row.get("best_length_by_AUPR", ""))
        row["best_single_length_AUPR"] = fmt(best_single_aupr)
        row[f"{args.model_name}_minus_best_single_length_AUPR"] = (
            fmt(model_aupr - best_single_aupr) if model_aupr is not None and best_single_aupr is not None else ""
        )
        row["reference_best_baseline_model"] = (
            baseline_row.get("reference_best_baseline_model")
            or baseline_row.get("best_baseline_model")
            or ""
        )
        row["reference_best_baseline_AUPR"] = fmt(baseline_aupr)
        row[f"{args.model_name}_minus_reference_baseline_AUPR"] = (
            fmt(model_aupr - baseline_aupr) if model_aupr is not None and baseline_aupr is not None else ""
        )
        for length_idx, label in enumerate(length_labels):
            value = ""
            if (
                isinstance(true_length_gate, list)
                and class_idx < len(true_length_gate)
                and isinstance(true_length_gate[class_idx], list)
                and length_idx < len(true_length_gate[class_idx])
            ):
                value = fmt(true_length_gate[class_idx][length_idx])
            row[f"true_length_gate_{label}nt"] = value
        rows.append(row)

    write(out_dir / "multirm12_multi_length_fusion_per_class.tsv", rows)

    macro_row: dict[str, object] = {
        "model_name": args.model_name,
        "model_dir": str(model_dir),
        "best_epoch": metrics.get("best_epoch", "") if isinstance(metrics, dict) else "",
        "best_validation_score": fmt(metrics.get("best_validation_score", "") if isinstance(metrics, dict) else ""),
    }
    for key in ["macro_AUPR_ovr", "macro_AUC_ovr", "macro_F1", "MCC", "balanced_accuracy"]:
        macro_row[f"test_{key}"] = fmt(test.get(key, "") if isinstance(test, Mapping) else "")
    write(out_dir / "multirm12_multi_length_fusion_macro.tsv", [macro_row])
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
