#!/usr/bin/env python
"""Summarize 41/101/201 nt sequence-only experiments by modification class."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


VARIANTS = ["length_41", "length_101", "length_201"]
LENGTH_BY_VARIANT = {"length_41": 41, "length_101": 101, "length_201": 201}
MACRO_KEYS = ["macro_AUPR_ovr", "macro_AUC_ovr", "macro_F1", "MCC", "balanced_accuracy"]
CLASS_KEYS = ["support", "AUPR_ovr", "AUC_ovr", "F1", "Precision", "Recall", "MCC_ovr"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def number(obj: object) -> float | None:
    try:
        parsed = float(obj)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def value(obj: object) -> str:
    parsed = number(obj)
    return "" if parsed is None else f"{parsed:.6f}"


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
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--class-names", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    classes = [name.strip() for name in args.class_names.split(",") if name.strip()]
    metrics = {variant: load(root / variant / "metrics.json") for variant in VARIANTS}

    macro_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        data = metrics[variant]
        validation = data.get("validation", {}) if isinstance(data, dict) else {}
        test = data.get("test", {}) if isinstance(data, dict) else {}
        row: dict[str, object] = {
            "variant": variant,
            "sequence_length": LENGTH_BY_VARIANT[variant],
            "metrics_exists": bool(data),
            "best_epoch": data.get("best_epoch", "") if isinstance(data, dict) else "",
            "best_validation_score": value(data.get("best_validation_score", "")) if isinstance(data, dict) else "",
        }
        for key in MACRO_KEYS:
            row[f"validation_{key}"] = value(validation.get(key, "") if isinstance(validation, dict) else "")
            row[f"test_{key}"] = value(test.get(key, "") if isinstance(test, dict) else "")
        macro_rows.append(row)
    write(out_dir / "multirm12_sequence_length_macro.tsv", macro_rows)

    per_class_rows: list[dict[str, object]] = []
    for mod_type in classes:
        row: dict[str, object] = {"mod_type": mod_type}
        aupr_by_variant: dict[str, float] = {}
        for variant in VARIANTS:
            test = metrics[variant].get("test", {}) if metrics[variant] else {}
            per_class = test.get("per_class", {}) if isinstance(test, dict) else {}
            class_metrics = per_class.get(mod_type, {}) if isinstance(per_class, dict) else {}
            for key in CLASS_KEYS:
                row[f"{variant}_{key}"] = value(
                    class_metrics.get(key, "") if isinstance(class_metrics, dict) else ""
                )
            aupr = number(row.get(f"{variant}_AUPR_ovr"))
            if aupr is not None:
                aupr_by_variant[variant] = aupr
        baseline_41 = aupr_by_variant.get("length_41")
        for variant in ["length_101", "length_201"]:
            current = aupr_by_variant.get(variant)
            row[f"{variant}_minus_41_AUPR"] = (
                value(current - baseline_41)
                if current is not None and baseline_41 is not None
                else ""
            )
        if aupr_by_variant:
            best_variant, best_aupr = max(aupr_by_variant.items(), key=lambda item: item[1])
            row["best_length"] = LENGTH_BY_VARIANT[best_variant]
            row["best_AUPR_ovr"] = value(best_aupr)
        else:
            row["best_length"] = ""
            row["best_AUPR_ovr"] = ""
        per_class_rows.append(row)
    write(out_dir / "multirm12_sequence_length_per_class.tsv", per_class_rows)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
