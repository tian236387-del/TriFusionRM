#!/usr/bin/env python
"""Summarize sequence-only, real RNAfold and shuffled-RNAfold experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


VARIANTS = ["sequence_only", "rnafold_mfe", "rnafold_mfe_shuffled"]
MACRO_KEYS = ["macro_AUPR_ovr", "macro_AUC_ovr", "macro_F1", "MCC", "balanced_accuracy"]
CLASS_KEYS = ["support", "AUPR_ovr", "AUC_ovr", "F1", "Precision", "Recall", "MCC_ovr"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def value(obj: object) -> str:
    try:
        number = float(obj)
    except (TypeError, ValueError):
        return ""
    return "" if math.isnan(number) else f"{number:.6f}"


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(text: object) -> float | None:
    try:
        out = float(text)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--class-names", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    classes = [name.strip() for name in args.class_names.split(",") if name.strip()]
    metrics = {name: load(root / name / "metrics.json") for name in VARIANTS}

    macro_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        data = metrics[variant]
        test = data.get("test", {}) if isinstance(data, dict) else {}
        validation = data.get("validation", {}) if isinstance(data, dict) else {}
        row: dict[str, object] = {
            "variant": variant,
            "metrics_exists": bool(data),
            "best_epoch": data.get("best_epoch", "") if isinstance(data, dict) else "",
            "best_validation_score": value(data.get("best_validation_score", "")) if isinstance(data, dict) else "",
        }
        for key in MACRO_KEYS:
            row[f"validation_{key}"] = value(validation.get(key, "") if isinstance(validation, dict) else "")
            row[f"test_{key}"] = value(test.get(key, "") if isinstance(test, dict) else "")
        macro_rows.append(row)
    write(out_dir / "multirm12_rnafold_ablation_macro.tsv", macro_rows)

    per_rows: list[dict[str, object]] = []
    for mod in classes:
        row: dict[str, object] = {"mod_type": mod}
        for variant in VARIANTS:
            test = metrics[variant].get("test", {}) if metrics[variant] else {}
            per = test.get("per_class", {}) if isinstance(test, dict) else {}
            cls = per.get(mod, {}) if isinstance(per, dict) else {}
            for key in CLASS_KEYS:
                row[f"{variant}_{key}"] = value(cls.get(key, "") if isinstance(cls, dict) else "")
        real = number(row.get("rnafold_mfe_AUPR_ovr"))
        seq = number(row.get("sequence_only_AUPR_ovr"))
        shuffled = number(row.get("rnafold_mfe_shuffled_AUPR_ovr"))
        row["real_minus_sequence_AUPR"] = value(real - seq) if real is not None and seq is not None else ""
        row["real_minus_shuffled_AUPR"] = value(real - shuffled) if real is not None and shuffled is not None else ""
        per_rows.append(row)
    write(out_dir / "multirm12_rnafold_ablation_per_class.tsv", per_rows)

    by_variant = {row["variant"]: row for row in macro_rows}
    seq_macro = number(by_variant["sequence_only"].get("test_macro_AUPR_ovr"))
    real_macro = number(by_variant["rnafold_mfe"].get("test_macro_AUPR_ovr"))
    shuffled_macro = number(by_variant["rnafold_mfe_shuffled"].get("test_macro_AUPR_ovr"))
    gain = [{
        "metric": "macro_AUPR_ovr",
        "sequence_only": value(seq_macro),
        "rnafold_mfe": value(real_macro),
        "rnafold_mfe_shuffled": value(shuffled_macro),
        "real_minus_sequence": value(real_macro - seq_macro) if real_macro is not None and seq_macro is not None else "",
        "real_minus_shuffled": value(real_macro - shuffled_macro) if real_macro is not None and shuffled_macro is not None else "",
        "interpretation": "RNAfold is supported only if real > sequence-only and real > shuffled on validation and test.",
    }]
    write(out_dir / "multirm12_rnafold_structure_gain.tsv", gain)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
