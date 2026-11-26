#!/usr/bin/env python
"""Summarize RBP positional hyperparameter sweep outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


VARIANTS = ["no_position", "no_position_rbp_pos", "no_position_shuffled_rbp_pos"]


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: List[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> Optional[float]:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def variant_rows(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_tsv(path)
    return {row.get("variant", ""): row for row in rows if row.get("variant")}


def diff(left: Optional[float], right: Optional[float]) -> str:
    if left is None or right is None:
        return ""
    return f"{left - right:.6f}"


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize RBP positional hyperparameter sweep.")
    parser.add_argument("--metrics-base", default="rmbase11_full_work/metrics/encode_top30_rbp_hparam_sweep")
    parser.add_argument("--config-table", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    metrics_base = Path(args.metrics_base)
    config_table = Path(args.config_table) if args.config_table else metrics_base / "rbp_hparam_sweep_configs.tsv"
    out = Path(args.out) if args.out else metrics_base / "rbp_hparam_sweep_summary.tsv"

    configs = read_tsv(config_table)
    rows: List[Dict[str, object]] = []
    for config in configs:
        tag = config.get("tag", "")
        if not tag:
            continue
        macro_path = metrics_base / tag / "rbp_positional_ablation_macro.tsv"
        by_variant = variant_rows(macro_path)
        row: Dict[str, object] = dict(config)
        row["macro_file"] = str(macro_path)
        row["macro_file_exists"] = macro_path.exists()
        values: Dict[str, Optional[float]] = {}
        for variant in VARIANTS:
            item = by_variant.get(variant, {})
            valid = as_float(item.get("validation_macro_AUPR_ovr", ""))
            test = as_float(item.get("test_macro_AUPR_ovr", ""))
            values[f"{variant}_valid"] = valid
            values[f"{variant}_test"] = test
            row[f"{variant}_best_epoch"] = item.get("best_epoch", "")
            row[f"{variant}_best_validation_score"] = item.get("best_validation_score", "")
            row[f"{variant}_validation_macro_AUPR_ovr"] = fmt(valid)
            row[f"{variant}_test_macro_AUPR_ovr"] = fmt(test)
            row[f"{variant}_test_macro_F1"] = item.get("test_macro_F1", "")
            row[f"{variant}_test_MCC"] = item.get("test_MCC", "")
        row["rbp_minus_main_validation_AUPR"] = diff(
            values.get("no_position_rbp_pos_valid"), values.get("no_position_valid")
        )
        row["rbp_minus_main_test_AUPR"] = diff(
            values.get("no_position_rbp_pos_test"), values.get("no_position_test")
        )
        row["rbp_minus_shuffled_validation_AUPR"] = diff(
            values.get("no_position_rbp_pos_valid"), values.get("no_position_shuffled_rbp_pos_valid")
        )
        row["rbp_minus_shuffled_test_AUPR"] = diff(
            values.get("no_position_rbp_pos_test"), values.get("no_position_shuffled_rbp_pos_test")
        )
        rows.append(row)

    def sort_key(row: Mapping[str, object]) -> tuple:
        valid = as_float(row.get("no_position_rbp_pos_validation_macro_AUPR_ovr", ""))
        rbp_delta = as_float(row.get("rbp_minus_shuffled_validation_AUPR", ""))
        return (valid if valid is not None else -1.0, rbp_delta if rbp_delta is not None else -1.0)

    rows.sort(key=sort_key, reverse=True)
    write_tsv(out, rows)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
