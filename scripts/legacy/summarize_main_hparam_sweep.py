#!/usr/bin/env python
"""Summarize validation-selected main-model hyperparameter runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


def load_result(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(
    name: str,
    result: Mapping[str, object],
    training_args: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    validation = result["validation"]
    test = result["test"]
    valid_am = validation["per_class"]["Am"]
    test_am = test["per_class"]["Am"]
    config = result.get("config", {})
    training_args = training_args or {}
    return {
        "run": name,
        "best_epoch": result.get("best_epoch", ""),
        "lr": training_args.get("lr", ""),
        "dropout": config.get("dropout", "") if isinstance(config, Mapping) else "",
        "weight_decay": training_args.get("weight_decay", ""),
        "class_weight_power": training_args.get("class_weight_power", ""),
        "validation_macro_AUPR": validation["macro_AUPR_ovr"],
        "validation_Am_AUPR": valid_am["AUPR_ovr"],
        "validation_Am_F1": valid_am["F1"],
        "validation_Am_MCC": valid_am["MCC_ovr"],
        "validation_selection_score": float(validation["macro_AUPR_ovr"]) + 0.10 * float(valid_am["F1"]),
        "test_macro_AUPR": test["macro_AUPR_ovr"],
        "test_Am_AUPR": test_am["AUPR_ovr"],
        "test_Am_F1": test_am["F1"],
        "test_Am_MCC": test_am["MCC_ovr"],
    }


def run(run_root: Path, control: Path, out_dir: Path) -> None:
    control_args_path = control.with_name("training_config.json")
    control_args = load_result(control_args_path) if control_args_path.exists() else {}
    rows: List[Dict[str, object]] = [metric_row("control", load_result(control), control_args)]
    for path in sorted(run_root.glob("*/metrics.json")):
        args_path = path.with_name("training_config.json")
        training_args = load_result(args_path) if args_path.exists() else {}
        rows.append(metric_row(path.parent.name, load_result(path), training_args))
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (out_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    candidates = [row for row in rows if row["run"] != "control"]
    best = max(candidates, key=lambda row: float(row["validation_selection_score"]))
    (out_dir / "best_validation_run.txt").write_text(str(best["run"]) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.run_root, args.control, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
