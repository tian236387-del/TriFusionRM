#!/usr/bin/env python
"""Summarize full, safe-context and sequence-only EvoRMD retraining."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional


MODES = ("full", "safe", "sequence_only")
METRICS = ("AUPR_ovr", "AUC_ovr", "F1", "Precision", "Recall", "MCC_ovr")


def load_metrics(run_root: Path) -> Dict[str, Mapping[str, object]]:
    return {
        mode: json.loads((run_root / mode / "metrics.json").read_text(encoding="utf-8"))
        for mode in MODES
    }


def write_per_class(path: Path, results: Mapping[str, Mapping[str, object]]) -> None:
    class_names = list(results["full"]["classes"])
    fields = ["mod_type", "support"] + [f"{mode}_{metric}" for mode in MODES for metric in METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for name in class_names:
            row = {"mod_type": name, "support": results["full"]["test"]["per_class"][name]["support"]}
            for mode in MODES:
                metrics = results[mode]["test"]["per_class"][name]
                for metric in METRICS:
                    row[f"{mode}_{metric}"] = metrics[metric]
            writer.writerow(row)


def write_overall(path: Path, results: Mapping[str, Mapping[str, object]]) -> None:
    fields = [
        "context_mode",
        "best_epoch",
        "best_validation_score",
        "test_macro_AUPR_ovr",
        "test_macro_F1",
        "test_MCC",
        "test_balanced_accuracy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for mode in MODES:
            result = results[mode]
            test = result["test"]
            writer.writerow(
                {
                    "context_mode": mode,
                    "best_epoch": result["best_epoch"],
                    "best_validation_score": result["best_validation_score"],
                    "test_macro_AUPR_ovr": test["macro_AUPR_ovr"],
                    "test_macro_F1": test["macro_F1"],
                    "test_MCC": test["MCC"],
                    "test_balanced_accuracy": test["balanced_accuracy"],
                }
            )


def write_markdown(path: Path, results: Mapping[str, Mapping[str, object]]) -> None:
    class_names = list(results["full"]["classes"])
    lines = [
        "# EvoRMD-adapted context retraining",
        "",
        "All variants are trained from scratch on the same CD-HIT80 strict split.",
        "The safe variant excludes `cell_list`, `seq_type_list`, and `support_num`.",
        "",
        "| Modification | Support | Full AUPR | Safe AUPR | Sequence-only AUPR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in class_names:
        full = results["full"]["test"]["per_class"][name]
        safe = results["safe"]["test"]["per_class"][name]
        sequence = results["sequence_only"]["test"]["per_class"][name]
        lines.append(
            f"| {name} | {full['support']} | {full['AUPR_ovr']:.6f} | "
            f"{safe['AUPR_ovr']:.6f} | {sequence['AUPR_ovr']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(run_root: Path, out_dir: Path) -> None:
    results = load_metrics(run_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_per_class(out_dir / "per_class_metrics.tsv", results)
    write_overall(out_dir / "overall_metrics.tsv", results)
    write_markdown(out_dir / "README.md", results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.run_root, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
