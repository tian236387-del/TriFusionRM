#!/usr/bin/env python
"""Summarize final-main-model mean and standard deviation across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


PER_CLASS_METRICS = ("AUPR_ovr", "AUC_ovr", "F1", "Precision", "Recall", "MCC_ovr")
MACRO_METRICS = ("macro_AUPR_ovr", "macro_F1", "MCC", "balanced_accuracy")


def load(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)), float(np.nanstd(array, ddof=1)) if len(array) > 1 else float("nan")


def run(seed42_model_dir: Path, run_root: Path, seeds: Sequence[int], out_dir: Path) -> None:
    results: Dict[int, Mapping[str, object]] = {}
    for seed in seeds:
        model_dir = seed42_model_dir if seed == 42 else run_root / f"seed{seed}"
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing seed {seed} metrics: {metrics_path}")
        results[seed] = load(metrics_path)
    classes = list(results[seeds[0]]["classes"])
    for seed, result in results.items():
        if list(result["classes"]) != classes:
            raise ValueError(f"Class order differs for seed {seed}")

    macro_rows: List[Dict[str, object]] = []
    for metric in MACRO_METRICS:
        values = [float(results[seed]["test"][metric]) for seed in seeds]
        mean, sd = mean_sd(values)
        row: Dict[str, object] = {"metric": metric, "mean": mean, "sd": sd}
        row.update({f"seed_{seed}": value for seed, value in zip(seeds, values)})
        macro_rows.append(row)

    class_rows: List[Dict[str, object]] = []
    for class_name in classes:
        support = int(results[seeds[0]]["test"]["per_class"][class_name]["support"])
        for metric in PER_CLASS_METRICS:
            values = [float(results[seed]["test"]["per_class"][class_name][metric]) for seed in seeds]
            mean, sd = mean_sd(values)
            row = {"mod_type": class_name, "support_seed42": support, "metric": metric, "mean": mean, "sd": sd}
            row.update({f"seed_{seed}": value for seed, value in zip(seeds, values)})
            class_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "final_main_multiseed_macro.tsv", macro_rows)
    write_tsv(out_dir / "final_main_multiseed_per_class.tsv", class_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42-model-dir", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--seeds", default="42,17,73")
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    run(args.seed42_model_dir, args.run_root, seeds, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
