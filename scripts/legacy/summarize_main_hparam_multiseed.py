#!/usr/bin/env python
"""Aggregate the validation-selected main-model configuration across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Optional


METRICS = ("AUPR_ovr", "F1", "MCC_ovr")


def load(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(seed42_metrics: Path, additional_root: Path, out_dir: Path, best_run: str) -> None:
    paths = [seed42_metrics] + sorted(additional_root.glob(f"{best_run}_seed*/metrics.json"))
    results = [load(path) for path in paths]
    if len(results) != 3:
        raise ValueError(f"Expected three seeds, found {len(results)}: {paths}")
    classes = list(results[0]["classes"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["mod_type", "support"]
    for metric in METRICS:
        fields.extend([f"mean_{metric}", f"sd_{metric}", f"min_{metric}", f"max_{metric}"])
    with (out_dir / "per_class_mean_sd.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for name in classes:
            support = int(results[0]["test"]["per_class"][name]["support"])
            row: Dict[str, object] = {"mod_type": name, "support": support}
            for metric in METRICS:
                values = [float(result["test"]["per_class"][name][metric]) for result in results]
                finite = [value for value in values if math.isfinite(value)]
                row[f"mean_{metric}"] = mean(finite) if finite else float("nan")
                row[f"sd_{metric}"] = stdev(finite) if len(finite) > 1 else 0.0
                row[f"min_{metric}"] = min(finite) if finite else float("nan")
                row[f"max_{metric}"] = max(finite) if finite else float("nan")
            writer.writerow(row)
    manifest = {
        "best_run": best_run,
        "selection_rule": "validation macro AUPR + 0.10 * validation Am F1",
        "seeds": [42, 17, 73],
        "metric_files": [str(path) for path in paths],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42-metrics", required=True, type=Path)
    parser.add_argument("--additional-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--best-run", required=True)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.seed42_metrics, args.additional_root, args.out_dir, args.best_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
