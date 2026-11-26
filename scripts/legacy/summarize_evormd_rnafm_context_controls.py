#!/usr/bin/env python
"""Summarize strict-split EvoRMD RNA-FM context-control experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


VARIANTS = [
    "full_metadata_enabled",
    "safe_context",
    "sequence_only",
    "full_metadata_shuffled",
]
METRICS = ["AUPR_ovr", "AUC_ovr", "F1", "Precision", "Recall", "MCC_ovr"]
MACRO_METRICS = ["accuracy", "balanced_accuracy", "macro_F1", "MCC", "macro_AUC_ovr", "macro_AUPR_ovr"]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: object) -> object:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return value


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})


def numeric(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def run(root: Path, out_dir: Path) -> None:
    results: Dict[str, Dict[str, object]] = {}
    for variant in VARIANTS:
        path = root / f"evormd_rnafm_official_core_{variant}_cdhit80_seed42" / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        results[variant] = read_json(path)

    summary_rows = []
    for variant in VARIANTS:
        result = results[variant]
        test = result["test"]
        assert isinstance(test, Mapping)
        row: Dict[str, object] = {
            "variant": variant,
            "best_epoch": result.get("best_epoch", ""),
            "best_validation_macro_AUPR": result.get("best_validation_score", ""),
        }
        row.update({f"test_{metric}": test.get(metric, "") for metric in MACRO_METRICS})
        summary_rows.append(row)

    classes = list(results["full_metadata_enabled"]["classes"])
    per_class_rows = []
    for name in classes:
        full_per_class = results["full_metadata_enabled"]["test"]["per_class"]
        assert isinstance(full_per_class, Mapping)
        full_values = full_per_class[name]
        assert isinstance(full_values, Mapping)
        row = {"mod_type": name, "support": full_values.get("support", "")}
        for variant in VARIANTS:
            per_class = results[variant]["test"]["per_class"]
            assert isinstance(per_class, Mapping)
            values = per_class[name]
            assert isinstance(values, Mapping)
            for metric in METRICS:
                row[f"{variant}_{metric}"] = values.get(metric, "")

        full_aupr = numeric(row["full_metadata_enabled_AUPR_ovr"])
        safe_aupr = numeric(row["safe_context_AUPR_ovr"])
        sequence_aupr = numeric(row["sequence_only_AUPR_ovr"])
        shuffled_aupr = numeric(row["full_metadata_shuffled_AUPR_ovr"])
        row["full_minus_safe_AUPR"] = full_aupr - safe_aupr
        row["safe_minus_sequence_AUPR"] = safe_aupr - sequence_aupr
        row["full_minus_shuffled_AUPR"] = full_aupr - shuffled_aupr
        per_class_rows.append(row)

    write_tsv(out_dir / "summary.tsv", summary_rows)
    write_tsv(out_dir / "per_class_metrics.tsv", per_class_rows)
    (out_dir / "README.md").write_text(
        "# EvoRMD RNA-FM context controls\n\n"
        "All runs use the same CD-HIT80 strict split, 41-nt sequence input, released RNA-FM T12 weights, "
        "training-only m6A/m5C downsampling, and unchanged validation/test partitions.\n\n"
        "- `full_metadata_enabled`: full available RMBase metadata.\n"
        "- `safe_context`: region/gene context only; excludes cell, assay/sequence type, and support count.\n"
        "- `sequence_only`: RNA-FM sequence representation only.\n"
        "- `full_metadata_shuffled`: complete metadata bundles globally permuted within each split.\n",
        encoding="utf-8",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="rmbase12_work/official_reproduction_benchmark")
    parser.add_argument(
        "--out-dir",
        default="rmbase12_work/metrics/evormd_rnafm_context_controls_cdhit80_seed42",
    )
    args = parser.parse_args(argv)
    run(Path(args.root), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
