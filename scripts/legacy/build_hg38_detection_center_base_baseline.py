#!/usr/bin/env python
"""Build an empirical center-base-only baseline on a frozen detection split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def central_base(row: dict[str, str]) -> str:
    value = row.get("center_base", "").upper().replace("U", "T")
    if value:
        return value
    sequence = row.get("sequence_201nt", "").upper().replace("U", "T")
    return sequence[len(sequence) // 2] if sequence else "N"


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "site_id", "true_mod_type", "true_class_index", "original_mod_type",
        "center_base", "pred_mod_type", "pred_class_index", "prob_unmodified", "prob_modified",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    with args.dataset.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get(args.split_column) == "train":
                counts[central_base(row)][int(float(row["det_label"]))] += 1
    probabilities = {
        base: values[1] / max(values[0] + values[1], 1)
        for base, values in counts.items()
    }
    global_positive = sum(values[1] for values in counts.values())
    global_total = sum(sum(values.values()) for values in counts.values())
    fallback = global_positive / max(global_total, 1)

    output = {"valid": [], "test": []}
    with args.dataset.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            split = row.get(args.split_column)
            if split not in output:
                continue
            base = central_base(row)
            probability = probabilities.get(base, fallback)
            prediction = int(probability >= 0.5)
            label = int(float(row["det_label"]))
            output[split].append(
                {
                    "site_id": row["site_id"],
                    "true_mod_type": "modified" if label else "unmodified",
                    "true_class_index": label,
                    "original_mod_type": row["original_mod_type"],
                    "center_base": base,
                    "pred_mod_type": "modified" if prediction else "unmodified",
                    "pred_class_index": prediction,
                    "prob_unmodified": 1.0 - probability,
                    "prob_modified": probability,
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(args.output_dir / "valid_predictions.tsv", output["valid"])
    write_predictions(args.output_dir / "test_predictions.tsv", output["test"])
    summary = {
        "model": "training_empirical_center_base_only",
        "dataset": str(args.dataset),
        "split_column": args.split_column,
        "training_counts_by_center_base": {
            base: {"unmodified": values[0], "modified": values[1]}
            for base, values in sorted(counts.items())
        },
        "probability_modified_by_center_base": probabilities,
        "fallback_probability": fallback,
        "row_counts": {split: len(rows) for split, rows in output.items()},
    }
    (args.output_dir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-column", default="split_detection_cluster201")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
