#!/usr/bin/env python
"""Export row-aligned five-seed raw ensembles for frozen control variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aggregate_submission_strict_results import CLASSES, CONTROL_VARIANTS, evaluate, load_ensemble


def write_predictions(path: Path, ids: list[str], labels, scores) -> None:
    fields = ["site_id", "true_class_index", "true_mod_type", *[f"prob_{name}" for name in CLASSES]]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row_idx, site_id in enumerate(ids):
            row = {
                "site_id": site_id,
                "true_class_index": int(labels[row_idx]),
                "true_mod_type": CLASSES[int(labels[row_idx])],
            }
            row.update(
                {f"prob_{name}": float(scores[row_idx, class_idx]) for class_idx, name in enumerate(CLASSES)}
            )
            writer.writerow(row)
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    reference_ids = None
    reference_labels = None
    for variant in CONTROL_VARIANTS:
        ids, labels, scores = load_ensemble(args.run_root / variant, "test")
        if reference_ids is None:
            reference_ids, reference_labels = ids, labels
        elif ids != reference_ids or (labels != reference_labels).any():
            raise ValueError(f"Control predictions are not row-aligned: {variant}")
        write_predictions(output / f"{variant}_test_predictions_raw.tsv", ids, labels, scores)
        metrics[variant] = evaluate(labels, scores)
    (output / "control_ensemble_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"variants": list(metrics), "test_rows": len(reference_ids or [])}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("rmbase12_work/submission_strict_runs_20260806/main_controls"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rmbase12_work/submission_strict_final_20260806/control_ensemble_predictions"),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
