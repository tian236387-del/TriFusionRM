#!/usr/bin/env python
"""Evaluate modified-vs-unannotated predictions with a frozen threshold."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def read_predictions(path: Path, probability_field: str) -> dict[str, float]:
    output: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"site_id", probability_field}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Prediction file {path} is missing {sorted(missing)}")
        for row in reader:
            site_id = row["site_id"]
            if site_id in output:
                raise ValueError(f"Duplicate prediction site_id: {site_id}")
            output[site_id] = float(row[probability_field])
    return output


def attach_labels(
    dataset: Path,
    split_column: str,
    split_value: str,
    predictions: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    labels: list[int] = []
    probabilities: list[float] = []
    site_ids: list[str] = []
    modifications: list[str] = []
    seen: set[str] = set()
    with dataset.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"site_id", "det_label", "original_mod_type", split_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset {dataset} is missing {sorted(missing)}")
        for row in reader:
            if row.get(split_column) != split_value or row["site_id"] not in predictions:
                continue
            site_id = row["site_id"]
            seen.add(site_id)
            site_ids.append(site_id)
            labels.append(int(float(row["det_label"])))
            probabilities.append(predictions[site_id])
            modifications.append(row["original_mod_type"])
    missing_predictions = set(predictions) - seen
    if missing_predictions:
        examples = sorted(missing_predictions)[:5]
        raise ValueError(f"{len(missing_predictions)} predictions do not match {split_value} rows: {examples}")
    return (
        np.asarray(labels, dtype=np.int8),
        np.asarray(probabilities, dtype=np.float64),
        site_ids,
        modifications,
    )


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    scores = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(scores == np.nanmax(scores))
    return float(thresholds[int(best[-1])])


def reliability_rows(labels: np.ndarray, probabilities: np.ndarray, bins: int) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, bins - 1)
    indices = np.maximum(indices, 0)
    for index in range(bins):
        mask = indices == index
        count = int(mask.sum())
        output.append(
            {
                "bin": index,
                "left": float(edges[index]),
                "right": float(edges[index + 1]),
                "count": count,
                "mean_probability": float(probabilities[mask].mean()) if count else float("nan"),
                "observed_fraction": float(labels[mask].mean()) if count else float("nan"),
            }
        )
    return output


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    bins: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    predictions = (probabilities >= threshold).astype(np.int8)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    reliability = reliability_rows(labels, probabilities, bins)
    ece = sum(
        int(row["count"]) / max(len(labels), 1)
        * abs(float(row["mean_probability"]) - float(row["observed_fraction"]))
        for row in reliability
        if int(row["count"]) > 0
    )
    metrics = {
        "rows": len(labels),
        "positive_rows": positives,
        "negative_rows": negatives,
        "positive_prevalence": positives / max(len(labels), 1),
        "threshold": threshold,
        "AUPR": float(average_precision_score(labels, probabilities)) if positives and negatives else float("nan"),
        "AUC": float(roc_auc_score(labels, probabilities)) if positives and negatives else float("nan"),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
        "Specificity": true_negative / max(negatives, 1),
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "Brier": float(brier_score_loss(labels, probabilities)),
        "ECE": float(ece),
    }
    return metrics, reliability


def prevalence_rows(metrics: Mapping[str, object], values: Sequence[float]) -> list[dict[str, object]]:
    sensitivity = float(metrics["Recall"])
    false_positive_rate = 1.0 - float(metrics["Specificity"])
    output = []
    for prevalence in values:
        denominator = sensitivity * prevalence + false_positive_rate * (1.0 - prevalence)
        precision = sensitivity * prevalence / denominator if denominator > 0 else float("nan")
        output.append(
            {
                "assumed_prevalence": prevalence,
                "sensitivity": sensitivity,
                "specificity": 1.0 - false_positive_rate,
                "expected_precision": precision,
            }
        )
    return output


def per_modification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    modifications: Sequence[str],
    threshold: float,
    bins: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    modification_array = np.asarray(modifications, dtype=object)
    for mod_type in sorted(set(modifications)):
        mask = modification_array == mod_type
        metrics, _ = binary_metrics(labels[mask], probabilities[mask], threshold, bins)
        output.append({"mod_type": mod_type, **metrics})
    return output


def macro_modification_metrics(
    rows: Sequence[Mapping[str, object]],
    minimum_positive_rows: int,
) -> dict[str, object]:
    metric_names = ("AUPR", "AUC", "F1", "MCC")
    stable_rows = [
        row for row in rows
        if int(row["positive_rows"]) >= max(1, minimum_positive_rows)
    ]

    def mean_metric(selected: Sequence[Mapping[str, object]], metric: str) -> float:
        values = [float(row[metric]) for row in selected if math.isfinite(float(row[metric]))]
        return float(np.mean(values)) if values else float("nan")

    summary: dict[str, object] = {
        "all_group_count": len(rows),
        "stable_group_count": len(stable_rows),
        "minimum_positive_rows": max(1, minimum_positive_rows),
        "all_groups": [str(row["mod_type"]) for row in rows],
        "stable_groups": [str(row["mod_type"]) for row in stable_rows],
    }
    for metric in metric_names:
        summary[f"macro_all_{metric}"] = mean_metric(rows, metric)
        summary[f"macro_stable_{metric}"] = mean_metric(stable_rows, metric)
    return summary


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run(args: argparse.Namespace) -> None:
    probability_field = args.probability_field or f"prob_{args.positive_class}"
    valid_predictions = read_predictions(args.valid_predictions, probability_field)
    test_predictions = read_predictions(args.test_predictions, probability_field)
    valid_labels, valid_probabilities, _, valid_modifications = attach_labels(
        args.dataset, args.split_column, "valid", valid_predictions
    )
    test_labels, test_probabilities, test_ids, test_modifications = attach_labels(
        args.dataset, args.split_column, "test", test_predictions
    )
    threshold = select_f1_threshold(valid_labels, valid_probabilities)
    valid_metrics, valid_reliability = binary_metrics(valid_labels, valid_probabilities, threshold, args.bins)
    test_metrics, test_reliability = binary_metrics(test_labels, test_probabilities, threshold, args.bins)
    test_metrics_at_half, _ = binary_metrics(test_labels, test_probabilities, 0.5, args.bins)

    per_modification_valid = per_modification_metrics(
        valid_labels,
        valid_probabilities,
        valid_modifications,
        threshold,
        args.bins,
    )
    per_modification_test = per_modification_metrics(
        test_labels,
        test_probabilities,
        test_modifications,
        threshold,
        args.bins,
    )
    macro_modification_valid = macro_modification_metrics(
        per_modification_valid,
        args.minimum_positive_rows,
    )
    macro_modification_test = macro_modification_metrics(
        per_modification_test,
        args.minimum_positive_rows,
    )

    prevalence = prevalence_rows(test_metrics, args.prevalences)
    summary = {
        "task": "hg38_modified_vs_matched_unannotated_candidate_detection",
        "dataset": str(args.dataset),
        "split_column": args.split_column,
        "positive_class": args.positive_class,
        "probability_field": probability_field,
        "threshold_selection": "maximum validation F1; test labels were not used",
        "validation_selected_threshold": threshold,
        "validation": valid_metrics,
        "test": test_metrics,
        "test_at_threshold_0.5": test_metrics_at_half,
        "per_original_modification_validation": per_modification_valid,
        "per_original_modification_test": per_modification_test,
        "macro_original_modification_validation": macro_modification_valid,
        "macro_original_modification_test": macro_modification_test,
        "prevalence_adjusted_precision": prevalence,
        "negative_label_caveat": (
            "Negative records are matched candidates absent from the complete RMBase coordinate index, "
            "not experimentally verified unmodified sites."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "detection_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metric_fields = [
        "mod_type", "rows", "positive_rows", "negative_rows", "positive_prevalence", "threshold",
        "AUPR", "AUC", "F1", "Precision", "Recall", "Specificity", "Accuracy",
        "Balanced_accuracy", "MCC", "Brier", "ECE",
    ]
    write_tsv(
        args.output_dir / "detection_per_modification_valid.tsv",
        per_modification_valid,
        metric_fields,
    )
    write_tsv(
        args.output_dir / "detection_per_modification_test.tsv",
        per_modification_test,
        metric_fields,
    )
    write_tsv(
        args.output_dir / "detection_reliability_valid.tsv",
        valid_reliability,
        ["bin", "left", "right", "count", "mean_probability", "observed_fraction"],
    )
    write_tsv(
        args.output_dir / "detection_reliability_test.tsv",
        test_reliability,
        ["bin", "left", "right", "count", "mean_probability", "observed_fraction"],
    )
    write_tsv(
        args.output_dir / "detection_prevalence_adjusted_precision.tsv",
        prevalence,
        ["assumed_prevalence", "sensitivity", "specificity", "expected_precision"],
    )
    with (args.output_dir / "detection_test_predictions_audited.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["site_id", "original_mod_type", "det_label", "prob_modified", "pred_modified"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for site_id, mod_type, label, probability in zip(
            test_ids, test_modifications, test_labels, test_probabilities
        ):
            writer.writerow(
                {
                    "site_id": site_id,
                    "original_mod_type": mod_type,
                    "det_label": int(label),
                    "prob_modified": float(probability),
                    "pred_modified": int(probability >= threshold),
                }
            )
    print(json.dumps({"threshold": threshold, "validation": valid_metrics, "test": test_metrics}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--valid-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-column", default="split_detection_cluster201")
    parser.add_argument("--positive-class", default="modified")
    parser.add_argument("--probability-field", default="")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument(
        "--minimum-positive-rows",
        type=int,
        default=20,
        help="Minimum positive rows for a modification to enter the stable macro summary.",
    )
    parser.add_argument("--prevalences", type=float, nargs="+", default=[0.001, 0.01, 0.05, 0.10])
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
