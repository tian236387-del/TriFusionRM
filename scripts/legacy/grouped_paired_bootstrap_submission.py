#!/usr/bin/env python
"""Cluster-level paired bootstrap for frozen StructDualRM ensembles.

The same 201-nt cluster resample is applied to every compared model. This
preserves within-cluster dependence and supports paired confidence intervals
for per-class and macro one-vs-rest AUPR, F1 and MCC.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

try:
    from numba import njit, prange
except ImportError as exc:  # pragma: no cover - server dependency check
    raise SystemExit("numba is required for 2,000 grouped bootstrap iterations") from exc


CLASSES = ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m7G", "Y", "AtoI"]


def read_dataset_groups(path: Path, split_column: str, group_column: str) -> dict[str, str]:
    groups: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"site_id", split_column, group_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing fields: {sorted(missing)}")
        for row in reader:
            if row[split_column] == "test":
                groups[row["site_id"]] = row[group_column]
    if not groups:
        raise ValueError(f"No test rows in {path} for {split_column}")
    return groups


def read_predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids: list[str] = []
    labels: list[int] = []
    scores: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"site_id", "true_class_index", *[f"prob_{name}" for name in CLASSES]}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Prediction file {path} is missing fields: {sorted(missing)}")
        for row in reader:
            ids.append(row["site_id"])
            labels.append(int(row["true_class_index"]))
            scores.append([float(row[f"prob_{name}"]) for name in CLASSES])
    return ids, np.asarray(labels, dtype=np.int16), np.asarray(scores, dtype=np.float64)


def align_predictions(
    reference_ids: Sequence[str],
    reference_labels: np.ndarray,
    path: Path,
) -> np.ndarray:
    ids, labels, scores = read_predictions(path)
    if ids == list(reference_ids):
        if not np.array_equal(labels, reference_labels):
            raise ValueError(f"Labels differ from reference in {path}")
        return scores
    index = {site_id: idx for idx, site_id in enumerate(ids)}
    if len(index) != len(ids) or set(index) != set(reference_ids):
        raise ValueError(f"Site identifiers do not match the reference in {path}")
    order = np.asarray([index[site_id] for site_id in reference_ids], dtype=np.int64)
    if not np.array_equal(labels[order], reference_labels):
        raise ValueError(f"Aligned labels differ from reference in {path}")
    return scores[order]


def bootstrap_cluster_counts(iterations: int, cluster_count: int, seed: int) -> np.ndarray:
    """Generate exact nonparametric cluster-bootstrap multiplicities."""
    rng = np.random.default_rng(seed)
    counts = np.empty((iterations, cluster_count), dtype=np.uint16)
    for iteration in range(iterations):
        sampled = rng.integers(0, cluster_count, size=cluster_count, dtype=np.int64)
        value = np.bincount(sampled, minlength=cluster_count)
        if value.max(initial=0) > np.iinfo(np.uint16).max:
            raise OverflowError("Cluster bootstrap multiplicity exceeds uint16")
        counts[iteration] = value.astype(np.uint16)
    return counts


@njit(cache=False, parallel=True)
def binary_bootstrap_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    order: np.ndarray,
    tie_end: np.ndarray,
    row_cluster: np.ndarray,
    cluster_counts: np.ndarray,
) -> np.ndarray:
    iterations = cluster_counts.shape[0]
    n = y.shape[0]
    output = np.empty((iterations, 3), dtype=np.float64)
    for iteration in prange(iterations):
        total_positive = 0.0
        total_negative = 0.0
        true_positive_pred = 0.0
        false_positive = 0.0
        false_negative = 0.0
        true_negative = 0.0
        for row in range(n):
            weight = float(cluster_counts[iteration, row_cluster[row]])
            if y[row] == 1:
                total_positive += weight
                if pred[row] == 1:
                    true_positive_pred += weight
                else:
                    false_negative += weight
            else:
                total_negative += weight
                if pred[row] == 1:
                    false_positive += weight
                else:
                    true_negative += weight

        cumulative_weight = 0.0
        cumulative_positive = 0.0
        tie_positive = 0.0
        ap = 0.0
        if total_positive > 0.0 and total_negative > 0.0:
            for rank in range(n):
                row = order[rank]
                weight = float(cluster_counts[iteration, row_cluster[row]])
                cumulative_weight += weight
                if y[row] == 1:
                    cumulative_positive += weight
                    tie_positive += weight
                if tie_end[rank] == 1:
                    if tie_positive > 0.0 and cumulative_weight > 0.0:
                        ap += (tie_positive / total_positive) * (cumulative_positive / cumulative_weight)
                    tie_positive = 0.0
        else:
            ap = np.nan

        f1_denominator = 2.0 * true_positive_pred + false_positive + false_negative
        f1 = 2.0 * true_positive_pred / f1_denominator if f1_denominator > 0.0 else 0.0
        mcc_denominator = math.sqrt(
            (true_positive_pred + false_positive)
            * (true_positive_pred + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative)
        )
        mcc = (
            (true_positive_pred * true_negative - false_positive * false_negative) / mcc_denominator
            if mcc_denominator > 0.0
            else 0.0
        )
        output[iteration, 0] = ap
        output[iteration, 1] = f1
        output[iteration, 2] = mcc
    return output


def metric_distribution(
    labels: np.ndarray,
    scores: np.ndarray,
    row_cluster: np.ndarray,
    counts: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    pred_class = scores.argmax(axis=1)
    points: dict[str, dict[str, float]] = {}
    distributions: dict[str, dict[str, np.ndarray]] = {}
    for class_idx, name in enumerate(CLASSES):
        y = (labels == class_idx).astype(np.uint8)
        pred = (pred_class == class_idx).astype(np.uint8)
        order = np.argsort(-scores[:, class_idx], kind="mergesort").astype(np.int64)
        sorted_scores = scores[order, class_idx]
        tie_end = np.ones(len(order), dtype=np.uint8)
        if len(order) > 1:
            tie_end[:-1] = (sorted_scores[:-1] != sorted_scores[1:]).astype(np.uint8)
        values = binary_bootstrap_metrics(y, pred, order, tie_end, row_cluster, counts)
        points[name] = {
            "AUPR_ovr": float(average_precision_score(y, scores[:, class_idx])),
            "F1": float(f1_score(y, pred, zero_division=0)),
            "MCC_ovr": float(matthews_corrcoef(y, pred)),
        }
        distributions[name] = {
            "AUPR_ovr": values[:, 0],
            "F1": values[:, 1],
            "MCC_ovr": values[:, 2],
        }
    points["macro"] = {
        metric: float(np.mean([points[name][metric] for name in CLASSES]))
        for metric in ("AUPR_ovr", "F1", "MCC_ovr")
    }
    distributions["macro"] = {
        metric: np.nanmean(np.stack([distributions[name][metric] for name in CLASSES]), axis=0)
        for metric in ("AUPR_ovr", "F1", "MCC_ovr")
    }
    return points, distributions


def summarize_delta(
    baseline: str,
    scope: str,
    support: int,
    metric: str,
    main_point: float,
    baseline_point: float,
    delta: np.ndarray,
) -> dict[str, object]:
    finite = delta[np.isfinite(delta)]
    if not len(finite):
        return {
            "comparison": f"StructDualRM_vs_{baseline}",
            "baseline": baseline,
            "scope": scope,
            "support": support,
            "metric": metric,
            "StructDualRM": main_point,
            "baseline_value": baseline_point,
            "delta": main_point - baseline_point,
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "p_two_sided": float("nan"),
            "probability_delta_gt_zero": float("nan"),
        }
    less_equal = (np.count_nonzero(finite <= 0.0) + 1.0) / (len(finite) + 1.0)
    greater_equal = (np.count_nonzero(finite >= 0.0) + 1.0) / (len(finite) + 1.0)
    return {
        "comparison": f"StructDualRM_vs_{baseline}",
        "baseline": baseline,
        "scope": scope,
        "support": support,
        "metric": metric,
        "StructDualRM": main_point,
        "baseline_value": baseline_point,
        "delta": main_point - baseline_point,
        "bootstrap_mean_delta": float(finite.mean()),
        "ci95_low": float(np.percentile(finite, 2.5)),
        "ci95_high": float(np.percentile(finite, 97.5)),
        "p_two_sided": float(min(1.0, 2.0 * min(less_equal, greater_equal))),
        "probability_delta_gt_zero": float(np.mean(finite > 0.0)),
    }


def benjamini_hochberg(rows: list[dict[str, object]]) -> None:
    usable = [(idx, float(row["p_two_sided"])) for idx, row in enumerate(rows) if math.isfinite(float(row["p_two_sided"]))]
    usable.sort(key=lambda item: item[1])
    adjusted = [1.0] * len(usable)
    running = 1.0
    total = len(usable)
    for reverse_rank in range(total - 1, -1, -1):
        _, pvalue = usable[reverse_rank]
        running = min(running, pvalue * total / (reverse_rank + 1))
        adjusted[reverse_rank] = running
    for position, (row_idx, _) in enumerate(usable):
        rows[row_idx]["p_fdr_bh"] = adjusted[position]


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    model_specs: dict[str, Path] = {}
    for value in args.model:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--model must use name=prediction.tsv")
        model_specs[name] = Path(raw_path)
    if args.main_name not in model_specs:
        raise ValueError(f"--main-name {args.main_name!r} is absent from --model")

    reference_ids, labels, reference_scores = read_predictions(model_specs[args.main_name])
    group_by_id = read_dataset_groups(args.dataset, args.split_column, args.group_column)
    if set(reference_ids) != set(group_by_id):
        missing_predictions = set(group_by_id) - set(reference_ids)
        extra_predictions = set(reference_ids) - set(group_by_id)
        raise ValueError(
            f"Prediction/test split mismatch: missing={len(missing_predictions)} extra={len(extra_predictions)}"
        )
    cluster_names = sorted({group_by_id[site_id] for site_id in reference_ids})
    cluster_index = {name: idx for idx, name in enumerate(cluster_names)}
    row_cluster = np.asarray([cluster_index[group_by_id[site_id]] for site_id in reference_ids], dtype=np.int32)
    counts = bootstrap_cluster_counts(args.iterations, len(cluster_names), args.seed)

    scores_by_model = {args.main_name: reference_scores}
    for name, path in model_specs.items():
        if name != args.main_name:
            scores_by_model[name] = align_predictions(reference_ids, labels, path)

    points: dict[str, dict[str, dict[str, float]]] = {}
    distributions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name, scores in scores_by_model.items():
        points[name], distributions[name] = metric_distribution(labels, scores, row_cluster, counts)

    rows: list[dict[str, object]] = []
    for baseline in model_specs:
        if baseline == args.main_name:
            continue
        for scope in ["macro", *CLASSES]:
            support = len(labels) if scope == "macro" else int(np.count_nonzero(labels == CLASSES.index(scope)))
            for metric in ("AUPR_ovr", "F1", "MCC_ovr"):
                delta = distributions[args.main_name][scope][metric] - distributions[baseline][scope][metric]
                rows.append(
                    summarize_delta(
                        baseline,
                        scope,
                        support,
                        metric,
                        points[args.main_name][scope][metric],
                        points[baseline][scope][metric],
                        delta,
                    )
                )
    benjamini_hochberg(rows)
    for row in rows:
        row["significant_after_fdr"] = int(float(row.get("p_fdr_bh", 1.0)) < 0.05)
        row["direction"] = "higher" if float(row["delta"]) > 0 else "lower" if float(row["delta"]) < 0 else "tie"

    args.output.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output / "cluster_paired_bootstrap.tsv", rows)
    write_tsv(
        args.output / "cluster_paired_bootstrap_significant.tsv",
        [row for row in rows if int(row["significant_after_fdr"]) == 1],
    )
    np.savez_compressed(
        args.output / "bootstrap_design.npz",
        row_cluster=row_cluster,
        cluster_counts=counts,
    )
    provenance = {
        "dataset": str(args.dataset),
        "split_column": args.split_column,
        "group_column": args.group_column,
        "test_rows": len(labels),
        "test_clusters": len(cluster_names),
        "iterations": args.iterations,
        "seed": args.seed,
        "models": {name: str(path) for name, path in model_specs.items()},
        "resampling_unit": "complete 201-nt MMseqs2 cluster",
        "pairing": "identical cluster multiplicities for every model",
        "fdr": "Benjamini-Hochberg across all baseline, scope and metric comparisons",
    }
    (args.output / "bootstrap_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), **provenance}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="name=raw_ensemble_prediction.tsv")
    parser.add_argument("--main-name", default="StructDualRM")
    parser.add_argument("--split-column", default="split_cluster201")
    parser.add_argument("--group-column", default="cluster201_id")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
