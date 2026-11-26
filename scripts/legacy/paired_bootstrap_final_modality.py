#!/usr/bin/env python
"""Paired stratified bootstrap tests for final-model modality experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef


COMPARISONS = (
    ("structure_main", "sequence_structure", "sequence_only"),
    ("rbp_main", "sequence_rbp", "sequence_only"),
    ("joint_vs_sequence", "full", "sequence_only"),
    ("structure_given_rbp", "full", "sequence_rbp"),
    ("rbp_given_structure", "full", "sequence_structure"),
    ("real_vs_shuffled_structure", "full", "full_shuffled_structure"),
    ("real_vs_shuffled_rbp", "full", "full_shuffled_rbp"),
    ("pair_edges", "full", "full_no_pair_edges"),
    ("rbp_distance_bins", "full", "full_shuffled_rbp_bins"),
)


@dataclass(frozen=True)
class PredictionTable:
    site_ids: Tuple[str, ...]
    labels: np.ndarray
    class_names: Tuple[str, ...]
    probabilities: np.ndarray


def load_predictions(path: Path) -> PredictionTable:
    site_ids: List[str] = []
    labels: List[int] = []
    probability_rows: List[List[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header: {path}")
        probability_fields = [field for field in reader.fieldnames if field.startswith("prob_")]
        if not probability_fields or "site_id" not in reader.fieldnames or "true_class_index" not in reader.fieldnames:
            raise ValueError(f"Invalid prediction table: {path}")
        for row in reader:
            site_ids.append(str(row["site_id"]))
            labels.append(int(row["true_class_index"]))
            probability_rows.append([float(row[field]) for field in probability_fields])
    if len(site_ids) != len(set(site_ids)):
        raise ValueError(f"Duplicate site_id values in {path}")
    return PredictionTable(
        site_ids=tuple(site_ids),
        labels=np.asarray(labels, dtype=np.int16),
        class_names=tuple(field.removeprefix("prob_") for field in probability_fields),
        probabilities=np.asarray(probability_rows, dtype=np.float64),
    )


def align_to_reference(reference: PredictionTable, candidate: PredictionTable, path: Path) -> PredictionTable:
    if reference.class_names != candidate.class_names:
        raise ValueError(f"Class order differs in {path}")
    if reference.site_ids == candidate.site_ids:
        aligned = candidate
    else:
        candidate_index = {site_id: idx for idx, site_id in enumerate(candidate.site_ids)}
        missing = [site_id for site_id in reference.site_ids if site_id not in candidate_index]
        extra = set(candidate.site_ids).difference(reference.site_ids)
        if missing or extra:
            raise ValueError(
                f"Prediction site_id sets differ for {path}: missing={len(missing)}, extra={len(extra)}"
            )
        order = np.asarray([candidate_index[site_id] for site_id in reference.site_ids], dtype=np.int64)
        aligned = PredictionTable(
            site_ids=reference.site_ids,
            labels=candidate.labels[order],
            class_names=candidate.class_names,
            probabilities=candidate.probabilities[order],
        )
    if not np.array_equal(reference.labels, aligned.labels):
        raise ValueError(f"True labels differ after site_id alignment for {path}")
    return aligned


def stratified_bootstrap_weights(
    labels: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = np.zeros((iterations, labels.shape[0]), dtype=np.int16)
    for class_index in np.unique(labels):
        indices = np.flatnonzero(labels == class_index)
        draws = rng.integers(0, len(indices), size=(iterations, len(indices)))
        for iteration in range(iterations):
            weights[iteration, indices] = np.bincount(draws[iteration], minlength=len(indices))
    return weights


def weighted_average_precision_distribution(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.float64)
    sorted_weights = weights[:, order].astype(np.float64, copy=False)
    positive_weights = sorted_weights * sorted_labels[None, :]
    group_starts = np.concatenate(
        ([0], np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1)
    )
    if len(group_starts) != len(sorted_scores):
        sorted_weights = np.add.reduceat(sorted_weights, group_starts, axis=1)
        positive_weights = np.add.reduceat(positive_weights, group_starts, axis=1)
    cumulative_positive = np.cumsum(positive_weights, axis=1)
    cumulative_total = np.cumsum(sorted_weights, axis=1)
    total_positive = cumulative_positive[:, -1]
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total > 0,
    )
    return np.divide(
        np.sum(positive_weights * precision, axis=1),
        total_positive,
        out=np.full(total_positive.shape, np.nan, dtype=np.float64),
        where=total_positive > 0,
    )


def weighted_binary_metric_distribution(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    metric: str,
) -> np.ndarray:
    positive = labels.astype(bool)
    predicted_positive = predictions.astype(bool)
    integer_weights = weights.astype(np.int64, copy=False)
    tp = integer_weights @ (positive & predicted_positive).astype(np.int64)
    fp = integer_weights @ (~positive & predicted_positive).astype(np.int64)
    fn = integer_weights @ (positive & ~predicted_positive).astype(np.int64)
    tn = integer_weights @ (~positive & ~predicted_positive).astype(np.int64)
    if metric == "F1":
        denominator = 2 * tp + fp + fn
        return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=np.float64), where=denominator > 0)
    if metric == "MCC_ovr":
        numerator = tp * tn - fp * fn
        denominator = np.sqrt(
            (tp + fp).astype(np.float64)
            * (tp + fn).astype(np.float64)
            * (tn + fp).astype(np.float64)
            * (tn + fn).astype(np.float64)
        )
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator > 0,
        )
    raise ValueError(f"Unsupported metric: {metric}")


def point_metrics(table: PredictionTable, class_index: int) -> Dict[str, float]:
    binary_labels = (table.labels == class_index).astype(np.int8)
    predicted = (table.probabilities.argmax(axis=1) == class_index).astype(np.int8)
    if binary_labels.sum() == 0:
        aupr = float("nan")
    else:
        aupr = float(average_precision_score(binary_labels, table.probabilities[:, class_index]))
    return {
        "AUPR_ovr": aupr,
        "F1": float(f1_score(binary_labels, predicted, zero_division=0)),
        "MCC_ovr": float(matthews_corrcoef(binary_labels, predicted)),
    }


def summarize_delta(
    comparison: str,
    model_a: str,
    model_b: str,
    scope: str,
    support: int,
    metric: str,
    point_a: float,
    point_b: float,
    differences: np.ndarray,
) -> Dict[str, object]:
    finite = differences[np.isfinite(differences)]
    if finite.size == 0:
        low = high = probability = p_value = float("nan")
    else:
        low, high = np.quantile(finite, [0.025, 0.975])
        probability = float((np.count_nonzero(finite > 0) + 1) / (finite.size + 1))
        lower_tail = (np.count_nonzero(finite <= 0) + 1) / (finite.size + 1)
        upper_tail = (np.count_nonzero(finite >= 0) + 1) / (finite.size + 1)
        p_value = float(min(1.0, 2.0 * min(lower_tail, upper_tail)))
    return {
        "comparison": comparison,
        "model_a": model_a,
        "model_b": model_b,
        "scope": scope,
        "support": support,
        "metric": metric,
        "model_a_point": point_a,
        "model_b_point": point_b,
        "delta_a_minus_b": point_a - point_b,
        "bootstrap_delta_mean": float(np.nanmean(finite)) if finite.size else float("nan"),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_delta_gt_zero": probability,
        "p_two_sided": p_value,
        "p_fdr_bh": float("nan"),
    }


def benjamini_hochberg(rows: List[Dict[str, object]]) -> None:
    for metric in sorted({str(row["metric"]) for row in rows}):
        indices = [
            idx
            for idx, row in enumerate(rows)
            if row["metric"] == metric and np.isfinite(float(row["p_two_sided"]))
        ]
        if not indices:
            continue
        ordered = sorted(indices, key=lambda idx: float(rows[idx]["p_two_sided"]))
        adjusted = np.ones(len(ordered), dtype=np.float64)
        running = 1.0
        total = len(ordered)
        for rank_index in range(total - 1, -1, -1):
            row_index = ordered[rank_index]
            rank = rank_index + 1
            running = min(running, float(rows[row_index]["p_two_sided"]) * total / rank)
            adjusted[rank_index] = running
        for rank_index, row_index in enumerate(ordered):
            rows[row_index]["p_fdr_bh"] = float(adjusted[rank_index])


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_comparison(
    comparison: str,
    model_a_name: str,
    model_a: PredictionTable,
    model_b_name: str,
    model_b: PredictionTable,
    iterations: int,
    block_size: int,
    seed: int,
) -> List[Dict[str, object]]:
    class_count = len(model_a.class_names)
    point_a = [point_metrics(model_a, class_index) for class_index in range(class_count)]
    point_b = [point_metrics(model_b, class_index) for class_index in range(class_count)]
    distributions_a = {
        metric: np.full((class_count, iterations), np.nan, dtype=np.float64)
        for metric in ("AUPR_ovr", "F1", "MCC_ovr")
    }
    distributions_b = {
        metric: np.full((class_count, iterations), np.nan, dtype=np.float64)
        for metric in ("AUPR_ovr", "F1", "MCC_ovr")
    }
    rng = np.random.default_rng(seed)
    predicted_a = model_a.probabilities.argmax(axis=1)
    predicted_b = model_b.probabilities.argmax(axis=1)
    for start in range(0, iterations, block_size):
        stop = min(iterations, start + block_size)
        weights = stratified_bootstrap_weights(model_a.labels, stop - start, rng)
        for class_index in range(class_count):
            binary_labels = (model_a.labels == class_index).astype(np.int8)
            if binary_labels.sum() > 0:
                distributions_a["AUPR_ovr"][class_index, start:stop] = weighted_average_precision_distribution(
                    binary_labels, model_a.probabilities[:, class_index], weights
                )
                distributions_b["AUPR_ovr"][class_index, start:stop] = weighted_average_precision_distribution(
                    binary_labels, model_b.probabilities[:, class_index], weights
                )
            distributions_a["F1"][class_index, start:stop] = weighted_binary_metric_distribution(
                binary_labels, predicted_a == class_index, weights, "F1"
            )
            distributions_b["F1"][class_index, start:stop] = weighted_binary_metric_distribution(
                binary_labels, predicted_b == class_index, weights, "F1"
            )
            distributions_a["MCC_ovr"][class_index, start:stop] = weighted_binary_metric_distribution(
                binary_labels, predicted_a == class_index, weights, "MCC_ovr"
            )
            distributions_b["MCC_ovr"][class_index, start:stop] = weighted_binary_metric_distribution(
                binary_labels, predicted_b == class_index, weights, "MCC_ovr"
            )

    rows: List[Dict[str, object]] = []
    evaluable = np.asarray(
        [np.count_nonzero(model_a.labels == class_index) > 0 for class_index in range(class_count)],
        dtype=bool,
    )
    for metric in ("AUPR_ovr", "F1", "MCC_ovr"):
        macro_a = np.nanmean(distributions_a[metric][evaluable], axis=0)
        macro_b = np.nanmean(distributions_b[metric][evaluable], axis=0)
        macro_point_a = float(np.nanmean([values[metric] for idx, values in enumerate(point_a) if evaluable[idx]]))
        macro_point_b = float(np.nanmean([values[metric] for idx, values in enumerate(point_b) if evaluable[idx]]))
        rows.append(
            summarize_delta(
                comparison,
                model_a_name,
                model_b_name,
                "macro_evaluable_classes",
                int(evaluable.sum()),
                metric,
                macro_point_a,
                macro_point_b,
                macro_a - macro_b,
            )
        )
        for class_index, class_name in enumerate(model_a.class_names):
            support = int(np.count_nonzero(model_a.labels == class_index))
            if support == 0:
                continue
            rows.append(
                summarize_delta(
                    comparison,
                    model_a_name,
                    model_b_name,
                    class_name,
                    support,
                    metric,
                    point_a[class_index][metric],
                    point_b[class_index][metric],
                    distributions_a[metric][class_index] - distributions_b[metric][class_index],
                )
            )
    return rows


def run(run_root: Path, full_model_dir: Path, out_dir: Path, iterations: int, block_size: int, seed: int) -> None:
    paths = {"full": full_model_dir / "test_predictions.tsv"}
    paths.update({name: run_root / name / "test_predictions.tsv" for name in {
        model_name for _, model_a, model_b in COMPARISONS for model_name in (model_a, model_b) if model_name != "full"
    }})
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} predictions: {path}")
    reference = load_predictions(paths["full"])
    tables = {"full": reference}
    for name, path in paths.items():
        if name != "full":
            tables[name] = align_to_reference(reference, load_predictions(path), path)

    rows: List[Dict[str, object]] = []
    for comparison_index, (comparison, model_a, model_b) in enumerate(COMPARISONS):
        rows.extend(
            bootstrap_comparison(
                comparison,
                model_a,
                tables[model_a],
                model_b,
                tables[model_b],
                iterations,
                block_size,
                seed + comparison_index * 1009,
            )
        )
    benjamini_hochberg(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "paired_bootstrap_modality.tsv", rows)
    significant = [
        row for row in rows
        if row["metric"] == "AUPR_ovr"
        and float(row["delta_a_minus_b"]) > 0
        and float(row["ci95_low"]) > 0
        and float(row["p_fdr_bh"]) < 0.05
    ]
    write_tsv(out_dir / "paired_bootstrap_positive_aupr_evidence.tsv", significant)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--full-model-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20250726)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.iterations < 50:
        raise ValueError("Use at least 50 bootstrap iterations.")
    if args.block_size < 1:
        raise ValueError("--block-size must be positive.")
    run(args.run_root, args.full_model_dir, args.out_dir, args.iterations, args.block_size, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
