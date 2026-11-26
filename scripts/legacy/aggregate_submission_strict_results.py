#!/usr/bin/env python
"""Aggregate the frozen five-seed StructDualRM submission experiments.

All calibration parameters are fitted on validation predictions. Test labels
are used only once to report frozen performance. The primary confirmatory task
contains 11 mutually exclusive classes; the 12-label coordinate-collapsed task
is summarized separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


CLASSES = ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m7G", "Y", "AtoI"]
SEEDS = [42, 17, 73, 101, 202]
FAMILIES = {
    "A": ["Am", "m1A", "m6A", "AtoI"],
    "C": ["Cm", "m5C"],
    "G": ["Gm", "m7G"],
    "U": ["Um", "m5U", "Y"],
}
CLASS_TO_BASE = {name: base for base, names in FAMILIES.items() for name in names}
MODEL_RUNS = {
    "StructDualRM": ("main_controls/full", "main"),
    "MultiRM": ("baselines/multirm_official", "baseline"),
    "TransRNAm": ("baselines/transrnam_official", "baseline"),
    "EvoRMD": ("baselines/evormd_safe", "baseline"),
    "RNA-FM sequence-only": ("baselines/rnafm_sequence_only", "modern_baseline"),
}
CONTROL_VARIANTS = [
    "sequence_only",
    "sequence_structure",
    "sequence_rbp",
    "full",
    "shuffled_structure",
    "shuffled_rbp",
    "shuffled_rbp_bins",
    "explicit_pair_edges",
    "capacity_matched_structure_cnn",
    "parameter_matched_rbp_mlp",
]
MACRO_KEYS = [
    "macro_AUPR_ovr",
    "macro_AUC_ovr",
    "macro_F1",
    "weighted_F1",
    "MCC",
    "balanced_accuracy",
    "accuracy",
]
PER_CLASS_KEYS = ["AUPR_ovr", "AUC_ovr", "F1", "Precision", "Recall", "MCC_ovr"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def mean_sd(values: Iterable[object]) -> tuple[float, float]:
    array = np.asarray([finite(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=np.float64), 1e-12, None)
    totals = clipped.sum(axis=1, keepdims=True)
    return clipped / np.clip(totals, 1e-12, None)


def read_predictions(path: Path, classes: Sequence[str] = CLASSES) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids: list[str] = []
    labels: list[int] = []
    scores: list[list[float]] = []
    index = {name: idx for idx, name in enumerate(classes)}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"site_id", *[f"prob_{name}" for name in classes]}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing prediction columns in {path}: {sorted(missing)}")
        for row in reader:
            ids.append(str(row["site_id"]))
            if row.get("true_class_index", "") != "":
                labels.append(int(row["true_class_index"]))
            elif row.get("true_mod_type", "") in index:
                labels.append(index[str(row["true_mod_type"])])
            else:
                raise ValueError(f"No usable true label in {path}")
            scores.append([float(row[f"prob_{name}"]) for name in classes])
    return ids, np.asarray(labels, dtype=np.int16), np.asarray(scores, dtype=np.float64)


def load_ensemble(root: Path, split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    reference_ids: list[str] | None = None
    reference_labels: np.ndarray | None = None
    members: list[np.ndarray] = []
    for seed in SEEDS:
        path = root / f"seed{seed}" / f"{split}_predictions.tsv"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        ids, labels, scores = read_predictions(path)
        if reference_ids is None:
            reference_ids, reference_labels = ids, labels
        elif ids != reference_ids or not np.array_equal(labels, reference_labels):
            raise ValueError(f"Seed predictions are not row-aligned: {path}")
        # Preserve independent sigmoid scores from the MultiRM/TransRNAm
        # reproductions. Row normalization changes one-vs-rest ranking and
        # therefore changes AUPR/AUC. Normalization is applied only by the
        # calibration functions below.
        members.append(scores)
    assert reference_ids is not None and reference_labels is not None
    return reference_ids, reference_labels, np.mean(np.stack(members), axis=0)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(normalize_scores(probs), 1e-12, 1.0)) / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def fit_temperature(labels: np.ndarray, probs: np.ndarray) -> float:
    grid = np.geomspace(0.05, 10.0, 401)
    losses = np.asarray([nll(labels, apply_temperature(probs, value)) for value in grid])
    best = int(losses.argmin())
    low = math.log(grid[max(0, best - 1)])
    high = math.log(grid[min(len(grid) - 1, best + 1)])
    for _ in range(40):
        left = (2 * low + high) / 3
        right = (low + 2 * high) / 3
        if nll(labels, apply_temperature(probs, math.exp(left))) <= nll(
            labels, apply_temperature(probs, math.exp(right))
        ):
            high = right
        else:
            low = left
    return float(math.exp((low + high) / 2))


def calibration_metrics(labels: np.ndarray, probs: np.ndarray, bins: int = 15) -> tuple[dict[str, float], list[dict[str, object]]]:
    probs = normalize_scores(probs)
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (pred == labels).astype(np.float64)
    onehot = np.eye(probs.shape[1], dtype=np.float64)[labels]
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    rows: list[dict[str, object]] = []
    for idx in range(bins):
        mask = (confidence >= edges[idx]) & (
            confidence <= edges[idx + 1] if idx == bins - 1 else confidence < edges[idx + 1]
        )
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else float("nan")
        empirical_accuracy = float(correct[mask].mean()) if count else float("nan")
        if count:
            ece += count / len(labels) * abs(mean_confidence - empirical_accuracy)
        rows.append(
            {
                "bin": idx + 1,
                "lower": edges[idx],
                "upper": edges[idx + 1],
                "count": count,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "absolute_gap": abs(mean_confidence - empirical_accuracy) if count else float("nan"),
            }
        )
    return {
        "ECE_15": float(ece),
        "Brier_multiclass": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "NLL": nll(labels, probs),
    }, rows


def safe_binary_metric(function, y: np.ndarray, value: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(function(y, value))
    except ValueError:
        return float("nan")


def evaluate(labels: np.ndarray, probs: np.ndarray, classes: Sequence[str] = CLASSES) -> dict:
    scores = np.asarray(probs, dtype=np.float64)
    pred = scores.argmax(axis=1)
    per_class: dict[str, dict[str, float | int]] = {}
    for idx, name in enumerate(classes):
        y = (labels == idx).astype(np.int8)
        q = (pred == idx).astype(np.int8)
        per_class[name] = {
            "support": int(y.sum()),
            "AUPR_ovr": safe_binary_metric(average_precision_score, y, scores[:, idx]),
            "AUC_ovr": safe_binary_metric(roc_auc_score, y, scores[:, idx]),
            "F1": float(f1_score(y, q, zero_division=0)),
            "Precision": float(precision_score(y, q, zero_division=0)),
            "Recall": float(recall_score(y, q, zero_division=0)),
            "MCC_ovr": float(matthews_corrcoef(y, q)) if len(np.unique(y)) > 1 else float("nan"),
        }
    return {
        "support": int(len(labels)),
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_F1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "weighted_F1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, pred)),
        "macro_AUPR_ovr": float(np.nanmean([per_class[name]["AUPR_ovr"] for name in classes])),
        "macro_AUC_ovr": float(np.nanmean([per_class[name]["AUC_ovr"] for name in classes])),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, pred, labels=list(range(len(classes)))).tolist(),
    }


def restrict_macro_summary(metrics: Mapping[str, object], supported_classes: Sequence[str]) -> dict:
    """Recompute macro summaries over classes represented in a lockbox.

    Predictions remain an unconstrained 11-class argmax, so assigning a sample
    to an unsupported class is still counted as an error. Only the averaging
    set is restricted; this prevents absent classes from entering macro F1 as
    artificial zeros.
    """
    supported = [name for name in supported_classes if name in CLASSES]
    if not supported:
        raise ValueError("At least one supported class is required")
    per_class = metrics["per_class"]
    assert isinstance(per_class, Mapping)
    support = np.asarray([float(per_class[name]["support"]) for name in supported], dtype=np.float64)  # type: ignore[index]
    f1_values = np.asarray([finite(per_class[name]["F1"]) for name in supported], dtype=np.float64)  # type: ignore[index]
    output = dict(metrics)
    output["macro_F1"] = float(np.nanmean(f1_values))
    output["weighted_F1"] = float(np.nansum(f1_values * support) / max(float(support.sum()), 1.0))
    output["balanced_accuracy"] = float(
        np.nanmean([finite(per_class[name]["Recall"]) for name in supported])  # type: ignore[index]
    )
    output["macro_AUPR_ovr"] = float(
        np.nanmean([finite(per_class[name]["AUPR_ovr"]) for name in supported])  # type: ignore[index]
    )
    output["macro_AUC_ovr"] = float(
        np.nanmean([finite(per_class[name]["AUC_ovr"]) for name in supported])  # type: ignore[index]
    )
    output["macro_averaging_classes"] = supported
    output["n_macro_classes"] = len(supported)
    return output


def family_metrics(labels: np.ndarray, probs: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for base, names in FAMILIES.items():
        indices = np.asarray([CLASSES.index(name) for name in names], dtype=int)
        mask = np.isin(labels, indices)
        local_labels = np.asarray([int(np.where(indices == value)[0][0]) for value in labels[mask]], dtype=int)
        local_probs = normalize_scores(probs[mask][:, indices])
        local_metrics = evaluate(local_labels, local_probs, names)
        metric_rows.append(
            {
                "family": base,
                "classes": ",".join(names),
                "support": int(mask.sum()),
                **{key: local_metrics[key] for key in MACRO_KEYS},
            }
        )
        matrix = np.asarray(local_metrics["confusion_matrix"], dtype=int)
        for true_idx, true_name in enumerate(names):
            for pred_idx, pred_name in enumerate(names):
                confusion_rows.append(
                    {
                        "family": base,
                        "true_class": true_name,
                        "predicted_class": pred_name,
                        "count": int(matrix[true_idx, pred_idx]),
                        "row_fraction": float(matrix[true_idx, pred_idx] / max(1, matrix[true_idx].sum())),
                    }
                )
    return metric_rows, confusion_rows


def load_dataset_rows(path: Path, needed_splits: Sequence[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    rows: dict[str, dict[str, str]] = {}
    fields: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        required = {"site_id", "mod_type", "center_base", *needed_splits}
        missing = required - set(fields)
        if missing:
            raise ValueError(f"Dataset lacks required fields: {sorted(missing)}")
        for row in reader:
            rows[str(row["site_id"])] = row
    return rows, fields


def center_base_predictions(
    dataset_rows: Mapping[str, Mapping[str, str]],
    ids: Sequence[str],
    train_split: str,
) -> np.ndarray:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in dataset_rows.values():
        if row.get(train_split) == "train":
            counts[str(row["center_base"]).upper().replace("T", "U")][str(row["mod_type"])] += 1
    output = np.zeros((len(ids), len(CLASSES)), dtype=np.float64)
    for row_idx, site_id in enumerate(ids):
        base = str(dataset_rows[site_id]["center_base"]).upper().replace("T", "U")
        family = FAMILIES.get(base, [])
        denominator = sum(counts[base].get(name, 0) + 1 for name in family)
        for name in family:
            output[row_idx, CLASSES.index(name)] = (counts[base].get(name, 0) + 1) / max(denominator, 1)
    return normalize_scores(output)


def flatten_metrics(model: str, scope: str, metrics: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for key in MACRO_KEYS:
        rows.append({"model": model, "scope": scope, "metric": key, "value": metrics.get(key, float("nan"))})
    return rows


def metric_rows_by_class(model: str, calibration: str, metrics: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    per_class = metrics["per_class"]
    assert isinstance(per_class, Mapping)
    for name in CLASSES:
        values = per_class[name]
        assert isinstance(values, Mapping)
        rows.append({"model": model, "calibration": calibration, "mod_type": name, **values})
    return rows


def aggregate_seed_metrics(run_root: Path, model: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seed_rows: list[dict[str, object]] = []
    metrics_by_seed: list[dict] = []
    for seed in SEEDS:
        path = run_root / f"seed{seed}" / "metrics.json"
        metrics = read_json(path)
        metrics_by_seed.append(metrics)
        test = metrics["test"]
        seed_rows.append(
            {
                "model": model,
                "seed": seed,
                "best_epoch": metrics.get("best_epoch", ""),
                **{f"test_{key}": test.get(key, float("nan")) for key in MACRO_KEYS},
            }
        )
    summary = []
    for key in MACRO_KEYS:
        mean, sd = mean_sd(item["test"].get(key) for item in metrics_by_seed)
        summary.append({"model": model, "metric": key, "mean": mean, "sd": sd, "n_seeds": len(SEEDS)})
    return seed_rows, summary


def aggregate_model_per_class(run_root: Path, model: str) -> list[dict[str, object]]:
    loaded = [read_json(run_root / f"seed{seed}" / "metrics.json") for seed in SEEDS]
    rows: list[dict[str, object]] = []
    for name in CLASSES:
        for metric in PER_CLASS_KEYS:
            mean, sd = mean_sd(item["test"]["per_class"][name].get(metric) for item in loaded)
            rows.append(
                {
                    "model": model,
                    "mod_type": name,
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "n_seeds": len(loaded),
                }
            )
    return rows


def aggregate_control_metrics(run_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    for variant in CONTROL_VARIANTS:
        loaded = [read_json(run_root / variant / f"seed{seed}" / "metrics.json") for seed in SEEDS]
        for split in ("validation", "test"):
            for metric in MACRO_KEYS:
                mean, sd = mean_sd(item[split].get(metric) for item in loaded)
                summary.append(
                    {"variant": variant, "split": split, "metric": metric, "mean": mean, "sd": sd, "n_seeds": 5}
                )
        for name in CLASSES:
            for metric in PER_CLASS_KEYS:
                mean, sd = mean_sd(item["test"]["per_class"][name].get(metric) for item in loaded)
                per_class_rows.append(
                    {"variant": variant, "mod_type": name, "metric": metric, "mean": mean, "sd": sd, "n_seeds": 5}
                )
    return summary, per_class_rows


def paired_control_effects(run_root: Path) -> list[dict[str, object]]:
    loaded = {
        variant: {seed: read_json(run_root / variant / f"seed{seed}" / "metrics.json") for seed in SEEDS}
        for variant in CONTROL_VARIANTS
    }
    contrasts = {
        "structure_without_rbp": {"sequence_structure": 1.0, "sequence_only": -1.0},
        "structure_with_rbp": {"full": 1.0, "sequence_rbp": -1.0},
        "rbp_without_structure": {"sequence_rbp": 1.0, "sequence_only": -1.0},
        "rbp_with_structure": {"full": 1.0, "sequence_structure": -1.0},
        "structure_rbp_interaction": {
            "full": 1.0,
            "sequence_structure": -1.0,
            "sequence_rbp": -1.0,
            "sequence_only": 1.0,
        },
        "gat_vs_capacity_matched_cnn": {"full": 1.0, "capacity_matched_structure_cnn": -1.0},
        "rbp_attention_vs_parameter_matched_mlp": {"full": 1.0, "parameter_matched_rbp_mlp": -1.0},
        "observed_vs_shuffled_structure": {"full": 1.0, "shuffled_structure": -1.0},
        "observed_vs_shuffled_rbp_samples": {"full": 1.0, "shuffled_rbp": -1.0},
        "ordered_vs_shuffled_rbp_bins": {"full": 1.0, "shuffled_rbp_bins": -1.0},
        "no_pair_edges_vs_explicit_pair_edges": {"full": 1.0, "explicit_pair_edges": -1.0},
    }
    rows: list[dict[str, object]] = []
    scopes = [("macro", "", metric) for metric in MACRO_KEYS]
    scopes.extend(("class", name, metric) for name in CLASSES for metric in PER_CLASS_KEYS)
    for contrast, coefficients in contrasts.items():
        for scope, name, metric in scopes:
            values = []
            for seed in SEEDS:
                value = 0.0
                for variant, coefficient in coefficients.items():
                    test = loaded[variant][seed]["test"]
                    source = test if scope == "macro" else test["per_class"][name]
                    value += coefficient * finite(source.get(metric))
                values.append(value)
            mean, sd = mean_sd(values)
            rows.append(
                {
                    "contrast": contrast,
                    "scope": scope,
                    "mod_type": name,
                    "metric": metric,
                    "mean_paired_delta": mean,
                    "sd_paired_delta": sd,
                    "n_seeds": len(SEEDS),
                    "positive_seeds": sum(value > 0 for value in values if math.isfinite(value)),
                    "coefficients": ";".join(f"{variant}:{coefficient:+g}" for variant, coefficient in coefficients.items()),
                }
            )
    return rows


def efficiency_rows(run_roots: Mapping[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys = [
        "parameter_count",
        "trainable_parameter_count",
        "flops_per_sample",
        "training_seconds",
        "training_peak_memory_bytes",
        "samples_per_second",
        "latency_ms_per_sample",
    ]
    for model, root in run_roots.items():
        loaded = [read_json(root / f"seed{seed}" / "metrics.json").get("efficiency", {}) for seed in SEEDS]
        for key in keys:
            values = []
            for item in loaded:
                if not isinstance(item, Mapping):
                    continue
                value = item.get(key)
                if value is None and key == "latency_ms_per_sample":
                    value = item.get("milliseconds_per_sample")
                values.append(value)
            mean, sd = mean_sd(values)
            rows.append({"model": model, "metric": key, "mean": mean, "sd": sd, "n_seeds": 5})
    return rows


def aggregate_holdouts(run_root: Path, output: Path) -> None:
    rows: list[dict[str, object]] = []
    for label in ("gene_holdout", "transcript_holdout", "study_holdout"):
        loaded: list[dict] = []
        valid_supported: set[str] | None = None
        test_supported: set[str] | None = None
        for seed in SEEDS:
            root = run_root / label / f"seed{seed}"
            _, valid_labels, _ = read_predictions(root / "valid_predictions.tsv")
            _, test_labels, test_probs = read_predictions(root / "test_predictions.tsv")
            current_valid = {CLASSES[idx] for idx in np.unique(valid_labels)}
            current_test = {CLASSES[idx] for idx in np.unique(test_labels)}
            valid_supported = current_valid if valid_supported is None else valid_supported & current_valid
            test_supported = current_test if test_supported is None else test_supported & current_test
            loaded.append(restrict_macro_summary(evaluate(test_labels, test_probs), sorted(current_valid & current_test)))
        supported = [name for name in CLASSES if name in (valid_supported or set()) and name in (test_supported or set())]
        for metric in MACRO_KEYS:
            mean, sd = mean_sd(item.get(metric) for item in loaded)
            rows.append(
                {
                    "holdout": label,
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "n_seeds": len(loaded),
                    "supported_classes": ",".join(supported),
                    "n_supported_classes": len(supported),
                    "aggregation_scope": "validation_and_test_supported_classes",
                }
            )
    write_tsv(output, rows)


def aggregate_multilabel(run_root: Path, output: Path) -> None:
    loaded = [read_json(run_root / f"seed{seed}" / "metrics.json") for seed in SEEDS]
    rows: list[dict[str, object]] = []
    keys = [
        "macro_AUPR",
        "macro_AUC",
        "macro_F1",
        "micro_F1",
        "macro_precision",
        "macro_recall",
        "macro_jaccard",
        "subset_accuracy",
        "hamming_loss",
        "mean_true_cardinality",
        "mean_predicted_cardinality",
    ]
    for key in keys:
        mean, sd = mean_sd(item["test"].get(key) for item in loaded)
        rows.append({"scope": "coordinate_multilabel_12", "metric": key, "mean": mean, "sd": sd, "n_seeds": 5})
    for name in ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "Y", "AtoI"]:
        for key in ("AUPR", "AUC", "F1", "Precision", "Recall", "MCC"):
            mean, sd = mean_sd(item["test"]["per_class"][name].get(key) for item in loaded)
            rows.append(
                {
                    "scope": f"class:{name}",
                    "metric": key,
                    "mean": mean,
                    "sd": sd,
                    "n_seeds": 5,
                }
            )
    write_tsv(output, rows)


def build_comparison(per_model: Mapping[str, Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    baseline_names = [name for name in MODEL_RUNS if name != "StructDualRM"]
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for metric in PER_CLASS_KEYS:
        wins = ties = losses = 0
        for name in CLASSES:
            main = finite(per_model["StructDualRM"]["per_class"][name][metric])  # type: ignore[index]
            values = {
                model: finite(per_model[model]["per_class"][name][metric])  # type: ignore[index]
                for model in baseline_names
            }
            usable = {model: value for model, value in values.items() if math.isfinite(value)}
            best_model = max(usable, key=usable.get) if usable else ""
            best = usable.get(best_model, float("nan"))
            difference = main - best if math.isfinite(main) and math.isfinite(best) else float("nan")
            status = "tie" if math.isfinite(difference) and abs(difference) <= 1e-12 else "win" if difference > 0 else "loss"
            wins += status == "win"
            ties += status == "tie"
            losses += status == "loss"
            rows.append(
                {
                    "mod_type": name,
                    "support": per_model["StructDualRM"]["per_class"][name]["support"],  # type: ignore[index]
                    "metric": metric,
                    "StructDualRM": main,
                    **values,
                    "best_baseline": best_model,
                    "best_baseline_value": best,
                    "difference": difference,
                    "status": status,
                }
            )
        summaries.append({"metric": metric, "wins": wins, "ties": ties, "losses": losses, "classes": len(CLASSES)})
    return rows, summaries


def plot_outputs(output: Path, calibrated: Mapping[str, Mapping[str, object]], reliability: Sequence[Mapping[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    sns.set_theme(style="whitegrid", context="paper")
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(CLASSES))
    width = 0.16
    for offset, model in enumerate(MODEL_RUNS):
        values = [calibrated[model]["per_class"][name]["AUPR_ovr"] for name in CLASSES]  # type: ignore[index]
        ax.bar(x + (offset - 2) * width, values, width=width, label=model)
    ax.set_xticks(x, CLASSES, rotation=35, ha="right")
    ax.set_ylabel("AUPR")
    ax.set_ylim(0, 1.03)
    ax.legend(ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "per_class_aupr_models.png", dpi=320)
    plt.close(fig)

    full_matrix = np.asarray(calibrated["StructDualRM"]["confusion_matrix"], dtype=float)
    normalized = full_matrix / np.clip(full_matrix.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    sns.heatmap(normalized, cmap="mako", vmin=0, vmax=1, xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(figure_dir / "structdualrm_confusion_matrix.png", dpi=320)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    for model in MODEL_RUNS:
        rows = [row for row in reliability if row["model"] == model and row["calibration"] == "temperature_scaled"]
        rows = [row for row in rows if int(row["count"]) > 0]
        ax.plot(
            [float(row["mean_confidence"]) for row in rows],
            [float(row["empirical_accuracy"]) for row in rows],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=model,
        )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Mean confidence", ylabel="Empirical accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "reliability_curves.png", dpi=320)
    plt.close(fig)


def sha256_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.tsv":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest})
    write_tsv(root / "sha256_manifest.tsv", rows, ["relative_path", "bytes", "sha256"])


def run(args: argparse.Namespace) -> None:
    run_root = args.run_root
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    dataset_rows, _ = load_dataset_rows(
        args.dataset,
        ["split_cluster201", "split_gene_holdout", "split_transcript_holdout", "split_study_holdout"],
    )

    ensemble_payload: dict[str, dict[str, object]] = {}
    calibration_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    macro_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    family_confusion_rows: list[dict[str, object]] = []
    ensemble_predictions: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}

    for model, (relative, _) in MODEL_RUNS.items():
        root = run_root / relative
        valid_ids, valid_labels, valid_probs = load_ensemble(root, "valid")
        test_ids, test_labels, test_probs = load_ensemble(root, "test")
        if model == "StructDualRM":
            center_test = center_base_predictions(dataset_rows, test_ids, "split_cluster201")
        temperature = fit_temperature(valid_labels, valid_probs)
        calibrated_test = apply_temperature(test_probs, temperature)
        raw_metrics = evaluate(test_labels, test_probs)
        calibrated_metrics = evaluate(test_labels, calibrated_test)
        raw_calibration, raw_reliability = calibration_metrics(test_labels, test_probs)
        scaled_calibration, scaled_reliability = calibration_metrics(test_labels, calibrated_test)
        ensemble_payload[model] = {
            "temperature_selected_on_validation": temperature,
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
            "raw_calibration": raw_calibration,
            "calibrated_calibration": scaled_calibration,
        }
        ensemble_predictions[model] = (test_ids, test_labels, test_probs)
        macro_rows.extend(flatten_metrics(model, "raw_ensemble", raw_metrics))
        macro_rows.extend(flatten_metrics(model, "temperature_scaled_ensemble", calibrated_metrics))
        per_class_rows.extend(metric_rows_by_class(model, "raw", raw_metrics))
        per_class_rows.extend(metric_rows_by_class(model, "temperature_scaled", calibrated_metrics))
        for calibration_name, metrics_value in (("raw", raw_calibration), ("temperature_scaled", scaled_calibration)):
            calibration_rows.append(
                {
                    "model": model,
                    "calibration": calibration_name,
                    "temperature": 1.0 if calibration_name == "raw" else temperature,
                    **metrics_value,
                }
            )
        for calibration_name, rows_value in (("raw", raw_reliability), ("temperature_scaled", scaled_reliability)):
            reliability_rows.extend({"model": model, "calibration": calibration_name, **row} for row in rows_value)
        for score_type, family_scores in (("raw", test_probs), ("temperature_scaled", calibrated_test)):
            metrics_value, confusion_value = family_metrics(test_labels, family_scores)
            family_rows.extend({"model": model, "score_type": score_type, **row} for row in metrics_value)
            family_confusion_rows.extend(
                {"model": model, "score_type": score_type, **row} for row in confusion_value
            )

    main_ids, main_labels, _ = ensemble_predictions["StructDualRM"]
    center_test_metrics = evaluate(main_labels, center_test)
    center_test_calibration, center_reliability = calibration_metrics(main_labels, center_test)
    ensemble_payload["Center-base only"] = {
        "definition": "Laplace-smoothed training class priors conditional only on the central nucleotide",
        "calibrated": center_test_metrics,
        "calibrated_calibration": center_test_calibration,
    }
    macro_rows.extend(flatten_metrics("Center-base only", "training_prior", center_test_metrics))
    per_class_rows.extend(metric_rows_by_class("Center-base only", "training_prior", center_test_metrics))
    calibration_rows.append(
        {"model": "Center-base only", "calibration": "training_prior", "temperature": 1.0, **center_test_calibration}
    )
    reliability_rows.extend({"model": "Center-base only", "calibration": "training_prior", **row} for row in center_reliability)
    center_family, center_confusion = family_metrics(main_labels, center_test)
    family_rows.extend({"model": "Center-base only", "score_type": "training_prior", **row} for row in center_family)
    family_confusion_rows.extend(
        {"model": "Center-base only", "score_type": "training_prior", **row} for row in center_confusion
    )

    # Ranking/discrimination are primary on unmodified ensemble scores.
    # Temperature scaling is reported only for probabilistic calibration.
    primary_metrics = {name: payload["raw"] for name, payload in ensemble_payload.items() if name in MODEL_RUNS}
    comparison_rows, win_rows = build_comparison(primary_metrics)

    seed_rows: list[dict[str, object]] = []
    seed_summary: list[dict[str, object]] = []
    seed_per_class_summary: list[dict[str, object]] = []
    run_roots: dict[str, Path] = {}
    for model, (relative, _) in MODEL_RUNS.items():
        root = run_root / relative
        run_roots[model] = root
        individual, summary = aggregate_seed_metrics(root, model)
        seed_rows.extend(individual)
        seed_summary.extend(summary)
        seed_per_class_summary.extend(aggregate_model_per_class(root, model))

    control_summary, control_per_class = aggregate_control_metrics(run_root / "main_controls")
    aggregate_holdouts(run_root / "holdouts", output / "tables" / "holdout_five_seed_summary.tsv")
    aggregate_multilabel(run_root / "multilabel", output / "tables" / "multilabel_12_five_seed_summary.tsv")

    write_json(output / "ensemble_metrics.json", ensemble_payload)
    write_tsv(output / "tables" / "ensemble_macro_metrics.tsv", macro_rows)
    write_tsv(output / "tables" / "ensemble_per_class_metrics.tsv", per_class_rows)
    write_tsv(output / "tables" / "per_class_vs_best_baseline.tsv", comparison_rows)
    write_tsv(output / "tables" / "per_class_win_summary.tsv", win_rows)
    write_tsv(output / "tables" / "within_family_metrics.tsv", family_rows)
    write_tsv(output / "tables" / "within_family_confusion.tsv", family_confusion_rows)
    write_tsv(output / "tables" / "calibration_metrics.tsv", calibration_rows)
    write_tsv(output / "tables" / "reliability_bins.tsv", reliability_rows)
    write_tsv(output / "tables" / "single_model_per_seed.tsv", seed_rows)
    write_tsv(output / "tables" / "single_model_five_seed_summary.tsv", seed_summary)
    write_tsv(output / "tables" / "single_model_per_class_five_seed_summary.tsv", seed_per_class_summary)
    write_tsv(output / "tables" / "control_five_seed_summary.tsv", control_summary)
    write_tsv(output / "tables" / "control_per_class_five_seed_summary.tsv", control_per_class)
    write_tsv(output / "tables" / "control_paired_factorial_effects.tsv", paired_control_effects(run_root / "main_controls"))
    write_tsv(output / "tables" / "efficiency_five_seed_summary.tsv", efficiency_rows(run_roots))

    # The study-level holdout is the frozen lockbox; calibrate its five-seed ensemble independently.
    study_root = run_root / "holdouts" / "study_holdout"
    lock_valid_ids, lock_valid_labels, lock_valid_probs = load_ensemble(study_root, "valid")
    lock_test_ids, lock_test_labels, lock_test_probs = load_ensemble(study_root, "test")
    lock_temperature = fit_temperature(lock_valid_labels, lock_valid_probs)
    lock_scaled = apply_temperature(lock_test_probs, lock_temperature)
    lock_raw_calibration, lock_raw_bins = calibration_metrics(lock_test_labels, lock_test_probs)
    lock_scaled_calibration, lock_scaled_bins = calibration_metrics(lock_test_labels, lock_scaled)
    lock_supported = [
        name
        for idx, name in enumerate(CLASSES)
        if np.any(lock_valid_labels == idx) and np.any(lock_test_labels == idx)
    ]
    lock_payload = {
        "definition": "Frozen test components of split_study_holdout; no study component appears in train or validation",
        "supported_classes": lock_supported,
        "missing_classes": [name for name in CLASSES if name not in lock_supported],
        "macro_averaging_rule": "Only classes represented in both validation and test; prediction remains an unconstrained 11-class argmax",
        "temperature_selected_on_validation": lock_temperature,
        "raw_metrics": restrict_macro_summary(evaluate(lock_test_labels, lock_test_probs), lock_supported),
        "calibrated_metrics": restrict_macro_summary(evaluate(lock_test_labels, lock_scaled), lock_supported),
        "raw_calibration": lock_raw_calibration,
        "calibrated_calibration": lock_scaled_calibration,
        "valid_rows": len(lock_valid_ids),
        "test_rows": len(lock_test_ids),
    }
    write_json(output / "lockbox_metrics.json", lock_payload)
    write_tsv(
        output / "tables" / "lockbox_reliability_bins.tsv",
        [{"calibration": "raw", **row} for row in lock_raw_bins]
        + [{"calibration": "temperature_scaled", **row} for row in lock_scaled_bins],
    )

    # Raw equal-weight ensemble scores are retained for paired ranking
    # bootstrap; calibrated probabilities are written separately.
    prediction_dir = output / "ensemble_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for model, (ids, labels, raw_scores) in ensemble_predictions.items():
        safe_name = model.lower().replace(" ", "_").replace("-", "_")
        calibrated_scores = apply_temperature(
            raw_scores,
            float(ensemble_payload[model]["temperature_selected_on_validation"]),
        )
        fields = ["site_id", "true_class_index", "true_mod_type", *[f"prob_{name}" for name in CLASSES]]
        for suffix, scores in (("raw", raw_scores), ("calibrated", calibrated_scores)):
            path = prediction_dir / f"{safe_name}_test_predictions_{suffix}.tsv"
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                for idx, site_id in enumerate(ids):
                    row: dict[str, object] = {
                        "site_id": site_id,
                        "true_class_index": int(labels[idx]),
                        "true_mod_type": CLASSES[int(labels[idx])],
                    }
                    row.update(
                        {f"prob_{name}": float(scores[idx, class_idx]) for class_idx, name in enumerate(CLASSES)}
                    )
                    writer.writerow(row)
            temporary.replace(path)

    plot_outputs(output, primary_metrics, reliability_rows)
    sha256_manifest(output)
    print(json.dumps({"output": str(output), "models": list(MODEL_RUNS), "classes": CLASSES, "seeds": SEEDS}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("rmbase12_work/submission_strict_runs_20260806"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("rmbase12_work/submission_strict/multirm11_hg38_conflict_clean_strict_splits.tsv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rmbase12_work/submission_strict_final_20260806"),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
