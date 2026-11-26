#!/usr/bin/env python
"""Validation-tuned late fusion for strict RMBase11 models."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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


CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]


def read_prediction(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(row)
    site_ids = [row["site_id"] for row in rows]
    labels = np.asarray([int(float(row["true_class_index"])) for row in rows], dtype=np.int64)
    probs = np.asarray([[float(row[f"prob_{name}"]) for name in CLASS_NAMES] for row in rows], dtype=np.float64)
    return site_ids, labels, probs


def align_predictions(paths: Sequence[Path]) -> Tuple[List[str], np.ndarray, List[np.ndarray]]:
    loaded = [read_prediction(path) for path in paths]
    common_ids = set(loaded[0][0])
    for site_ids, _, _ in loaded[1:]:
        common_ids &= set(site_ids)
    ordered_ids = [site_id for site_id in loaded[0][0] if site_id in common_ids]
    if not ordered_ids:
        raise ValueError("No overlapping site_id values across prediction files.")

    aligned_probs = []
    labels_ref = None
    for site_ids, labels, probs in loaded:
        idx_by_id = {site_id: idx for idx, site_id in enumerate(site_ids)}
        indices = [idx_by_id[site_id] for site_id in ordered_ids]
        aligned_labels = labels[indices]
        if labels_ref is None:
            labels_ref = aligned_labels
        elif not np.array_equal(labels_ref, aligned_labels):
            raise ValueError("Aligned labels differ across prediction files.")
        aligned_probs.append(probs[indices])
    assert labels_ref is not None
    return ordered_ids, labels_ref, aligned_probs


def normalize(probs: np.ndarray) -> np.ndarray:
    out = probs.copy()
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return out / row_sums


def metric_or_nan(fn, *args) -> float:
    try:
        return float(fn(*args))
    except ValueError:
        return float("nan")


def evaluate(probs: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    pred = probs.argmax(axis=1)
    onehot = np.eye(len(CLASS_NAMES), dtype=np.int32)[labels]
    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        y = (labels == idx).astype(int)
        p = probs[:, idx]
        pred_bin = (pred == idx).astype(int)
        per_class[name] = {
            "support": int(y.sum()),
            "AUC_ovr": metric_or_nan(roc_auc_score, y, p) if len(np.unique(y)) > 1 else float("nan"),
            "AUPR_ovr": metric_or_nan(average_precision_score, y, p) if len(np.unique(y)) > 1 else float("nan"),
            "F1": metric_or_nan(lambda a, b: f1_score(a, b, zero_division=0), y, pred_bin),
            "Precision": metric_or_nan(lambda a, b: precision_score(a, b, zero_division=0), y, pred_bin),
            "Recall": metric_or_nan(lambda a, b: recall_score(a, b, zero_division=0), y, pred_bin),
            "MCC_ovr": metric_or_nan(matthews_corrcoef, y, pred_bin),
        }
    return {
        "accuracy": metric_or_nan(accuracy_score, labels, pred),
        "balanced_accuracy": metric_or_nan(balanced_accuracy_score, labels, pred),
        "macro_F1": metric_or_nan(lambda a, b: f1_score(a, b, average="macro", zero_division=0), labels, pred),
        "weighted_F1": metric_or_nan(lambda a, b: f1_score(a, b, average="weighted", zero_division=0), labels, pred),
        "MCC": metric_or_nan(matthews_corrcoef, labels, pred),
        "macro_AUC_ovr": metric_or_nan(lambda a, b: roc_auc_score(a, b, multi_class="ovr", average="macro"), onehot, normalize(probs)),
        "macro_AUPR_ovr": float(np.nanmean([per_class[name]["AUPR_ovr"] for name in CLASS_NAMES])),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
    }


def combine(probs_list: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    out = np.zeros_like(probs_list[0], dtype=np.float64)
    for probs, weight in zip(probs_list, weights):
        out += float(weight) * probs
    return out


def generate_weight_grid(n: int, step: float) -> Iterable[Tuple[float, ...]]:
    units = int(round(1.0 / step))
    if n == 1:
        yield (1.0,)
        return
    for cuts in itertools.product(range(units + 1), repeat=n - 1):
        used = sum(cuts)
        if used > units:
            continue
        weights = list(cuts) + [units - used]
        yield tuple(value / units for value in weights)


def write_predictions(path: Path, site_ids: Sequence[str], labels: np.ndarray, probs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["site_id", "true_class_index", "true_mod_type", "pred_class_index", "pred_mod_type"] + [f"prob_{name}" for name in CLASS_NAMES]
    pred = probs.argmax(axis=1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, site_id in enumerate(site_ids):
            row = {
                "site_id": site_id,
                "true_class_index": int(labels[idx]),
                "true_mod_type": CLASS_NAMES[int(labels[idx])],
                "pred_class_index": int(pred[idx]),
                "pred_mod_type": CLASS_NAMES[int(pred[idx])],
            }
            row.update({f"prob_{name}": float(probs[idx, j]) for j, name in enumerate(CLASS_NAMES)})
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build validation-tuned ensemble from strict model predictions.")
    parser.add_argument("--names", required=True, help="Comma-separated model names.")
    parser.add_argument("--valid", required=True, help="Comma-separated validation prediction TSVs.")
    parser.add_argument("--test", required=True, help="Comma-separated test prediction TSVs.")
    parser.add_argument("--out-dir", default="rmbase11_full_work/models/M7_late_fusion_mmseqs80_full")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--metric", choices=["macro_AUPR_ovr", "macro_F1", "MCC", "balanced_accuracy"], default="macro_AUPR_ovr")
    args = parser.parse_args(argv)

    names = [item.strip() for item in args.names.split(",") if item.strip()]
    valid_paths = [Path(item.strip()) for item in args.valid.split(",") if item.strip()]
    test_paths = [Path(item.strip()) for item in args.test.split(",") if item.strip()]
    if not (len(names) == len(valid_paths) == len(test_paths)):
        raise ValueError("names, valid and test must have the same length.")

    valid_ids, valid_labels, valid_probs_list = align_predictions(valid_paths)
    best_weights = None
    best_metrics = None
    best_score = float("-inf")
    tried = 0
    for weights in generate_weight_grid(len(names), args.step):
        probs = combine(valid_probs_list, weights)
        metrics = evaluate(probs, valid_labels)
        score = float(metrics[args.metric])
        tried += 1
        if score > best_score:
            best_score = score
            best_weights = weights
            best_metrics = metrics
    assert best_weights is not None and best_metrics is not None

    test_ids, test_labels, test_probs_list = align_predictions(test_paths)
    test_probs = combine(test_probs_list, best_weights)
    test_metrics = evaluate(test_probs, test_labels)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(out_dir / "valid_predictions.tsv", valid_ids, valid_labels, combine(valid_probs_list, best_weights))
    write_predictions(out_dir / "test_predictions.tsv", test_ids, test_labels, test_probs)
    payload = {
        "variant": "M7_late_fusion",
        "display_name": "M7 validation-tuned late fusion",
        "models": names,
        "valid_prediction_paths": [str(path) for path in valid_paths],
        "test_prediction_paths": [str(path) for path in test_paths],
        "weight_step": args.step,
        "weights": {name: weight for name, weight in zip(names, best_weights)},
        "grid_candidates": tried,
        "selection_metric": args.metric,
        "validation": best_metrics,
        "test": test_metrics,
        "row_counts": {"valid": len(valid_labels), "test": len(test_labels)},
        "caveats": [
            "Weights were selected on the validation set only and then frozen for test evaluation.",
            "This is an inference-level late-fusion architecture, not a single end-to-end neural model.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# M7 validation-tuned late fusion\n\n"
        f"Models: {', '.join(names)}\n\n"
        f"Weights selected on validation `{args.metric}`: {payload['weights']}.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
