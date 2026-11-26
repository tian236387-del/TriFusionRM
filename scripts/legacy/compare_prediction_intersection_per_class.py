#!/usr/bin/env python
"""Compare prediction TSV files on their shared site_id intersection."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


MetricRow = Dict[str, object]
PredictionMap = Dict[str, Mapping[str, str]]


def parse_model_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--pred entries must use name=path")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("model name cannot be empty")
    return name, Path(path)


def read_predictions(path: Path) -> PredictionMap:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: PredictionMap = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "site_id" not in reader.fieldnames:
            raise ValueError(f"Prediction TSV must contain site_id: {path}")
        for row in reader:
            site_id = str(row.get("site_id", ""))
            if site_id:
                rows[site_id] = row
    return rows


def infer_classes(rows_by_model: Mapping[str, PredictionMap], requested: Sequence[str]) -> List[str]:
    if requested:
        return list(requested)
    classes: List[str] = []
    for rows in rows_by_model.values():
        if not rows:
            continue
        first = next(iter(rows.values()))
        classes = [field.removeprefix("prob_") for field in first if field.startswith("prob_")]
        if classes:
            return classes
    raise ValueError("Could not infer classes from prob_* columns.")


def float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def class_metrics(rows: Sequence[Mapping[str, str]], classes: Sequence[str], mod_type: str) -> MetricRow:
    y_true = np.asarray([1 if row.get("true_mod_type") == mod_type else 0 for row in rows], dtype=np.int32)
    support = int(y_true.sum())
    prob_col = f"prob_{mod_type}"
    y_score = np.asarray([float_or_nan(row.get(prob_col, "nan")) for row in rows], dtype=np.float64)
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)

    pred_classes = [str(row.get("pred_mod_type", "")) for row in rows]
    if not any(pred_classes):
        prob_matrix = np.asarray(
            [[float_or_nan(row.get(f"prob_{name}", "nan")) for name in classes] for row in rows],
            dtype=np.float64,
        )
        prob_matrix = np.nan_to_num(prob_matrix, nan=-math.inf)
        pred_classes = [classes[int(idx)] for idx in prob_matrix.argmax(axis=1)]
    y_pred = np.asarray([1 if pred == mod_type else 0 for pred in pred_classes], dtype=np.int32)

    if support == 0:
        aupr = auc = float("nan")
    elif support == len(rows):
        aupr = 1.0
        auc = float("nan")
    else:
        aupr = float(average_precision_score(y_true, y_score))
        auc = float(roc_auc_score(y_true, y_score))

    return {
        "support": support,
        "AUPR_ovr": aupr,
        "AUC_ovr": auc,
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "MCC_ovr": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def fmt(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return value


def write_tsv(path: Path, rows: Sequence[MetricRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})


def compare(models: Sequence[Tuple[str, Path]], classes: Sequence[str]) -> List[MetricRow]:
    rows_by_model = {name: read_predictions(path) for name, path in models}
    common_ids = set.intersection(*(set(rows) for rows in rows_by_model.values()))
    if not common_ids:
        raise ValueError("Prediction files have no shared site_id values.")
    ordered_ids = sorted(common_ids)
    class_names = infer_classes(rows_by_model, classes)

    out_rows: List[MetricRow] = []
    for mod_type in class_names:
        out: MetricRow = {"mod_type": mod_type, "intersection_rows": len(ordered_ids)}
        best_aupr = float("-inf")
        best_model = ""
        for model_name, _ in models:
            pred_rows = [rows_by_model[model_name][site_id] for site_id in ordered_ids]
            metrics = class_metrics(pred_rows, class_names, mod_type)
            if "support" not in out:
                out["support"] = metrics["support"]
            for key, value in metrics.items():
                if key == "support":
                    continue
                out[f"{model_name}_{key}"] = value
            aupr = float(metrics["AUPR_ovr"])
            if not math.isnan(aupr) and aupr > best_aupr:
                best_aupr = aupr
                best_model = model_name
        out["best_AUPR_model"] = best_model
        out["best_AUPR"] = best_aupr if best_model else float("nan")
        out_rows.append(out)
    return out_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", action="append", type=parse_model_arg, required=True, help="name=path")
    parser.add_argument("--classes", default="", help="Comma-separated class order. Defaults to prob_* columns.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    rows = compare(args.pred, classes)
    write_tsv(Path(args.out), rows)
    print(f"Wrote {len(rows)} per-class rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
