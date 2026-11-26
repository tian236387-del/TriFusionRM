#!/usr/bin/env python
"""Evaluate predictions within same-center-base modification groups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


BASE_GROUPS = {
    "A-derived": ["Am", "m1A", "m6A"],
    "C-derived": ["Cm", "m5C"],
    "G-derived": ["Gm", "m7G"],
    "U-derived": ["Um", "D", "Y", "m5U"],
}


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def metric_or_nan(fn, *args) -> float:
    try:
        return float(fn(*args))
    except ValueError:
        return float("nan")


def evaluate_group(rows: Sequence[Dict[str, str]], classes: Sequence[str]) -> Dict[str, object]:
    sub = [row for row in rows if row["true_mod_type"] in classes]
    y = np.asarray([classes.index(row["true_mod_type"]) for row in sub], dtype=np.int64)
    probs = np.asarray([[float(row[f"prob_{cls}"]) for cls in classes] for row in sub], dtype=np.float64)
    denom = probs.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    probs = probs / denom
    pred = probs.argmax(axis=1)
    per_class = {}
    macro_aupr_values = []
    for idx, cls in enumerate(classes):
        yy = (y == idx).astype(int)
        pp = probs[:, idx]
        pred_bin = (pred == idx).astype(int)
        aupr = metric_or_nan(average_precision_score, yy, pp) if len(np.unique(yy)) > 1 else float("nan")
        macro_aupr_values.append(aupr)
        per_class[cls] = {
            "support": int(yy.sum()),
            "AUPR": aupr,
            "F1": metric_or_nan(lambda a, b: f1_score(a, b, zero_division=0), yy, pred_bin),
        }
    onehot = np.eye(len(classes), dtype=np.int32)[y]
    return {
        "sample_count": len(sub),
        "classes": list(classes),
        "accuracy": metric_or_nan(accuracy_score, y, pred),
        "balanced_accuracy": metric_or_nan(balanced_accuracy_score, y, pred),
        "macro_F1": metric_or_nan(lambda a, b: f1_score(a, b, average="macro", zero_division=0), y, pred),
        "macro_AUC_ovr": metric_or_nan(lambda a, b: roc_auc_score(a, b, multi_class="ovr", average="macro"), onehot, probs)
        if len(classes) > 2
        else metric_or_nan(roc_auc_score, y, probs[:, 1]),
        "macro_AUPR": float(np.nanmean(macro_aupr_values)),
        "per_class": per_class,
    }


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run(args: argparse.Namespace) -> Dict[str, object]:
    rows = read_tsv(Path(args.predictions))
    summary_rows = []
    details = {}
    for group_name, classes in BASE_GROUPS.items():
        result = evaluate_group(rows, classes)
        details[group_name] = result
        summary_rows.append(
            {
                "model": args.model_name,
                "base_group": group_name,
                "classes": ",".join(classes),
                "sample_count": result["sample_count"],
                "accuracy": result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "macro_F1": result["macro_F1"],
                "macro_AUPR": result["macro_AUPR"],
                "macro_AUC_ovr": result["macro_AUC_ovr"],
            }
        )
    write_tsv(
        Path(args.out_tsv),
        summary_rows,
        ["model", "base_group", "classes", "sample_count", "accuracy", "balanced_accuracy", "macro_F1", "macro_AUPR", "macro_AUC_ovr"],
    )
    payload = {"model": args.model_name, "predictions": args.predictions, "groups": details}
    Path(args.out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Within-base evaluation from prediction TSV.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--out-json", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
