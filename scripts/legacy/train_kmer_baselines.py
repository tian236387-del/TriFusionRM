#!/usr/bin/env python
"""Train sequence-only k-mer baselines on the frozen m1A/m6A benchmark split."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

try:
    from scripts.train_structdualrm import calibrate_thresholds, evaluate
except ModuleNotFoundError:  # pragma: no cover - command-line execution from scripts/
    from train_structdualrm import calibrate_thresholds, evaluate


ALPHABET = "ACGT"


def all_kmers(k: int) -> List[str]:
    return ["".join(parts) for parts in itertools.product(ALPHABET, repeat=k)]


def make_kmer_features(sequences: Sequence[str], k_values: Sequence[int] = (1, 2, 3, 4), center_width: int = 5) -> Tuple[np.ndarray, List[str]]:
    kmer_sets = {k: all_kmers(k) for k in k_values}
    names: List[str] = []
    for k in k_values:
        names.extend([f"k{k}_{kmer}" for kmer in kmer_sets[k]])
    center_kmers = all_kmers(center_width)
    names.extend([f"center{center_width}_{kmer}" for kmer in center_kmers])
    names.extend(["gc_fraction", "center_is_drach"])

    index = {name: idx for idx, name in enumerate(names)}
    arr = np.zeros((len(sequences), len(names)), dtype=np.float32)

    for row_idx, raw_seq in enumerate(sequences):
        seq = raw_seq.upper().replace("U", "T")
        for k in k_values:
            for pos in range(0, max(0, len(seq) - k + 1)):
                kmer = seq[pos : pos + k]
                col = index.get(f"k{k}_{kmer}")
                if col is not None:
                    arr[row_idx, col] += 1.0

        center = len(seq) // 2
        half = center_width // 2
        centered = seq[center - half : center + half + 1]
        col = index.get(f"center{center_width}_{centered}")
        if col is not None:
            arr[row_idx, col] = 1.0

        valid = [base for base in seq if base in ALPHABET]
        if valid:
            arr[row_idx, index["gc_fraction"]] = (valid.count("G") + valid.count("C")) / len(valid)
        if len(centered) == 5:
            d, r, a, c, h = centered
            arr[row_idx, index["center_is_drach"]] = float(d in "AGT" and r in "AG" and a == "A" and c == "C" and h in "ACT")

    return arr, names


def read_split_rows(path: Path, split_column: str) -> Dict[str, List[Dict[str, str]]]:
    groups = {"train": [], "valid": [], "test": []}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            split = row.get(split_column, "")
            if split in groups:
                groups[split].append(row)
    return groups


def labels_from_rows(rows: Sequence[Dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row["label_m1A"]), float(row["label_m6A"])] for row in rows], dtype=np.float32)


def train_binary_heads(model_kind: str, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, n_estimators: int, seed: int) -> np.ndarray:
    probs = []
    for task_idx in range(y_train.shape[1]):
        y = y_train[:, task_idx]
        if model_kind == "logreg":
            model = LogisticRegression(class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=seed)
        elif model_kind == "extratrees":
            model = ExtraTreesClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=seed, n_jobs=-1)
        else:
            raise ValueError(f"unknown model_kind: {model_kind}")
        model.fit(x_train, y)
        probs.append(model.predict_proba(x_eval)[:, 1])
    return np.vstack(probs).T.astype(np.float32)


def write_predictions(path: Path, rows: Sequence[Dict[str, str]], labels: np.ndarray, probs: np.ndarray, thresholds: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["site_id", "label_m1A", "label_m6A", "pred_m1A_prob", "pred_m6A_prob", "calibrated_m1A_label", "calibrated_m6A_label"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    "site_id": row["site_id"],
                    "label_m1A": int(labels[idx, 0]),
                    "label_m6A": int(labels[idx, 1]),
                    "pred_m1A_prob": float(probs[idx, 0]),
                    "pred_m6A_prob": float(probs[idx, 1]),
                    "calibrated_m1A_label": int(probs[idx, 0] >= thresholds["m1A"]),
                    "calibrated_m6A_label": int(probs[idx, 1] >= thresholds["m6A"]),
                }
            )


def run_baseline(model_kind: str, rows_by_split: Dict[str, List[Dict[str, str]]], out_dir: Path, n_estimators: int, seed: int) -> None:
    train_rows = rows_by_split["train"]
    valid_rows = rows_by_split["valid"]
    test_rows = rows_by_split["test"]
    x_train, feature_names = make_kmer_features([row["sequence"] for row in train_rows])
    x_valid, _ = make_kmer_features([row["sequence"] for row in valid_rows])
    x_test, _ = make_kmer_features([row["sequence"] for row in test_rows])
    y_train = labels_from_rows(train_rows)
    y_valid = labels_from_rows(valid_rows)
    y_test = labels_from_rows(test_rows)

    valid_probs = train_binary_heads(model_kind, x_train, y_train, x_valid, n_estimators=n_estimators, seed=seed)
    test_probs = train_binary_heads(model_kind, x_train, y_train, x_test, n_estimators=n_estimators, seed=seed)
    thresholds = calibrate_thresholds(valid_probs, y_valid)

    variant = f"kmer_{model_kind}"
    metrics = {
        "model_name": variant,
        "variant": variant,
        "feature_count": len(feature_names),
        "row_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "validation": evaluate(valid_probs, y_valid, thresholds),
        "test": evaluate(test_probs, y_test, thresholds),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "thresholds_validation.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    write_predictions(out_dir / "valid_predictions.tsv", valid_rows, y_valid, valid_probs, thresholds)
    write_predictions(out_dir / "test_predictions.tsv", test_rows, y_test, test_probs, thresholds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train k-mer baselines for external benchmark comparison.")
    parser.add_argument("--split-file", default="05_datasets/benchmark_splits/m1A_m6A_split_random_seed42.tsv")
    parser.add_argument("--split-column", default="split_random_seed42")
    parser.add_argument("--out-root", default="12_benchmarks")
    parser.add_argument("--models", nargs="+", default=["logreg", "extratrees"], choices=["logreg", "extratrees"])
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows_by_split = read_split_rows(Path(args.split_file), args.split_column)
    for model_kind in args.models:
        out_dir = Path(args.out_root) / f"kmer_{model_kind}"
        run_baseline(model_kind, rows_by_split, out_dir, n_estimators=args.n_estimators, seed=args.seed)
        print(f"wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
