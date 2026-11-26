#!/usr/bin/env python
"""Train multiclass k-mer baselines on the RMBase hg38 11-class dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
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
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def normalize_sequence(seq: str) -> str:
    return str(seq or "").upper().replace("U", "T")


def kmers(seq: str, k: int) -> Iterable[str]:
    for idx in range(0, max(0, len(seq) - k + 1)):
        yield seq[idx : idx + k]


def build_vocab(rows: Sequence[Dict[str, str]], ks: Sequence[int], min_count: int) -> Dict[str, int]:
    counts = Counter()
    for row in rows:
        seq = normalize_sequence(row["sequence_41nt"])
        for k in ks:
            counts.update(f"{k}:{mer}" for mer in kmers(seq, k))
    vocab = {mer: idx for idx, (mer, count) in enumerate(sorted(counts.items())) if count >= min_count}
    return vocab


def vectorize(rows: Sequence[Dict[str, str]], vocab: Dict[str, int], ks: Sequence[int]) -> np.ndarray:
    x = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        seq = normalize_sequence(row["sequence_41nt"])
        for k in ks:
            for mer in kmers(seq, k):
                key = f"{k}:{mer}"
                col = vocab.get(key)
                if col is not None:
                    x[row_idx, col] += 1.0
    lengths = x.sum(axis=1, keepdims=True)
    lengths[lengths < 1] = 1.0
    return x / lengths


def labels(rows: Sequence[Dict[str, str]]) -> np.ndarray:
    return np.asarray([int(float(row["class_index"])) for row in rows], dtype=np.int64)


def metric_or_nan(fn, *args) -> float:
    try:
        return float(fn(*args))
    except ValueError:
        return float("nan")


def evaluate(probs: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    pred = probs.argmax(axis=1)
    onehot = np.eye(len(CLASS_NAMES), dtype=np.int32)[y]
    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        y_bin = (y == idx).astype(int)
        pred_bin = (pred == idx).astype(int)
        per_class[name] = {
            "support": int(y_bin.sum()),
            "AUC_ovr": metric_or_nan(roc_auc_score, y_bin, probs[:, idx]) if len(np.unique(y_bin)) > 1 else float("nan"),
            "AUPR_ovr": metric_or_nan(average_precision_score, y_bin, probs[:, idx]) if len(np.unique(y_bin)) > 1 else float("nan"),
            "F1": metric_or_nan(lambda a, b: f1_score(a, b, zero_division=0), y_bin, pred_bin),
            "Precision": metric_or_nan(lambda a, b: precision_score(a, b, zero_division=0), y_bin, pred_bin),
            "Recall": metric_or_nan(lambda a, b: recall_score(a, b, zero_division=0), y_bin, pred_bin),
            "MCC_ovr": metric_or_nan(matthews_corrcoef, y_bin, pred_bin),
        }
    return {
        "accuracy": metric_or_nan(accuracy_score, y, pred),
        "balanced_accuracy": metric_or_nan(balanced_accuracy_score, y, pred),
        "macro_F1": metric_or_nan(lambda a, b: f1_score(a, b, average="macro", zero_division=0), y, pred),
        "weighted_F1": metric_or_nan(lambda a, b: f1_score(a, b, average="weighted", zero_division=0), y, pred),
        "MCC": metric_or_nan(matthews_corrcoef, y, pred),
        "macro_AUC_ovr": metric_or_nan(lambda a, b: roc_auc_score(a, b, average="macro"), onehot, probs),
        "macro_AUPR_ovr": float(np.nanmean([per_class[name]["AUPR_ovr"] for name in CLASS_NAMES])),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
    }


def write_predictions(path: Path, rows: Sequence[Dict[str, str]], probs: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["site_id", "true_mod_type", "true_class_index", "pred_mod_type", "pred_class_index"] + [f"prob_{name}" for name in CLASS_NAMES]
    pred = probs.argmax(axis=1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, row in enumerate(rows):
            out = {
                "site_id": row["site_id"],
                "true_mod_type": row["mod_type"],
                "true_class_index": int(y[idx]),
                "pred_mod_type": CLASS_NAMES[int(pred[idx])],
                "pred_class_index": int(pred[idx]),
            }
            out.update({f"prob_{name}": float(probs[idx, j]) for j, name in enumerate(CLASS_NAMES)})
            writer.writerow(out)


def run(args: argparse.Namespace) -> Dict[str, object]:
    rows = read_tsv(Path(args.dataset))
    train_rows = [row for row in rows if row.get(args.split) == "train"]
    valid_rows = [row for row in rows if row.get(args.split) == "valid"]
    test_rows = [row for row in rows if row.get(args.split) == "test"]
    ks = [int(k) for k in args.kmers.split(",") if k.strip()]
    vocab = build_vocab(train_rows, ks, args.min_count)
    x_train = vectorize(train_rows, vocab, ks)
    y_train = labels(train_rows)
    x_valid = vectorize(valid_rows, vocab, ks)
    y_valid = labels(valid_rows)
    x_test = vectorize(test_rows, vocab, ks)
    y_test = labels(test_rows)

    if args.model == "logistic":
        clf = LogisticRegression(class_weight="balanced", max_iter=args.max_iter, solver="lbfgs", random_state=args.seed)
    elif args.model == "extratrees":
        clf = ExtraTreesClassifier(n_estimators=args.n_estimators, class_weight="balanced", random_state=args.seed, n_jobs=-1)
    else:
        raise ValueError(args.model)
    clf.fit(x_train, y_train)
    valid_probs = clf.predict_proba(x_valid)
    test_probs = clf.predict_proba(x_test)

    metrics = {
        "dataset": args.dataset,
        "variant": f"kmer_{args.model}_{args.kmers.replace(',', '')}",
        "task": "RMBase_hg38_11_class_multiclass",
        "classes": CLASS_NAMES,
        "row_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "vocab_size": len(vocab),
        "kmers": ks,
        "validation": evaluate(valid_probs, y_valid),
        "test": evaluate(test_probs, y_test),
        "caveats": [
            "This is a sequence-only classical baseline on the same RMBase hg38 11-class split.",
            "m5U and D have very small post-filter sample sizes; their per-class metrics should be interpreted cautiously.",
        ],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions(out_dir / "valid_predictions.tsv", valid_rows, valid_probs, y_valid)
    write_predictions(out_dir / "test_predictions.tsv", test_rows, test_probs, y_test)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RMBase 11-class k-mer baseline.")
    parser.add_argument("--dataset", default="rmbase11_work/datasets/rmbase11_hg38_41nt_cdhit80_like.tsv")
    parser.add_argument("--out-dir", default="rmbase11_work/models/kmer_logistic")
    parser.add_argument("--split", default="split_random")
    parser.add_argument("--model", choices=["logistic", "extratrees"], default="logistic")
    parser.add_argument("--kmers", default="3,4")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = run(args)
    print(json.dumps({k: metrics[k] for k in ["variant", "row_counts", "vocab_size", "test"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
