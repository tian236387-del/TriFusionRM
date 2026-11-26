#!/usr/bin/env python
"""Train a scalable position-aware hashed k-mer detection baseline."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


def normalize_sequence(value: object) -> str:
    return str(value or "").strip().upper().replace("U", "T")


def positional_tokens(sequence: str) -> str:
    return " ".join(f"p{index:03d}{base}" for index, base in enumerate(sequence))


def iter_split_chunks(
    path: Path,
    split_column: str,
    split_value: str,
    sequence_column: str,
    chunk_size: int,
) -> Iterable[tuple[list[str], np.ndarray, list[str], list[str]]]:
    sequences: list[str] = []
    labels: list[int] = []
    site_ids: list[str] = []
    modifications: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"site_id", "det_label", "original_mod_type", sequence_column, split_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing fields: {sorted(missing)}")
        for row in reader:
            if row.get(split_column) != split_value:
                continue
            sequence = normalize_sequence(row.get(sequence_column))
            if not sequence or set(sequence) - set("ACGT"):
                raise ValueError(f"Invalid sequence for {row['site_id']}")
            sequences.append(sequence)
            labels.append(int(float(row["det_label"])))
            site_ids.append(row["site_id"])
            modifications.append(row["original_mod_type"])
            if len(sequences) >= chunk_size:
                yield sequences, np.asarray(labels, dtype=np.int8), site_ids, modifications
                sequences, labels, site_ids, modifications = [], [], [], []
    if sequences:
        yield sequences, np.asarray(labels, dtype=np.int8), site_ids, modifications


def transform(
    sequences: Sequence[str],
    kmer_vectorizer: HashingVectorizer,
    position_vectorizer: HashingVectorizer,
):
    kmer = kmer_vectorizer.transform(sequences)
    position = position_vectorizer.transform(positional_tokens(sequence) for sequence in sequences)
    return hstack([kmer, position], format="csr", dtype=np.float32)


def predict_split(
    model: SGDClassifier,
    dataset: Path,
    split_column: str,
    split_value: str,
    sequence_column: str,
    chunk_size: int,
    kmer_vectorizer: HashingVectorizer,
    position_vectorizer: HashingVectorizer,
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    site_ids: list[str] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    modifications: list[str] = []
    for sequences, chunk_labels, chunk_ids, chunk_modifications in iter_split_chunks(
        dataset, split_column, split_value, sequence_column, chunk_size
    ):
        features = transform(sequences, kmer_vectorizer, position_vectorizer)
        probabilities.append(model.predict_proba(features)[:, 1].astype(np.float32))
        labels.append(chunk_labels)
        site_ids.extend(chunk_ids)
        modifications.extend(chunk_modifications)
    if not labels:
        raise ValueError(f"No rows found for split {split_value!r}")
    return site_ids, np.concatenate(labels), np.concatenate(probabilities), modifications


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    scores = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(scores == np.nanmax(scores))
    return float(thresholds[int(best[-1])])


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probabilities >= threshold
    return {
        "AUPR": float(average_precision_score(labels, probabilities)),
        "AUC": float(roc_auc_score(labels, probabilities)),
        "F1": float(f1_score(labels, prediction, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, prediction)),
        "threshold": threshold,
    }


def write_predictions(
    path: Path,
    site_ids: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    modifications: Sequence[str],
    threshold: float,
) -> None:
    fields = [
        "site_id", "true_mod_type", "true_class_index", "original_mod_type",
        "pred_mod_type", "pred_class_index", "prob_unmodified", "prob_modified",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for site_id, label, probability, modification in zip(site_ids, labels, probabilities, modifications):
            predicted = int(probability >= threshold)
            writer.writerow(
                {
                    "site_id": site_id,
                    "true_mod_type": "modified" if label else "unmodified",
                    "true_class_index": int(label),
                    "original_mod_type": modification,
                    "pred_mod_type": "modified" if predicted else "unmodified",
                    "pred_class_index": predicted,
                    "prob_unmodified": float(1.0 - probability),
                    "prob_modified": float(probability),
                }
            )


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kmer_vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(args.min_k, args.max_k),
        n_features=args.kmer_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    position_vectorizer = HashingVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        n_features=args.position_features,
        alternate_sign=False,
        binary=True,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        learning_rate="optimal",
        average=True,
        random_state=args.seed,
    )
    first = True
    chunks_seen = 0
    rows_seen = 0
    for epoch in range(1, args.epochs + 1):
        for chunk_index, (sequences, labels, _, _) in enumerate(
            iter_split_chunks(
                args.dataset, args.split_column, "train", args.sequence_column, args.chunk_size
            ),
            start=1,
        ):
            features = transform(sequences, kmer_vectorizer, position_vectorizer)
            order = np.arange(len(labels))
            np.random.default_rng(args.seed + epoch * 100000 + chunk_index).shuffle(order)
            if first:
                model.partial_fit(features[order], labels[order], classes=np.asarray([0, 1], dtype=np.int8))
                first = False
            else:
                model.partial_fit(features[order], labels[order])
            chunks_seen += 1
            rows_seen += len(labels)
            if chunk_index % 5 == 0:
                print(f"epoch={epoch} chunks={chunk_index} rows={rows_seen}", flush=True)

    valid = predict_split(
        model, args.dataset, args.split_column, "valid", args.sequence_column, args.chunk_size,
        kmer_vectorizer, position_vectorizer,
    )
    test = predict_split(
        model, args.dataset, args.split_column, "test", args.sequence_column, args.chunk_size,
        kmer_vectorizer, position_vectorizer,
    )
    threshold = select_threshold(valid[1], valid[2])
    summary = {
        "model": "position_aware_hashed_kmer_logistic_sgd",
        "dataset": str(args.dataset),
        "split_column": args.split_column,
        "sequence_column": args.sequence_column,
        "seed": args.seed,
        "epochs": args.epochs,
        "rows_seen_across_epochs": rows_seen,
        "chunks_seen": chunks_seen,
        "validation": metrics(valid[1], valid[2], threshold),
        "test": metrics(test[1], test[2], threshold),
        "threshold_selection": "maximum validation F1",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    write_predictions(args.output_dir / "valid_predictions.tsv", *valid, threshold)
    write_predictions(args.output_dir / "test_predictions.tsv", *test, threshold)
    joblib.dump(model, args.output_dir / "model.joblib")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-column", default="split_detection_cluster201")
    parser.add_argument("--sequence-column", default="sequence_201nt")
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--kmer-features", type=int, default=262144)
    parser.add_argument("--position-features", type=int, default=65536)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
