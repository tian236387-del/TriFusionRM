#!/usr/bin/env python
"""Parallel cluster bootstrap for frozen multimodal ablations.

The bootstrap unit is a CD-HIT80 cluster.  Scores are kept fixed and the
same resampling weights are used for every model, class, and scope.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score


CLASSES = ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "Y", "AtoI"]
FAMILIES = {
    "A": ["Am", "m1A", "m6A", "m6Am", "AtoI"],
    "C": ["Cm", "m5C"],
    "G": ["Gm", "m7G"],
    "U": ["Um", "m5U", "Y"],
}
MODEL_ORDER = [
    "sequence_only",
    "sequence_structure",
    "sequence_rbp",
    "full",
    "full_shuffled_structure",
    "full_shuffled_rbp",
    "full_no_pair_edges",
    "full_shuffled_rbp_bins",
]
COMPARISONS = [
    ("structure_increment", "sequence_structure", "sequence_only"),
    ("rbp_increment", "sequence_rbp", "sequence_only"),
    ("joint_increment", "full", "sequence_only"),
    ("structure_given_rbp", "full", "sequence_rbp"),
    ("rbp_given_structure", "full", "sequence_structure"),
    ("real_structure_minus_shuffled", "full", "full_shuffled_structure"),
    ("real_rbp_minus_shuffled", "full", "full_shuffled_rbp"),
    ("pair_edges_contribution", "full", "full_no_pair_edges"),
    ("rbp_distance_bins_contribution", "full", "full_shuffled_rbp_bins"),
]
SCOPES = ("standard", "same_base")

_PROBS = None
_LABELS = None
_WEIGHTS = None
_CLASS_NAMES = CLASSES
_BLOCK_SIZE = 64


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_cluster_map(dataset: Path, split_column: str, split_value: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with dataset.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get(split_column) == split_value:
                mapping[row["site_id"]] = row["cdhit80_cluster_id"]
    return mapping


def load_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_tsv(path)
    site_ids = np.asarray([row["site_id"] for row in rows], dtype=object)
    labels = np.asarray([CLASSES.index(row["true_mod_type"]) for row in rows], dtype=np.int16)
    probabilities = np.asarray(
        [[float(row[f"prob_{name}"]) for name in CLASSES] for row in rows], dtype=np.float32
    )
    probabilities = np.clip(probabilities, 1e-7, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return site_ids, labels, probabilities


def initialise_worker(probs_path: str, labels_path: str, weights_path: str, shapes: Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]], block_size: int) -> None:
    global _PROBS, _LABELS, _WEIGHTS, _BLOCK_SIZE
    probs_shape, labels_shape, weights_shape = shapes
    _PROBS = np.memmap(probs_path, mode="r", dtype=np.float32, shape=probs_shape)
    _LABELS = np.memmap(labels_path, mode="r", dtype=np.int16, shape=labels_shape)
    _WEIGHTS = np.load(weights_path, mmap_mode="r")
    _BLOCK_SIZE = block_size


def scope_indices(scope: str, class_index: int) -> Tuple[np.ndarray, np.ndarray, int]:
    assert _LABELS is not None
    class_name = CLASSES[class_index]
    if scope == "standard":
        indices = np.arange(len(_LABELS), dtype=np.int64)
        family_indices = np.asarray([class_index], dtype=np.int16)
        local_index = 0
    elif scope == "same_base":
        family_names = next(names for names in FAMILIES.values() if class_name in names)
        family_indices = np.asarray([CLASSES.index(name) for name in family_names], dtype=np.int16)
        indices = np.flatnonzero(np.isin(_LABELS, family_indices))
        local_index = family_names.index(class_name)
    else:
        raise ValueError(scope)
    return indices, family_indices, local_index


def weighted_ap(scores: np.ndarray, binary: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
    """Compute AP for every bootstrap row-weight vector in the shared memmap."""
    assert _WEIGHTS is not None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_binary = binary[order].astype(np.float32, copy=False)
    starts = np.concatenate(([0], np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1))
    results = np.empty(_WEIGHTS.shape[0], dtype=np.float32)
    for start in range(0, _WEIGHTS.shape[0], _BLOCK_SIZE):
        stop = min(_WEIGHTS.shape[0], start + _BLOCK_SIZE)
        if row_indices.size == _WEIGHTS.shape[1]:
            weights = np.asarray(_WEIGHTS[start:stop], dtype=np.float32)[:, order]
        else:
            weights = np.asarray(_WEIGHTS[start:stop][:, row_indices], dtype=np.float32)[:, order]
        positive_weights = weights * sorted_binary[None, :]
        if len(starts) != len(sorted_scores):
            weights = np.add.reduceat(weights, starts, axis=1)
            positive_weights = np.add.reduceat(positive_weights, starts, axis=1)
        cumulative_positive = np.cumsum(positive_weights, axis=1, dtype=np.float32)
        cumulative_total = np.cumsum(weights, axis=1, dtype=np.float32)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        total_positive = cumulative_positive[:, -1]
        values = np.divide(
            np.sum(positive_weights * precision, axis=1, dtype=np.float32),
            total_positive,
            out=np.full(stop - start, np.nan, dtype=np.float32),
            where=total_positive > 0,
        )
        results[start:stop] = values
    return results


def worker(task: Tuple[str, int, int]) -> Tuple[str, int, int, np.ndarray]:
    scope, class_index, model_index = task
    assert _PROBS is not None and _LABELS is not None
    row_indices, family_indices, local_index = scope_indices(scope, class_index)
    class_labels = _LABELS[row_indices]
    binary = (class_labels == class_index).astype(np.int8)
    if not binary.any():
        return scope, class_index, model_index, np.full(_WEIGHTS.shape[0], np.nan, dtype=np.float32)
    if scope == "standard":
        scores = np.asarray(_PROBS[model_index, row_indices, class_index], dtype=np.float32)
    else:
        family_probabilities = np.asarray(_PROBS[model_index, row_indices][:, family_indices], dtype=np.float32)
        family_probabilities /= family_probabilities.sum(axis=1, keepdims=True).clip(min=1e-7)
        scores = family_probabilities[:, local_index]
    values = weighted_ap(scores, binary, row_indices)
    return scope, class_index, model_index, values


def bh(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.ones_like(p)
    running = 1.0
    for index in range(len(order) - 1, -1, -1):
        original = order[index]
        running = min(running, p[original] * len(p) / (index + 1))
        adjusted[original] = running
    return np.clip(adjusted, 0.0, 1.0)


def empirical_p(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    lower = (np.count_nonzero(finite <= 0) + 1) / (len(finite) + 1)
    upper = (np.count_nonzero(finite >= 0) + 1) / (len(finite) + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def distribution_summary(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "bootstrap_mean_delta": float(np.mean(finite)),
        "bootstrap_sd_delta": float(np.std(finite, ddof=1)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_two_sided": empirical_p(finite),
    }


def observed_scope_ap(scope: str, class_index: int, probabilities: np.ndarray, labels: np.ndarray) -> float:
    class_name = CLASSES[class_index]
    if scope == "standard":
        mask = np.ones(len(labels), dtype=bool)
        scores = probabilities[:, class_index]
    else:
        family_names = next(names for names in FAMILIES.values() if class_name in names)
        family_indices = np.asarray([CLASSES.index(name) for name in family_names], dtype=np.int16)
        mask = np.isin(labels, family_indices)
        family_probabilities = probabilities[mask][:, family_indices]
        family_probabilities /= family_probabilities.sum(axis=1, keepdims=True).clip(min=1e-7)
        scores = family_probabilities[:, family_names.index(class_name)]
    binary = (labels[mask] == class_index).astype(np.int8)
    return float(average_precision_score(binary, scores)) if binary.any() else float("nan")


def generate_weights(
    dataset: Path,
    site_ids: np.ndarray,
    labels: np.ndarray,
    split_column: str,
    split_value: str,
    replicates: int,
    seed: int,
    path: Path,
) -> Tuple[np.ndarray, Dict[str, int], int, int]:
    cluster_map = load_cluster_map(dataset, split_column, split_value)
    missing = [str(value) for value in site_ids if str(value) not in cluster_map]
    if missing:
        raise ValueError(f"Missing cluster IDs: {missing[:5]} (n={len(missing)})")
    cluster_ids = np.asarray([cluster_map[str(value)] for value in site_ids], dtype=object)
    unique_clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    class_cluster_indices = {
        index: np.unique(inverse[labels == index])
        for index in range(len(CLASSES))
        if np.count_nonzero(labels == index) > 0
    }
    weights = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(replicates, len(labels)))
    rng = np.random.default_rng(seed)
    completed = 0
    attempted = 0
    while completed < replicates:
        request = min(256, replicates - completed + 64)
        counts = rng.multinomial(len(unique_clusters), np.full(len(unique_clusters), 1.0 / len(unique_clusters)), size=request)
        attempted += request
        valid = np.ones(request, dtype=bool)
        for cluster_indices in class_cluster_indices.values():
            valid &= counts[:, cluster_indices].sum(axis=1) > 0
        selected = counts[valid]
        if not len(selected):
            continue
        selected = selected[: replicates - completed]
        weights[completed : completed + len(selected)] = selected[:, inverse].astype(np.int32)
        completed += len(selected)
    weights.flush()
    del weights
    support = {name: int(np.count_nonzero(labels == index)) for index, name in enumerate(CLASSES)}
    return inverse, support, len(unique_clusters), attempted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--factorial-root", type=Path, required=True)
    parser.add_argument("--full-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-column", default="split_cdhit80_cluster")
    parser.add_argument("--split-value", default="test")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.replicates < 100 or args.workers < 1 or args.block_size < 1:
        raise ValueError("Invalid bootstrap settings")

    args.output.mkdir(parents=True, exist_ok=True)
    work = args.output / "work"
    work.mkdir(parents=True, exist_ok=True)
    paths = {
        model: (args.full_model_dir if model == "full" else args.factorial_root / model) / "test_predictions.tsv"
        for model in MODEL_ORDER
    }
    site_ids: np.ndarray | None = None
    labels: np.ndarray | None = None
    probabilities: List[np.ndarray] = []
    for model in MODEL_ORDER:
        current_ids, current_labels, current_probabilities = load_predictions(paths[model])
        if site_ids is None:
            site_ids, labels = current_ids, current_labels
        elif not np.array_equal(site_ids, current_ids) or not np.array_equal(labels, current_labels):
            raise ValueError(f"Prediction pairing failed for {model}")
        probabilities.append(current_probabilities)
    assert site_ids is not None and labels is not None
    probability_array = np.asarray(probabilities, dtype=np.float32)
    probs_path = work / "probabilities.dat"
    probs_memmap = np.memmap(probs_path, mode="w+", dtype=np.float32, shape=probability_array.shape)
    probs_memmap[:] = probability_array
    probs_memmap.flush()
    del probs_memmap
    labels_path = work / "labels.dat"
    labels_memmap = np.memmap(labels_path, mode="w+", dtype=np.int16, shape=labels.shape)
    labels_memmap[:] = labels
    labels_memmap.flush()
    del labels_memmap
    weights_path = work / "cluster_bootstrap_weights.npy"
    inverse, support, cluster_count, attempted = generate_weights(
        args.dataset, site_ids, labels, args.split_column, args.split_value, args.replicates, args.random_seed, weights_path
    )
    weights_shape = (args.replicates, len(labels))
    shapes = (probability_array.shape, labels.shape, weights_shape)

    observed: Dict[Tuple[str, int, int], float] = {}
    for scope in SCOPES:
        for class_index in range(len(CLASSES)):
            if support[CLASSES[class_index]] == 0:
                continue
            for model_index, model in enumerate(MODEL_ORDER):
                observed[(scope, class_index, model_index)] = observed_scope_ap(
                    scope, class_index, probability_array[model_index], labels
                )

    tasks = [
        (scope, class_index, model_index)
        for scope in SCOPES
        for class_index, class_name in enumerate(CLASSES)
        if support[class_name] > 0
        for model_index in range(len(MODEL_ORDER))
    ]
    distributions: Dict[Tuple[str, int, int], np.ndarray] = {}
    context = mp.get_context("fork")
    worker_count = min(args.workers, len(tasks))
    print(f"parallel_workers={worker_count} tasks={len(tasks)} replicates={args.replicates}", flush=True)
    with context.Pool(
        processes=worker_count,
        initializer=initialise_worker,
        initargs=(str(probs_path), str(labels_path), str(weights_path), shapes, args.block_size),
    ) as pool:
        for index, (scope, class_index, model_index, values) in enumerate(
            pool.imap_unordered(worker, tasks), start=1
        ):
            distributions[(scope, class_index, model_index)] = values
            if index % 16 == 0 or index == len(tasks):
                print(f"completed_tasks={index}/{len(tasks)}", flush=True)

    macro_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []
    for scope in SCOPES:
        for comparison, model_a, model_b in COMPARISONS:
            model_a_index = MODEL_ORDER.index(model_a)
            model_b_index = MODEL_ORDER.index(model_b)
            delta_by_class = {
                class_index: distributions[(scope, class_index, model_a_index)]
                - distributions[(scope, class_index, model_b_index)]
                for class_index, class_name in enumerate(CLASSES)
                if support[class_name] > 0
            }
            macro_delta = np.nanmean(np.asarray(list(delta_by_class.values())), axis=0)
            evaluable_indices = list(delta_by_class)
            observed_a = float(np.mean([observed[(scope, index, model_a_index)] for index in evaluable_indices]))
            observed_b = float(np.mean([observed[(scope, index, model_b_index)] for index in evaluable_indices]))
            finite = macro_delta[np.isfinite(macro_delta)]
            low, high = np.quantile(finite, [0.025, 0.975])
            macro_rows.append(
                {
                    "scope": scope,
                    "comparison": comparison,
                    "model_a": model_a,
                    "model_b": model_b,
                    "replicates": args.replicates,
                    "evaluable_classes": len(evaluable_indices),
                    "observed_model_a_macro_AUPR": observed_a,
                    "observed_model_b_macro_AUPR": observed_b,
                    "observed_delta_macro_AUPR": observed_a - observed_b,
                    "bootstrap_mean_delta": float(np.mean(finite)),
                    "bootstrap_sd_delta": float(np.std(finite, ddof=1)),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "p_two_sided": empirical_p(finite),
                }
            )
            for class_index, values in delta_by_class.items():
                finite_class = values[np.isfinite(values)]
                low, high = np.quantile(finite_class, [0.025, 0.975])
                class_rows.append(
                    {
                        "scope": scope,
                        "comparison": comparison,
                        "model_a": model_a,
                        "model_b": model_b,
                        "class": CLASSES[class_index],
                        "support": support[CLASSES[class_index]],
                        "replicates": len(finite_class),
                        "observed_model_a_AUPR": observed[(scope, class_index, model_a_index)],
                        "observed_model_b_AUPR": observed[(scope, class_index, model_b_index)],
                        "observed_delta_AUPR": observed[(scope, class_index, model_a_index)] - observed[(scope, class_index, model_b_index)],
                        "bootstrap_mean_delta": float(np.mean(finite_class)),
                        "bootstrap_sd_delta": float(np.std(finite_class, ddof=1)),
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                        "p_two_sided": empirical_p(finite_class),
                    }
                )

    for scope in SCOPES:
        indices = [index for index, row in enumerate(macro_rows) if row["scope"] == scope]
        for index, value in zip(indices, bh([float(macro_rows[index]["p_two_sided"]) for index in indices])):
            macro_rows[index]["fdr_bh_within_scope"] = float(value)
        indices = [index for index, row in enumerate(class_rows) if row["scope"] == scope]
        for index, value in zip(indices, bh([float(class_rows[index]["p_two_sided"]) for index in indices])):
            class_rows[index]["fdr_bh_within_scope"] = float(value)

    write_tsv(args.output / "cluster_bootstrap_macro_aupr.tsv", macro_rows)
    write_tsv(args.output / "cluster_bootstrap_per_class_aupr.tsv", class_rows)
    positive = [
        row for row in class_rows
        if float(row["observed_delta_AUPR"]) > 0
        and float(row["ci95_low"]) > 0
        and float(row["fdr_bh_within_scope"]) < 0.05
    ]
    write_tsv(args.output / "cluster_bootstrap_positive_per_class_evidence.tsv", positive)
    audit = {
        "status": "PASS",
        "dataset": str(args.dataset.resolve()),
        "split": args.split_value,
        "rows": len(labels),
        "clusters": cluster_count,
        "replicates": args.replicates,
        "attempted_replicates": attempted,
        "random_seed": args.random_seed,
        "resampling_unit": "cdhit80_cluster_id",
        "worker_count": worker_count,
        "block_size": args.block_size,
        "models": {model: str(paths[model].resolve()) for model in MODEL_ORDER},
        "scopes": {
            "standard": "OvR over every test row",
            "same_base": "OvR after restricting rows and renormalizing probabilities within the center-base family",
        },
        "class_support": support,
        "class_cluster_support": {
            name: int(len(np.unique(inverse[labels == index])))
            for index, name in enumerate(CLASSES)
        },
        "bootstrap_conditioning": "Cluster draws omitting all positives of any evaluable class were redrawn.",
        "fdr": "Benjamini-Hochberg separately for macro comparisons and all per-class comparisons within each scope.",
    }
    (args.output / "cluster_bootstrap_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
