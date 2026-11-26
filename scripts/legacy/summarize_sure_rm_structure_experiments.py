#!/usr/bin/env python
"""Aggregate SURE-RM seeds, per-modification metrics and matched-seed effects."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


MODEL_PATTERN = re.compile(r"^(sure_.+)_seed(\d+)$")
SUMMARY_METRICS = (
    "validation_macro_stable_AUPR",
    "test_macro_stable_AUPR",
    "test_macro_stable_AUC",
    "test_macro_stable_F1",
    "test_macro_stable_MCC",
    "test_overall_AUPR",
    "test_overall_Brier",
    "test_overall_ECE",
)
PAIR_COMPARISONS = (
    ("sure_s3_probability_graph", "sure_s2_probability_nodes", "probability_edges_vs_nodes"),
    ("sure_s3_probability_graph", "sure_control_sample_shuffled", "real_vs_sample_shuffled"),
    ("sure_s3_probability_graph", "sure_control_node_shuffled", "real_vs_node_shuffled"),
    ("sure_s3_probability_graph", "sure_control_weight_shuffled", "real_vs_weight_shuffled"),
    ("sure_s3_probability_graph", "sure_control_degree_rewired", "real_vs_degree_rewired"),
    ("sure_s3_probability_graph", "sure_control_capacity_cnn", "gat_vs_capacity_cnn"),
    ("sure_s4_multiscale_probability", "sure_s3_probability_graph", "multiscale_vs_single_scale"),
    ("sure_s4_multiscale_probability", "sure_control_multiscale_misaligned", "aligned_vs_misaligned_scales"),
)


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def mean_sd(values: Iterable[float]) -> tuple[float, float, int]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    return float(finite.mean()), float(finite.std(ddof=1)) if finite.size > 1 else 0.0, int(finite.size)


def flatten_metrics(payload: Mapping[str, object]) -> dict[str, float]:
    validation = payload["macro_original_modification_validation"]  # type: ignore[index]
    test_macro = payload["macro_original_modification_test"]  # type: ignore[index]
    test = payload["test"]  # type: ignore[index]
    return {
        "validation_macro_stable_AUPR": float(validation["macro_stable_AUPR"]),  # type: ignore[index]
        "test_macro_stable_AUPR": float(test_macro["macro_stable_AUPR"]),  # type: ignore[index]
        "test_macro_stable_AUC": float(test_macro["macro_stable_AUC"]),  # type: ignore[index]
        "test_macro_stable_F1": float(test_macro["macro_stable_F1"]),  # type: ignore[index]
        "test_macro_stable_MCC": float(test_macro["macro_stable_MCC"]),  # type: ignore[index]
        "test_overall_AUPR": float(test["AUPR"]),  # type: ignore[index]
        "test_overall_Brier": float(test["Brier"]),  # type: ignore[index]
        "test_overall_ECE": float(test["ECE"]),  # type: ignore[index]
    }


def run(args: argparse.Namespace) -> None:
    records: dict[tuple[str, int], dict[str, object]] = {}
    for path in sorted(args.models_dir.glob("sure_*_seed*/audited/detection_metrics.json")):
        match = MODEL_PATTERN.match(path.parent.parent.name)
        if not match:
            continue
        variant, seed_text = match.groups()
        payload = json.loads(path.read_text(encoding="utf-8"))
        records[(variant, int(seed_text))] = payload
    if not records:
        raise FileNotFoundError(f"No SURE-RM audited metrics found under {args.models_dir}")

    by_variant: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    for (variant, seed), payload in records.items():
        by_variant[variant].append((seed, payload))

    seed_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for variant, items in sorted(by_variant.items()):
        flat = [(seed, flatten_metrics(payload)) for seed, payload in sorted(items)]
        for seed, values in flat:
            seed_rows.append({"variant": variant, "seed": seed, **values})
        aggregate: dict[str, object] = {"variant": variant, "seed_count": len(items)}
        for metric in SUMMARY_METRICS:
            value_mean, value_sd, count = mean_sd(values[metric] for _, values in flat)
            aggregate[f"mean_{metric}"] = value_mean
            aggregate[f"sd_{metric}"] = value_sd
            aggregate[f"n_{metric}"] = count
        aggregate_rows.append(aggregate)

        per_class: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for _, payload in items:
            for row in payload["per_original_modification_test"]:  # type: ignore[index]
                per_class[str(row["mod_type"])].append(row)
        for mod_type, rows in sorted(per_class.items()):
            output: dict[str, object] = {
                "variant": variant,
                "mod_type": mod_type,
                "seed_count": len(rows),
                "positive_rows": rows[0]["positive_rows"],
            }
            for metric in ("AUPR", "AUC", "F1", "MCC", "Brier", "ECE"):
                value_mean, value_sd, _ = mean_sd(float(row[metric]) for row in rows)
                output[f"mean_{metric}"] = value_mean
                output[f"sd_{metric}"] = value_sd
            class_rows.append(output)

    effect_rows: list[dict[str, object]] = []
    for model_a, model_b, comparison in PAIR_COMPARISONS:
        shared_seeds = sorted(
            seed for variant, seed in records if variant == model_a and (model_b, seed) in records
        )
        if not shared_seeds:
            continue
        for metric in SUMMARY_METRICS:
            deltas = [
                flatten_metrics(records[(model_a, seed)])[metric]
                - flatten_metrics(records[(model_b, seed)])[metric]
                for seed in shared_seeds
            ]
            delta_mean, delta_sd, count = mean_sd(deltas)
            effect_rows.append(
                {
                    "comparison": comparison,
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "paired_seed_count": count,
                    "mean_delta_a_minus_b": delta_mean,
                    "sd_delta": delta_sd,
                    "seed_deltas": ",".join(f"{seed}:{delta:.8g}" for seed, delta in zip(shared_seeds, deltas)),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "sure_rm_seed_metrics.tsv", seed_rows)
    write_tsv(args.output_dir / "sure_rm_variant_mean_sd.tsv", aggregate_rows)
    write_tsv(args.output_dir / "sure_rm_per_modification_mean_sd.tsv", class_rows)
    write_tsv(args.output_dir / "sure_rm_matched_seed_effects.tsv", effect_rows)

    candidate_rows = [row for row in aggregate_rows if "control" not in str(row["variant"])]
    selected = max(
        candidate_rows,
        key=lambda row: float(row["mean_validation_macro_stable_AUPR"]),
    )
    selection = {
        "selection_rule": "highest mean validation macro-stable AUPR across completed seeds; controls excluded",
        "selected_variant": selected["variant"],
        "completed_seed_count": selected["seed_count"],
        "mean_validation_macro_stable_AUPR": selected["mean_validation_macro_stable_AUPR"],
        "test_metrics_excluded_from_selection": True,
    }
    (args.output_dir / "sure_rm_validation_selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("rmbase12_work/detection_strict_20260814/models"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rmbase12_work/detection_strict_20260814/summary/sure_rm"),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
