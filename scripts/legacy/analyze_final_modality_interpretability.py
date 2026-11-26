#!/usr/bin/env python
"""Summarize structure strata and RBP distance evidence for the final model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

from paired_bootstrap_final_modality import align_to_reference, load_predictions


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def loop_span_group(value: float) -> str:
    if value <= 0:
        return "0"
    if value <= 5:
        return "1_5"
    if value <= 10:
        return "6_10"
    if value <= 20:
        return "11_20"
    return "gt20"


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_structure(path: Path, site_ids: set[str]) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "site_id" not in reader.fieldnames:
            raise ValueError(f"Invalid RNAfold table: {path}")
        for row in reader:
            site_id = str(row["site_id"])
            if site_id in site_ids:
                rows[site_id] = row
    missing = site_ids.difference(rows)
    if missing:
        raise ValueError(f"RNAfold rows missing for {len(missing)} test sites; first={next(iter(missing))}")
    return rows


def summarize_true_class_gains(full, sequence, structure_by_id) -> List[Dict[str, object]]:
    full_true = full.probabilities[np.arange(len(full.labels)), full.labels]
    sequence_true = sequence.probabilities[np.arange(len(sequence.labels)), sequence.labels]
    delta = full_true - sequence_true
    rows: List[Dict[str, object]] = []
    groups: Dict[tuple[str, str, str], List[float]] = {}
    for index, site_id in enumerate(full.site_ids):
        class_name = full.class_names[int(full.labels[index])]
        structure = structure_by_id[site_id]
        center_state = "paired" if parse_float(structure.get("center_paired")) >= 0.5 else "unpaired"
        span_group = loop_span_group(parse_float(structure.get("loop_span_around_center")))
        groups.setdefault((class_name, center_state, span_group), []).append(float(delta[index]))
    for (class_name, center_state, span_group), values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "mod_type": class_name,
                "center_state": center_state,
                "loop_span_group": span_group,
                "support": len(values),
                "mean_full_minus_sequence_true_probability": float(array.mean()),
                "median_full_minus_sequence_true_probability": float(np.median(array)),
                "fraction_positive_gain": float(np.mean(array > 0)),
                "standard_error": float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else float("nan"),
            }
        )
    return rows


def summarize_stratified_aupr(full, sequence, structure_by_id) -> List[Dict[str, object]]:
    center_states = np.asarray(
        ["paired" if parse_float(structure_by_id[site_id].get("center_paired")) >= 0.5 else "unpaired" for site_id in full.site_ids]
    )
    span_groups = np.asarray(
        [loop_span_group(parse_float(structure_by_id[site_id].get("loop_span_around_center"))) for site_id in full.site_ids]
    )
    rows: List[Dict[str, object]] = []
    for class_index, class_name in enumerate(full.class_names):
        for stratum_name, stratum_values in (("center_state", center_states), ("loop_span_group", span_groups)):
            for value in sorted(set(stratum_values.tolist())):
                mask = stratum_values == value
                binary = (full.labels[mask] == class_index).astype(np.int8)
                support = int(binary.sum())
                negatives = int(mask.sum() - support)
                if support == 0 or negatives == 0:
                    full_aupr = sequence_aupr = float("nan")
                else:
                    full_aupr = float(average_precision_score(binary, full.probabilities[mask, class_index]))
                    sequence_aupr = float(average_precision_score(binary, sequence.probabilities[mask, class_index]))
                rows.append(
                    {
                        "mod_type": class_name,
                        "stratum": stratum_name,
                        "stratum_value": value,
                        "positive_support": support,
                        "negative_support": negatives,
                        "full_AUPR": full_aupr,
                        "sequence_only_AUPR": sequence_aupr,
                        "full_minus_sequence_AUPR": full_aupr - sequence_aupr,
                    }
                )
    return rows


def summarize_rbp_attention(path: Path) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"mod_type", "rbp", "bin_left", "bin_right", "mean_attention", "distance_bin"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Invalid RBP attention table: {path}")
        for row in reader:
            grouped.setdefault((str(row["mod_type"]), str(row["rbp"])), []).append(row)
    rows: List[Dict[str, object]] = []
    by_class: Dict[str, List[Dict[str, object]]] = {}
    for (mod_type, rbp), bin_rows in grouped.items():
        center = [
            parse_float(row["mean_attention"])
            for row in bin_rows
            if parse_float(row["bin_left"]) >= -50 and parse_float(row["bin_right"]) <= 50
        ]
        distal = [
            parse_float(row["mean_attention"])
            for row in bin_rows
            if not (parse_float(row["bin_left"]) >= -50 and parse_float(row["bin_right"]) <= 50)
        ]
        all_values = [parse_float(row["mean_attention"]) for row in bin_rows]
        best = max(bin_rows, key=lambda row: parse_float(row["mean_attention"]))
        row: Dict[str, object] = {
            "mod_type": mod_type,
            "rbp": rbp,
            "mean_attention_all_bins": float(np.mean(all_values)),
            "mean_attention_center_100nt": float(np.mean(center)) if center else float("nan"),
            "mean_attention_distal": float(np.mean(distal)) if distal else float("nan"),
            "center_minus_distal_attention": float(np.mean(center) - np.mean(distal)) if center and distal else float("nan"),
            "top_distance_bin": best["distance_bin"],
            "top_bin_attention": parse_float(best["mean_attention"]),
            "attention_rank_within_class": 0,
            "center_enrichment_rank_within_class": 0,
        }
        by_class.setdefault(mod_type, []).append(row)
    for class_rows in by_class.values():
        for rank, row in enumerate(sorted(class_rows, key=lambda item: float(item["mean_attention_all_bins"]), reverse=True), 1):
            row["attention_rank_within_class"] = rank
        for rank, row in enumerate(sorted(class_rows, key=lambda item: float(item["center_minus_distal_attention"]), reverse=True), 1):
            row["center_enrichment_rank_within_class"] = rank
        rows.extend(class_rows)
    return sorted(rows, key=lambda row: (str(row["mod_type"]), int(row["attention_rank_within_class"])))


def run(full_model_dir: Path, sequence_only_dir: Path, rnafold: Path, out_dir: Path) -> None:
    full_path = full_model_dir / "test_predictions.tsv"
    sequence_path = sequence_only_dir / "test_predictions.tsv"
    full = load_predictions(full_path)
    sequence = align_to_reference(full, load_predictions(sequence_path), sequence_path)
    structure_by_id = load_structure(rnafold, set(full.site_ids))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(
        out_dir / "structure_true_class_probability_gain.tsv",
        summarize_true_class_gains(full, sequence, structure_by_id),
    )
    write_tsv(
        out_dir / "structure_stratified_aupr.tsv",
        summarize_stratified_aupr(full, sequence, structure_by_id),
    )
    attention_path = full_model_dir / "test_rbp_positional_attention.tsv"
    if attention_path.is_file():
        attention_rows = summarize_rbp_attention(attention_path)
        write_tsv(out_dir / "rbp_distance_attention_summary.tsv", attention_rows)
        write_tsv(
            out_dir / "rbp_top5_per_class.tsv",
            [row for row in attention_rows if int(row["attention_rank_within_class"]) <= 5],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-model-dir", required=True, type=Path)
    parser.add_argument("--sequence-only-dir", required=True, type=Path)
    parser.add_argument("--rnafold", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.full_model_dir, args.sequence_only_dir, args.rnafold, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
