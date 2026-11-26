#!/usr/bin/env python
"""Summarize structure-source ablations for MotifStructFusionV4."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_VARIANTS = [
    "sequence_only",
    "rnafold_mfe",
    "rnafold_mfe_shuffled",
    "rnafold_prob",
    "rnafold_prob_shuffled",
    "rnafold_loop",
    "rnafold_loop_shuffled",
    "rnaplfold_accessibility",
    "rnaplfold_accessibility_shuffled",
    "rnastructure_fold",
    "rnastructure_fold_shuffled",
]

MACRO_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_AUPR_ovr",
    "macro_AUC_ovr",
    "macro_F1",
    "weighted_F1",
    "MCC",
]

PER_CLASS_KEYS = ["support", "AUPR_ovr", "AUC_ovr", "F1", "Precision", "Recall", "MCC_ovr"]


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def nested_get(data: Mapping[str, object], path: Sequence[str], default: object = "") -> object:
    current: object = data
    for item in path:
        if not isinstance(current, Mapping) or item not in current:
            return default
        current = current[item]
    return current


def as_float(value: object) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_f):
        return None
    return value_f


def fmt(value: object) -> object:
    value_f = as_float(value)
    if value_f is None:
        return ""
    return f"{value_f:.6f}"


def diff(left: object, right: object) -> object:
    left_f = as_float(left)
    right_f = as_float(right)
    if left_f is None or right_f is None:
        return ""
    return f"{left_f - right_f:.6f}"


def diff_float(left: object, right: object) -> Optional[float]:
    left_f = as_float(left)
    right_f = as_float(right)
    if left_f is None or right_f is None:
        return None
    return left_f - right_f


def parse_variants(value: str) -> List[str]:
    if not value.strip():
        return list(DEFAULT_VARIANTS)
    return [item.strip() for item in value.split(",") if item.strip()]


def get_classes(metrics_by_variant: Mapping[str, Mapping[str, object]]) -> List[str]:
    seen: List[str] = []
    for metrics in metrics_by_variant.values():
        per_class = nested_get(metrics, ["test", "per_class"], {})
        if not isinstance(per_class, Mapping):
            continue
        for name in per_class.keys():
            if str(name) not in seen:
                seen.append(str(name))
    return seen


def metric(metrics_by_variant: Mapping[str, Mapping[str, object]], variant: str, split: str, key: str) -> object:
    return nested_get(metrics_by_variant.get(variant, {}), [split, key], "")


def class_metric(metrics_by_variant: Mapping[str, Mapping[str, object]], variant: str, mod_type: str, key: str) -> object:
    return nested_get(metrics_by_variant.get(variant, {}), ["test", "per_class", mod_type, key], "")


def build_macro_rows(root: Path, variants: Sequence[str], metrics_by_variant: Mapping[str, Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sequence_test_aupr = metric(metrics_by_variant, "sequence_only", "test", "macro_AUPR_ovr")
    for variant in variants:
        metrics = metrics_by_variant.get(variant, {})
        row_counts = metrics.get("row_counts", {}) if isinstance(metrics, Mapping) else {}
        early = metrics.get("early_stopping", {}) if isinstance(metrics, Mapping) else {}
        if not isinstance(row_counts, Mapping):
            row_counts = {}
        if not isinstance(early, Mapping):
            early = {}
        row: Dict[str, object] = {
            "variant": variant,
            "metrics_file": str(root / variant / "metrics.json"),
            "exists": bool(metrics),
            "structure_file": metrics.get("rnafold", "") if isinstance(metrics, Mapping) else "",
            "dataset": metrics.get("dataset", "") if isinstance(metrics, Mapping) else "",
            "split": metrics.get("split", "") if isinstance(metrics, Mapping) else "",
            "best_epoch": metrics.get("best_epoch", "") if isinstance(metrics, Mapping) else "",
            "best_validation_score": fmt(metrics.get("best_validation_score", "") if isinstance(metrics, Mapping) else ""),
            "early_stopped": early.get("early_stopped", ""),
            "train_rows": row_counts.get("train", ""),
            "valid_rows": row_counts.get("valid", ""),
            "test_rows": row_counts.get("test", ""),
        }
        for split in ["validation", "test"]:
            for key in MACRO_KEYS:
                row[f"{split}_{key}"] = fmt(metric(metrics_by_variant, variant, split, key))
        row["test_macro_AUPR_minus_sequence_only"] = diff(metric(metrics_by_variant, variant, "test", "macro_AUPR_ovr"), sequence_test_aupr)
        rows.append(row)
    return rows


def shuffled_for_variant(variant: str) -> str:
    if variant.endswith("_shuffled"):
        return ""
    candidate = f"{variant}_shuffled"
    aliases = {
        "rnafold_mfe": "rnafold_mfe_shuffled",
        "rnafold_prob": "rnafold_prob_shuffled",
        "rnafold_loop": "rnafold_loop_shuffled",
        "rnaplfold_accessibility": "rnaplfold_accessibility_shuffled",
        "rnastructure_fold": "rnastructure_fold_shuffled",
    }
    return aliases.get(variant, candidate)


def build_per_class_matrix(
    variants: Sequence[str],
    classes: Sequence[str],
    metrics_by_variant: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for mod_type in classes:
        row: Dict[str, object] = {
            "mod_type": mod_type,
            "support": fmt(class_metric(metrics_by_variant, "sequence_only", mod_type, "support")),
        }
        for variant in variants:
            for key in PER_CLASS_KEYS:
                row[f"{variant}_{key}"] = fmt(class_metric(metrics_by_variant, variant, mod_type, key))
        for variant in variants:
            if variant == "sequence_only" or variant.endswith("_shuffled"):
                continue
            row[f"{variant}_minus_sequence_only_AUPR"] = diff(
                class_metric(metrics_by_variant, variant, mod_type, "AUPR_ovr"),
                class_metric(metrics_by_variant, "sequence_only", mod_type, "AUPR_ovr"),
            )
            shuffled = shuffled_for_variant(variant)
            if shuffled in variants:
                row[f"{variant}_minus_{shuffled}_AUPR"] = diff(
                    class_metric(metrics_by_variant, variant, mod_type, "AUPR_ovr"),
                    class_metric(metrics_by_variant, shuffled, mod_type, "AUPR_ovr"),
                )
        rows.append(row)
    return rows


def build_decision_rows(
    variants: Sequence[str],
    classes: Sequence[str],
    metrics_by_variant: Mapping[str, Mapping[str, object]],
    min_support: int,
    min_delta: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for variant in variants:
        if variant == "sequence_only" or variant.endswith("_shuffled"):
            continue
        shuffled = shuffled_for_variant(variant)
        row: Dict[str, object] = {
            "variant": variant,
            "shuffled_control": shuffled if shuffled in variants else "",
            "test_macro_AUPR": fmt(metric(metrics_by_variant, variant, "test", "macro_AUPR_ovr")),
            "test_macro_AUPR_minus_sequence_only": diff(
                metric(metrics_by_variant, variant, "test", "macro_AUPR_ovr"),
                metric(metrics_by_variant, "sequence_only", "test", "macro_AUPR_ovr"),
            ),
            "test_macro_AUPR_minus_shuffled": diff(
                metric(metrics_by_variant, variant, "test", "macro_AUPR_ovr"),
                metric(metrics_by_variant, shuffled, "test", "macro_AUPR_ovr"),
            )
            if shuffled in variants
            else "",
        }
        supported = 0
        wins_vs_seq = 0
        wins_vs_shuffled = 0
        stable_classes: List[str] = []
        for mod_type in classes:
            support = as_float(class_metric(metrics_by_variant, "sequence_only", mod_type, "support"))
            if support is None or support < min_support:
                continue
            supported += 1
            delta_seq = diff_float(
                class_metric(metrics_by_variant, variant, mod_type, "AUPR_ovr"),
                class_metric(metrics_by_variant, "sequence_only", mod_type, "AUPR_ovr"),
            )
            delta_shuf = diff_float(
                class_metric(metrics_by_variant, variant, mod_type, "AUPR_ovr"),
                class_metric(metrics_by_variant, shuffled, mod_type, "AUPR_ovr"),
            ) if shuffled in variants else None
            if delta_seq is not None and delta_seq >= min_delta:
                wins_vs_seq += 1
            if delta_shuf is not None and delta_shuf >= min_delta:
                wins_vs_shuffled += 1
            if (
                delta_seq is not None
                and delta_seq >= min_delta
                and (delta_shuf is None or delta_shuf >= min_delta)
            ):
                stable_classes.append(mod_type)
        row["supported_classes_evaluated"] = supported
        row[f"classes_with_AUPR_gain_ge_{min_delta:g}_vs_sequence"] = wins_vs_seq
        row[f"classes_with_AUPR_gain_ge_{min_delta:g}_vs_shuffled"] = wins_vs_shuffled
        row["stable_positive_classes"] = ",".join(stable_classes)
        rows.append(row)
    return rows


def build_report(out_dir: Path, decision_rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Structure Ablation Summary",
        "",
        "Use `test_macro_AUPR_minus_sequence_only` and `test_macro_AUPR_minus_shuffled` together.",
        "A structure source is useful only if real features beat both sequence-only and shuffled-structure controls.",
        "",
        "## Output files",
        "",
        f"- `{out_dir / 'structure_ablation_macro_metrics.tsv'}`",
        f"- `{out_dir / 'structure_ablation_per_class_matrix.tsv'}`",
        f"- `{out_dir / 'structure_ablation_decision_table.tsv'}`",
        "",
        "## Decision Table",
        "",
        "| variant | macro AUPR | vs sequence | vs shuffled | stable positive classes |",
        "|---|---:|---:|---:|---|",
    ]
    for row in decision_rows:
        lines.append(
            "| {variant} | {aupr} | {dseq} | {dshuf} | {classes} |".format(
                variant=row.get("variant", ""),
                aupr=row.get("test_macro_AUPR", ""),
                dseq=row.get("test_macro_AUPR_minus_sequence_only", ""),
                dshuf=row.get("test_macro_AUPR_minus_shuffled", ""),
                classes=row.get("stable_positive_classes", ""),
            )
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize structure-source ablation experiments.")
    parser.add_argument("--root", default="rmbase11_full_work/models/structure_ablation_cdhit80")
    parser.add_argument("--out-dir", default="rmbase11_full_work/metrics/structure_ablation_cdhit80")
    parser.add_argument("--variants", default="")
    parser.add_argument("--min-support", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=0.005)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    variants = parse_variants(args.variants)
    metrics_by_variant = {variant: load_json(root / variant / "metrics.json") for variant in variants}
    classes = get_classes(metrics_by_variant)
    macro_rows = build_macro_rows(root, variants, metrics_by_variant)
    matrix_rows = build_per_class_matrix(variants, classes, metrics_by_variant)
    decision_rows = build_decision_rows(variants, classes, metrics_by_variant, args.min_support, args.min_delta)

    write_tsv(out_dir / "structure_ablation_macro_metrics.tsv", macro_rows)
    write_tsv(out_dir / "structure_ablation_per_class_matrix.tsv", matrix_rows)
    write_tsv(out_dir / "structure_ablation_decision_table.tsv", decision_rows)
    (out_dir / "structure_ablation_summary.md").write_text(build_report(out_dir, decision_rows), encoding="utf-8")

    missing = [variant for variant, metrics in metrics_by_variant.items() if not metrics]
    if missing:
        print("Missing metrics.json for: " + ", ".join(missing))
    print(f"Wrote structure ablation summary to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
