#!/usr/bin/env python
"""Summarize the five MotifStructFusionV4 ablation experiments.

Expected directory layout:

    <root>/full/metrics.json
    <root>/sequence_only/metrics.json
    <root>/shuffled_structure/metrics.json
    <root>/no_position/metrics.json
    <root>/with_rbp/metrics.json

Optional variants such as ``with_rbp_positional`` are included when requested.
The script writes macro-level and per-class TSV tables that are ready for
paper figures or supplementary tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_VARIANTS = ["sequence_only", "full", "shuffled_structure", "no_position", "with_rbp"]
OPTIONAL_VARIANTS = ["with_rbp_positional", "shuffled_rbp_positional"]
DEFAULT_CLASSES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]

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


def load_tsv_by_key(path: Path, key: str) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or key not in reader.fieldnames:
            return {}
        return {row[key]: row for row in reader}


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


def fmt_float(value: object) -> object:
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


def get_classes(metrics_by_variant: Mapping[str, Mapping[str, object]]) -> List[str]:
    seen: List[str] = []
    for metrics in metrics_by_variant.values():
        per_class = nested_get(metrics, ["test", "per_class"], {})
        if not isinstance(per_class, Mapping):
            continue
        for name in per_class.keys():
            if str(name) not in seen:
                seen.append(str(name))
    return seen or DEFAULT_CLASSES


def get_per_class_value(metrics: Mapping[str, object], mod_type: str, key: str) -> object:
    value = nested_get(metrics, ["test", "per_class", mod_type, key], "")
    return value


def get_variant_dirs(root: Path, variants_arg: Optional[str], include_optional: bool) -> List[str]:
    if variants_arg:
        return [item.strip() for item in variants_arg.split(",") if item.strip()]
    variants = list(DEFAULT_VARIANTS)
    if include_optional:
        variants.extend([name for name in OPTIONAL_VARIANTS if (root / name / "metrics.json").exists()])
    return variants


def build_macro_rows(root: Path, variants: Sequence[str], metrics_by_variant: Mapping[str, Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for variant in variants:
        metrics = metrics_by_variant.get(variant, {})
        row_counts = metrics.get("row_counts", {})
        early = metrics.get("early_stopping", {})
        if not isinstance(row_counts, Mapping):
            row_counts = {}
        if not isinstance(early, Mapping):
            early = {}
        row: Dict[str, object] = {
            "variant": variant,
            "metrics_file": str(root / variant / "metrics.json"),
            "exists": bool(metrics),
            "dataset": metrics.get("dataset", ""),
            "split": metrics.get("split", ""),
            "best_epoch": metrics.get("best_epoch", ""),
            "best_validation_score": fmt_float(metrics.get("best_validation_score", "")),
            "early_stopped": early.get("early_stopped", ""),
            "early_stop_patience": early.get("patience", ""),
            "early_stop_min_delta": early.get("min_delta", ""),
            "train_rows": row_counts.get("train", ""),
            "valid_rows": row_counts.get("valid", ""),
            "test_rows": row_counts.get("test", ""),
        }
        for split_name in ["validation", "test"]:
            split_metrics = metrics.get(split_name, {})
            if not isinstance(split_metrics, Mapping):
                split_metrics = {}
            for key in MACRO_KEYS:
                row[f"{split_name}_{key}"] = fmt_float(split_metrics.get(key, ""))
        rows.append(row)
    return rows


def build_per_class_rows(
    root: Path,
    variants: Sequence[str],
    classes: Sequence[str],
    metrics_by_variant: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for variant in variants:
        metrics = metrics_by_variant.get(variant, {})
        for mod_type in classes:
            row: Dict[str, object] = {"variant": variant, "mod_type": mod_type}
            for key in PER_CLASS_KEYS:
                row[key] = fmt_float(get_per_class_value(metrics, mod_type, key))
            rows.append(row)
    return rows


def build_aupr_matrix(
    root: Path,
    variants: Sequence[str],
    classes: Sequence[str],
    metrics_by_variant: Mapping[str, Mapping[str, object]],
    full_variant: str,
    sequence_variant: str,
    shuffled_structure_variant: str,
    no_position_variant: str,
    rbp_variant: str,
    rbp_positional_variant: str,
) -> List[Dict[str, object]]:
    gate_tables = {
        variant: {
            "structure": load_tsv_by_key(root / variant / "test_per_class_structure_gate.tsv", "mod_type"),
            "rbp_positional": load_tsv_by_key(root / variant / "test_per_class_rbp_positional_gate.tsv", "mod_type"),
            "modality": load_tsv_by_key(root / variant / "test_per_class_modality_gate.tsv", "mod_type"),
        }
        for variant in variants
    }
    rows: List[Dict[str, object]] = []
    full_metrics = metrics_by_variant.get(full_variant, {})
    for mod_type in classes:
        row: Dict[str, object] = {
            "mod_type": mod_type,
            "support": fmt_float(get_per_class_value(full_metrics, mod_type, "support")),
        }
        for variant in variants:
            metrics = metrics_by_variant.get(variant, {})
            row[f"{variant}_AUPR"] = fmt_float(get_per_class_value(metrics, mod_type, "AUPR_ovr"))
            row[f"{variant}_AUC"] = fmt_float(get_per_class_value(metrics, mod_type, "AUC_ovr"))
            row[f"{variant}_F1"] = fmt_float(get_per_class_value(metrics, mod_type, "F1"))
            row[f"{variant}_MCC_ovr"] = fmt_float(get_per_class_value(metrics, mod_type, "MCC_ovr"))

        if sequence_variant in variants and full_variant in variants:
            row[f"{full_variant}_minus_{sequence_variant}_AUPR"] = diff(
                get_per_class_value(full_metrics, mod_type, "AUPR_ovr"),
                get_per_class_value(metrics_by_variant.get(sequence_variant, {}), mod_type, "AUPR_ovr"),
            )
        if shuffled_structure_variant in variants and full_variant in variants:
            row[f"{full_variant}_minus_{shuffled_structure_variant}_AUPR"] = diff(
                get_per_class_value(full_metrics, mod_type, "AUPR_ovr"),
                get_per_class_value(metrics_by_variant.get(shuffled_structure_variant, {}), mod_type, "AUPR_ovr"),
            )
        if no_position_variant in variants and full_variant in variants:
            row[f"{full_variant}_minus_{no_position_variant}_AUPR"] = diff(
                get_per_class_value(full_metrics, mod_type, "AUPR_ovr"),
                get_per_class_value(metrics_by_variant.get(no_position_variant, {}), mod_type, "AUPR_ovr"),
            )
        if rbp_variant in variants and full_variant in variants:
            row[f"{rbp_variant}_minus_{full_variant}_AUPR"] = diff(
                get_per_class_value(metrics_by_variant.get(rbp_variant, {}), mod_type, "AUPR_ovr"),
                get_per_class_value(full_metrics, mod_type, "AUPR_ovr"),
            )
        if rbp_positional_variant in variants and full_variant in variants:
            row[f"{rbp_positional_variant}_minus_{full_variant}_AUPR"] = diff(
                get_per_class_value(metrics_by_variant.get(rbp_positional_variant, {}), mod_type, "AUPR_ovr"),
                get_per_class_value(full_metrics, mod_type, "AUPR_ovr"),
            )

        full_structure_gate = gate_tables.get(full_variant, {}).get("structure", {}).get(mod_type, {})
        row[f"{full_variant}_structure_gate_mean"] = full_structure_gate.get("mean_gate_for_class_logit", "")
        row[f"{full_variant}_structure_gate_true_samples"] = full_structure_gate.get("mean_gate_on_true_samples", "")

        gate_rbp_variant = rbp_positional_variant if rbp_positional_variant in variants else rbp_variant
        modality_gate = gate_tables.get(gate_rbp_variant, {}).get("modality", {}).get(mod_type, {})
        rbp_pos_gate = gate_tables.get(gate_rbp_variant, {}).get("rbp_positional", {}).get(mod_type, {})
        row["class_gate_sequence_mean"] = modality_gate.get("mean_sequence_gate_for_class_logit", "")
        row["class_gate_structure_mean"] = modality_gate.get("mean_structure_gate_for_class_logit", "")
        row["class_gate_rbp_positional_mean"] = modality_gate.get("mean_rbp_pos_gate_for_class_logit", "")
        row["rbp_positional_gate_mean"] = rbp_pos_gate.get("mean_gate_for_class_logit", "")
        row["rbp_positional_gate_true_samples"] = rbp_pos_gate.get("mean_gate_on_true_samples", "")
        rows.append(row)
    return rows


def build_markdown_report(out_dir: Path, variants: Sequence[str], macro_rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Five-experiment metric summary",
        "",
        "## Variants",
        "",
    ]
    descriptions = {
        "sequence_only": "Sequence-only baseline.",
        "full": "Sequence plus RNAfold structure with class-specific gates.",
        "shuffled_structure": "Structure-control model with shuffled RNAfold features.",
        "no_position": "Position-ablation model without position or center-distance coding.",
        "with_rbp": "Full model plus global RBP features.",
        "with_rbp_positional": "Full model plus positional RBP features.",
        "shuffled_rbp_positional": "RBP positional-control model with shuffled positional RBP features.",
    }
    for variant in variants:
        lines.append(f"- `{variant}`: {descriptions.get(variant, 'Custom variant.')}")
    lines.extend(
        [
            "",
            "## Main output files",
            "",
            f"- `{out_dir / 'five_experiment_macro_metrics.tsv'}`",
            f"- `{out_dir / 'five_experiment_per_class_metrics.tsv'}`",
            f"- `{out_dir / 'five_experiment_per_class_aupr_matrix.tsv'}`",
            "",
            "## Test macro metrics",
            "",
            "| variant | test macro AUPR | test macro AUC | test macro F1 | test MCC | best epoch | early stopped |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in macro_rows:
        lines.append(
            "| {variant} | {aupr} | {auc} | {f1} | {mcc} | {epoch} | {stopped} |".format(
                variant=row.get("variant", ""),
                aupr=row.get("test_macro_AUPR_ovr", ""),
                auc=row.get("test_macro_AUC_ovr", ""),
                f1=row.get("test_macro_F1", ""),
                mcc=row.get("test_MCC", ""),
                epoch=row.get("best_epoch", ""),
                stopped=row.get("early_stopped", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Recommended comparisons",
            "",
            "- `full_minus_sequence_only_AUPR`: net contribution of RNAfold structure.",
            "- `full_minus_shuffled_structure_AUPR`: whether real structure beats a shuffled-structure control.",
            "- `full_minus_no_position_AUPR`: contribution of position and center-distance coding.",
            "- `with_rbp_minus_full_AUPR`: extra value from global RBP features.",
            "- `with_rbp_positional_minus_full_AUPR`: extra value from positional RBP features, when available.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize five MotifStructFusionV4 experiment outputs.")
    parser.add_argument(
        "--root",
        default="rmbase11_full_work/models/motif_struct_fusion_v4_class_modality_gate_v3",
        help="Directory containing variant subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        default="rmbase11_full_work/metrics/five_experiment_summary",
        help="Output directory for summary tables.",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variant list. Defaults to the five main experiments.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also include optional variants when their metrics.json files exist.",
    )
    parser.add_argument("--full-variant", default="full", help="Variant used as the full sequence+structure model.")
    parser.add_argument("--sequence-variant", default="sequence_only", help="Variant used as the sequence-only baseline.")
    parser.add_argument("--shuffled-structure-variant", default="shuffled_structure", help="Variant used as the shuffled-structure control.")
    parser.add_argument("--no-position-variant", default="no_position", help="Variant used as the no-position ablation.")
    parser.add_argument("--rbp-variant", default="with_rbp", help="Variant used as the global RBP model.")
    parser.add_argument("--rbp-positional-variant", default="with_rbp_positional", help="Variant used as the positional RBP model.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    variants = get_variant_dirs(root, args.variants or None, args.include_optional)
    metrics_by_variant = {variant: load_json(root / variant / "metrics.json") for variant in variants}
    classes = get_classes(metrics_by_variant)

    macro_rows = build_macro_rows(root, variants, metrics_by_variant)
    per_class_rows = build_per_class_rows(root, variants, classes, metrics_by_variant)
    aupr_matrix_rows = build_aupr_matrix(
        root,
        variants,
        classes,
        metrics_by_variant,
        args.full_variant,
        args.sequence_variant,
        args.shuffled_structure_variant,
        args.no_position_variant,
        args.rbp_variant,
        args.rbp_positional_variant,
    )

    write_tsv(out_dir / "five_experiment_macro_metrics.tsv", macro_rows)
    write_tsv(out_dir / "five_experiment_per_class_metrics.tsv", per_class_rows)
    write_tsv(out_dir / "five_experiment_per_class_aupr_matrix.tsv", aupr_matrix_rows)
    (out_dir / "five_experiment_summary.md").write_text(
        build_markdown_report(out_dir, variants, macro_rows),
        encoding="utf-8",
    )

    missing = [variant for variant, metrics in metrics_by_variant.items() if not metrics]
    if missing:
        print("Missing metrics.json for: " + ", ".join(missing))
    print(f"Wrote summary tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
