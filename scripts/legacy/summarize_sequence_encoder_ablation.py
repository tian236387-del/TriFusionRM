#!/usr/bin/env python
"""Summarize motif-vs-Transformer sequence encoder ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


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
    return "" if value_f is None else f"{value_f:.6f}"


def diff(left: object, right: object) -> object:
    left_f = as_float(left)
    right_f = as_float(right)
    if left_f is None or right_f is None:
        return ""
    return f"{left_f - right_f:.6f}"


def nested_get(data: Mapping[str, object], path: Sequence[str], default: object = "") -> object:
    current: object = data
    for item in path:
        if not isinstance(current, Mapping) or item not in current:
            return default
        current = current[item]
    return current


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def default_items(reference_root: Path, root: Path) -> List[Tuple[str, Path]]:
    return [
        ("motif_sequence_only", reference_root / "sequence_only" / "metrics.json"),
        ("motif_rnafold_mfe", reference_root / "rnafold_mfe" / "metrics.json"),
        ("motif_rnafold_mfe_shuffled", reference_root / "rnafold_mfe_shuffled" / "metrics.json"),
        ("hybrid_sequence_only", root / "transformer_hybrid_sequence_only" / "metrics.json"),
        ("hybrid_rnafold_mfe", root / "transformer_hybrid_rnafold_mfe" / "metrics.json"),
        ("hybrid_rnafold_mfe_shuffled", root / "transformer_hybrid_rnafold_mfe_shuffled" / "metrics.json"),
    ]


def get_classes(metrics_by_name: Mapping[str, Mapping[str, object]]) -> List[str]:
    seen: List[str] = []
    for metrics in metrics_by_name.values():
        per_class = nested_get(metrics, ["test", "per_class"], {})
        if not isinstance(per_class, Mapping):
            continue
        for name in per_class.keys():
            name_s = str(name)
            if name_s not in seen:
                seen.append(name_s)
    return seen


def metric(metrics_by_name: Mapping[str, Mapping[str, object]], name: str, split: str, key: str) -> object:
    return nested_get(metrics_by_name.get(name, {}), [split, key], "")


def class_metric(metrics_by_name: Mapping[str, Mapping[str, object]], name: str, mod_type: str, key: str) -> object:
    return nested_get(metrics_by_name.get(name, {}), ["test", "per_class", mod_type, key], "")


def build_macro_rows(items: Sequence[Tuple[str, Path]], metrics_by_name: Mapping[str, Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    motif_seq = metric(metrics_by_name, "motif_sequence_only", "test", "macro_AUPR_ovr")
    motif_full = metric(metrics_by_name, "motif_rnafold_mfe", "test", "macro_AUPR_ovr")
    hybrid_seq = metric(metrics_by_name, "hybrid_sequence_only", "test", "macro_AUPR_ovr")
    hybrid_full = metric(metrics_by_name, "hybrid_rnafold_mfe", "test", "macro_AUPR_ovr")
    hybrid_shuf = metric(metrics_by_name, "hybrid_rnafold_mfe_shuffled", "test", "macro_AUPR_ovr")
    for name, path in items:
        metrics = metrics_by_name.get(name, {})
        row: Dict[str, object] = {
            "variant": name,
            "metrics_file": str(path),
            "exists": bool(metrics),
            "best_epoch": metrics.get("best_epoch", "") if isinstance(metrics, Mapping) else "",
            "best_validation_score": fmt(metrics.get("best_validation_score", "") if isinstance(metrics, Mapping) else ""),
        }
        for split in ["validation", "test"]:
            for key in MACRO_KEYS:
                row[f"{split}_{key}"] = fmt(metric(metrics_by_name, name, split, key))
        row["test_macro_AUPR_minus_motif_sequence_only"] = diff(metric(metrics_by_name, name, "test", "macro_AUPR_ovr"), motif_seq)
        row["test_macro_AUPR_minus_motif_rnafold_mfe"] = diff(metric(metrics_by_name, name, "test", "macro_AUPR_ovr"), motif_full)
        rows.append(row)
    rows.append(
        {
            "variant": "delta_hybrid_encoder_on_sequence_only",
            "metrics_file": "",
            "exists": True,
            "test_macro_AUPR_ovr": diff(hybrid_seq, motif_seq),
        }
    )
    rows.append(
        {
            "variant": "delta_hybrid_full_vs_motif_full",
            "metrics_file": "",
            "exists": True,
            "test_macro_AUPR_ovr": diff(hybrid_full, motif_full),
        }
    )
    rows.append(
        {
            "variant": "delta_hybrid_structure_vs_hybrid_sequence_only",
            "metrics_file": "",
            "exists": True,
            "test_macro_AUPR_ovr": diff(hybrid_full, hybrid_seq),
        }
    )
    rows.append(
        {
            "variant": "delta_hybrid_structure_vs_hybrid_shuffled",
            "metrics_file": "",
            "exists": True,
            "test_macro_AUPR_ovr": diff(hybrid_full, hybrid_shuf),
        }
    )
    return rows


def build_per_class_rows(classes: Sequence[str], metrics_by_name: Mapping[str, Mapping[str, object]]) -> List[Dict[str, object]]:
    variants = [
        "motif_sequence_only",
        "motif_rnafold_mfe",
        "motif_rnafold_mfe_shuffled",
        "hybrid_sequence_only",
        "hybrid_rnafold_mfe",
        "hybrid_rnafold_mfe_shuffled",
    ]
    rows: List[Dict[str, object]] = []
    for mod_type in classes:
        row: Dict[str, object] = {
            "mod_type": mod_type,
            "support": fmt(class_metric(metrics_by_name, "motif_sequence_only", mod_type, "support")),
        }
        for variant in variants:
            for key in PER_CLASS_KEYS:
                row[f"{variant}_{key}"] = fmt(class_metric(metrics_by_name, variant, mod_type, key))
        row["hybrid_sequence_only_minus_motif_sequence_only_AUPR"] = diff(
            class_metric(metrics_by_name, "hybrid_sequence_only", mod_type, "AUPR_ovr"),
            class_metric(metrics_by_name, "motif_sequence_only", mod_type, "AUPR_ovr"),
        )
        row["hybrid_rnafold_mfe_minus_motif_rnafold_mfe_AUPR"] = diff(
            class_metric(metrics_by_name, "hybrid_rnafold_mfe", mod_type, "AUPR_ovr"),
            class_metric(metrics_by_name, "motif_rnafold_mfe", mod_type, "AUPR_ovr"),
        )
        row["hybrid_rnafold_mfe_minus_hybrid_sequence_only_AUPR"] = diff(
            class_metric(metrics_by_name, "hybrid_rnafold_mfe", mod_type, "AUPR_ovr"),
            class_metric(metrics_by_name, "hybrid_sequence_only", mod_type, "AUPR_ovr"),
        )
        row["hybrid_rnafold_mfe_minus_hybrid_shuffled_AUPR"] = diff(
            class_metric(metrics_by_name, "hybrid_rnafold_mfe", mod_type, "AUPR_ovr"),
            class_metric(metrics_by_name, "hybrid_rnafold_mfe_shuffled", mod_type, "AUPR_ovr"),
        )
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize motif-vs-Transformer sequence encoder ablations.")
    parser.add_argument("--reference-root", default="rmbase11_full_work/models/structure_ablation_cdhit80")
    parser.add_argument("--root", default="rmbase11_full_work/models/sequence_encoder_ablation_cdhit80")
    parser.add_argument("--out-dir", default="rmbase11_full_work/metrics/sequence_encoder_ablation_cdhit80")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    reference_root = Path(args.reference_root)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    items = default_items(reference_root, root)
    metrics_by_name = {name: load_json(path) for name, path in items}
    classes = get_classes(metrics_by_name)
    macro_rows = build_macro_rows(items, metrics_by_name)
    per_class_rows = build_per_class_rows(classes, metrics_by_name)
    write_tsv(out_dir / "sequence_encoder_ablation_macro_metrics.tsv", macro_rows)
    write_tsv(out_dir / "sequence_encoder_ablation_per_class_matrix.tsv", per_class_rows)
    missing = [name for name, metrics in metrics_by_name.items() if not metrics]
    if missing:
        print("Missing metrics.json for: " + ", ".join(missing))
    print(f"Wrote sequence encoder ablation summary to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
