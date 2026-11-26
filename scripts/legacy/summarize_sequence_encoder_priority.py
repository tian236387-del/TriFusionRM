#!/usr/bin/env python
"""Summarize priority sequence-encoder experiments for StructDualRM."""

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
FOCUS_CLASSES = ("Gm", "m1A", "m7G")


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def fmt(value: object) -> str:
    value_f = as_float(value)
    return "" if value_f is None else f"{value_f:.6f}"


def diff(left: object, right: object) -> str:
    left_f = as_float(left)
    right_f = as_float(right)
    if left_f is None or right_f is None:
        return ""
    return f"{left_f - right_f:.6f}"


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


def parse_variants(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def collect_items(root: Path, variants: Sequence[str], reference_name: str, reference_metrics: str) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    if reference_metrics:
        items.append((reference_name, Path(reference_metrics)))
    for variant in variants:
        items.append((variant, root / variant / "metrics.json"))
    return items


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


def early_stopped(metrics: Mapping[str, object]) -> object:
    value = nested_get(metrics, ["early_stopping", "early_stopped"], "")
    if value == "":
        return ""
    return bool(value)


def build_macro_rows(
    items: Sequence[Tuple[str, Path]],
    metrics_by_name: Mapping[str, Mapping[str, object]],
    reference_name: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    reference_macro = metric(metrics_by_name, reference_name, "test", "macro_AUPR_ovr")
    best_macro = None
    best_name = ""
    for name, path in items:
        metrics = metrics_by_name.get(name, {})
        macro = as_float(metric(metrics_by_name, name, "test", "macro_AUPR_ovr"))
        if macro is not None and (best_macro is None or macro > best_macro):
            best_macro = macro
            best_name = name
        row: Dict[str, object] = {
            "variant": name,
            "metrics_file": str(path),
            "exists": bool(metrics),
            "seq_encoder": nested_get(metrics, ["config", "seq_encoder"], ""),
            "seq_residual_encoder": nested_get(metrics, ["config", "seq_residual_encoder"], ""),
            "seq_residual_scale": fmt(nested_get(metrics, ["config", "seq_residual_scale"], "")),
            "seq_residual_focus_classes": ",".join(str(x) for x in metrics.get("seq_residual_focus_classes", []))
            if isinstance(metrics, Mapping)
            else "",
            "best_epoch": metrics.get("best_epoch", "") if isinstance(metrics, Mapping) else "",
            "best_validation_score": fmt(metrics.get("best_validation_score", "") if isinstance(metrics, Mapping) else ""),
            "early_stopped": early_stopped(metrics) if isinstance(metrics, Mapping) else "",
        }
        for split in ("validation", "test"):
            for key in MACRO_KEYS:
                row[f"{split}_{key}"] = fmt(metric(metrics_by_name, name, split, key))
        row["test_macro_AUPR_minus_reference"] = diff(metric(metrics_by_name, name, "test", "macro_AUPR_ovr"), reference_macro)
        rows.append(row)
    if best_macro is not None:
        rows.append(
            {
                "variant": "best_by_test_macro_AUPR",
                "metrics_file": "",
                "exists": True,
                "seq_encoder": best_name,
                "test_macro_AUPR_ovr": f"{best_macro:.6f}",
                "test_macro_AUPR_minus_reference": diff(best_macro, reference_macro),
            }
        )
    return rows


def build_per_class_rows(
    classes: Sequence[str],
    variants: Sequence[str],
    metrics_by_name: Mapping[str, Mapping[str, object]],
    reference_name: str,
) -> List[Dict[str, object]]:
    names = [reference_name] + [variant for variant in variants if variant != reference_name]
    rows: List[Dict[str, object]] = []
    for mod_type in classes:
        row: Dict[str, object] = {
            "mod_type": mod_type,
            "support": "",
        }
        for name in names:
            support = class_metric(metrics_by_name, name, mod_type, "support")
            if row["support"] == "" and fmt(support):
                row["support"] = fmt(support)
            for key in PER_CLASS_KEYS:
                row[f"{name}_{key}"] = fmt(class_metric(metrics_by_name, name, mod_type, key))
            row[f"{name}_minus_reference_AUPR"] = diff(
                class_metric(metrics_by_name, name, mod_type, "AUPR_ovr"),
                class_metric(metrics_by_name, reference_name, mod_type, "AUPR_ovr"),
            )
        best_name = ""
        best_aupr: Optional[float] = None
        for name in names:
            value = as_float(class_metric(metrics_by_name, name, mod_type, "AUPR_ovr"))
            if value is not None and (best_aupr is None or value > best_aupr):
                best_aupr = value
                best_name = name
        row["best_variant_by_AUPR"] = best_name
        row["best_AUPR"] = "" if best_aupr is None else f"{best_aupr:.6f}"
        rows.append(row)
    return rows


def build_focus_rows(
    variants: Sequence[str],
    metrics_by_name: Mapping[str, Mapping[str, object]],
    reference_name: str,
) -> List[Dict[str, object]]:
    names = [reference_name] + [variant for variant in variants if variant != reference_name]
    rows: List[Dict[str, object]] = []
    for mod_type in FOCUS_CLASSES:
        for name in names:
            rows.append(
                {
                    "mod_type": mod_type,
                    "variant": name,
                    "support": fmt(class_metric(metrics_by_name, name, mod_type, "support")),
                    "AUPR_ovr": fmt(class_metric(metrics_by_name, name, mod_type, "AUPR_ovr")),
                    "F1": fmt(class_metric(metrics_by_name, name, mod_type, "F1")),
                    "MCC_ovr": fmt(class_metric(metrics_by_name, name, mod_type, "MCC_ovr")),
                    "AUPR_minus_reference": diff(
                        class_metric(metrics_by_name, name, mod_type, "AUPR_ovr"),
                        class_metric(metrics_by_name, reference_name, mod_type, "AUPR_ovr"),
                    ),
                }
            )
    return rows


def build_report(
    macro_rows: Sequence[Mapping[str, object]],
    focus_rows: Sequence[Mapping[str, object]],
    missing: Sequence[str],
) -> str:
    lines = ["# Sequence Encoder Priority Summary", ""]
    if missing:
        lines.append("Missing metrics.json for: " + ", ".join(missing))
        lines.append("")
    lines.append("## Macro AUPR")
    for row in macro_rows:
        if row.get("variant") == "best_by_test_macro_AUPR":
            lines.append(
                f"- Best: {row.get('seq_encoder', '')} test macro AUPR={row.get('test_macro_AUPR_ovr', '')}, "
                f"delta={row.get('test_macro_AUPR_minus_reference', '')}"
            )
        elif row.get("exists"):
            lines.append(
                f"- {row.get('variant', '')}: test macro AUPR={row.get('test_macro_AUPR_ovr', '')}, "
                f"delta={row.get('test_macro_AUPR_minus_reference', '')}"
            )
    lines.append("")
    lines.append("## Focus Classes")
    for row in focus_rows:
        if row.get("AUPR_ovr"):
            lines.append(
                f"- {row.get('mod_type', '')} / {row.get('variant', '')}: "
                f"AUPR={row.get('AUPR_ovr', '')}, F1={row.get('F1', '')}, delta={row.get('AUPR_minus_reference', '')}"
            )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize priority sequence encoder experiments.")
    parser.add_argument("--root", default="rmbase11_full_work/models/sequence_encoder_priority_cdhit80")
    parser.add_argument("--out-dir", default="rmbase11_full_work/metrics/sequence_encoder_priority_cdhit80")
    parser.add_argument("--variants", default="kmer_extended_rbp,dilated_rbp,hybrid_transformer_rbp")
    parser.add_argument("--reference-name", default="current_main_rbp")
    parser.add_argument("--reference-metrics", default="")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    variants = parse_variants(args.variants)
    items = collect_items(root, variants, args.reference_name, args.reference_metrics)
    metrics_by_name = {name: load_json(path) for name, path in items}
    classes = get_classes(metrics_by_name)
    macro_rows = build_macro_rows(items, metrics_by_name, args.reference_name)
    per_class_rows = build_per_class_rows(classes, variants, metrics_by_name, args.reference_name)
    focus_rows = build_focus_rows(variants, metrics_by_name, args.reference_name)
    write_tsv(out_dir / "sequence_encoder_priority_macro.tsv", macro_rows)
    write_tsv(out_dir / "sequence_encoder_priority_per_class.tsv", per_class_rows)
    write_tsv(out_dir / "sequence_encoder_priority_focus_classes.tsv", focus_rows)
    missing = [name for name, metrics in metrics_by_name.items() if not metrics]
    (out_dir / "sequence_encoder_priority_report.md").write_text(
        build_report(macro_rows, focus_rows, missing),
        encoding="utf-8",
    )
    if missing:
        print("Missing metrics.json for: " + ", ".join(missing))
    print(f"Wrote sequence encoder priority summary to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
