"""Summarize positional RBP ablation metrics for StructDualRM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


METRIC_KEYS = [
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
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_tsv_by_key(path: Path, key: str) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle, delimiter="\t") if key in row}


def fmt_float(value: object, digits: int = 6) -> object:
    try:
        if value in {"", None}:
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return value


def as_float(value: object) -> Optional[float]:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def diff(left: object, right: object) -> object:
    left_f = as_float(left)
    right_f = as_float(right)
    if left_f is None or right_f is None:
        return ""
    return f"{left_f - right_f:.6f}"


def metric_section(metrics: Mapping[str, object], section: str) -> Mapping[str, object]:
    values = metrics.get(section, {})
    return values if isinstance(values, Mapping) else {}


def per_class(metrics: Mapping[str, object], mod_type: str) -> Mapping[str, object]:
    test = metric_section(metrics, "test")
    values = test.get("per_class", {})
    if not isinstance(values, Mapping):
        return {}
    item = values.get(mod_type, {})
    return item if isinstance(item, Mapping) else {}


def get_classes(metrics_by_variant: Mapping[str, Mapping[str, object]]) -> List[str]:
    for metrics in metrics_by_variant.values():
        test = metric_section(metrics, "test")
        values = test.get("per_class", {})
        if isinstance(values, Mapping) and values:
            return [str(key) for key in values.keys()]
    return []


def build_macro_rows(
    root: Path,
    variants: Sequence[str],
    metrics_by_variant: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for variant in variants:
        metrics = metrics_by_variant.get(variant, {})
        validation = metric_section(metrics, "validation")
        test = metric_section(metrics, "test")
        row: Dict[str, object] = {
            "variant": variant,
            "metrics_file": str(root / variant / "metrics.json"),
            "exists": bool(metrics),
            "dataset": metrics.get("dataset", ""),
            "split": metrics.get("split", ""),
            "best_epoch": metrics.get("best_epoch", ""),
            "best_validation_score": fmt_float(metrics.get("best_validation_score", "")),
            "early_stopped": metric_section(metrics, "early_stopping").get("early_stopped", ""),
            "train_rows": metrics.get("train_rows", ""),
            "valid_rows": metrics.get("valid_rows", ""),
            "test_rows": metrics.get("test_rows", ""),
        }
        for key in METRIC_KEYS:
            row[f"validation_{key}"] = fmt_float(validation.get(key, ""))
            row[f"test_{key}"] = fmt_float(test.get(key, ""))
        rows.append(row)
    return rows


def build_per_class_rows(
    root: Path,
    variants: Sequence[str],
    classes: Sequence[str],
    metrics_by_variant: Mapping[str, Mapping[str, object]],
    baseline_variant: str,
    rbp_variant: str,
    shuffled_rbp_variant: str,
) -> List[Dict[str, object]]:
    rbp_gate_table = load_tsv_by_key(root / rbp_variant / "test_per_class_rbp_positional_gate.tsv", "mod_type")
    rbp_modality_gate = load_tsv_by_key(root / rbp_variant / "test_per_class_modality_gate.tsv", "mod_type")

    rows: List[Dict[str, object]] = []
    baseline_metrics = metrics_by_variant.get(baseline_variant, {})
    rbp_metrics = metrics_by_variant.get(rbp_variant, {})
    shuffled_metrics = metrics_by_variant.get(shuffled_rbp_variant, {})
    for mod_type in classes:
        row: Dict[str, object] = {
            "mod_type": mod_type,
            "support": fmt_float(per_class(baseline_metrics, mod_type).get("support", "")),
        }
        for variant in variants:
            values = per_class(metrics_by_variant.get(variant, {}), mod_type)
            for key in PER_CLASS_KEYS:
                row[f"{variant}_{key}"] = fmt_float(values.get(key, ""))

        base_aupr = per_class(baseline_metrics, mod_type).get("AUPR_ovr", "")
        rbp_aupr = per_class(rbp_metrics, mod_type).get("AUPR_ovr", "")
        shuf_aupr = per_class(shuffled_metrics, mod_type).get("AUPR_ovr", "")
        row[f"{rbp_variant}_minus_{baseline_variant}_AUPR"] = diff(rbp_aupr, base_aupr)
        row[f"{rbp_variant}_minus_{shuffled_rbp_variant}_AUPR"] = diff(rbp_aupr, shuf_aupr)
        row[f"{shuffled_rbp_variant}_minus_{baseline_variant}_AUPR"] = diff(shuf_aupr, base_aupr)

        gate_row = rbp_gate_table.get(mod_type, {})
        modality_row = rbp_modality_gate.get(mod_type, {})
        row["rbp_positional_gate_mean"] = gate_row.get("mean_rbp_pos_gate_for_class_logit", "")
        row["rbp_positional_gate_true_samples"] = gate_row.get("mean_rbp_pos_gate_on_true_samples", "")
        row["class_gate_sequence_mean"] = modality_row.get("mean_sequence_gate_for_class_logit", "")
        row["class_gate_structure_mean"] = modality_row.get("mean_structure_gate_for_class_logit", "")
        row["class_gate_rbp_positional_mean"] = modality_row.get("mean_rbp_pos_gate_for_class_logit", "")
        rows.append(row)
    return rows


def build_report(out_dir: Path, macro_rows: Sequence[Mapping[str, object]], per_class_rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# RBP positional ablation summary",
        "",
        "## Output files",
        "",
        f"- `{out_dir / 'rbp_positional_ablation_macro.tsv'}`",
        f"- `{out_dir / 'rbp_positional_ablation_per_class.tsv'}`",
        "",
        "## Test macro metrics",
        "",
        "| variant | test macro AUPR | test macro AUC | test macro F1 | test MCC | best epoch | early stopped |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
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
            "## How to read",
            "",
            "- Positive `no_position_rbp_pos_minus_no_position_AUPR` means positional RBP features improve that class over the main model.",
            "- Positive `no_position_rbp_pos_minus_no_position_shuffled_rbp_pos_AUPR` means real RBP positions beat the mismatched RBP control.",
            "- Classes with very small support should be treated as weak evidence.",
            "",
            "## Per-class AUPR deltas",
            "",
            "| mod_type | support | RBP - main | RBP - shuffled RBP | shuffled RBP - main |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in per_class_rows:
        lines.append(
            "| {mod} | {support} | {d1} | {d2} | {d3} |".format(
                mod=row.get("mod_type", ""),
                support=row.get("support", ""),
                d1=row.get("no_position_rbp_pos_minus_no_position_AUPR", ""),
                d2=row.get("no_position_rbp_pos_minus_no_position_shuffled_rbp_pos_AUPR", ""),
                d3=row.get("no_position_shuffled_rbp_pos_minus_no_position_AUPR", ""),
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize positional RBP ablation outputs.")
    parser.add_argument("--root", default="rmbase11_full_work/models/rbp_positional_ablation_cdhit80")
    parser.add_argument("--out-dir", default="rmbase11_full_work/metrics/rbp_positional_ablation_cdhit80")
    parser.add_argument("--baseline-variant", default="no_position")
    parser.add_argument("--rbp-variant", default="no_position_rbp_pos")
    parser.add_argument("--shuffled-rbp-variant", default="no_position_shuffled_rbp_pos")
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variants. Defaults to baseline,RBP,RBP-shuffled.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    variants = (
        [item.strip() for item in args.variants.split(",") if item.strip()]
        if args.variants
        else [args.baseline_variant, args.rbp_variant, args.shuffled_rbp_variant]
    )
    metrics_by_variant = {variant: load_json(root / variant / "metrics.json") for variant in variants}
    classes = get_classes(metrics_by_variant)

    macro_rows = build_macro_rows(root, variants, metrics_by_variant)
    per_class_rows = build_per_class_rows(
        root,
        variants,
        classes,
        metrics_by_variant,
        args.baseline_variant,
        args.rbp_variant,
        args.shuffled_rbp_variant,
    )

    write_tsv(out_dir / "rbp_positional_ablation_macro.tsv", macro_rows)
    write_tsv(out_dir / "rbp_positional_ablation_per_class.tsv", per_class_rows)
    (out_dir / "rbp_positional_ablation_report.md").write_text(
        build_report(out_dir, macro_rows, per_class_rows),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
