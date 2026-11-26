#!/usr/bin/env python
"""Summarize CD-HIT80 strict baselines against the StructDualRM main model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


BASELINE_MODELS = ["multirm_official", "transrnam_official", "evormd_adapted"]
DEFAULT_CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
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
    return json.loads(path.read_text(encoding="utf-8"))


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


def nested_get(data: Mapping[str, object], path: Sequence[str], default: object = "") -> object:
    current: object = data
    for item in path:
        if not isinstance(current, Mapping) or item not in current:
            return default
        current = current[item]
    return current


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect_metrics(baseline_root: Path, proposed_dir: Path) -> Dict[str, Dict[str, object]]:
    metrics: Dict[str, Dict[str, object]] = {}
    for model in BASELINE_MODELS:
        metrics[model] = load_json(baseline_root / model / "metrics.json")
    metrics["StructDualRM_no_position"] = load_json(proposed_dir / "metrics.json")
    return metrics


def get_class_names(metrics_by_model: Mapping[str, Mapping[str, object]], explicit: str = "") -> List[str]:
    if explicit.strip():
        return [name.strip() for name in explicit.split(",") if name.strip()]
    for metrics in metrics_by_model.values():
        classes = metrics.get("classes", [])
        if isinstance(classes, list) and classes:
            return [str(name) for name in classes]
    seen: List[str] = []
    for metrics in metrics_by_model.values():
        per_class = nested_get(metrics, ["test", "per_class"], {})
        if not isinstance(per_class, Mapping):
            continue
        for name in per_class.keys():
            if str(name) not in seen:
                seen.append(str(name))
    return seen or list(DEFAULT_CLASS_NAMES)


def source_label(model: str, metrics: Mapping[str, object]) -> str:
    if model == "StructDualRM_no_position":
        return "proposed_main_model"
    source = metrics.get("source_info", {})
    if isinstance(source, Mapping):
        return str(source.get("source_type", "architecture_adapted_baseline"))
    return "architecture_adapted_baseline"


def macro_rows(metrics_by_model: Mapping[str, Mapping[str, object]], baseline_root: Path, proposed_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model, metrics in metrics_by_model.items():
        model_dir = proposed_dir if model == "StructDualRM_no_position" else baseline_root / model
        test = metrics.get("test", {})
        valid = metrics.get("validation", {})
        row_counts = metrics.get("row_counts", {})
        if not isinstance(test, Mapping):
            test = {}
        if not isinstance(valid, Mapping):
            valid = {}
        if not isinstance(row_counts, Mapping):
            row_counts = {}
        row: Dict[str, object] = {
            "model": model,
            "source_type": source_label(model, metrics),
            "exists": bool(metrics),
            "model_dir": str(model_dir),
            "dataset": metrics.get("dataset", ""),
            "split": metrics.get("split_column", metrics.get("split", "")),
            "objective": metrics.get("objective", ""),
            "best_epoch": metrics.get("best_epoch", ""),
            "best_validation_score": fmt(metrics.get("best_validation_score", "")),
            "train_rows": row_counts.get("train", ""),
            "valid_rows": row_counts.get("valid", ""),
            "test_rows": row_counts.get("test", ""),
        }
        for key in MACRO_KEYS:
            row[f"valid_{key}"] = fmt(valid.get(key, ""))
            row[f"test_{key}"] = fmt(test.get(key, ""))
        rows.append(row)
    return rows


def per_class_rows(
    metrics_by_model: Mapping[str, Mapping[str, object]],
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    proposed = metrics_by_model.get("StructDualRM_no_position", {})
    for mod_type in class_names:
        row: Dict[str, object] = {"mod_type": mod_type}
        support = nested_get(proposed, ["test", "per_class", mod_type, "support"], "")
        if support == "":
            for metrics in metrics_by_model.values():
                support = nested_get(metrics, ["test", "per_class", mod_type, "support"], "")
                if support != "":
                    break
        row["support"] = fmt(support)
        for model, metrics in metrics_by_model.items():
            for key in PER_CLASS_KEYS[1:]:
                row[f"{model}_{key}"] = fmt(nested_get(metrics, ["test", "per_class", mod_type, key], ""))
        proposed_aupr = as_float(nested_get(proposed, ["test", "per_class", mod_type, "AUPR_ovr"], ""))
        baseline_auprs = [
            as_float(nested_get(metrics_by_model.get(model, {}), ["test", "per_class", mod_type, "AUPR_ovr"], ""))
            for model in BASELINE_MODELS
        ]
        valid_baselines = [value for value in baseline_auprs if value is not None]
        if proposed_aupr is not None and valid_baselines:
            row["StructDualRM_minus_best_baseline_AUPR"] = f"{proposed_aupr - max(valid_baselines):.6f}"
            row["best_baseline_AUPR"] = f"{max(valid_baselines):.6f}"
            row["best_baseline_model"] = BASELINE_MODELS[baseline_auprs.index(max(valid_baselines))]
        else:
            row["StructDualRM_minus_best_baseline_AUPR"] = ""
            row["best_baseline_AUPR"] = ""
            row["best_baseline_model"] = ""
        rows.append(row)
    return rows


def build_report(
    out_dir: Path,
    rows: Sequence[Mapping[str, object]],
    class_rows: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# CD-HIT80 Strict Baseline Comparison",
        "",
        "This report compares three architecture-adapted baselines against the current StructDualRM main model under the same RMBase11 CD-HIT80 strict split.",
        "",
        "## Macro Metrics",
        "",
        "| Model | Source | Test macro AUPR | Test macro AUC | Test macro F1 | Test MCC | Best epoch |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {source_type} | {aupr} | {auc} | {f1} | {mcc} | {epoch} |".format(
                model=row.get("model", ""),
                source_type=row.get("source_type", ""),
                aupr=row.get("test_macro_AUPR_ovr", ""),
                auc=row.get("test_macro_AUC_ovr", ""),
                f1=row.get("test_macro_F1", ""),
                mcc=row.get("test_MCC", ""),
                epoch=row.get("best_epoch", ""),
            )
        )
    wins = 0
    total = 0
    for row in class_rows:
        delta = as_float(row.get("StructDualRM_minus_best_baseline_AUPR", ""))
        if delta is None:
            continue
        total += 1
        wins += int(delta > 0)
    lines.extend(
        [
            "",
            "## Per-class Summary",
            "",
            f"StructDualRM exceeds the best baseline in {wins}/{total} evaluable modification classes by per-class AUPR.",
            "",
            "## Output Files",
            "",
            f"- `{out_dir / 'cdhit80_baseline_vs_structdualrm_macro.tsv'}`",
            f"- `{out_dir / 'cdhit80_baseline_vs_structdualrm_per_class.tsv'}`",
            "",
            "## Reproduction Boundary",
            "",
            "- MultiRM and TransRNAm are architecture-adapted reproductions using the unified RMBase11-CD-HIT80 loader and metric code.",
            "- EvoRMD is adapted because local RNA-FM package/weights may not be available; the adapter keeps the sequence/context fusion setting but replaces unavailable RNA-FM dependencies.",
            "- The comparison is therefore a unified-data benchmark rather than a claim of exact paper-score reproduction.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize CD-HIT80 baselines versus StructDualRM.")
    parser.add_argument(
        "--baseline-root",
        default="rmbase11_full_work/official_reproduction_benchmark/cdhit80_strict_split",
    )
    parser.add_argument(
        "--proposed-dir",
        default="rmbase11_full_work/models/motif_struct_fusion_v4_cdhit80/no_position",
    )
    parser.add_argument(
        "--out-dir",
        default="rmbase11_full_work/metrics/cdhit80_baseline_vs_structdualrm",
    )
    parser.add_argument("--class-names", default="", help="Optional comma-separated class order; inferred from metrics if omitted.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    baseline_root = Path(args.baseline_root)
    proposed_dir = Path(args.proposed_dir)
    out_dir = Path(args.out_dir)
    metrics_by_model = collect_metrics(baseline_root, proposed_dir)
    class_names = get_class_names(metrics_by_model, args.class_names)
    rows = macro_rows(metrics_by_model, baseline_root, proposed_dir)
    class_rows = per_class_rows(metrics_by_model, class_names)
    write_tsv(out_dir / "cdhit80_baseline_vs_structdualrm_macro.tsv", rows)
    write_tsv(out_dir / "cdhit80_baseline_vs_structdualrm_per_class.tsv", class_rows)
    (out_dir / "cdhit80_baseline_vs_structdualrm_report.md").write_text(
        build_report(out_dir, rows, class_rows),
        encoding="utf-8",
    )
    missing = [model for model, metrics in metrics_by_model.items() if not metrics]
    if missing:
        print("Missing metrics.json for: " + ", ".join(missing))
    print(f"Wrote summary tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
