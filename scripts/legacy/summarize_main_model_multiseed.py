#!/usr/bin/env python
"""Summarize multi-seed stability runs for the main StructDualRM model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


MACRO_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_AUPR_ovr",
    "macro_AUC_ovr",
    "macro_F1",
    "weighted_F1",
    "MCC",
]

PER_CLASS_KEYS = ["AUPR_ovr", "AUC_ovr", "F1", "Precision", "Recall", "MCC_ovr"]
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


def mean_std(values: Sequence[float]) -> Dict[str, object]:
    if not values:
        return {"n": 0, "mean": "", "std": "", "min": "", "max": ""}
    mean_v = statistics.mean(values)
    std_v = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": f"{mean_v:.6f}",
        "std": f"{std_v:.6f}",
        "min": f"{min(values):.6f}",
        "max": f"{max(values):.6f}",
    }


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


def parse_seeds(raw: str) -> List[str]:
    return [item.strip() for item in raw.replace(",", " ").split() if item.strip()]


def collect_metrics(root: Path, seeds: Sequence[str]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for seed in seeds:
        out[seed] = load_json(root / f"seed_{seed}" / "metrics.json")
    return out


def class_names(metrics_by_seed: Mapping[str, Mapping[str, object]]) -> List[str]:
    names: List[str] = []
    for metrics in metrics_by_seed.values():
        per_class = nested_get(metrics, ["test", "per_class"], {})
        if not isinstance(per_class, Mapping):
            continue
        for name in per_class:
            name_s = str(name)
            if name_s not in names:
                names.append(name_s)
    return names


def build_seed_rows(root: Path, seeds: Sequence[str], metrics_by_seed: Mapping[str, Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        path = root / f"seed_{seed}" / "metrics.json"
        metrics = metrics_by_seed.get(seed, {})
        row: Dict[str, object] = {
            "seed": seed,
            "metrics_file": str(path),
            "exists": bool(metrics),
            "dataset": metrics.get("dataset", "") if isinstance(metrics, Mapping) else "",
            "split": metrics.get("split", "") if isinstance(metrics, Mapping) else "",
            "best_epoch": metrics.get("best_epoch", "") if isinstance(metrics, Mapping) else "",
            "best_validation_score": fmt(metrics.get("best_validation_score", "") if isinstance(metrics, Mapping) else ""),
            "early_stopped": nested_get(metrics, ["early_stopping", "early_stopped"], ""),
            "train_rows": nested_get(metrics, ["row_counts", "train"], ""),
            "valid_rows": nested_get(metrics, ["row_counts", "valid"], ""),
            "test_rows": nested_get(metrics, ["row_counts", "test"], ""),
        }
        for split in ("validation", "test"):
            for key in MACRO_KEYS:
                row[f"{split}_{key}"] = fmt(nested_get(metrics, [split, key], ""))
        rows.append(row)
    return rows


def build_macro_summary(metrics_by_seed: Mapping[str, Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split in ("validation", "test"):
        for key in MACRO_KEYS:
            values = [
                value
                for metrics in metrics_by_seed.values()
                for value in [as_float(nested_get(metrics, [split, key], ""))]
                if value is not None
            ]
            stats = mean_std(values)
            rows.append({"split": split, "metric": key, **stats})
    return rows


def build_per_class_summary(
    classes: Sequence[str],
    metrics_by_seed: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for mod_type in classes:
        row: Dict[str, object] = {"mod_type": mod_type}
        supports = [
            value
            for metrics in metrics_by_seed.values()
            for value in [as_float(nested_get(metrics, ["test", "per_class", mod_type, "support"], ""))]
            if value is not None
        ]
        support_stats = mean_std(supports)
        row["support_mean"] = support_stats["mean"]
        row["support_std"] = support_stats["std"]
        for key in PER_CLASS_KEYS:
            values = [
                value
                for metrics in metrics_by_seed.values()
                for value in [as_float(nested_get(metrics, ["test", "per_class", mod_type, key], ""))]
                if value is not None
            ]
            stats = mean_std(values)
            label = key.replace("_ovr", "")
            row[f"{label}_mean"] = stats["mean"]
            row[f"{label}_std"] = stats["std"]
            row[f"{label}_min"] = stats["min"]
            row[f"{label}_max"] = stats["max"]
        rows.append(row)
    return rows


def build_focus_rows(classes: Sequence[str], per_class_rows: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    wanted = set(FOCUS_CLASSES)
    available = {str(row.get("mod_type", "")): row for row in per_class_rows}
    return [available[name] for name in classes if name in wanted and name in available]


def build_report(
    seeds: Sequence[str],
    seed_rows: Sequence[Mapping[str, object]],
    macro_summary: Sequence[Mapping[str, object]],
    focus_rows: Sequence[Mapping[str, object]],
) -> str:
    completed = [row for row in seed_rows if row.get("exists")]
    missing = [str(row.get("seed", "")) for row in seed_rows if not row.get("exists")]
    lines = [
        "# Main Model Multi-seed Stability",
        "",
        f"Requested seeds: {', '.join(seeds)}",
        f"Completed seeds: {len(completed)}",
    ]
    if missing:
        lines.append(f"Missing seeds: {', '.join(missing)}")
    lines.extend(["", "## Test Macro Metrics"])
    for row in macro_summary:
        if row.get("split") == "test" and row.get("metric") in {"macro_AUPR_ovr", "macro_F1", "MCC", "balanced_accuracy"}:
            lines.append(
                f"- {row.get('metric')}: mean={row.get('mean')}, std={row.get('std')}, "
                f"min={row.get('min')}, max={row.get('max')}, n={row.get('n')}"
            )
    lines.extend(["", "## Focus Classes"])
    for row in focus_rows:
        lines.append(
            f"- {row.get('mod_type')}: AUPR={row.get('AUPR_mean')} +/- {row.get('AUPR_std')}, "
            f"F1={row.get('F1_mean')} +/- {row.get('F1_std')}, "
            f"MCC={row.get('MCC_mean')} +/- {row.get('MCC_std')}"
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize main-model multi-seed stability runs.")
    parser.add_argument("--root", default="rmbase11_full_work/models/main_model_multiseed_cdhit80")
    parser.add_argument("--out-dir", default="rmbase11_full_work/metrics/main_model_multiseed_cdhit80")
    parser.add_argument("--seeds", default="42,13,17,23,37")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    seeds = parse_seeds(args.seeds)
    metrics_by_seed = collect_metrics(root, seeds)
    classes = class_names(metrics_by_seed)
    seed_rows = build_seed_rows(root, seeds, metrics_by_seed)
    macro_summary = build_macro_summary(metrics_by_seed)
    per_class_summary = build_per_class_summary(classes, metrics_by_seed)
    focus_rows = build_focus_rows(classes, per_class_summary)
    write_tsv(out_dir / "main_model_multiseed_by_seed.tsv", seed_rows)
    write_tsv(out_dir / "main_model_multiseed_macro_mean_std.tsv", macro_summary)
    write_tsv(out_dir / "main_model_multiseed_per_class_mean_std.tsv", per_class_summary)
    write_tsv(out_dir / "main_model_multiseed_focus_classes.tsv", focus_rows)
    (out_dir / "main_model_multiseed_report.md").write_text(
        build_report(seeds, seed_rows, macro_summary, focus_rows),
        encoding="utf-8",
    )
    missing = [seed for seed, metrics in metrics_by_seed.items() if not metrics]
    if missing:
        print("Missing metrics.json for seeds: " + ", ".join(missing))
    print(f"Wrote multi-seed stability summary to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
