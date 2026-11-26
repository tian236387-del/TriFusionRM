#!/usr/bin/env python
"""Summarize proposed PairedGAT-RBP fusion experiments against baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
BASELINE_MODELS = ["multirm_official", "transrnam_official", "evormd_adapted"]
PROPOSED_ORDER = [
    "full",
    "sequence_only",
    "sequence_structure",
    "sequence_rbp",
    "no_position",
    "no_gate",
    "shuffled_structure",
    "shuffled_rbp",
    "with_rbp",
]


def load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        print(f"missing: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric_block(metrics: Mapping[str, object]) -> Mapping[str, object]:
    test = metrics.get("test", {})
    if isinstance(test, Mapping):
        return test
    return {}


def row_from_metrics(setting: str, model: str, family: str, metrics: Mapping[str, object], path: Path) -> Dict[str, object]:
    test = metric_block(metrics)
    row_counts = metrics.get("row_counts", {})
    config = metrics.get("config", {})
    gate = metrics.get("test_gate_mean", "")
    gate_seq = gate_structure = gate_rbp = ""
    if isinstance(gate, list) and len(gate) >= 3:
        gate_seq, gate_structure, gate_rbp = gate[:3]
    return {
        "setting": setting,
        "model": model,
        "family": family,
        "best_epoch": metrics.get("best_epoch", ""),
        "train_rows": row_counts.get("train", "") if isinstance(row_counts, Mapping) else "",
        "valid_rows": row_counts.get("valid", "") if isinstance(row_counts, Mapping) else "",
        "test_rows": row_counts.get("test", "") if isinstance(row_counts, Mapping) else "",
        "test_macro_AUPR": test.get("macro_AUPR_ovr", ""),
        "test_macro_AUC": test.get("macro_AUC_ovr", ""),
        "test_macro_F1": test.get("macro_F1", ""),
        "test_balanced_accuracy": test.get("balanced_accuracy", ""),
        "test_MCC": test.get("MCC", ""),
        "gate_sequence": gate_seq,
        "gate_structure": gate_structure,
        "gate_rbp": gate_rbp,
        "use_structure": config.get("use_structure", "") if isinstance(config, Mapping) else "",
        "use_rbp": config.get("use_rbp", "") if isinstance(config, Mapping) else "",
        "use_position": config.get("use_position", "") if isinstance(config, Mapping) else "",
        "use_gate": config.get("use_gate", "") if isinstance(config, Mapping) else "",
        "model_dir": str(path.parent).replace("\\", "/"),
    }


def per_class_rows(setting: str, model: str, family: str, metrics: Mapping[str, object]) -> List[Dict[str, object]]:
    test = metric_block(metrics)
    per_class = test.get("per_class", {})
    if not isinstance(per_class, Mapping):
        return []
    rows: List[Dict[str, object]] = []
    for name in CLASS_NAMES:
        item = per_class.get(name, {})
        if not isinstance(item, Mapping):
            item = {}
        rows.append(
            {
                "setting": setting,
                "model": model,
                "family": family,
                "mod_type": name,
                "support": item.get("support", ""),
                "AUPR_ovr": item.get("AUPR_ovr", ""),
                "AUC_ovr": item.get("AUC_ovr", ""),
                "F1": item.get("F1", ""),
                "Precision": item.get("Precision", ""),
                "Recall": item.get("Recall", ""),
                "MCC_ovr": item.get("MCC_ovr", ""),
            }
        )
    return rows


def wide_per_class_row(setting: str, model: str, family: str, metrics: Mapping[str, object]) -> Dict[str, object]:
    test = metric_block(metrics)
    per_class = test.get("per_class", {})
    row: Dict[str, object] = {"setting": setting, "model": model, "family": family}
    for name in CLASS_NAMES:
        item = per_class.get(name, {}) if isinstance(per_class, Mapping) else {}
        row[name] = item.get("AUPR_ovr", "") if isinstance(item, Mapping) else ""
    row["macro_AUPR"] = test.get("macro_AUPR_ovr", "")
    return row


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"skip empty table: {path}")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PairedGAT-RBP fusion experiments.")
    parser.add_argument("--proposed-root", default="rmbase11_full_work/models/paired_gat_rbp_fusion_strict_experiments")
    parser.add_argument("--baseline-root", default="rmbase11_full_work/official_reproduction_benchmark/strict_mmseqs80_split")
    parser.add_argument("--out-dir", default="rmbase11_full_work/metrics/paired_gat_rbp_fusion_strict")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    proposed_root = Path(args.proposed_root)
    baseline_root = Path(args.baseline_root)
    out_dir = Path(args.out_dir)

    main_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []
    per_class_long: List[Dict[str, object]] = []
    per_class_wide: List[Dict[str, object]] = []

    for model in BASELINE_MODELS:
        path = baseline_root / model / "metrics.json"
        metrics = load_json(path)
        if metrics is None:
            continue
        main_rows.append(row_from_metrics("strict_mmseqs80_split", model, "baseline", metrics, path))
        per_class_long.extend(per_class_rows("strict_mmseqs80_split", model, "baseline", metrics))
        per_class_wide.append(wide_per_class_row("strict_mmseqs80_split", model, "baseline", metrics))

    for model in PROPOSED_ORDER:
        path = proposed_root / model / "metrics.json"
        metrics = load_json(path)
        if metrics is None:
            continue
        row = row_from_metrics("strict_mmseqs80_split", f"PairedGAT_RBP_{model}", "proposed", metrics, path)
        if model == "full":
            main_rows.append(row)
        ablation_rows.append(row)
        per_class_long.extend(per_class_rows("strict_mmseqs80_split", f"PairedGAT_RBP_{model}", "proposed", metrics))
        per_class_wide.append(wide_per_class_row("strict_mmseqs80_split", f"PairedGAT_RBP_{model}", "proposed", metrics))

    known = set(PROPOSED_ORDER)
    if proposed_root.exists():
        for child in sorted(proposed_root.iterdir()):
            if not child.is_dir() or child.name in known:
                continue
            path = child / "metrics.json"
            metrics = load_json(path)
            if metrics is None:
                continue
            row = row_from_metrics("strict_mmseqs80_split", f"PairedGAT_RBP_{child.name}", "proposed", metrics, path)
            ablation_rows.append(row)
            per_class_long.extend(per_class_rows("strict_mmseqs80_split", f"PairedGAT_RBP_{child.name}", "proposed", metrics))
            per_class_wide.append(wide_per_class_row("strict_mmseqs80_split", f"PairedGAT_RBP_{child.name}", "proposed", metrics))

    write_tsv(out_dir / "strict_main_model_comparison.tsv", main_rows)
    write_tsv(out_dir / "strict_paired_gat_rbp_ablation.tsv", ablation_rows)
    write_tsv(out_dir / "strict_all_models_per_class_metrics_long.tsv", per_class_long)
    write_tsv(out_dir / "strict_all_models_per_class_AUPR_wide.tsv", per_class_wide)

    print(f"Wrote summary tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
