#!/usr/bin/env python
"""Build per-class structure-gain tables for MotifStructFusionV4."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional


CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
VARIANTS = ["full", "sequence_only", "shuffled_structure", "no_position", "with_rbp", "with_rbp_positional"]


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_gate_table(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["mod_type"]: row for row in csv.DictReader(handle, delimiter="\t")}


def per_class(metrics: Mapping[str, object], name: str) -> Mapping[str, object]:
    test = metrics.get("test", {})
    if not isinstance(test, Mapping):
        return {}
    values = test.get("per_class", {})
    if not isinstance(values, Mapping):
        return {}
    item = values.get(name, {})
    return item if isinstance(item, Mapping) else {}


def metric(metrics_by_variant: Mapping[str, Mapping[str, object]], variant: str, mod_type: str, key: str) -> object:
    item = per_class(metrics_by_variant.get(variant, {}), mod_type)
    return item.get(key, "")


def as_float(value: object) -> Optional[float]:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def difference(left: object, right: object) -> object:
    left_f = as_float(left)
    right_f = as_float(right)
    if left_f is None or right_f is None:
        return ""
    return left_f - right_f


def write_tsv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare per-class MotifStructFusionV4 structure effects.")
    parser.add_argument("--root", default="rmbase11_full_work/models/motif_struct_fusion_v4_structure_only")
    parser.add_argument("--out-dir", default="rmbase11_full_work/metrics/motif_struct_fusion_v4_structure_only_strict")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    metrics_by_variant = {variant: load_json(root / variant / "metrics.json") for variant in VARIANTS}
    gate_by_class = load_gate_table(root / "full" / "test_per_class_structure_gate.tsv")
    modality_gate_by_class = load_gate_table(root / "with_rbp_positional" / "test_per_class_modality_gate.tsv")

    rows = []
    for mod_type in CLASS_NAMES:
        seq_aupr = metric(metrics_by_variant, "sequence_only", mod_type, "AUPR_ovr")
        full_aupr = metric(metrics_by_variant, "full", mod_type, "AUPR_ovr")
        shuffled_aupr = metric(metrics_by_variant, "shuffled_structure", mod_type, "AUPR_ovr")
        no_position_aupr = metric(metrics_by_variant, "no_position", mod_type, "AUPR_ovr")
        rbp_aupr = metric(metrics_by_variant, "with_rbp", mod_type, "AUPR_ovr")
        rbp_pos_aupr = metric(metrics_by_variant, "with_rbp_positional", mod_type, "AUPR_ovr")
        gate_row = gate_by_class.get(mod_type, {})
        modality_gate_row = modality_gate_by_class.get(mod_type, {})
        rows.append(
            {
                "mod_type": mod_type,
                "support": metric(metrics_by_variant, "full", mod_type, "support"),
                "sequence_only_AUPR": seq_aupr,
                "full_AUPR": full_aupr,
                "shuffled_structure_AUPR": shuffled_aupr,
                "no_position_AUPR": no_position_aupr,
                "with_rbp_AUPR": rbp_aupr,
                "with_rbp_positional_AUPR": rbp_pos_aupr,
                "full_minus_sequence_only_AUPR": difference(full_aupr, seq_aupr),
                "full_minus_shuffled_structure_AUPR": difference(full_aupr, shuffled_aupr),
                "with_rbp_minus_sequence_only_AUPR": difference(rbp_aupr, seq_aupr),
                "with_rbp_positional_minus_sequence_only_AUPR": difference(rbp_pos_aupr, seq_aupr),
                "with_rbp_positional_minus_full_AUPR": difference(rbp_pos_aupr, full_aupr),
                "full_F1": metric(metrics_by_variant, "full", mod_type, "F1"),
                "full_MCC_ovr": metric(metrics_by_variant, "full", mod_type, "MCC_ovr"),
                "mean_structure_gate_for_class_logit": gate_row.get("mean_gate_for_class_logit", ""),
                "mean_structure_gate_on_true_samples": gate_row.get("mean_gate_on_true_samples", ""),
                "class_gate_structure_prior": modality_gate_row.get("structure_gate_prior", ""),
                "class_gate_rbp_positional_prior": modality_gate_row.get("rbp_positional_gate_prior", ""),
                "class_gate_sequence_mean": modality_gate_row.get("mean_sequence_gate_for_class_logit", ""),
                "class_gate_structure_mean": modality_gate_row.get("mean_structure_gate_for_class_logit", ""),
                "class_gate_rbp_positional_mean": modality_gate_row.get("mean_rbp_pos_gate_for_class_logit", ""),
                "true_sample_structure_gate": modality_gate_row.get("mean_structure_gate_on_true_samples", ""),
                "true_sample_rbp_positional_gate": modality_gate_row.get("mean_rbp_pos_gate_on_true_samples", ""),
            }
        )

    write_tsv(out_dir / "v4_per_class_structure_gain.tsv", rows)
    print(f"Wrote {out_dir / 'v4_per_class_structure_gain.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
