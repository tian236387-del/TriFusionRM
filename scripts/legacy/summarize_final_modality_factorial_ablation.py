#!/usr/bin/env python
"""Summarize final-model sequence/structure/RBP factorial and matched controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


VARIANT_DIRS = {
    "sequence_only": "sequence_only",
    "sequence_structure": "sequence_structure",
    "sequence_rbp": "sequence_rbp",
    "full_shuffled_structure": "full_shuffled_structure",
    "full_shuffled_rbp": "full_shuffled_rbp",
    "full_no_pair_edges": "full_no_pair_edges",
    "full_shuffled_rbp_bins": "full_shuffled_rbp_bins",
}
METRICS = ("AUPR_ovr", "AUC_ovr", "F1", "Precision", "Recall", "MCC_ovr")


def load(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(run_root: Path, full_model_dir: Path, out_dir: Path) -> None:
    results: Dict[str, Mapping[str, object]] = {"full": load(full_model_dir / "metrics.json")}
    results.update({name: load(run_root / directory / "metrics.json") for name, directory in VARIANT_DIRS.items()})
    classes = list(results["full"]["classes"])
    out_dir.mkdir(parents=True, exist_ok=True)

    macro_rows: List[Dict[str, object]] = []
    for name, result in results.items():
        test = result["test"]
        macro_rows.append(
            {
                "variant": name,
                "best_epoch": result["best_epoch"],
                "validation_macro_AUPR_ovr": result["validation"]["macro_AUPR_ovr"],
                "test_macro_AUPR_ovr": test["macro_AUPR_ovr"],
                "test_macro_F1": test["macro_F1"],
                "test_MCC": test["MCC"],
                "test_balanced_accuracy": test["balanced_accuracy"],
            }
        )
    write_tsv(out_dir / "factorial_macro.tsv", macro_rows)

    per_class_rows: List[Dict[str, object]] = []
    decision_rows: List[Dict[str, object]] = []
    for class_name in classes:
        full = results["full"]["test"]["per_class"][class_name]
        sequence = results["sequence_only"]["test"]["per_class"][class_name]
        sequence_structure = results["sequence_structure"]["test"]["per_class"][class_name]
        sequence_rbp = results["sequence_rbp"]["test"]["per_class"][class_name]
        shuffled_structure = results["full_shuffled_structure"]["test"]["per_class"][class_name]
        shuffled_rbp = results["full_shuffled_rbp"]["test"]["per_class"][class_name]
        no_pair_edges = results["full_no_pair_edges"]["test"]["per_class"][class_name]
        shuffled_bins = results["full_shuffled_rbp_bins"]["test"]["per_class"][class_name]
        row: Dict[str, object] = {"mod_type": class_name, "support": full["support"]}
        for variant, result in results.items():
            values = result["test"]["per_class"][class_name]
            for metric in METRICS:
                row[f"{variant}_{metric}"] = values[metric]
        per_class_rows.append(row)

        structure_increment = float(sequence_structure["AUPR_ovr"]) - float(sequence["AUPR_ovr"])
        rbp_increment = float(sequence_rbp["AUPR_ovr"]) - float(sequence["AUPR_ovr"])
        joint_increment = float(full["AUPR_ovr"]) - float(sequence["AUPR_ovr"])
        interaction = (
            float(full["AUPR_ovr"])
            - float(sequence_structure["AUPR_ovr"])
            - float(sequence_rbp["AUPR_ovr"])
            + float(sequence["AUPR_ovr"])
        )
        decision_rows.append(
            {
                "mod_type": class_name,
                "support": full["support"],
                "structure_increment_AUPR": structure_increment,
                "rbp_increment_AUPR": rbp_increment,
                "joint_increment_AUPR": joint_increment,
                "structure_rbp_interaction_AUPR": interaction,
                "real_structure_minus_shuffled_AUPR": float(full["AUPR_ovr"]) - float(shuffled_structure["AUPR_ovr"]),
                "real_rbp_minus_shuffled_AUPR": float(full["AUPR_ovr"]) - float(shuffled_rbp["AUPR_ovr"]),
                "pair_edges_contribution_AUPR": float(full["AUPR_ovr"]) - float(no_pair_edges["AUPR_ovr"]),
                "rbp_distance_bins_contribution_AUPR": float(full["AUPR_ovr"]) - float(shuffled_bins["AUPR_ovr"]),
                "structure_supported": structure_increment > 0 and float(full["AUPR_ovr"]) > float(shuffled_structure["AUPR_ovr"]),
                "rbp_supported": rbp_increment > 0 and float(full["AUPR_ovr"]) > float(shuffled_rbp["AUPR_ovr"]),
            }
        )
    write_tsv(out_dir / "factorial_per_class.tsv", per_class_rows)
    write_tsv(out_dir / "modality_evidence_decision.tsv", decision_rows)

    readme = [
        "# Final modality factorial ablation",
        "",
        "All variants use the final 41/101/201-nt sequence architecture, the same CD-HIT80 split, and seed 42.",
        "The shuffled controls preserve model capacity and feature marginals while breaking sample-level correspondence.",
        "Pair-edge removal retains RNAfold node/scalar inputs; RBP-bin shuffling retains RBP values but breaks distance semantics.",
        "Statistical significance must be assessed with paired bootstrap comparisons on common test predictions.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--full-model-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.run_root, args.full_model_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
