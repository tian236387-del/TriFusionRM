#!/usr/bin/env python
"""Summarize strict no-metadata models with and without RBP features."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


MODELS = [
    ("M0_gene_no_metadata_full", "M0 Transformer-only", "gene split", "sequence only"),
    ("M4_gene_no_metadata_full", "M4 StructDualRM", "gene split", "sequence + structure"),
    ("M4_gene_no_metadata_rbp_full", "M4 StructDualRM + RBP", "gene split", "sequence + structure + RBP"),
    ("M0_mmseqs80_no_metadata_full", "M0 Transformer-only", "MMseqs2 80% cluster split", "sequence only"),
    ("M0_mmseqs80_no_metadata_rbp_full", "M0 Transformer-only + RBP", "MMseqs2 80% cluster split", "sequence + RBP"),
    ("M4_mmseqs80_no_metadata_full", "M4 StructDualRM", "MMseqs2 80% cluster split", "sequence + structure"),
    ("M4_mmseqs80_no_metadata_rbp_full", "M4 StructDualRM + RBP", "MMseqs2 80% cluster split", "sequence + structure + RBP"),
]


def load_metrics(model_dir: Path) -> Dict[str, object]:
    return json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> int:
    root = Path("rmbase11_full_work")
    out_dir = root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for model_name, label, split_label, feature_set in MODELS:
        model_dir = root / "models" / model_name
        if not (model_dir / "metrics.json").exists():
            continue
        metrics = load_metrics(model_dir)
        test = metrics["test"]
        rows.append(
            {
                "model_dir": model_name,
                "model": label,
                "split": split_label,
                "feature_set": feature_set,
                "epochs": len(metrics.get("history", [])),
                "use_metadata": metrics.get("use_metadata"),
                "use_rbp_features": metrics.get("use_rbp_features", False),
                "rbp_feature_count": metrics.get("rbp_feature_count", 0),
                "train_rows": metrics["row_counts"]["train"],
                "valid_rows": metrics["row_counts"]["valid"],
                "test_rows": metrics["row_counts"]["test"],
                "accuracy": test["accuracy"],
                "balanced_accuracy": test["balanced_accuracy"],
                "macro_F1": test["macro_F1"],
                "weighted_F1": test["weighted_F1"],
                "MCC": test["MCC"],
                "macro_AUPR": test["macro_AUPR_ovr"],
                "macro_AUC": test["macro_AUC_ovr"],
            }
        )

    fields = list(rows[0].keys())
    table_path = out_dir / "full_strict_rbp_model_metrics.tsv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    by_name = {row["model_dir"]: row for row in rows}
    comparisons = []
    for base_name, rbp_name in [
        ("M4_gene_no_metadata_full", "M4_gene_no_metadata_rbp_full"),
        ("M4_mmseqs80_no_metadata_full", "M4_mmseqs80_no_metadata_rbp_full"),
    ]:
        if base_name not in by_name or rbp_name not in by_name:
            continue
        base = by_name[base_name]
        rbp = by_name[rbp_name]
        comparisons.append(
            {
                "split": rbp["split"],
                "delta_balanced_accuracy": float(rbp["balanced_accuracy"]) - float(base["balanced_accuracy"]),
                "delta_macro_F1": float(rbp["macro_F1"]) - float(base["macro_F1"]),
                "delta_MCC": float(rbp["MCC"]) - float(base["MCC"]),
                "delta_macro_AUPR": float(rbp["macro_AUPR"]) - float(base["macro_AUPR"]),
                "delta_macro_AUC": float(rbp["macro_AUC"]) - float(base["macro_AUC"]),
            }
        )
    delta_path = out_dir / "full_strict_rbp_delta_vs_m4.tsv"
    with delta_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0].keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)

    coverage_path = root / "rbp" / "rbp_feature_coverage.tsv"
    manifest_path = root / "rbp" / "encode_eclip_file_manifest.tsv"
    report_path = out_dir / "full_strict_rbp_feature_report.md"
    lines = [
        "# Full RMBase11 RBP Feature Branch Results",
        "",
        "This report summarizes the reviewer-safe no-metadata experiments after adding an auxiliary RBP binding feature branch to StructDualRM.",
        "",
        "## RBP input",
        "",
        "- Dataset: full RMBase11 hg38 11-class dataset, 945,219 sites.",
        "- Peak source used in this run: locally available complete ENCODE eCLIP GRCh38 narrowPeak files.",
        "- Targets included: ELAVL1 and HNRNPC in K562. One partially downloaded HNRNPC HepG2 file was excluded because gzip validation failed.",
        "- Features: any-overlap, unique-target count, merged-peak count within ±50/±100/±200 nt, target-specific ±100/±200 nt overlap and log10 nearest distance.",
        "- Important limitation: this is a pilot RBP branch because additional ENCODE downloads were blocked in the current environment. It should be expanded to a broader RBP panel before being framed as a final biological-context model.",
        "",
        "## Test-set metrics",
        "",
        "| Model | Split | Features | Bal. Acc | Macro F1 | MCC | Macro AUPR | Macro AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {split} | {feature_set} | {balanced_accuracy} | {macro_F1} | {MCC} | {macro_AUPR} | {macro_AUC} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Delta relative to M4 without RBP",
            "",
            "| Split | ΔBal. Acc | ΔMacro F1 | ΔMCC | ΔMacro AUPR | ΔMacro AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {split} | {delta_balanced_accuracy:.6g} | {delta_macro_F1:.6g} | {delta_MCC:.6g} | {delta_macro_AUPR:.6g} | {delta_macro_AUC:.6g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Metrics table: `{table_path}`",
            f"- Delta table: `{delta_path}`",
            f"- RBP coverage: `{coverage_path}`",
            f"- RBP manifest: `{manifest_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {table_path}")
    print(f"Wrote {delta_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
