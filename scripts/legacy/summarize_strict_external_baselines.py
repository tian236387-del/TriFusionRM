#!/usr/bin/env python
"""Summarize strict MMseqs2 external baselines against StructDualRM."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


RUNS = [
    ("M0 Transformer-only", "internal_ablation", "models/M0_mmseqs80_no_metadata_full", "sequence only"),
    ("M4 StructDualRM", "proposed_structure", "models/M4_mmseqs80_no_metadata_full", "sequence + RNAfold structure"),
    ("M0 + RBP", "internal_rbp_control", "models/M0_mmseqs80_no_metadata_rbp_full", "sequence + local RBP"),
    ("M4 StructDualRM + RBP", "proposed_rbp_pilot", "models/M4_mmseqs80_no_metadata_rbp_full", "sequence + RNAfold structure + local RBP"),
    ("M5 label-attention StructDualRM", "architecture_revision", "models/M5_label_attention_bce_rbp_mmseqs80_full", "nucleotide Transformer + structure/RBP + label attention + BCE"),
    ("M6 MultiRM-Struct/RBP hybrid", "architecture_revision", "models/M6_multirm_struct_rbp_mmseqs80_full", "3-mer BiLSTM label attention + RNAfold/RBP features"),
    ("MultiRM-style retrained", "external_baseline", "baselines/multirm_style_mmseqs80_strict_full", "3-mer embedding + BiLSTM + label-specific attention"),
    ("TransRNAm-style retrained", "external_baseline", "baselines/transrnam_style_mmseqs80_strict_full", "3-mer embedding + Transformer pooling + CNN head"),
    ("EvoRMD-lite safe-context", "external_baseline", "baselines/evormd_lite_safe_context_mmseqs80_strict_full", "sequence encoder + conservative context"),
]


def load_metrics(path: Path) -> Dict[str, object]:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def metric_row(root: Path, name: str, role: str, rel_dir: str, features: str) -> Dict[str, object]:
    metrics = load_metrics(root / rel_dir)
    test = metrics["test"]
    valid = metrics["validation"]
    hist = metrics.get("history", [])
    best_epoch = ""
    if hist:
        best_epoch = max(hist, key=lambda item: item["validation"].get("macro_AUPR_ovr", float("-inf")))["epoch"]
    return {
        "rank_by_macro_AUPR": 0,
        "model": name,
        "role": role,
        "model_dir": str(root / rel_dir),
        "feature_summary": features,
        "epochs": len(hist),
        "best_epoch_by_valid_macro_AUPR": best_epoch,
        "train_rows": metrics["row_counts"]["train"],
        "valid_rows": metrics["row_counts"]["valid"],
        "test_rows": metrics["row_counts"]["test"],
        "test_accuracy": test["accuracy"],
        "test_balanced_accuracy": test["balanced_accuracy"],
        "test_macro_F1": test["macro_F1"],
        "test_weighted_F1": test["weighted_F1"],
        "test_MCC": test["MCC"],
        "test_macro_AUPR": test["macro_AUPR_ovr"],
        "test_macro_AUC": test["macro_AUC_ovr"],
        "valid_macro_AUPR": valid["macro_AUPR_ovr"],
    }


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = Path("rmbase11_full_work")
    out_dir = root / "metrics"
    rows = [metric_row(root, *run) for run in RUNS]
    rows.sort(key=lambda row: float(row["test_macro_AUPR"]), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank_by_macro_AUPR"] = idx

    table_path = out_dir / "strict_mmseqs80_external_baseline_comparison.tsv"
    write_tsv(table_path, rows)

    best = rows[0]
    m4_rbp = next(row for row in rows if row["model"] == "M4 StructDualRM + RBP")
    multirm = next(row for row in rows if row["model"] == "MultiRM-style retrained")
    m5 = next(row for row in rows if row["model"] == "M5 label-attention StructDualRM")
    m6 = next(row for row in rows if row["model"] == "M6 MultiRM-Struct/RBP hybrid")

    report_path = out_dir / "strict_mmseqs80_external_baseline_report.md"
    lines = [
        "# Strict MMseqs2 External Baseline Comparison",
        "",
        "## Setting",
        "",
        "- Dataset: full RMBase11 hg38, 945,219 sites.",
        "- Split: MMseqs2 80% sequence-cluster split, no random-site leakage.",
        "- Epochs: 50 for all listed neural baselines.",
        "- Selection metric: validation macro AUPR; table reports held-out test metrics.",
        "- External baselines are architecture-style retrained implementations on the same processed dataset, not cross-paper copied scores.",
        "",
        "## Main result",
        "",
        f"The highest strict test macro AUPR in the current run is **{float(best['test_macro_AUPR']):.4f}** from **{best['model']}**.",
        f"The current best StructDualRM-family variant is **M4 StructDualRM + RBP**, with macro AUPR **{float(m4_rbp['test_macro_AUPR']):.4f}**, macro F1 **{float(m4_rbp['test_macro_F1']):.4f}**, balanced accuracy **{float(m4_rbp['test_balanced_accuracy']):.4f}** and MCC **{float(m4_rbp['test_MCC']):.4f}**.",
        f"Compared with MultiRM-style, M4 + RBP is lower by **{float(multirm['test_macro_AUPR']) - float(m4_rbp['test_macro_AUPR']):.4f} macro AUPR** and **{float(multirm['test_macro_F1']) - float(m4_rbp['test_macro_F1']):.4f} macro F1**.",
        f"The two architecture revisions did not close the gap: M5 reached **{float(m5['test_macro_AUPR']):.4f}** macro AUPR, and M6 reached **{float(m6['test_macro_AUPR']):.4f}** macro AUPR.",
        "",
        "This means the current StructDualRM model is not yet the strongest under the strict external-baseline setting. It still improves over TransRNAm-style and over the original M4 without RBP, but MultiRM-style is currently the benchmark to beat.",
        "",
        "## Metrics",
        "",
        "| Rank | Model | Role | Macro AUPR | Macro F1 | Balanced Acc | MCC | Macro AUC |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {rank_by_macro_AUPR} | {model} | {role} | {test_macro_AUPR:.4f} | {test_macro_F1:.4f} | {test_balanced_accuracy:.4f} | {test_MCC:.4f} | {test_macro_AUC:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- MultiRM-style benefits from label-specific attention and a BCE-style multi-label objective, which appears more robust for rare classes under the strict cluster split.",
            "- M4 + RBP improves over M4 without RBP, especially in macro AUPR, showing that local RBP annotation is useful, but the current RBP panel is only ELAVL1/HNRNPC from locally available complete eCLIP files.",
            "- M5 showed that simply adding label-specific attention and BCE on top of the nucleotide-level StructDualRM encoder is not sufficient.",
            "- M6 showed that directly appending RNAfold/RBP features to the MultiRM-style backbone can overfit validation and does not improve held-out test macro AUPR.",
            "- The next architecture iteration should use safer regularized fusion: pretrain/retain the MultiRM-style sequence branch, add structure as a small residual logit correction, and constrain auxiliary-feature contribution instead of concatenating a large auxiliary vector into every class head.",
            "- EvoRMD-lite safe-context remains an adapted-lite baseline, not an official RNA-FM EvoRMD reproduction.",
            "",
            "## Files",
            "",
            f"- Comparison table: `{table_path}`",
            "- Model outputs are under `rmbase11_full_work/models` and `rmbase11_full_work/baselines`.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {table_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
