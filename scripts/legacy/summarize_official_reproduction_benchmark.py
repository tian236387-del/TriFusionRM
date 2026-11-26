#!/usr/bin/env python
"""Summarize official-architecture random-vs-strict benchmark results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


RUNS = [
    ("paper_like_random_split", "multirm_official"),
    ("paper_like_random_split", "transrnam_official"),
    ("paper_like_random_split", "evormd_adapted"),
    ("strict_mmseqs80_split", "multirm_official"),
    ("strict_mmseqs80_split", "transrnam_official"),
    ("strict_mmseqs80_split", "evormd_adapted"),
]


def load_metrics(path: Path) -> Dict[str, object]:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def row_for(root: Path, setting: str, model: str) -> Dict[str, object]:
    model_dir = root / setting / model
    metrics = load_metrics(model_dir)
    test = metrics["test"]
    valid = metrics["validation"]
    source = metrics["source_info"]
    hist = metrics.get("history", [])
    return {
        "setting": setting,
        "model": model,
        "source_type": source["source_type"],
        "split_column": metrics["split_column"],
        "objective": metrics["objective"],
        "epochs": len(hist),
        "best_epoch": metrics["best_epoch"],
        "train_rows": metrics["row_counts"]["train"],
        "valid_rows": metrics["row_counts"]["valid"],
        "test_rows": metrics["row_counts"]["test"],
        "valid_macro_AUPR": valid["macro_AUPR_ovr"],
        "test_macro_AUPR": test["macro_AUPR_ovr"],
        "test_macro_AUC": test["macro_AUC_ovr"],
        "test_macro_F1": test["macro_F1"],
        "test_balanced_accuracy": test["balanced_accuracy"],
        "test_MCC": test["MCC"],
        "model_dir": str(model_dir),
        "limitations": source["limitations"],
    }


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = Path("rmbase11_full_work/official_reproduction_benchmark")
    rows = []
    for setting, model in RUNS:
        metrics_path = root / setting / model / "metrics.json"
        if metrics_path.exists():
            rows.append(row_for(root, setting, model))
    if not rows:
        raise FileNotFoundError("No metrics.json files found for official reproduction benchmark")

    table_path = root / "official_reproduction_random_vs_strict_comparison.tsv"
    write_tsv(table_path, rows)

    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["setting"]] = row

    lines = [
        "# Official-Architecture Random-vs-Strict Benchmark",
        "",
        "## Purpose",
        "",
        "This benchmark separates two questions:",
        "",
        "1. How do official or official-architecture baseline models perform under a paper-like random split?",
        "2. How much performance changes when the same models are evaluated under the stricter MMseqs2-80 cluster split used by this project?",
        "",
        "## Reproduction boundary",
        "",
        "- MultiRM: portable reproduction of the official `model_v3` architecture.",
        "- TransRNAm: portable reproduction of the official `model_v11_test` architecture.",
        "- EvoRMD: adapted reproduction only. The local environment does not contain the `fm` RNA-FM package or RNA-FM weights, so RNA-FM is replaced by a trainable nucleotide Transformer while retaining context fusion and attention pooling.",
        "- All models use the same RMBase11 hg38 table and identical metric code within each split setting.",
        "",
        "## Main table",
        "",
        "| Setting | Model | Source type | Macro AUPR | Macro F1 | Balanced Acc | MCC | Best epoch |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {setting} | {model} | {source_type} | {test_macro_AUPR:.4f} | {test_macro_F1:.4f} | {test_balanced_accuracy:.4f} | {test_MCC:.4f} | {best_epoch} |".format(
                **row
            )
        )
    lines.extend(["", "## Random-to-strict changes", ""])
    for model, settings in by_model.items():
        if "paper_like_random_split" not in settings or "strict_mmseqs80_split" not in settings:
            continue
        random_row = settings["paper_like_random_split"]
        strict_row = settings["strict_mmseqs80_split"]
        delta = float(strict_row["test_macro_AUPR"]) - float(random_row["test_macro_AUPR"])
        lines.append(
            f"- `{model}`: macro AUPR random={float(random_row['test_macro_AUPR']):.4f}, "
            f"strict={float(strict_row['test_macro_AUPR']):.4f}, delta={delta:+.4f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation guidance",
            "",
            "Random-split results are useful as a supplementary compatibility benchmark, but they should not be used as the strongest claim of generalization because similar sequences and neighboring genomic sites can cross partitions.",
            "",
            "Strict MMseqs2-80 cluster-split results are more conservative and should be treated as the primary evidence when arguing that the project uses a more rigorous preprocessing and evaluation protocol than many random-split benchmarks.",
            "",
            f"Full table: `{table_path}`",
        ]
    )
    report_path = root / "official_reproduction_random_vs_strict_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {table_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
