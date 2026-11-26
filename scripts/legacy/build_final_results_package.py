#!/usr/bin/env python
"""Collect final experiments, tables, figures, logs, and code into one package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FACTOR_VARIANTS = (
    "sequence_only",
    "sequence_structure",
    "sequence_rbp",
    "full_shuffled_structure",
    "full_shuffled_rbp",
    "full_no_pair_edges",
    "full_shuffled_rbp_bins",
)
FACTOR_TABLES = (
    "factorial_macro.tsv",
    "factorial_per_class.tsv",
    "modality_evidence_decision.tsv",
    "paired_bootstrap_modality.tsv",
    "paired_bootstrap_positive_aupr_evidence.tsv",
    "structure_true_class_probability_gain.tsv",
    "structure_stratified_aupr.tsv",
    "rbp_distance_attention_summary.tsv",
    "rbp_top5_per_class.tsv",
)
MULTISEED_TABLES = (
    "final_main_multiseed_macro.tsv",
    "final_main_multiseed_per_class.tsv",
)
CODE_FILES = (
    "scripts/train_paired_gat_rbp_fusion.py",
    "scripts/train_motif_struct_fusion_v4.py",
    "scripts/run_final_modality_factorial_ablation.sh",
    "scripts/summarize_final_modality_factorial_ablation.py",
    "scripts/paired_bootstrap_final_modality.py",
    "scripts/analyze_final_modality_interpretability.py",
    "scripts/run_final_main_multiseed.sh",
    "scripts/summarize_final_main_multiseed.py",
    "scripts/run_final_results_evidence_suite.sh",
    "scripts/build_final_results_package.py",
    "scripts/wait_and_package_final_results.sh",
    "tests/test_modality_control_permutations.py",
    "tests/test_paired_bootstrap_final_modality.py",
)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return
        except OSError:
            pass
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def collect_tree(source: Path, destination: Path) -> int:
    if not source.is_dir():
        return 0
    count = 0
    for path in source.rglob("*"):
        if path.is_file():
            link_or_copy(path, destination / path.relative_to(source))
            count += 1
    return count


def collect_selected(source_dir: Path, names: Sequence[str], destination: Path) -> int:
    count = 0
    for name in names:
        source = source_dir / name
        if source.is_file():
            link_or_copy(source, destination / name)
            count += 1
    return count


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotate_heatmap(axis: plt.Axes, values: np.ndarray, digits: int = 3) -> None:
    threshold = float(np.nanmax(np.abs(values))) * 0.55 if np.isfinite(values).any() else 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.{digits}f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > threshold else "black",
                )


def plot_baseline_comparison(path: Path, figure_dir: Path) -> bool:
    if not path.is_file():
        return False
    rows = [row for row in read_tsv(path) if int(float(row["support"])) > 0]
    classes = [row["mod_type"] for row in rows]
    series = (
        ("MultiRM", "multirm_official_AUPR_ovr", "#4C78A8"),
        ("TransRNAm", "transrnam_official_AUPR_ovr", "#F58518"),
        ("EvoRMD-safe", "evormd_safe_AUPR_ovr", "#54A24B"),
        ("Final model", "current_main_AUPR_ovr", "#E45756"),
    )
    positions = np.arange(len(classes))
    width = 0.19
    fig, axis = plt.subplots(figsize=(13, 5.8))
    for index, (label, field, color) in enumerate(series):
        values = [float(row[field]) for row in rows]
        axis.bar(positions + (index - 1.5) * width, values, width, label=label, color=color)
    axis.set_xticks(positions, classes)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("AUPR")
    axis.set_title("Per-class AUPR on the strict common test set")
    axis.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, figure_dir / "figure_01_baseline_per_class_aupr")
    return True


def plot_factorial_heatmap(path: Path, figure_dir: Path) -> bool:
    if not path.is_file():
        return False
    rows = [row for row in read_tsv(path) if int(float(row["support"])) > 0]
    classes = [row["mod_type"] for row in rows]
    variants = (
        ("Sequence only", "sequence_only"),
        ("Sequence + structure", "sequence_structure"),
        ("Sequence + RBP", "sequence_rbp"),
        ("Full", "full"),
    )
    values = np.asarray(
        [[float(row[f"{prefix}_AUPR_ovr"]) for row in rows] for _, prefix in variants],
        dtype=np.float64,
    )
    fig, axis = plt.subplots(figsize=(13, 4.8))
    image = axis.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(classes)), classes)
    axis.set_yticks(range(len(variants)), [label for label, _ in variants])
    axis.set_title("2 x 2 modality factorial experiment: per-class AUPR")
    annotate_heatmap(axis, values)
    fig.colorbar(image, ax=axis, label="AUPR", fraction=0.025, pad=0.02)
    save_figure(fig, figure_dir / "figure_02_modality_factorial_aupr")
    return True


def plot_modality_effects(path: Path, figure_dir: Path) -> bool:
    if not path.is_file():
        return False
    rows = [row for row in read_tsv(path) if int(float(row["support"])) > 0]
    classes = [row["mod_type"] for row in rows]
    effects = (
        ("Structure main", "structure_increment_AUPR"),
        ("RBP main", "rbp_increment_AUPR"),
        ("Joint", "joint_increment_AUPR"),
        ("Struct x RBP", "structure_rbp_interaction_AUPR"),
        ("Real struct - shuffled", "real_structure_minus_shuffled_AUPR"),
        ("Real RBP - shuffled", "real_rbp_minus_shuffled_AUPR"),
        ("Pair-edge contribution", "pair_edges_contribution_AUPR"),
        ("RBP-bin contribution", "rbp_distance_bins_contribution_AUPR"),
    )
    values = np.asarray([[float(row[field]) for row in rows] for _, field in effects], dtype=np.float64)
    limit = max(0.01, float(np.nanmax(np.abs(values))))
    fig, axis = plt.subplots(figsize=(13, 6.5))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(classes)), classes)
    axis.set_yticks(range(len(effects)), [label for label, _ in effects])
    axis.set_title("Per-class modality effects and matched controls")
    annotate_heatmap(axis, values)
    fig.colorbar(image, ax=axis, label="AUPR difference", fraction=0.025, pad=0.02)
    save_figure(fig, figure_dir / "figure_03_modality_effects")
    return True


def plot_structure_gain(path: Path, figure_dir: Path) -> bool:
    if not path.is_file():
        return False
    rows = read_tsv(path)
    grouped: Dict[tuple[str, str], List[tuple[int, float]]] = {}
    for row in rows:
        grouped.setdefault((row["mod_type"], row["center_state"]), []).append(
            (int(row["support"]), float(row["mean_full_minus_sequence_true_probability"]))
        )
    classes = sorted({key[0] for key in grouped})
    states = ("paired", "unpaired")
    values = np.full((len(states), len(classes)), np.nan, dtype=np.float64)
    for state_index, state in enumerate(states):
        for class_index, class_name in enumerate(classes):
            entries = grouped.get((class_name, state), [])
            if entries:
                support = sum(item[0] for item in entries)
                values[state_index, class_index] = sum(item[0] * item[1] for item in entries) / support
    positions = np.arange(len(classes))
    fig, axis = plt.subplots(figsize=(12.5, 5.5))
    axis.bar(positions - 0.19, values[0], 0.38, label="Paired center", color="#4C78A8")
    axis.bar(positions + 0.19, values[1], 0.38, label="Unpaired center", color="#E45756")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, classes)
    axis.set_ylabel("Full - sequence true-class probability")
    axis.set_title("Structure-related score gain by center pairing state")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, figure_dir / "figure_04_structure_state_gain")
    return True


def plot_rbp_center_enrichment(path: Path, figure_dir: Path) -> bool:
    if not path.is_file():
        return False
    rows = read_tsv(path)
    classes = sorted({row["mod_type"] for row in rows})
    rbps = sorted({row["rbp"] for row in rows})
    lookup = {
        (row["mod_type"], row["rbp"]): float(row["center_minus_distal_attention"])
        for row in rows
    }
    values = np.asarray([[lookup.get((class_name, rbp), np.nan) for rbp in rbps] for class_name in classes])
    limit = max(1e-4, float(np.nanmax(np.abs(values))))
    fig, axis = plt.subplots(figsize=(16, 6.5))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(rbps)), rbps, rotation=65, ha="right", fontsize=8)
    axis.set_yticks(range(len(classes)), classes)
    axis.set_title("RBP attention enrichment: center bins minus distal bins")
    fig.colorbar(image, ax=axis, label="Attention difference", fraction=0.02, pad=0.02)
    save_figure(fig, figure_dir / "figure_05_rbp_center_enrichment")
    return True


def plot_multiseed(path: Path, figure_dir: Path) -> bool:
    if not path.is_file():
        return False
    rows = [row for row in read_tsv(path) if row["metric"] == "AUPR_ovr" and int(row["support_seed42"]) > 0]
    classes = [row["mod_type"] for row in rows]
    means = np.asarray([float(row["mean"]) for row in rows])
    standard_deviations = np.asarray([float(row["sd"]) for row in rows])
    positions = np.arange(len(classes))
    fig, axis = plt.subplots(figsize=(11.5, 5.5))
    axis.errorbar(positions, means, yerr=standard_deviations, fmt="o", capsize=4, color="#4C78A8")
    axis.set_xticks(positions, classes)
    axis.set_ylim(max(0, float(np.nanmin(means - standard_deviations)) - 0.05), 1.02)
    axis.set_ylabel("AUPR mean +/- SD")
    axis.set_title("Final-model stability across seeds 17, 42, and 73")
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, figure_dir / "figure_06_multiseed_stability")
    return True


def expected_paths(
    factor_run_root: Path,
    factor_metrics_dir: Path,
    multiseed_run_root: Path,
    multiseed_metrics_dir: Path,
) -> List[Path]:
    paths = [factor_metrics_dir / name for name in FACTOR_TABLES]
    paths.extend(multiseed_metrics_dir / name for name in MULTISEED_TABLES)
    for variant in FACTOR_VARIANTS:
        paths.extend(
            (
                factor_run_root / variant / "metrics.json",
                factor_run_root / variant / "test_predictions.tsv",
            )
        )
    for seed in (17, 73):
        paths.extend(
            (
                multiseed_run_root / f"seed{seed}" / "metrics.json",
                multiseed_run_root / f"seed{seed}" / "test_predictions.tsv",
            )
        )
    return paths


def build_manifest(package_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    manifest_relative = Path("08_manifest/package_manifest.tsv")
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or path.relative_to(package_dir) == manifest_relative:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows.append(
            {
                "relative_path": path.relative_to(package_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    package_dir: Path = args.package_dir
    directories = {
        "overview": package_dir / "00_overview",
        "tables": package_dir / "01_tables",
        "figures": package_dir / "02_figures",
        "statistics": package_dir / "03_statistics",
        "runs": package_dir / "04_models_predictions_configs",
        "logs": package_dir / "05_logs",
        "code": package_dir / "06_reproducibility_code",
        "docs": package_dir / "07_documentation",
        "manifest": package_dir / "08_manifest",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    required = expected_paths(
        args.factor_run_root,
        args.factor_metrics_dir,
        args.multiseed_run_root,
        args.multiseed_metrics_dir,
    )
    missing = [path for path in required if not path.is_file()]
    if args.require_complete and missing:
        raise FileNotFoundError(f"Final experiment suite is incomplete; missing {len(missing)} files. First: {missing[0]}")

    collected: Dict[str, int] = {}
    collected["factor_runs"] = collect_tree(args.factor_run_root, directories["runs"] / "factorial_seed42")
    collected["multiseed_runs"] = collect_tree(args.multiseed_run_root, directories["runs"] / "final_main_multiseed")
    collected["reference_model"] = collect_tree(args.full_model_dir, directories["runs"] / "reference_final_model_seed42")
    collected["factor_tables"] = collect_selected(
        args.factor_metrics_dir, FACTOR_TABLES, directories["tables"] / "factorial_and_controls"
    )
    collected["multiseed_tables"] = collect_selected(
        args.multiseed_metrics_dir, MULTISEED_TABLES, directories["tables"] / "multiseed"
    )
    if args.baseline_table.is_file():
        link_or_copy(args.baseline_table, directories["tables"] / "baseline" / args.baseline_table.name)
    for name in ("paired_bootstrap_modality.tsv", "paired_bootstrap_positive_aupr_evidence.tsv"):
        source = args.factor_metrics_dir / name
        if source.is_file():
            link_or_copy(source, directories["statistics"] / name)
    if args.log_file.is_file():
        link_or_copy(args.log_file, directories["logs"] / args.log_file.name)
    pid_file = args.log_file.with_suffix(".pid")
    if pid_file.is_file():
        link_or_copy(pid_file, directories["logs"] / pid_file.name)
    if args.results_document.is_file():
        link_or_copy(args.results_document, directories["docs"] / args.results_document.name)
    for relative in CODE_FILES:
        source = args.project_root / relative
        if source.is_file():
            link_or_copy(source, directories["code"] / relative)

    generated_figures: List[str] = []
    figure_jobs = (
        (plot_baseline_comparison, args.baseline_table),
        (plot_factorial_heatmap, args.factor_metrics_dir / "factorial_per_class.tsv"),
        (plot_modality_effects, args.factor_metrics_dir / "modality_evidence_decision.tsv"),
        (plot_structure_gain, args.factor_metrics_dir / "structure_true_class_probability_gain.tsv"),
        (plot_rbp_center_enrichment, args.factor_metrics_dir / "rbp_distance_attention_summary.tsv"),
        (plot_multiseed, args.multiseed_metrics_dir / "final_main_multiseed_per_class.tsv"),
    )
    for index, (plotter, source) in enumerate(figure_jobs, 1):
        if plotter(source, directories["figures"]):
            generated_figures.append(f"figure_{index:02d}")

    status = {
        "complete": not missing,
        "missing_file_count": len(missing),
        "missing_files": [str(path) for path in missing],
        "collected_file_counts": collected,
        "generated_figure_groups": generated_figures,
        "storage_note": "Large run files are hard-linked when source and package are on the same filesystem.",
    }
    (directories["overview"] / "collection_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = f"""# 最终实验结果包

本目录集中保存最终主模型新增实验的结果、表格、图片、统计检验、预测文件、模型配置、日志和复现代码。

- 当前状态：{'完整' if not missing else '训练中，已归档当前可用文件'}
- 缺失文件数：{len(missing)}
- `01_tables`：baseline、2×2 因子实验、匹配对照和多 seed 表格。
- `02_figures`：PNG 300 dpi 与 PDF 矢量图。
- `03_statistics`：逐类配对 bootstrap、95% CI、p 值和 FDR。
- `04_models_predictions_configs`：训练变体、预测、指标、模型和配置。
- `05_logs`：服务器训练与归档日志。
- `06_reproducibility_code`：本次实验使用的代码快照。
- `07_documentation`：Results 章节证据链文稿。
- `08_manifest`：文件大小和 SHA256 清单。

注意：m6Am 严格测试集阳性支持为 0，不报告有效测试性能；m5U 支持极少，结果需谨慎解释。
"""
    (directories["overview"] / "README.md").write_text(readme, encoding="utf-8")
    write_tsv(directories["manifest"] / "package_manifest.tsv", build_manifest(package_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--factor-run-root", required=True, type=Path)
    parser.add_argument("--factor-metrics-dir", required=True, type=Path)
    parser.add_argument("--multiseed-run-root", required=True, type=Path)
    parser.add_argument("--multiseed-metrics-dir", required=True, type=Path)
    parser.add_argument("--full-model-dir", required=True, type=Path)
    parser.add_argument("--baseline-table", required=True, type=Path)
    parser.add_argument("--results-document", required=True, type=Path)
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
