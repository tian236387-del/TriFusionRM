#!/usr/bin/env python
"""Audit StructDualRM experiments without selecting models from test performance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


COMMON_DATASET = "rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv"
CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "Y", "AtoI"]
EVALUABLE_CLASSES = [name for name in CLASS_NAMES if name != "m6Am"]


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def value(mapping: object, key: str, default: object = "") -> object:
    return mapping.get(key, default) if isinstance(mapping, Mapping) else default


def number(raw: object) -> float:
    try:
        result = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return result


def fmt(raw: object, digits: int = 6) -> str:
    result = number(raw)
    return f"{result:.{digits}f}" if math.isfinite(result) else "NA"


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def classify(path: Path, metrics: Mapping[str, object]) -> tuple[str, str, bool]:
    name = path.as_posix().lower()
    model_name = path.parent.name.lower()
    dataset = str(metrics.get("dataset", ""))
    classes = list(metrics.get("classes", []))
    if "smoke" in name or "probe" in name:
        return "smoke", "短流程代码检查，不代表收敛性能", False
    if dataset != COMMON_DATASET or classes != CLASS_NAMES:
        return "different_data", "数据集、标签顺序或样本空间与最终严格实验不同", False
    if "shuffled" in name:
        return "negative_control", "打乱实验用于检验信息对应关系，不能部署或参与选模", False
    if "m6a_cap_interference_controls" in name:
        return "distribution_diagnostic", "改变m6A采样上限，仅用于类别干扰诊断", False
    if "evormd_full_teacher" in name or number(value(metrics.get("distillation"), "m1a_weight", 0)) > 0:
        return "leakage_sensitive", "教师含实验来源元数据偏倚，只能作敏感性诊断", False
    if "two_stage" in name or "calibrator" in name or "reranker" in name:
        return "post_processing", "后处理模型不属于端到端主架构，单独报告", False
    if "sequence_rbp_multiseed_20260730" in name or (
        "final_modality_factorial_ablation_seed42_20260726" in name and model_name == "sequence_rbp"
    ):
        return "final_candidate", "Matched sequence-plus-RBP candidate evaluated with the same pure validation rule", True
    if "no_pair_edges_multiseed_20260730" in name or (
        "final_modality_factorial_ablation_seed42_20260726" in name and model_name == "full_no_pair_edges"
    ):
        return "final_candidate", "干净三模态简化候选；最终资格取决于多seed验证", True
    if "final_main_multiseed_20260726" in name or (
        model_name == "multi_length_independent" and "multirm12_multi_length_fusion_cdhit80" in name
    ):
        return "reference_model", "当前三模态参考架构", True
    if "sequence_only" in name or "sequence_structure" in name or "sequence_rbp" in name:
        return "core_ablation", "同口径核心模态消融", False
    if "no_pair_edges" in name or "no_pair" in name:
        return "exploratory_candidate", "Unmatched no-pair experiment outside the predeclared three-seed candidate set", False
    if "afamily" in name or "m1a_aux" in name:
        return "exploratory_candidate", "类别特异头或重采样实验；需纯验证规则和多seed复核", False
    if "hparam" in name:
        return "exploratory_candidate", "超参数实验使用类别重点复合选模，不能与纯Macro AUPR直接排序", False
    if "multirm12_sequence_length_ablation" in name or model_name == "multi_length_shared":
        return "architecture_ablation", "序列长度或参数共享消融", False
    return "other", "未纳入最终候选排名的其他实验", False


def group_name(path: Path, category: str) -> str:
    name = path.as_posix().lower()
    if category == "final_candidate":
        if "sequence_rbp" in name:
            return "sequence_rbp"
        return "three_modal_no_pair_edges"
    if category == "reference_model":
        return "three_modal_with_pair_edges"
    if "main_hparam" in name:
        return "am_focused_hparam"
    if "afamily_embedded_head_w05" in name:
        return "afamily_oversampled"
    if "afamily_early_class" in name:
        return "afamily_early_fusion"
    return path.parent.name


def experiment_seed(path: Path, metrics: Mapping[str, object]) -> object:
    config = metrics.get("config", {})
    stored = value(config, "seed", metrics.get("seed", ""))
    if stored not in ("", None):
        return stored
    for component in reversed(path.parts):
        match = re.fullmatch(r"seed[_-]?(\d+)", component, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    if path.name == "multi_length_independent" or "full_no_pair_edges" in path.as_posix():
        return 42
    return ""


def audit(metrics_paths: Iterable[Path], out_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    for path in sorted(metrics_paths):
        try:
            metrics = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        validation = metrics.get("validation", {})
        test = metrics.get("test", {})
        config = metrics.get("config", {})
        category, reason, eligible = classify(path, metrics)
        group = group_name(path.parent, category)
        row = {
            "experiment": path.parent.as_posix(),
            "group": group,
            "category": category,
            "eligible_for_final_selection": "yes" if eligible else "no",
            "reason": reason,
            "dataset": metrics.get("dataset", ""),
            "split": metrics.get("split", ""),
            "num_classes": len(metrics.get("classes", [])),
            "seed": experiment_seed(path.parent, metrics),
            "best_epoch": metrics.get("best_epoch", ""),
            "selection_metric": metrics.get("selection_metric", ""),
            "selection_focus_class": metrics.get("selection_focus_class", ""),
            "selection_focus_metric": metrics.get("selection_focus_metric", ""),
            "selection_focus_weight": metrics.get("selection_focus_weight", ""),
            "best_selection_score": fmt(metrics.get("best_validation_score")),
            "validation_macro_AUPR": fmt(value(validation, "macro_AUPR_ovr")),
            "test_macro_AUPR": fmt(value(test, "macro_AUPR_ovr")),
            "test_macro_F1": fmt(value(test, "macro_F1")),
            "test_global_MCC": fmt(value(test, "MCC")),
            "test_balanced_accuracy": fmt(value(test, "balanced_accuracy")),
            "uses_structure": value(config, "use_structure", ""),
            "uses_rbp_positional": value(config, "use_rbp_positional", ""),
            "afamily_aux_weight": metrics.get("afamily_aux_weight", ""),
            "afamily_residual_scale": metrics.get("afamily_residual_scale", ""),
            "oversampling_enabled": value(metrics.get("oversampling"), "enabled", False),
        }
        rows.append(row)
        for split_name, split_metrics in [("validation", validation), ("test", test)]:
            class_metrics = value(split_metrics, "per_class", {})
            for class_name in CLASS_NAMES:
                current = value(class_metrics, class_name, {})
                per_class_rows.append(
                    {
                        "experiment": path.parent.as_posix(),
                        "group": group,
                        "category": category,
                        "split": split_name,
                        "mod_type": class_name,
                        "support": value(current, "support", ""),
                        "AUPR_ovr": fmt(value(current, "AUPR_ovr")),
                        "AUC_ovr": fmt(value(current, "AUC_ovr")),
                        "F1": fmt(value(current, "F1")),
                        "Precision": fmt(value(current, "Precision")),
                        "Recall": fmt(value(current, "Recall")),
                        "MCC_ovr": fmt(value(current, "MCC_ovr")),
                    }
                )

    rows.sort(key=lambda row: (row["category"], row["group"], row["experiment"]))
    write_tsv(out_dir / "all_experiments_audit.tsv", rows)
    write_tsv(out_dir / "all_experiments_per_class.tsv", per_class_rows)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["eligible_for_final_selection"] == "yes":
            grouped[str(row["group"])].append(row)
    candidate_rows: list[dict[str, object]] = []
    for group, members in sorted(grouped.items()):
        valid = [score for row in members if math.isfinite(score := number(row["validation_macro_AUPR"]))]
        test_aupr = [score for row in members if math.isfinite(score := number(row["test_macro_AUPR"]))]
        test_f1 = [score for row in members if math.isfinite(score := number(row["test_macro_F1"]))]
        test_mcc = [score for row in members if math.isfinite(score := number(row["test_global_MCC"]))]
        if not valid:
            continue
        candidate_rows.append(
            {
                "candidate": group,
                "num_seeds": len(valid),
                "validation_macro_AUPR_mean": fmt(sum(valid) / len(valid)),
                "validation_macro_AUPR_min": fmt(min(valid)),
                "test_macro_AUPR_mean": fmt(sum(test_aupr) / len(test_aupr)) if test_aupr else "NA",
                "test_macro_F1_mean": fmt(sum(test_f1) / len(test_f1)) if test_f1 else "NA",
                "test_global_MCC_mean": fmt(sum(test_mcc) / len(test_mcc)) if test_mcc else "NA",
                "selection_status": "eligible" if len(valid) >= 3 else "needs_at_least_3_seeds",
            }
        )
    write_tsv(out_dir / "final_candidate_multiseed_summary.tsv", candidate_rows)

    report = [
        "# StructDualRM experiment audit",
        "",
        f"- Parsed experiments: {len(rows)}",
        f"- Common final dataset: `{COMMON_DATASET}`",
        "- Final architecture selection uses validation Macro AUPR across matched seeds; test metrics are reported only after selection.",
        "- Shuffled controls, changed-distribution diagnostics, metadata-sensitive distillation, smoke runs and post-processing models are never eligible for final selection.",
        "- `m6Am` has no positive sample in the common strict test set and is excluded from evaluable-class macro interpretation.",
        "",
        "The candidate summary is provisional until every candidate has at least three matched seeds.",
    ]
    (out_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=Path("rmbase12_work/models"))
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit(args.models_root.rglob("metrics.json"), args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
