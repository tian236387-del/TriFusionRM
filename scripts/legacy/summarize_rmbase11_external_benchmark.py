#!/usr/bin/env python
"""Create tables and a Chinese report for the RMBase11 external-model benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]


MODEL_SPECS = [
    {
        "variant": "StructDualRM_M4",
        "display_name": "StructDualRM full",
        "path": "rmbase11_full_work/models/M4_structdualrm_full/metrics.json",
        "source_type": "proposed",
        "official_repo": "",
        "benchmark_role": "main",
        "comparability": "同数据、同split；使用序列、RNAfold结构、结构图和上下文/门控特征",
        "leakage_risk": "low",
    },
    {
        "variant": "StructDualRM_M0",
        "display_name": "Transformer sequence-only",
        "path": "rmbase11_full_work/models/M0_transformer_only_full/metrics.json",
        "source_type": "internal_ablation",
        "official_repo": "",
        "benchmark_role": "ablation",
        "comparability": "同数据、同split；仅序列Transformer，用于消融",
        "leakage_risk": "low",
    },
    {
        "variant": "MultiRM_style_full",
        "display_name": "MultiRM-style retrained",
        "path": "rmbase11_full_work/baselines/multirm_style_full/metrics.json",
        "source_type": "style_reimplementation",
        "official_repo": "https://github.com/Tsedao/MultiRM",
        "benchmark_role": "main_baseline",
        "comparability": "同数据、同split；保留3-mer embedding、BiLSTM、label-specific attention思想",
        "leakage_risk": "low",
    },
    {
        "variant": "TransRNAm_style_full",
        "display_name": "TransRNAm-style retrained",
        "path": "rmbase11_full_work/baselines/transrnam_style_full/metrics.json",
        "source_type": "style_reimplementation",
        "official_repo": "https://github.com/lennylv/TransRNAm",
        "benchmark_role": "main_baseline",
        "comparability": "同数据、同split；保留3-mer embedding、Transformer pooling、CNN classifier思想",
        "leakage_risk": "low",
    },
    {
        "variant": "EvoRMD_lite_full_context",
        "display_name": "EvoRMD-lite adapted full-context",
        "path": "rmbase11_full_work/baselines/evormd_lite_full/metrics.json",
        "source_type": "adapted_lite_reimplementation",
        "official_repo": "https://github.com/Gardeina/EvoRMD",
        "benchmark_role": "context_rich_diagnostic",
        "comparability": "同数据、同split；保留上下文融合到attention前，但未使用RNA-FM",
        "leakage_risk": "high",
    },
]


SUMMARY_FIELDS = [
    "rank_by_macro_AUPR",
    "variant",
    "display_name",
    "benchmark_role",
    "source_type",
    "official_repo",
    "model_dir",
    "train_rows",
    "valid_rows",
    "test_rows",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_F1",
    "test_weighted_F1",
    "test_MCC",
    "test_macro_AUC_ovr",
    "test_macro_AUPR_ovr",
    "valid_macro_AUPR_ovr",
    "best_epoch_by_valid_macro_AUPR",
    "leakage_risk",
    "comparability",
]

PER_CLASS_FIELDS = [
    "variant",
    "display_name",
    "mod_type",
    "support",
    "AUC_ovr",
    "AUPR_ovr",
    "F1",
    "Precision",
    "Recall",
    "MCC_ovr",
]

CODE_AUDIT_FIELDS = [
    "model",
    "official_repo",
    "local_source",
    "official_code_status",
    "same_data_action",
    "caveat",
]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def best_epoch(metrics: Dict[str, object]) -> object:
    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        return ""
    best = max(
        history,
        key=lambda item: float(item.get("validation", {}).get("macro_AUPR_ovr", -1.0)),
    )
    return best.get("epoch", "")


def summarize_one(spec: Dict[str, str]) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    path = Path(spec["path"])
    metrics = read_json(path)
    test = metrics.get("test", {})
    valid = metrics.get("validation", {})
    rows = metrics.get("row_counts", {})
    summary = {
        "variant": spec["variant"],
        "display_name": spec["display_name"],
        "benchmark_role": spec["benchmark_role"],
        "source_type": spec["source_type"],
        "official_repo": spec["official_repo"],
        "model_dir": str(path.parent),
        "train_rows": rows.get("train", ""),
        "valid_rows": rows.get("valid", ""),
        "test_rows": rows.get("test", ""),
        "test_accuracy": test.get("accuracy", ""),
        "test_balanced_accuracy": test.get("balanced_accuracy", ""),
        "test_macro_F1": test.get("macro_F1", ""),
        "test_weighted_F1": test.get("weighted_F1", ""),
        "test_MCC": test.get("MCC", ""),
        "test_macro_AUC_ovr": test.get("macro_AUC_ovr", ""),
        "test_macro_AUPR_ovr": test.get("macro_AUPR_ovr", ""),
        "valid_macro_AUPR_ovr": valid.get("macro_AUPR_ovr", ""),
        "best_epoch_by_valid_macro_AUPR": best_epoch(metrics),
        "leakage_risk": spec["leakage_risk"],
        "comparability": spec["comparability"],
    }
    per_class = []
    for mod_type in CLASS_NAMES:
        item = test.get("per_class", {}).get(mod_type, {})
        row = {"variant": spec["variant"], "display_name": spec["display_name"], "mod_type": mod_type}
        row.update(item)
        per_class.append(row)
    return summary, per_class


def make_code_audit() -> List[Dict[str, object]]:
    return [
        {
            "model": "MultiRM",
            "official_repo": "https://github.com/Tsedao/MultiRM",
            "local_source": "external/MultiRM-master",
            "official_code_status": "官方仓库可获得；原始代码针对旧版PyTorch和51/101 nt多标签输入。",
            "same_data_action": "在RMBase hg38 11类41 nt split上重训3-mer embedding + BiLSTM + label-specific attention适配版。",
            "caveat": "标注为style_reimplementation；不能写成官方权重原样重训。",
        },
        {
            "model": "TransRNAm",
            "official_repo": "https://github.com/lennylv/TransRNAm",
            "local_source": "rmbase11_full_work/baseline_repos/TransRNAm-main",
            "official_code_status": "官方仓库可获得；脚本依赖原作者Data_song、JSON编码矩阵，并写死旧GPU编号/路径。",
            "same_data_action": "在同一RMBase hg38 split上重训3-mer embedding + Transformer pooling + CNN适配版。",
            "caveat": "标注为style_reimplementation；不是原脚本无修改直接运行。",
        },
        {
            "model": "EvoRMD",
            "official_repo": "https://github.com/Gardeina/EvoRMD",
            "local_source": "rmbase11_full_work/baseline_repos/EvoRMD-main",
            "official_code_status": "官方仓库可获得；依赖RNA-FM/fm、pandas和原始生物上下文数据；当前Python环境缺少fm/pandas。",
            "same_data_action": "运行EvoRMD-lite适配版：保留上下文融合到attention前，但以可训练序列编码器替代RNA-FM。",
            "caveat": "full-context版本使用cell_list/seq_type_list/support_num等RMBase元数据，存在标签泄漏风险，只作为context-rich诊断，不作为主公平baseline。",
        },
    ]


def write_report(path: Path, summaries: Sequence[Dict[str, object]], code_audit_path: Path, summary_path: Path, per_class_path: Path) -> None:
    main_rows = [row for row in summaries if row["benchmark_role"] in {"main", "main_baseline", "ablation"}]
    diagnostic_rows = [row for row in summaries if row["benchmark_role"] == "context_rich_diagnostic"]
    m4 = next(row for row in summaries if row["variant"] == "StructDualRM_M4")
    m0 = next(row for row in summaries if row["variant"] == "StructDualRM_M0")
    multirm = next(row for row in summaries if row["variant"] == "MultiRM_style_full")
    trans = next(row for row in summaries if row["variant"] == "TransRNAm_style_full")
    evo = next(row for row in summaries if row["variant"] == "EvoRMD_lite_full_context")

    lines = [
        "# RMBase hg38 11类外部模型对标实验报告",
        "",
        "## 1. 实验目的",
        "",
        "本次实验的目标是把 StructDualRM 放到与公开模型同一数据条件下比较，而不是直接引用不同论文、不同数据集、不同负样本策略下的分数。对标对象为 MultiRM、TransRNAm 和 EvoRMD。所有可运行结果均使用 `rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites.tsv`，固定 `split_random`：train 756,175、valid 94,522、test 94,522。",
        "",
        "需要特别强调：这是一套 hg38 RMBase 11类实验，不是之前的 hg19 m1A/m6A 双位点流程。",
        "",
        "## 2. 官方代码与可运行性",
        "",
        f"官方代码审计表：`{code_audit_path}`。",
        "",
        "- MultiRM 官方仓库可获得，但原始代码面向旧版 PyTorch、51/101 nt 多标签输入；本次运行的是同结构思想的 3-mer BiLSTM label-attention 适配版。",
        "- TransRNAm 官方仓库可获得，但原脚本依赖作者提供的 Data_song、JSON 编码矩阵，并写死旧 GPU/path；本次运行的是同结构思想的 3-mer Transformer/CNN 适配版。",
        "- EvoRMD 官方仓库可获得，但当前环境没有 `fm/RNA-FM` 和 `pandas`，且原始上下文数据字段不等同于 RMBase 当前表；本次运行的是 EvoRMD-lite 适配版。",
        "",
        "## 3. 主结果",
        "",
        f"完整汇总表：`{summary_path}`。每类 one-vs-rest 指标：`{per_class_path}`。",
        "",
        "| 模型 | 角色 | Accuracy | Balanced Acc | Macro F1 | MCC | Macro AUC | Macro AUPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            "| {display_name} | {role} | {acc} | {bal} | {f1} | {mcc} | {auc} | {aupr} |".format(
                display_name=row["display_name"],
                role=row["benchmark_role"],
                acc=fmt(row["test_accuracy"]),
                bal=fmt(row["test_balanced_accuracy"]),
                f1=fmt(row["test_macro_F1"]),
                mcc=fmt(row["test_MCC"]),
                auc=fmt(row["test_macro_AUC_ovr"]),
                aupr=fmt(row["test_macro_AUPR_ovr"]),
            )
        )
    lines.extend(
        [
            "",
            "主公平对比中，StructDualRM full 的 test macro AUPR 为 "
            f"{fmt(m4['test_macro_AUPR_ovr'])}，高于 MultiRM-style 的 {fmt(multirm['test_macro_AUPR_ovr'])} "
            f"和 TransRNAm-style 的 {fmt(trans['test_macro_AUPR_ovr'])}。相对于内部 sequence-only Transformer 消融 M0，"
            f"M4 的 macro AUPR 从 {fmt(m0['test_macro_AUPR_ovr'])} 提高到 {fmt(m4['test_macro_AUPR_ovr'])}，"
            f"balanced accuracy 从 {fmt(m0['test_balanced_accuracy'])} 提高到 {fmt(m4['test_balanced_accuracy'])}。",
            "",
            "## 4. EvoRMD-lite full-context 诊断结果",
            "",
            "| 模型 | 角色 | Accuracy | Balanced Acc | Macro F1 | MCC | Macro AUC | Macro AUPR | 泄漏风险 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in diagnostic_rows:
        lines.append(
            "| {display_name} | {role} | {acc} | {bal} | {f1} | {mcc} | {auc} | {aupr} | {risk} |".format(
                display_name=row["display_name"],
                role=row["benchmark_role"],
                acc=fmt(row["test_accuracy"]),
                bal=fmt(row["test_balanced_accuracy"]),
                f1=fmt(row["test_macro_F1"]),
                mcc=fmt(row["test_MCC"]),
                auc=fmt(row["test_macro_AUC_ovr"]),
                aupr=fmt(row["test_macro_AUPR_ovr"]),
                risk=row["leakage_risk"],
            )
        )
    lines.extend(
        [
            "",
            f"EvoRMD-lite full-context 的 test macro AUPR 为 {fmt(evo['test_macro_AUPR_ovr'])}，数值高于 StructDualRM。这个结果不能简单解释为 EvoRMD 思路全面优于 StructDualRM，因为该适配版使用了 `cell_list`、`seq_type_list`、`support_num` 等 RMBase 元数据。对于“已知修饰位点的11类分类”来说，这些字段可能携带实验来源和数据库注释偏倚，和标签强相关；在真实未知候选位点预测时也未必可用。因此它更适合作为 context-rich 上限/泄漏敏感性诊断，而不应作为论文主结论中的公平 baseline。",
            "",
            "## 5. 可以写进论文的结论口径",
            "",
            "建议写法：",
            "",
            "> To avoid cross-study benchmark drift, we retrained architecture-style implementations of MultiRM and TransRNAm on the same RMBase hg38 11-class split. StructDualRM achieved higher macro AUPR and macro F1 than these sequence-window baselines, supporting the added value of RNAfold-derived structural features and structure-aware fusion. An EvoRMD-lite context-rich diagnostic run achieved near-saturated performance, but this result was not used as a primary fair baseline because RMBase metadata may encode experimental-source information unavailable for de novo candidate prediction.",
            "",
            "不建议写法：",
            "",
            "- 不要写“已经完整原版复现 EvoRMD 官方模型”，因为当前没有 RNA-FM/fm 环境。",
            "- 不要直接把 EvoRMD-lite full-context 的 0.9923 macro AUPR 当作公平比较结果。",
            "- 不要把这些结果和论文原表里的分数直接混比；这里的优势来自同数据同 split。",
            "",
            "## 6. 输出文件",
            "",
            f"- 主汇总表：`{summary_path}`",
            f"- 每类指标表：`{per_class_path}`",
            f"- 代码/可运行性审计表：`{code_audit_path}`",
            "- 模型目录：`rmbase11_full_work/models` 和 `rmbase11_full_work/baselines`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(path: Path, summaries: Sequence[Dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_rows = [row for row in summaries if row["benchmark_role"] in {"main", "main_baseline", "ablation"}]
    labels = [str(row["display_name"]).replace(" ", "\n") for row in plot_rows]
    aupr = [float(row["test_macro_AUPR_ovr"]) for row in plot_rows]
    f1 = [float(row["test_macro_F1"]) for row in plot_rows]
    colors = ["#2F5597" if row["variant"] == "StructDualRM_M4" else "#8FAADC" for row in plot_rows]
    x = range(len(plot_rows))

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    width = 0.36
    ax.bar([i - width / 2 for i in x], aupr, width=width, label="Macro AUPR", color=colors)
    ax.bar([i + width / 2 for i in x], f1, width=width, label="Macro F1", color="#B4C7E7")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.65, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Same-split RMBase hg38 11-class benchmark")
    ax.legend(frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, object]:
    summaries: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []
    for spec in MODEL_SPECS:
        if not Path(spec["path"]).exists():
            continue
        summary, per_class = summarize_one(spec)
        summaries.append(summary)
        per_class_rows.extend(per_class)

    summaries.sort(key=lambda row: float(row.get("test_macro_AUPR_ovr", -1.0)), reverse=True)
    for rank, row in enumerate(summaries, start=1):
        row["rank_by_macro_AUPR"] = rank

    out_dir = Path(args.out_dir)
    summary_path = out_dir / "external_model_benchmark_comparison.tsv"
    per_class_path = out_dir / "external_model_benchmark_per_class.tsv"
    code_audit_path = out_dir / "external_model_code_audit.tsv"
    report_path = Path(args.report)
    plot_path = Path(args.plot)

    write_tsv(summary_path, summaries, SUMMARY_FIELDS)
    write_tsv(per_class_path, per_class_rows, PER_CLASS_FIELDS)
    write_tsv(code_audit_path, make_code_audit(), CODE_AUDIT_FIELDS)
    write_report(report_path, summaries, code_audit_path, summary_path, per_class_path)
    make_plot(plot_path, summaries)

    payload = {
        "summary_tsv": str(summary_path),
        "per_class_tsv": str(per_class_path),
        "code_audit_tsv": str(code_audit_path),
        "report_md": str(report_path),
        "plot_png": str(plot_path) if plot_path.exists() else "",
        "model_count": len(summaries),
    }
    (out_dir / "external_model_benchmark_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize RMBase11 external model benchmark.")
    parser.add_argument("--out-dir", default="rmbase11_full_work/metrics")
    parser.add_argument("--report", default="rmbase11_full_work/RMBase11_hg38_外部模型对标报告.md")
    parser.add_argument("--plot", default="rmbase11_full_work/metrics/external_model_benchmark_macro_scores.png")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
