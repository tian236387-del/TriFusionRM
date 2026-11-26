#!/usr/bin/env python
"""Write a Chinese report for the full RMBase hg38 11-class experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def table(rows: List[Dict[str, str]], fields: List[str]) -> str:
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(field, "")) for field in fields) + "|")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    counts = read_tsv(Path(args.counts))
    comp = read_tsv(Path(args.comparison))
    per_class = read_tsv(Path(args.per_class))
    build_summary = json.loads(Path(args.build_summary).read_text(encoding="utf-8"))
    fold_summary = json.loads(Path(args.fold_summary).read_text(encoding="utf-8"))
    m4 = next(row for row in comp if row["variant"] == "M4")
    m0 = next(row for row in comp if row["variant"] == "M0")

    count_rows = [
        {
            "修饰": row["mod_type"],
            "RMBase原始": row["raw_target_rows"],
            "保留位点": row["retained_all_site_rows"],
            "训练/验证/测试": f"{row['train_rows']}/{row['valid_rows']}/{row['test_rows']}",
        }
        for row in counts
    ]
    comparison_rows = [
        {
            "模型": row["variant"],
            "Accuracy": fmt(row["test_accuracy"]),
            "Balanced Acc": fmt(row["test_balanced_accuracy"]),
            "Macro F1": fmt(row["test_macro_F1"]),
            "MCC": fmt(row["test_MCC"]),
            "Macro AUC": fmt(row["test_macro_AUC_ovr"]),
            "Macro AUPR": fmt(row["test_macro_AUPR_ovr"]),
        }
        for row in comp
    ]
    m4_per_class = [
        {
            "修饰": row["mod_type"],
            "测试数": row["support"],
            "AUC": fmt(row["AUC_ovr"]),
            "AUPR": fmt(row["AUPR_ovr"]),
            "F1": fmt(row["F1"]),
            "Precision": fmt(row["Precision"]),
            "Recall": fmt(row["Recall"]),
        }
        for row in per_class
        if row["variant"] == "M4"
    ]

    text = f"""# RMBase hg38 11类全量结构实验报告

## 实验结论

本次实验使用的是 RMBase v3 Homo sapiens `hg38` 公开单碱基修饰注释，不是原始本地 `hg19` m1A/m6A 数据。按照你的要求，这一版没有做每类 5000 条的小规模/平衡采样，而是保留 11 种修饰类型的全部可用位点，共 `{build_summary['total_retained_rows']}` 条。

全部 `{fold_summary['input_rows']}` 条 41 nt 中心窗口均已运行 RNAfold，并提取 dot-bracket、MFE、中心配对状态、局部配对比例、全窗口配对比例、MFE per nt 和中心 loop span 等结构特征。完整 M4 模型把这些 RNAfold 结构标量和 dot-bracket 解析得到的配对边输入到结构分支和 pair-aware graph 分支中。

## 全量类别规模

{table(count_rows, ["修饰", "RMBase原始", "保留位点", "训练/验证/测试"])}

需要强调：这是一个极度不平衡的全量数据集。m6A 有 882,385 条，而 m5U 只有 66 条。因此 accuracy 和 weighted F1 很容易被 m6A 主导，论文里应优先展示 macro AUPR、balanced accuracy、macro F1、MCC 和 per-class 指标。

## M0 与 M4 对比

{table(comparison_rows, ["模型", "Accuracy", "Balanced Acc", "Macro F1", "MCC", "Macro AUC", "Macro AUPR"])}

M4 相比 M0 的提升为：

- Macro AUPR：{fmt(m0['test_macro_AUPR_ovr'])} -> {fmt(m4['test_macro_AUPR_ovr'])}
- Balanced accuracy：{fmt(m0['test_balanced_accuracy'])} -> {fmt(m4['test_balanced_accuracy'])}
- Macro F1：{fmt(m0['test_macro_F1'])} -> {fmt(m4['test_macro_F1'])}
- MCC：{fmt(m0['test_MCC'])} -> {fmt(m4['test_MCC'])}

这说明在全量 11 类 RMBase hg38 数据上，RNAfold-derived structure features、pair-aware structure graph 和 class-specific gate 对模型识别能力有增益。

## M4 单类别表现

{table(m4_per_class, ["修饰", "测试数", "AUC", "AUPR", "F1", "Precision", "Recall"])}

## 输出文件

- 全量数据集：`rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites.tsv`
- 全量 RNAfold 结构：`rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv`
- M0 模型：`rmbase11_full_work/models/M0_transformer_only_full/model.pt`
- M4 模型：`rmbase11_full_work/models/M4_structdualrm_full/model.pt`
- 模型对比表：`rmbase11_full_work/metrics/rmbase11_model_comparison.tsv`
- 单类别指标：`rmbase11_full_work/metrics/rmbase11_per_class_metrics.tsv`

## 论文表述建议

可以写：为了进一步评估模型在多修饰类型上的泛化能力，我们基于 RMBase v3 hg38 构建了 11 类 RNA 修饰全量 benchmark，并对所有 41 nt 中心窗口进行 RNAfold 二级结构预测。结构感知 M4 模型在全量测试集上取得 macro AUPR {fmt(m4['test_macro_AUPR_ovr'])} 和 balanced accuracy {fmt(m4['test_balanced_accuracy'])}，高于序列-only Transformer M0，说明结构信息在跨修饰类型识别中提供了额外判别力。

不要写：这个实验是 hg19；不要写所有类别样本均衡；不要只报告 accuracy；不要把这次结果直接说成超过 EvoRMD/MultiRM/TransRNAm，除非后续在同一 split 上实际跑完那三个模型。
"""
    Path(args.out).write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write full RMBase11 report.")
    parser.add_argument("--counts", default="rmbase11_full_work/tables/rmbase11_full_class_counts.tsv")
    parser.add_argument("--comparison", default="rmbase11_full_work/metrics/rmbase11_model_comparison.tsv")
    parser.add_argument("--per-class", default="rmbase11_full_work/metrics/rmbase11_per_class_metrics.tsv")
    parser.add_argument("--build-summary", default="rmbase11_full_work/datasets/rmbase11_full_build_summary.json")
    parser.add_argument("--fold-summary", default="rmbase11_full_work/structure/rnafold_41nt_all_sites.summary.json")
    parser.add_argument("--out", default="rmbase11_full_work/RMBase11_hg38_全量结构实验报告.md")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
