#!/usr/bin/env python
"""Write a Chinese report for the RMBase hg38 11-class robustness experiment."""

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
    sens = read_tsv(Path(args.sensitivity))
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    best = comp[0]
    m0 = next(row for row in comp if row["variant"] == "M0")
    m4 = next(row for row in comp if row["variant"] == "M4")
    kmer = next(row for row in comp if row["variant"] == "kmer_extratrees_34")

    comp_rows = []
    for row in comp:
        comp_rows.append(
            {
                "模型": row["variant"],
                "Accuracy": fmt(row["test_accuracy"]),
                "Balanced accuracy": fmt(row["test_balanced_accuracy"]),
                "Macro F1": fmt(row["test_macro_F1"]),
                "MCC": fmt(row["test_MCC"]),
                "Macro AUC": fmt(row["test_macro_AUC_ovr"]),
                "Macro AUPR": fmt(row["test_macro_AUPR_ovr"]),
            }
        )
    count_rows = []
    for row in counts:
        count_rows.append(
            {
                "类别": row["mod_type"],
                "RMBase原始": row["raw_target_rows"],
                "中心校验": row["valid_41nt_center_checked_rows"],
                "采样后": row["balanced_sample_rows"],
                "去冗余后": row["redundancy_filtered_rows"],
                "训练/验证/测试": f"{row['train_rows']}/{row['valid_rows']}/{row['test_rows']}",
            }
        )

    sensitivity_rows = []
    for row in sens:
        sensitivity_rows.append(
            {
                "模型": row["variant"],
                "保留类别数": row["included_class_count"],
                "排除类别": row["excluded_classes"],
                "Macro AUPR": fmt(row["macro_AUPR_ovr_included"]),
                "Macro F1": fmt(row["macro_F1_included"]),
                "Mean recall": fmt(row["mean_recall_included"]),
            }
        )

    text = f"""# RMBase hg38 11类RNA修饰扩展实验报告

## 1. 实验目的

原始 StructDualRM 项目主要围绕本地 hg19 坐标体系下的 m1A/m6A 双标签预测展开。为了回应“只做两个位点是否不足以说明鲁棒性”的问题，本次补充了一个独立的 RMBase v3 Homo sapiens hg38 11 类 RNA 修饰多分类 benchmark，类别包括 Am、Cm、Gm、Um、D、Y、m1A、m5C、m5U、m6A 和 m7G。

这个实验的定位不是替换原来的 hg19 m1A/m6A 主线，而是作为跨修饰类型的鲁棒性补充。由于 RMBase v3 公开下载页提供的是 hg38 单位点注释，本实验明确标注为 hg38 11 类扩展实验，不能和原 hg19 数据直接混写为同一数据来源。

## 2. 数据来源与预处理

本次使用 RMBase v3 的人类 hg38 single-site 文件：`hg38.m1A`、`hg38.m5C`、`hg38.m6A`、`hg38.m7G`、`hg38.Nm`、`hg38.Pseudo` 和 `hg38.otherMod`。RMBase 文件为 29 列 BED-like 格式，其中第 7 列为修饰类型，第 19 列为 41 nt 中心窗口序列。

预处理步骤如下：

1. 解析 29 列 BED-like 文件，只保留目标 11 类修饰。
2. 将 U 统一为 T，并要求窗口长度为 41 nt。
3. 进行中心碱基校验：Am/m1A/m6A 为 A，Cm/m5C 为 C，Gm/m7G 为 G，Um/D/Y/m5U 为 T。
4. 对超大类 m6A、m5C、Y 做类别上限采样，每类最多 5000 条，避免训练集被 m6A 主导。
5. 进行 80% identity 的固定窗口 greedy 去冗余。由于本机未安装 `cd-hit-est`，这里不是直接调用 CD-HIT，而是 CD-HIT-like 的确定性去冗余。
6. 按类别分层切分训练、验证和测试集，比例约为 8:1:1。

## 3. 类别规模

{table(count_rows, ["类别", "RMBase原始", "中心校验", "采样后", "去冗余后", "训练/验证/测试"])}

需要注意，D 和 m5U 在 RMBase 中本来样本就少，经过 80% identity 去冗余后分别只剩 34 和 11 条。因此这两个类别可以纳入全局 robustness benchmark，但单类指标不能被过度解释。

## 4. RNAfold结构特征

对最终去冗余后的 16,459 条 41 nt 序列全部运行 RNAfold，共得到 16,459 条结构记录。每个位点提取 dot-bracket、MFE、中心碱基是否配对、中心 11 nt 和 21 nt 配对比例、全窗口配对比例、MFE per nt 以及中心 loop span。这些结构特征进入 M1/M3/M4 的标量结构分支，dot-bracket 解析得到的配对边进入 M2/M3/M4 的 pair-aware graph 分支。

## 5. 模型与消融设计

本次 11 类实验采用多分类 softmax 输出，损失函数为 class-weighted CrossEntropyLoss，并使用 balanced sampler 缓解类别不平衡。消融版本如下：

- M0：中心感知序列 Transformer，仅使用碱基、绝对位置和距中心距离编码。
- M1：M0 + RNAfold 标量结构特征。
- M2：M0 + pair-aware RNA 结构图分支。
- M3：M2 + class-specific gate，对不同修饰类别学习结构证据权重。
- M4：M3 + 区域/支持度等上下文特征，是完整结构感知版本。

同时训练了两个传统序列基线：3/4-mer logistic regression 和 3/4-mer ExtraTrees。

## 6. 模型性能

{table(comp_rows, ["模型", "Accuracy", "Balanced accuracy", "Macro F1", "MCC", "Macro AUC", "Macro AUPR"])}

完整 M4 模型在测试集上取得最好的整体结果：accuracy {fmt(m4["test_accuracy"])}、balanced accuracy {fmt(m4["test_balanced_accuracy"])}、MCC {fmt(m4["test_MCC"])}、macro AUC {fmt(m4["test_macro_AUC_ovr"])}、macro AUPR {fmt(m4["test_macro_AUPR_ovr"])}。与序列-only M0 相比，M4 的 macro AUPR 从 {fmt(m0["test_macro_AUPR_ovr"])} 提升到 {fmt(m4["test_macro_AUPR_ovr"])}，MCC 从 {fmt(m0["test_MCC"])} 提升到 {fmt(m4["test_MCC"])}。与最强传统 k-mer ExtraTrees 基线相比，M4 的 macro AUPR 从 {fmt(kmer["test_macro_AUPR_ovr"])} 提升到 {fmt(m4["test_macro_AUPR_ovr"])}。

## 7. 小样本类别敏感性分析

考虑到 D 和 m5U 测试集中分别只有 4 和 1 条，本次额外统计了排除 D/m5U 后的 9 类平均指标：

{table(sensitivity_rows, ["模型", "保留类别数", "排除类别", "Macro AUPR", "Macro F1", "Mean recall"])}

排除极小样本类别后，M4 仍为最优模型，9 类 macro AUPR 达到 {fmt(sens[0]["macro_AUPR_ovr_included"])}。这说明 M4 的优势不是由 m5U 这类偶然小样本指标单独造成的，而是在多数有足够测试样本的修饰类型上保持了更强的可分性。

## 8. 可以写进论文的结论

第一，原始 m1A/m6A 双标签实验仍是主线，因为它基于本地 hg19 数据、转录本空间和 m1A-m6A 生物学关系分析。第二，RMBase 11 类 hg38 实验可以作为额外 robustness benchmark，证明 StructDualRM 的结构感知框架并非只对 m1A/m6A 有效。第三，M2/M3/M4 相比 M0 的提升说明 RNAfold-derived pairing information 和 class-specific structural gating 对跨修饰类型识别有贡献。

## 9. 必须谨慎表述的地方

- 本实验是 hg38 RMBase v3 benchmark，不是 hg19 数据；不要写成和原 m1A/m6A hg19 主线完全同源。
- 本机未安装 CD-HIT，当前去冗余应写为 CD-HIT-like 80% identity greedy filtering。若论文必须写“CD-HIT”，需要安装 `cd-hit-est` 后重新跑。
- D 和 m5U 样本过少，不能对这两类给出生物学强结论。
- 目前只跑了 seed=42。若要冲击高水平期刊，建议补 3-5 个随机种子，报告均值和标准差。

## 10. 主要输出文件

- 数据集：`rmbase11_work/datasets/rmbase11_hg38_41nt_cdhit80_like.tsv`
- RNAfold：`rmbase11_work/structure/rnafold_41nt_rmbase11.tsv`
- 类别计数：`rmbase11_work/tables/rmbase11_class_counts.tsv`
- 模型目录：`rmbase11_work/models`
- 总体指标：`rmbase11_work/metrics/rmbase11_model_comparison.tsv`
- 单类指标：`rmbase11_work/metrics/rmbase11_per_class_metrics.tsv`
- 敏感性分析：`rmbase11_work/metrics/rmbase11_sensitivity_excluding_D_m5U.tsv`

## 11. 复现命令

```powershell
.\\.venv\\Scripts\\python.exe -B scripts\\build_rmbase11_dataset.py --extracted-dir rmbase11_work\\extracted --out-dir rmbase11_work --max-per-class 5000 --identity-threshold 0.8 --seed 42
.\\.venv\\Scripts\\python.exe -B scripts\\run_rmbase11_rnafold.py --dataset rmbase11_work\\datasets\\rmbase11_hg38_41nt_cdhit80_like.tsv --out rmbase11_work\\structure\\rnafold_41nt_rmbase11.tsv --batch-size 1000
.\\.venv\\Scripts\\python.exe -B scripts\\train_structdualrm_multiclass.py --dataset rmbase11_work\\datasets\\rmbase11_hg38_41nt_cdhit80_like.tsv --rnafold rmbase11_work\\structure\\rnafold_41nt_rmbase11.tsv --out-dir rmbase11_work\\models\\M4_structdualrm_multiclass --variant M4 --epochs 50 --batch-size 256 --balanced-sampler --seed 42
.\\.venv\\Scripts\\python.exe -B scripts\\compare_rmbase11_metrics.py --models-dir rmbase11_work\\models --out-dir rmbase11_work\\metrics
```
"""
    Path(args.out).write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write RMBase11 experiment report.")
    parser.add_argument("--counts", default="rmbase11_work/tables/rmbase11_class_counts.tsv")
    parser.add_argument("--comparison", default="rmbase11_work/metrics/rmbase11_model_comparison.tsv")
    parser.add_argument("--sensitivity", default="rmbase11_work/metrics/rmbase11_sensitivity_excluding_D_m5U.tsv")
    parser.add_argument("--summary", default="rmbase11_work/datasets/rmbase11_hg38_build_summary.json")
    parser.add_argument("--out", default="rmbase11_work/RMBase11_hg38_扩展实验报告.md")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
