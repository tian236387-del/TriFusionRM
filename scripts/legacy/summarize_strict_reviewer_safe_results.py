#!/usr/bin/env python
"""Summarize reviewer-safe RMBase11 strict experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


MODEL_SPECS = [
    {
        "analysis_tier": "Main strict",
        "model": "M4_no_metadata",
        "setting": "CD-HIT-like 80% redundancy-filtered, no metadata",
        "path": "rmbase11_work/models/M4_no_metadata_cdhit80/metrics.json",
        "note": "Sequence + RNAfold scalar features + RNAfold pair graph + gate; region/support/gene-known disabled.",
    },
    {
        "analysis_tier": "Main strict baseline",
        "model": "M0_sequence_only",
        "setting": "CD-HIT-like 80% redundancy-filtered",
        "path": "rmbase11_work/models/M0_structdualrm_multiclass/metrics.json",
        "note": "Sequence transformer only; same data and split.",
    },
    {
        "analysis_tier": "Supplementary diagnostic",
        "model": "M4_with_metadata",
        "setting": "CD-HIT-like 80% redundancy-filtered, metadata enabled",
        "path": "rmbase11_work/models/M4_structdualrm_multiclass/metrics.json",
        "note": "Uses region/support/gene-known context; useful as diagnostic but not the cleanest main claim.",
    },
    {
        "analysis_tier": "Supplementary full benchmark",
        "model": "M4_full_random_with_metadata",
        "setting": "Full all-site RMBase11 random split, metadata enabled",
        "path": "rmbase11_full_work/models/M4_structdualrm_full/metrics.json",
        "note": "Complete all-site benchmark; higher score but more vulnerable to imbalance/redundancy concerns.",
    },
    {
        "analysis_tier": "Supplementary full baseline",
        "model": "M0_full_random",
        "setting": "Full all-site RMBase11 random split",
        "path": "rmbase11_full_work/models/M0_transformer_only_full/metrics.json",
        "note": "Sequence-only full random-split ablation.",
    },
]


WITHIN_BASE_SPECS = [
    ("M4_no_metadata", "rmbase11_work/metrics/within_base_M4_no_metadata_cdhit80.tsv"),
    ("M4_with_metadata", "rmbase11_work/metrics/within_base_M4_with_metadata_cdhit80.tsv"),
    ("M0_sequence_only", "rmbase11_work/metrics/within_base_M0_sequence_cdhit80.tsv"),
]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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


def summarize_models() -> List[Dict[str, object]]:
    rows = []
    for spec in MODEL_SPECS:
        path = Path(spec["path"])
        if not path.exists():
            continue
        metrics = read_json(path)
        test = metrics["test"]
        counts = metrics.get("row_counts", {})
        rows.append(
            {
                "analysis_tier": spec["analysis_tier"],
                "model": spec["model"],
                "setting": spec["setting"],
                "train_rows": counts.get("train", ""),
                "valid_rows": counts.get("valid", ""),
                "test_rows": counts.get("test", ""),
                "use_metadata": metrics.get("use_metadata", "not_recorded"),
                "balanced_accuracy": test.get("balanced_accuracy", ""),
                "macro_F1": test.get("macro_F1", ""),
                "MCC": test.get("MCC", ""),
                "macro_AUPR": test.get("macro_AUPR_ovr", ""),
                "macro_AUC": test.get("macro_AUC_ovr", ""),
                "note": spec["note"],
            }
        )
    return rows


def summarize_within_base() -> List[Dict[str, object]]:
    rows = []
    for model, path_str in WITHIN_BASE_SPECS:
        path = Path(path_str)
        if not path.exists():
            continue
        for row in read_tsv(path):
            row = dict(row)
            row["model"] = model
            rows.append(row)
    return rows


def write_svg(path: Path, strict_rows: Sequence[Dict[str, object]]) -> None:
    plot_rows = [r for r in strict_rows if r["model"] in {"M4_no_metadata", "M0_sequence_only", "M4_with_metadata"}]
    colors = {
        "M4_no_metadata": "#2563eb",
        "M0_sequence_only": "#94a3b8",
        "M4_with_metadata": "#f59e0b",
    }
    width, height = 920, 360
    left, top, bar_h, gap = 250, 70, 28, 42
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="28" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#111827">Reviewer-safe RMBase11 strict benchmark</text>',
        '<text x="28" y="58" font-family="Arial" font-size="13" fill="#4b5563">CD-HIT-like 80% redundancy-filtered set; bars show macro AUPR.</text>',
    ]
    for idx, row in enumerate(plot_rows):
        y = top + idx * (bar_h + gap)
        score = float(row["macro_AUPR"])
        f1 = float(row["macro_F1"])
        length = int((width - left - 130) * score)
        label = row["model"]
        color = colors.get(label, "#64748b")
        lines.extend(
            [
                f'<text x="28" y="{y + 19}" font-family="Arial" font-size="14" fill="#111827">{label}</text>',
                f'<rect x="{left}" y="{y}" width="{width-left-130}" height="{bar_h}" fill="#eef2f7"/>',
                f'<rect x="{left}" y="{y}" width="{length}" height="{bar_h}" fill="{color}"/>',
                f'<text x="{left + length + 8}" y="{y + 19}" font-family="Arial" font-size="13" fill="#111827">AUPR {fmt(score)} / F1 {fmt(f1)}</text>',
            ]
        )
    lines.extend(
        [
            '<text x="28" y="320" font-family="Arial" font-size="12" fill="#6b7280">Main claim should use M4_no_metadata vs M0_sequence_only; metadata-enabled models are diagnostic/supplementary.</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, model_rows: Sequence[Dict[str, object]], within_rows: Sequence[Dict[str, object]], out_dir: Path) -> None:
    by_model = {row["model"]: row for row in model_rows}
    m4 = by_model["M4_no_metadata"]
    m0 = by_model["M0_sequence_only"]
    m4_meta = by_model["M4_with_metadata"]
    full = by_model["M4_full_random_with_metadata"]
    lines = [
        "# RMBase11 严格可信结果汇总",
        "",
        "## 1. 为什么要重跑严格实验",
        "",
        "原始 full RMBase11 random split 的分数很高，但它同时存在类别极度不平衡、随机位点划分、序列相似性冗余、中心碱基差异和元数据可能泄漏等风险。因此，论文主结果不应只依赖 full random benchmark，而应使用更严格的 metadata-ablated 和 redundancy-aware 结果。",
        "",
        "本次重新整理后的推荐层次是：",
        "",
        "- 主结果：CD-HIT-like 80% 去冗余数据集，M4_no_metadata 对比 sequence-only M0，并报告 macro-AUPR、macro-F1、balanced accuracy 和 within-base 指标。",
        "- 补充结果：full all-site random split 和 metadata-enabled M4，用于说明模型在完整 benchmark 设置下也表现较好，但不作为最强泛化结论。",
        "",
        "## 2. 主结果表",
        "",
        "| 层级 | 模型 | 设置 | Train | Valid | Test | Metadata | Balanced Acc | Macro F1 | MCC | Macro AUPR | Macro AUC |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            f"| {row['analysis_tier']} | {row['model']} | {row['setting']} | {row['train_rows']} | {row['valid_rows']} | {row['test_rows']} | {row['use_metadata']} | {fmt(row['balanced_accuracy'])} | {fmt(row['macro_F1'])} | {fmt(row['MCC'])} | {fmt(row['macro_AUPR'])} | {fmt(row['macro_AUC'])} |"
        )

    delta_aupr = float(m4["macro_AUPR"]) - float(m0["macro_AUPR"])
    delta_f1 = float(m4["macro_F1"]) - float(m0["macro_F1"])
    meta_delta = float(m4_meta["macro_AUPR"]) - float(m4["macro_AUPR"])
    full_delta = float(full["macro_AUPR"]) - float(m4["macro_AUPR"])
    lines.extend(
        [
            "",
            "## 3. 关键解释",
            "",
            f"在 CD-HIT-like 80% 去冗余数据集上，M4_no_metadata 的 macro AUPR 为 {fmt(m4['macro_AUPR'])}，高于 sequence-only M0 的 {fmt(m0['macro_AUPR'])}，差值为 {fmt(delta_aupr)}；macro F1 从 {fmt(m0['macro_F1'])} 提高到 {fmt(m4['macro_F1'])}，差值为 {fmt(delta_f1)}。这说明即使去掉 region/support/gene-known 等元数据，RNAfold 二级结构和结构图分支仍然提供了独立增益。",
            "",
            f"metadata-enabled M4 的 macro AUPR 为 {fmt(m4_meta['macro_AUPR'])}，比 no-metadata M4 高 {fmt(meta_delta)}。这说明上下文/数据库证据相关字段确实会抬高结果，因此更适合放在补充材料，不适合作为主结论的唯一依据。",
            "",
            f"full all-site random split 的 M4 macro AUPR 为 {fmt(full['macro_AUPR'])}，比严格 no-metadata 去冗余结果高 {fmt(full_delta)}。这个差距正好支持我们的审稿人式判断：原始 full benchmark 可以作为完整可比较设置，但不能单独证明泛化能力。",
            "",
            "## 4. Within-base 评估",
            "",
            "| 模型 | 中心碱基组 | 类别 | 样本数 | Balanced Acc | Macro F1 | Macro AUPR | Macro AUC |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in within_rows:
        lines.append(
            f"| {row['model']} | {row['base_group']} | {row['classes']} | {row['sample_count']} | {fmt(row['balanced_accuracy'])} | {fmt(row['macro_F1'])} | {fmt(row['macro_AUPR'])} | {fmt(row['macro_AUC_ovr'])} |"
        )
    lines.extend(
        [
            "",
            "within-base 评估的意义是排除模型只靠中心碱基区分 A/C/G/U 家族的可能。结果显示，在 C-derived、G-derived 和 U-derived 组内，M4_no_metadata 的 macro AUPR 均高于 M0；A-derived 组中 M0 的 macro AUPR 略高，但 M4_no_metadata 的 balanced accuracy 更高。整体上，这支持结构信息不是简单依赖中心碱基带来的表观优势。",
            "",
            "## 5. 推荐论文表述",
            "",
            "> In the full random-split RMBase11 benchmark, StructDualRM achieved high predictive performance. However, because random site-level splitting and database-derived metadata may inflate apparent performance under severe class imbalance, we treated this result as a supplementary benchmark. For the main robustness analysis, we evaluated a metadata-ablated StructDualRM model on a CD-HIT-like 80% redundancy-filtered dataset and further assessed classification within same-center-base modification groups. Although absolute scores decreased under this stricter setting, the structure-aware model remained superior to the sequence-only transformer baseline, supporting an independent contribution of RNAfold-derived secondary-structure information to multi-type RNA modification prediction.",
            "",
            "## 6. 输出文件",
            "",
            f"- 主结果表：`{out_dir / 'strict_reviewer_safe_model_metrics.tsv'}`",
            f"- within-base 表：`{out_dir / 'strict_reviewer_safe_within_base_metrics.tsv'}`",
            f"- 图：`{out_dir / 'strict_reviewer_safe_macro_scores.svg'}`",
            "",
            "## 7. 仍建议补充的实验",
            "",
            "本轮已经完成了 CD-HIT-like 80% 去冗余数据集上的 50 epoch M4_no_metadata 主实验，以及 within-base 评估。full all-site 数据集的 `split_gene` 也已经生成在 `rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_strict_splits.tsv`，但 full gene-level no-metadata 50 epoch 尚未纳入本主表。全量 `split_cdhit80_cluster` 因近百万序列聚类耗时较长，本轮未完成；当前 CD-HIT-like 证据来自已完成的 class-capped redundancy-filtered 数据集。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def run(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.out_dir)
    model_rows = summarize_models()
    within_rows = summarize_within_base()
    model_tsv = out_dir / "strict_reviewer_safe_model_metrics.tsv"
    within_tsv = out_dir / "strict_reviewer_safe_within_base_metrics.tsv"
    svg_path = out_dir / "strict_reviewer_safe_macro_scores.svg"
    report_path = out_dir / "strict_reviewer_safe_results_report.md"
    write_tsv(
        model_tsv,
        model_rows,
        [
            "analysis_tier",
            "model",
            "setting",
            "train_rows",
            "valid_rows",
            "test_rows",
            "use_metadata",
            "balanced_accuracy",
            "macro_F1",
            "MCC",
            "macro_AUPR",
            "macro_AUC",
            "note",
        ],
    )
    write_tsv(
        within_tsv,
        within_rows,
        ["model", "base_group", "classes", "sample_count", "accuracy", "balanced_accuracy", "macro_F1", "macro_AUPR", "macro_AUC_ovr"],
    )
    write_svg(svg_path, model_rows)
    write_report(report_path, model_rows, within_rows, out_dir)
    payload = {
        "model_tsv": str(model_tsv),
        "within_base_tsv": str(within_tsv),
        "figure_svg": str(svg_path),
        "report_md": str(report_path),
    }
    (out_dir / "strict_reviewer_safe_results_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize strict reviewer-safe RMBase11 results.")
    parser.add_argument("--out-dir", default="rmbase11_work/metrics")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
