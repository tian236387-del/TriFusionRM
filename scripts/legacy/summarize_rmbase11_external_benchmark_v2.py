#!/usr/bin/env python
"""Summarize same-split RMBase11 benchmarks, including safe EvoRMD-lite."""

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
        "role": "proposed_main",
        "source_type": "proposed",
        "repo": "",
        "leakage_risk": "low",
        "note": "Sequence + RNAfold structure + structure graph + gated fusion.",
    },
    {
        "variant": "StructDualRM_M0",
        "display_name": "Transformer sequence-only",
        "path": "rmbase11_full_work/models/M0_transformer_only_full/metrics.json",
        "role": "internal_ablation",
        "source_type": "internal_ablation",
        "repo": "",
        "leakage_risk": "low",
        "note": "Sequence-only transformer ablation on the same split.",
    },
    {
        "variant": "MultiRM_style_full",
        "display_name": "MultiRM-style retrained",
        "path": "rmbase11_full_work/baselines/multirm_style_full/metrics.json",
        "role": "main_baseline",
        "source_type": "style_reimplementation",
        "repo": "https://github.com/Tsedao/MultiRM",
        "leakage_risk": "low",
        "note": "3-mer embedding + BiLSTM + label-specific attention adapted to 11-class same split.",
    },
    {
        "variant": "TransRNAm_style_full",
        "display_name": "TransRNAm-style retrained",
        "path": "rmbase11_full_work/baselines/transrnam_style_full/metrics.json",
        "role": "main_baseline",
        "source_type": "style_reimplementation",
        "repo": "https://github.com/lennylv/TransRNAm",
        "leakage_risk": "low",
        "note": "3-mer embedding + transformer pooling + CNN head adapted to 11-class same split.",
    },
    {
        "variant": "EvoRMD_lite_safe_context",
        "display_name": "EvoRMD-lite safe-context",
        "path": "rmbase11_full_work/baselines/evormd_lite_safe_context_full/metrics.json",
        "role": "secondary_baseline",
        "source_type": "adapted_lite_reimplementation",
        "repo": "https://github.com/Gardeina/EvoRMD",
        "leakage_risk": "medium",
        "note": "Context-before-attention idea, no RNA-FM, conservative context only: region/gene_type/gene_known.",
    },
    {
        "variant": "EvoRMD_lite_full_context",
        "display_name": "EvoRMD-lite full-context",
        "path": "rmbase11_full_work/baselines/evormd_lite_full/metrics.json",
        "role": "context_rich_diagnostic",
        "source_type": "adapted_lite_reimplementation",
        "repo": "https://github.com/Gardeina/EvoRMD",
        "leakage_risk": "high",
        "note": "Context-before-attention idea, no RNA-FM, uses RMBase metadata including cell_list, seq_type_list and support_num.",
    },
]

SUMMARY_FIELDS = [
    "rank_by_macro_AUPR",
    "variant",
    "display_name",
    "role",
    "source_type",
    "repo",
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
    "note",
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

AUDIT_FIELDS = ["model", "official_repo", "local_source", "official_status", "same_data_action", "caveat"]


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
    best = max(history, key=lambda x: float(x.get("validation", {}).get("macro_AUPR_ovr", -1.0)))
    return best.get("epoch", "")


def summarize(spec: Dict[str, str]) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    path = Path(spec["path"])
    metrics = read_json(path)
    test = metrics.get("test", {})
    valid = metrics.get("validation", {})
    counts = metrics.get("row_counts", {})
    row = {
        "variant": spec["variant"],
        "display_name": spec["display_name"],
        "role": spec["role"],
        "source_type": spec["source_type"],
        "repo": spec["repo"],
        "model_dir": str(path.parent),
        "train_rows": counts.get("train", ""),
        "valid_rows": counts.get("valid", ""),
        "test_rows": counts.get("test", ""),
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
        "note": spec["note"],
    }
    per_class = []
    test_per_class = test.get("per_class", {})
    for name in CLASS_NAMES:
        item = test_per_class.get(name, {})
        class_row = {"variant": spec["variant"], "display_name": spec["display_name"], "mod_type": name}
        class_row.update(item)
        per_class.append(class_row)
    return row, per_class


def audit_rows() -> List[Dict[str, object]]:
    return [
        {
            "model": "MultiRM",
            "official_repo": "https://github.com/Tsedao/MultiRM",
            "local_source": "external/MultiRM-master",
            "official_status": "Repository available; original code targets old PyTorch and 51/101 nt multi-label input.",
            "same_data_action": "Retrained an architecture-style 3-mer BiLSTM label-attention baseline on the RMBase hg38 11-class split.",
            "caveat": "Report as style_reimplementation, not exact official retraining.",
        },
        {
            "model": "TransRNAm",
            "official_repo": "https://github.com/lennylv/TransRNAm",
            "local_source": "rmbase11_full_work/baseline_repos/TransRNAm-main",
            "official_status": "Repository available; original script depends on author Data_song JSON matrices and hard-coded legacy GPU/path settings.",
            "same_data_action": "Retrained an architecture-style 3-mer Transformer/CNN baseline on the same split.",
            "caveat": "Report as style_reimplementation, not unmodified official script execution.",
        },
        {
            "model": "EvoRMD",
            "official_repo": "https://github.com/Gardeina/EvoRMD",
            "local_source": "rmbase11_full_work/baseline_repos/EvoRMD-main",
            "official_status": "Repository available; official version requires RNA-FM/fm, pandas and original biological context stack; current venv lacks fm/pandas.",
            "same_data_action": "Ran EvoRMD-lite adaptations preserving context-before-attention, replacing RNA-FM with a trainable sequence encoder.",
            "caveat": "safe-context is a secondary adapted baseline; full-context uses RMBase metadata and is treated as leakage-risk diagnostic only.",
        },
    ]


def write_svg(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    plot_rows = [r for r in rows if r["role"] in {"proposed_main", "internal_ablation", "main_baseline", "secondary_baseline"}]
    width, height = 980, 460
    left, top, bar_h, gap = 260, 60, 26, 34
    max_score = 1.0
    colors = {
        "StructDualRM_M4": "#1f4e79",
        "StructDualRM_M0": "#9dc3e6",
        "MultiRM_style_full": "#70ad47",
        "TransRNAm_style_full": "#a9d18e",
        "EvoRMD_lite_safe_context": "#f4b183",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="32" font-family="Arial" font-size="22" font-weight="700" fill="#1f2937">Same-split RMBase hg38 11-class benchmark</text>',
        '<text x="30" y="54" font-family="Arial" font-size="13" fill="#4b5563">Bars show test macro AUPR; labels also report macro F1.</text>',
    ]
    for i, row in enumerate(plot_rows):
        y = top + i * (bar_h + gap)
        score = float(row["test_macro_AUPR_ovr"])
        f1 = float(row["test_macro_F1"])
        length = int((width - left - 120) * score / max_score)
        color = colors.get(str(row["variant"]), "#6b7280")
        label = str(row["display_name"])
        lines.extend(
            [
                f'<text x="30" y="{y + 18}" font-family="Arial" font-size="14" fill="#111827">{label}</text>',
                f'<rect x="{left}" y="{y}" width="{width - left - 120}" height="{bar_h}" fill="#edf2f7"/>',
                f'<rect x="{left}" y="{y}" width="{length}" height="{bar_h}" fill="{color}"/>',
                f'<text x="{left + length + 8}" y="{y + 18}" font-family="Arial" font-size="13" fill="#111827">AUPR {fmt(score)} / F1 {fmt(f1)}</text>',
            ]
        )
    lines.extend(
        [
            '<text x="30" y="410" font-family="Arial" font-size="12" fill="#6b7280">EvoRMD-lite full-context is excluded from the main plot because RMBase metadata may encode experimental-source leakage.</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, rows: Sequence[Dict[str, object]], summary_path: Path, per_class_path: Path, audit_path: Path, svg_path: Path) -> None:
    by_variant = {str(r["variant"]): r for r in rows}
    m4 = by_variant["StructDualRM_M4"]
    m0 = by_variant["StructDualRM_M0"]
    multirm = by_variant["MultiRM_style_full"]
    trans = by_variant["TransRNAm_style_full"]
    evo_safe = by_variant["EvoRMD_lite_safe_context"]
    evo_full = by_variant["EvoRMD_lite_full_context"]
    main_rows = [r for r in rows if r["role"] in {"proposed_main", "internal_ablation", "main_baseline", "secondary_baseline"}]

    lines = [
        "# RMBase hg38 11类外部模型对标实验报告",
        "",
        "## 1. 实验目的",
        "",
        "本次实验不是把不同论文里的分数直接拿来比较，而是把 StructDualRM、MultiRM、TransRNAm 和 EvoRMD 的同架构/适配实现放在同一份 RMBase hg38 11类数据、同一个 `split_random` 下评估。数据规模为 train 756,175、valid 94,522、test 94,522。",
        "",
        "需要注意：这是 hg38 RMBase 11类鲁棒性实验，不是之前的 hg19 m1A/m6A 双标签流程。",
        "",
        "## 2. 官方代码状态",
        "",
        f"代码审计表见：`{audit_path}`。",
        "",
        "- MultiRM：官方仓库已下载，本次保留 3-mer embedding、BiLSTM 和 label-specific attention 思想，做同数据重训。",
        "- TransRNAm：官方仓库已下载，本次保留 3-mer embedding、Transformer pooling 和 CNN head 思想，去掉原脚本硬编码输入。",
        "- EvoRMD：官方仓库已下载，但当前环境缺少 `fm/RNA-FM` 与 `pandas`，因此未声称完整原版复现；本次做 EvoRMD-lite 适配版。",
        "",
        "## 3. 主对比结果",
        "",
        f"完整汇总表：`{summary_path}`。每类指标：`{per_class_path}`。图：`{svg_path}`。",
        "",
        "| 模型 | 角色 | Accuracy | Balanced Acc | Macro F1 | MCC | Macro AUC | Macro AUPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['display_name']} | {row['role']} | {fmt(row['test_accuracy'])} | {fmt(row['test_balanced_accuracy'])} | {fmt(row['test_macro_F1'])} | {fmt(row['test_MCC'])} | {fmt(row['test_macro_AUC_ovr'])} | {fmt(row['test_macro_AUPR_ovr'])} |"
        )
    lines.extend(
        [
            "",
            f"主对比中，StructDualRM full 的 test macro AUPR 为 {fmt(m4['test_macro_AUPR_ovr'])}，高于 MultiRM-style 的 {fmt(multirm['test_macro_AUPR_ovr'])}、TransRNAm-style 的 {fmt(trans['test_macro_AUPR_ovr'])}，也高于 EvoRMD-lite safe-context 的 {fmt(evo_safe['test_macro_AUPR_ovr'])}。",
            f"相对于内部序列消融 M0，M4 的 macro AUPR 从 {fmt(m0['test_macro_AUPR_ovr'])} 提高到 {fmt(m4['test_macro_AUPR_ovr'])}，balanced accuracy 从 {fmt(m0['test_balanced_accuracy'])} 提高到 {fmt(m4['test_balanced_accuracy'])}。",
            "",
            "## 4. EvoRMD-lite 的解释边界",
            "",
            "| 版本 | Accuracy | Balanced Acc | Macro F1 | MCC | Macro AUPR | 风险说明 |",
            "|---|---:|---:|---:|---:|---:|---|",
            f"| safe-context | {fmt(evo_safe['test_accuracy'])} | {fmt(evo_safe['test_balanced_accuracy'])} | {fmt(evo_safe['test_macro_F1'])} | {fmt(evo_safe['test_MCC'])} | {fmt(evo_safe['test_macro_AUPR_ovr'])} | 只用 region/gene_type/gene_known，作为次级适配 baseline |",
            f"| full-context | {fmt(evo_full['test_accuracy'])} | {fmt(evo_full['test_balanced_accuracy'])} | {fmt(evo_full['test_macro_F1'])} | {fmt(evo_full['test_MCC'])} | {fmt(evo_full['test_macro_AUPR_ovr'])} | 使用 cell_list/seq_type_list/support_num，存在明显元数据泄漏风险 |",
            "",
            "full-context 分数接近饱和，不应写成公平 baseline。它更适合作为一个审稿人视角的诊断：RMBase 元数据能携带实验来源和数据库注释偏倚，因此如果把这些字段直接喂给模型，模型可能学到标签来源而不是真正的序列/结构规律。",
            "",
            "## 5. 论文中建议的表述",
            "",
            "建议写：",
            "",
            "> To avoid cross-study benchmark drift, we retrained architecture-style implementations of MultiRM, TransRNAm and an EvoRMD-lite safe-context model on the same RMBase hg38 11-class split. StructDualRM achieved higher macro AUPR and macro F1 than these adapted baselines, supporting the added value of RNAfold-derived structural features and structure-aware fusion. An EvoRMD-lite full-context diagnostic run achieved near-saturated performance, but was not used as a primary fair baseline because RMBase metadata may encode experimental-source information unavailable for de novo candidate prediction.",
            "",
            "不要写：",
            "",
            "- 不要写“完整原版复现 EvoRMD 官方模型”，因为当前没有 RNA-FM/fm 环境。",
            "- 不要把 EvoRMD-lite full-context 的 0.9923 macro AUPR 当作公平比较结果。",
            "- 不要和论文原表分数直接混比；本文对标结论应限定为同数据、同 split。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def run(args: argparse.Namespace) -> Dict[str, object]:
    summaries: List[Dict[str, object]] = []
    per_class: List[Dict[str, object]] = []
    for spec in MODEL_SPECS:
        if not Path(spec["path"]).exists():
            continue
        row, class_rows = summarize(spec)
        summaries.append(row)
        per_class.extend(class_rows)

    summaries.sort(key=lambda r: float(r["test_macro_AUPR_ovr"]), reverse=True)
    for idx, row in enumerate(summaries, start=1):
        row["rank_by_macro_AUPR"] = idx

    out_dir = Path(args.out_dir)
    summary_path = out_dir / "external_model_benchmark_comparison_v2.tsv"
    per_class_path = out_dir / "external_model_benchmark_per_class_v2.tsv"
    audit_path = out_dir / "external_model_code_audit_v2.tsv"
    svg_path = out_dir / "external_model_benchmark_macro_scores.svg"
    report_path = Path(args.report)

    write_tsv(summary_path, summaries, SUMMARY_FIELDS)
    write_tsv(per_class_path, per_class, PER_CLASS_FIELDS)
    write_tsv(audit_path, audit_rows(), AUDIT_FIELDS)
    write_svg(svg_path, summaries)
    write_report(report_path, summaries, summary_path, per_class_path, audit_path, svg_path)

    payload = {
        "summary_tsv": str(summary_path),
        "per_class_tsv": str(per_class_path),
        "audit_tsv": str(audit_path),
        "report_md": str(report_path),
        "figure_svg": str(svg_path),
        "model_count": len(summaries),
    }
    (out_dir / "external_model_benchmark_summary_v2.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize RMBase11 external benchmark.")
    parser.add_argument("--out-dir", default="rmbase11_full_work/metrics")
    parser.add_argument("--report", default="rmbase11_full_work/RMBase11_hg38_外部模型对标报告_v2.md")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
