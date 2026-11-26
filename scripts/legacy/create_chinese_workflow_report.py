from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "StructDualRM完整流程与实验分析报告.md"
REPORT_DOCS = ROOT / "docs" / "StructDualRM完整流程与实验分析报告.md"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: str | float, digits: int = 4) -> str:
    try:
        x = float(value)
    except Exception:
        return "" if value in ("", None) else str(value)
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    if abs(x) >= 0.001:
        return f"{x:.{digits}f}"
    return f"{x:.2e}"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(x).replace("\n", " ") for x in row) + "|")
    return "\n".join(lines)


def image(path: str, caption: str) -> str:
    return f"![{caption}]({path}){{width=6.5in}}\n\n*{caption}*\n"


def safe_rows(rows: list[dict[str, str]], keys: list[str], max_rows: int | None = None) -> list[list[str]]:
    out = []
    for row in rows[: max_rows or len(rows)]:
        out.append([fmt(row.get(k, "")) for k in keys])
    return out


def main() -> None:
    workflow = read_tsv(ROOT / "11_tables" / "workflow_summary.tsv")
    model = read_tsv(ROOT / "09_results" / "metrics" / "model_comparison.tsv")
    benchmark = read_tsv(ROOT / "09_results" / "metrics" / "external_benchmark_comparison.tsv")
    reader = read_tsv(ROOT / "11_tables" / "biology_summary" / "reader_enrichment.tsv")
    external = read_tsv(ROOT / "11_tables" / "biology_summary" / "external_validation_summary.tsv")
    spacing = read_tsv(ROOT / "11_tables" / "biology_summary" / "spacing_permutation_results.tsv")
    structure = read_tsv(ROOT / "11_tables" / "biology_summary" / "rnafold_structure_comparison.tsv")
    sequence = read_tsv(ROOT / "11_tables" / "biology_summary" / "sequence_feature_comparison.tsv")
    go = read_tsv(ROOT / "11_tables" / "biology_summary" / "go_enrichment.tsv")
    pathway = read_tsv(ROOT / "11_tables" / "biology_summary" / "kegg_reactome_enrichment.tsv")
    m4_cfg = read_json(ROOT / "08_models" / "M4_structdualrm_full" / "training_config.json")
    raw_manifest_rows = len(read_tsv(ROOT / "00_raw" / "raw_input_manifest.tsv")) if (ROOT / "00_raw" / "raw_input_manifest.tsv").exists() else 0

    m4 = next(row for row in model if row["variant"] == "M4")
    m0 = next(row for row in model if row["variant"] == "M0")
    multirm = next(row for row in benchmark if row["variant"] == "multirm_style_bilstm_attention")
    official = next(row for row in benchmark if row["variant"] == "multirm_official_101nt_zero_shot")

    workflow_table = md_table(
        ["阶段", "行数", "m1A+", "m6A+", "共阳性", "阴性", "备注"],
        [[r["stage"], r["rows"], r.get("m1A_positive", ""), r.get("m6A_positive", ""), r.get("co_positive", ""), r.get("negative", ""), r.get("notes", "")] for r in workflow],
    )

    model_table = md_table(
        ["模型", "含义", "macro AUPR", "macro MCC", "m1A AUPR", "m6A AUPR", "平均 F1"],
        [
            [
                r["variant"],
                {
                    "M0": "sequence-only Transformer",
                    "M1": "Transformer + global RNAfold structure",
                    "M2": "Transformer + pair-aware graph",
                    "M3": "gated StructDualRM",
                    "M4": "full StructDualRM",
                }.get(r["variant"], r["model_dir"]),
                fmt(r["test_macro_AUPR"]),
                fmt(r["test_macro_MCC"]),
                fmt(r["test_m1A_AUPR"]),
                fmt(r["test_m6A_AUPR"]),
                fmt(r["test_average_F1"]),
            ]
            for r in model
        ],
    )

    benchmark_table = md_table(
        ["模型/基线", "macro AUPR", "macro MCC", "m1A AUPR", "m6A AUPR", "平均 F1"],
        [
            [r["variant"], fmt(r["test_macro_AUPR"]), fmt(r["test_macro_MCC"]), fmt(r["test_m1A_AUPR"]), fmt(r["test_m6A_AUPR"]), fmt(r["test_average_F1"])]
            for r in benchmark
        ],
    )

    reader_table = md_table(
        ["比较", "左组 n", "右组 n", "reader OR", "reader FDR"],
        [[r["comparison"], r["left_total"], r["right_total"], fmt(r["odds_ratio"]), fmt(r["fdr_bh"])] for r in reader],
    )

    external_table = md_table(
        ["比较", "左组 n", "右组 n", "外部注释 OR", "外部注释 FDR"],
        [[r["comparison"], r["left_total"], r["right_total"], fmt(r["odds_ratio"]), fmt(r["fdr_bh"])] for r in external],
    )

    spacing_table = md_table(
        ["设计", "指标", "观察值", "随机均值", "观察/随机", "经验 P", "置换次数"],
        safe_rows(spacing, ["design", "metric", "observed", "null_mean", "observed_over_null", "empirical_p_one_sided", "n_permutations"]),
    )

    structure_keep = [
        r
        for r in structure
        if r["comparison"] == "m1A_CDS_prox25 vs m1A_CDS_distal50"
        and r["feature"] in {"gc_fraction_101nt", "MFE_per_nt", "paired_fraction_101nt", "central_11nt_paired_fraction", "central_21nt_paired_fraction", "loop_span_around_center"}
    ]
    structure_table = md_table(
        ["特征", "proximal 中位数", "distal 中位数", "效应量", "FDR"],
        [[r["feature"], fmt(r["left_median"]), fmt(r["right_median"]), fmt(r["effect_size"]), fmt(r["fdr_bh"])] for r in structure_keep],
    )

    sequence_table = md_table(
        ["比较", "motif 特征", "OR/效应", "FDR"],
        [[r["comparison"], r["feature"], fmt(r.get("odds_ratio", r.get("effect_size", ""))), fmt(r["fdr_bh"])] for r in sequence],
    )

    go_table = md_table(
        ["组别", "数据库", "term", "overlap", "P", "FDR"],
        [[r["group"], r["library"], r["term"][:70], f"{r['overlap_count']}/{r['target_size']}", fmt(r["p_value"]), fmt(r["fdr_bh"])] for r in go[:10]],
    )
    pathway_table = md_table(
        ["组别", "数据库", "term", "overlap", "P", "FDR"],
        [[r["group"], r["library"], r["term"][:70], f"{r['overlap_count']}/{r['target_size']}", fmt(r["p_value"]), fmt(r["fdr_bh"])] for r in pathway[:10]],
    )

    report = f"""# StructDualRM m1A/m6A 项目完整流程与实验分析报告

生成位置：`D:\\pythonprogram\\m1A_m6A`

本报告用于解释这个项目从原始 m1A/m6A 数据到模型训练、消融实验、外部 benchmark、生物学解释性分析、论文材料整理的完整链条。它不是论文正文，而是给后续写论文、答辩和修改 Word 稿使用的“项目说明书”。

## 1. 项目一句话总结

本项目构建了一个 hg19 坐标体系下的 m1A/m6A 双标签预测和解释流程。核心模型是 StructDualRM，它将 101 nt 序列窗口、RNAfold 二级结构、pair-aware graph、转录本区域上下文和 label-specific gate 结合起来，同时预测 m1A 和 m6A。最终完整模型 M4 在测试集上达到 macro AUPR `{fmt(m4['test_macro_AUPR'])}`、macro MCC `{fmt(m4['test_macro_MCC'])}`，并且通过 reader enrichment、外部 hg19 公共注释、结构特征比较和功能富集补充了生物学解释。

## 2. 为什么要这样设计整个流程

### 2.1 为什么使用 hg19

你的原始参考基因组和前期数据处理基于 hg19/h19，因此后续外部验证也应优先寻找 hg19 版本或能明确映射到 hg19 的数据。这样可以避免 hg19/hg38 坐标混用带来的假阳性 overlap 或错误位点映射。

### 2.2 为什么是 m1A 和 m6A 双标签

m1A 和 m6A 都是腺苷相关 RNA 修饰，但化学性质和生物学功能不同。单独预测每一种修饰只能回答“这个位置像不像某一种修饰位点”，而双标签模型可以进一步分析两类修饰在同一转录本、相近区域或局部结构中的关系。这正是你研究的创新点之一。

### 2.3 为什么引入 RNA 二级结构

m1A 会影响碱基配对能力，m6A 也和局部结构、reader binding 相关。因此只用序列 motif 可能不够。StructDualRM 的结构创新点在于把 RNAfold 局部结构、pair-aware graph 和全局结构统计一起纳入模型，让模型不仅看 A/C/G/U 序列，也看局部折叠状态和配对关系。

### 2.4 为什么不用 accuracy 作为核心指标

数据高度不平衡，m6A 阳性远多于 m1A，阴性池也很大。accuracy 容易被多数类主导，所以模型比较主要看 AUPR、MCC、F1、AUC、recall、specificity。其中 AUPR 对阳性稀有任务更有意义，MCC 对不平衡二分类更稳健。

## 3. 文件夹和数据层级

当前项目目录已按完整实验链补齐：

- `00_raw`：原始输入数据层，包含 m1A/m6A 顶层原始下载文件、raw manifest。递归 manifest 记录 `{raw_manifest_rows}` 个原始文件。
- `01_reference`：参考注释/参考文件。
- `02_parsed_sites`：解析后的 raw site 表。
- `03_hg19_sites`：过滤到 hg19 后的位点表。
- `04_transcript_space`：映射到 RefSeq transcript 空间后的位点表。
- `05_datasets`：训练用 joint dataset 和 negative pool。
- `06_structure`：RNAfold 结构特征。
- `07_graphs`：结构图或图特征。
- `08_models`：M0-M4 模型权重、预测、配置和指标。
- `09_results`：模型比较、benchmark、日志、校准等结果。
- `10_figures`：最终论文主图 Figure 1-6 的 SVG 和 PNG。
- `11_tables`：论文表格、workflow summary、生物学分析表。
- `12_benchmarks`：外部/同类模型 benchmark。
- `external`：外部公共数据、MultiRM 代码、hg19 external validation 数据。
- `submission`：Nature Communications/Springer Nature 模板稿、Word 稿、source data、压缩包。

## 4. 原始数据预处理流程

### 4.1 原始数据来源

原始 m1A 数据来自 `D:\\dataset\\m1A_gg`，原始 m6A 数据来自 `D:\\dataset\\m6A_gg`。现在项目的 `00_raw` 已保存顶层原始下载文件，并用 `raw_input_manifest.tsv` 记录完整递归来源。

这里没有把全部 22 GB 左右的递归解压文件都复制进项目，因为很多是 tar/gz 解压后的重复内容。项目保留顶层原始文件和完整 manifest，既能追踪来源，也避免不必要膨胀。

### 4.2 raw parsing

第一步是把不同来源和格式的 m1A/m6A 文件解析成统一 schema。这个阶段解决的问题是：不同 GEO 数据文件可能是 BED、txt、xlsx、peak table 等格式，字段名、坐标体系和修饰类型标注都不一致，不能直接合并。

产出文件：

- `02_parsed_sites/m1A_raw_parsed.tsv`
- `02_parsed_sites/m6A_raw_parsed.tsv`
- `11_tables/skipped_inputs.tsv`

`skipped_inputs.tsv` 记录被跳过的输入，例如非 candidate m1A 输入、hg18/mm9/mm10 数据、需要 liftover 的数据等。这一步很重要，因为它说明我们不是随意丢弃数据，而是按坐标体系和格式规则过滤。

### 4.3 hg19 过滤

解析后只保留明确属于 hg19 的记录。原因是你的主数据和参考基因组使用 hg19；如果混用 hg18、mm9、mm10 或 hg38，会导致坐标不可比。

产出文件：

- `03_hg19_sites/m1A_hg19.tsv`
- `03_hg19_sites/m6A_hg19.tsv`
- `11_tables/hg19_filter_qc.tsv`

### 4.4 RefSeq transcript 映射

基因组坐标本身不能直接表达一个修饰位点处于 5UTR、CDS 还是 3UTR，也不方便计算同一转录本内 m1A-m6A 的距离。因此把 hg19 位点映射到 RefSeq transcript coordinate。

产出文件：

- `04_transcript_space/m1A_refseq_mapped.tsv`
- `04_transcript_space/m6A_refseq_mapped.tsv`
- `04_transcript_space/m1A_m6A_joint_positive_sites.tsv`

## 5. 数据构建和 QC 结果

下面是整个数据构建流程的核心统计：

{workflow_table}

关键解释：

- m1A raw parsed 为 `32217`，经过 hg19 过滤后保留 `30636`，映射到 RefSeq 后为 `29059`。
- m6A raw parsed 为 `189658`，经过 hg19 过滤后保留 `189508`，映射到 RefSeq 后为 `170604`。
- m1A center-A 过滤后只剩 `4675`，说明 m1A 数据更稀有也更严格。
- joint positive sites 为 `62079`，其中 m1A 阳性 `4675`，m6A 阳性 `57418`，共阳性 `14`。
- negative pool 为 `186237`。
- joint dataset 为 `248316` 行。
- RNAfold 结构计算目前处理了 `50000` 个 101 nt 窗口。

这个 QC 说明你的任务是一个明显不平衡的双标签问题，尤其 m1A 是难点。

{image('10_figures/main_figures_png/Figure1_workflow_architecture.png', 'Figure 1：整体 workflow 和 StructDualRM 研究设计')}

## 6. 序列窗口、负样本和 RNAfold 结构

### 6.1 为什么用 101 nt 窗口

101 nt 窗口以候选 A 为中心，左右各取 50 nt。这种设置足够覆盖局部 motif、邻近结构和短程上下文，同时不会让窗口过长导致噪声过多。MultiRM 等 RNA modification predictor 也常用类似短窗口策略。

### 6.2 为什么做 center-A 过滤

m1A 和 m6A 都是腺苷修饰，因此预测窗口中心必须是 A。center-A 过滤保证模型学习的是“中心 A 是否被修饰”，而不是在任意窗口里寻找某个远处修饰。

### 6.3 为什么构建 negative pool

模型不能只学阳性位点，还要知道未修饰或未标注 A 的背景分布。negative pool 来自 RefSeq transcript 中匹配的 candidate adenines，并避开已知阳性位点。这一步用于构建监督学习中的负样本。

### 6.4 RNAfold 结构特征

RNAfold 结构包括：

- center 是否 paired；
- central 11 nt paired fraction；
- central 21 nt paired fraction；
- 101 nt paired fraction；
- MFE per nt；
- loop span around center；
- base-pair partner list；
- paired flags。

这些特征分成两类：一类进入 global structure MLP，另一类用于 pair-aware graph branch。

## 7. StructDualRM 模型搭建

### 7.1 M4 完整模型结构

M4 是完整 StructDualRM，输入包括：

- `seq`：101 nt 序列编码；
- `paired_flags`：每个位置是否配对；
- `pair_partner`：每个位置的 RNAfold 配对 partner；
- `struct`：6 个 RNAfold scalar features；
- `region`：5UTR/CDS/3UTR/UTR 区域；
- `context`：relative transcript position 和 log10 transcript length。

模型核心模块：

- base embedding + position embedding + center-distance embedding；
- Transformer encoder；
- center-aware attention pooling；
- pair-aware graph branch；
- global structure MLP；
- transcript context branch；
- m1A/m6A label-specific gates；
- m1A 和 m6A 两个 prediction heads。

{image('10_figures/main_figures_png/Figure6_structdualrm_architecture.png', 'Figure 6：StructDualRM M4 模型架构图')}

### 7.2 训练设置

M4 训练配置：

{md_table(['参数', '值'], [[k, v] for k, v in m4_cfg.items()])}

核心设置包括：

- epoch = `{m4_cfg['epochs']}`；
- batch size = `{m4_cfg['batch_size']}`；
- d_model = `{m4_cfg['d_model']}`；
- heads = `{m4_cfg['heads']}`；
- dropout = `{m4_cfg['dropout']}`；
- lr = `{m4_cfg['lr']}`；
- split = `{m4_cfg['split']}`；
- seed = `{m4_cfg['seed']}`。

## 8. 消融实验

### 8.1 消融实验是否已经完成

已经完成。消融实验对应 M0-M4：

- M0：只用 sequence Transformer；
- M1：加入 global RNAfold structure；
- M2：加入 pair-aware graph；
- M3：加入 gated StructDualRM；
- M4：完整模型，包含 sequence、graph、structure、context 和 label-specific gate。

### 8.2 消融实验结果

{model_table}

结论：

- M4 的 macro AUPR 为 `{fmt(m4['test_macro_AUPR'])}`，macro MCC 为 `{fmt(m4['test_macro_MCC'])}`，是五个模型里最强。
- M0 的 macro AUPR 为 `{fmt(m0['test_macro_AUPR'])}`，macro MCC 为 `{fmt(m0['test_macro_MCC'])}`。
- M4 相比 M0 的 macro AUPR 提升约 `{fmt(float(m4['test_macro_AUPR']) - float(m0['test_macro_AUPR']))}`。
- m1A AUPR 从 M0 的 `{fmt(m0['test_m1A_AUPR'])}` 提升到 M4 的 `{fmt(m4['test_m1A_AUPR'])}`，说明结构和上下文模块对稀有 m1A 标签尤其有帮助。

{image('10_figures/main_figures_png/Figure2_model_performance.png', 'Figure 2：模型消融和外部 benchmark 性能比较')}

### 8.3 为什么消融实验重要

如果只报告 M4 最好，审稿人会问：到底是 Transformer 起作用，还是结构、graph、gate、context 起作用？消融实验就是逐步移除或加入模块，证明每个模块的贡献。你现在的消融链条已经能支撑“完整结构-aware 模型带来提升”这个论点。

## 9. 同数据集 benchmark 和 MultiRM 对比

### 9.1 为什么要做同数据 benchmark

直接拿论文里的指标和你的指标比较是不公平的，因为数据集、窗口长度、阴性样本、坐标体系都可能不同。因此我们把 k-mer baseline、MultiRM-style baseline、official MultiRM zero-shot 和 StructDualRM 放在同一个 test split 上比较。

### 9.2 benchmark 结果

{benchmark_table}

核心结论：

- MultiRM-style BiLSTM attention 的 macro AUPR 为 `{fmt(multirm['test_macro_AUPR'])}`，macro MCC 为 `{fmt(multirm['test_macro_MCC'])}`。
- official MultiRM zero-shot 的 macro AUPR 为 `{fmt(official['test_macro_AUPR'])}`，macro MCC 为 `{fmt(official['test_macro_MCC'])}`。
- M4 StructDualRM 的 macro AUPR 为 `{fmt(m4['test_macro_AUPR'])}`，macro MCC 为 `{fmt(m4['test_macro_MCC'])}`。

这说明你的模型不是只比自己前面的版本强，也在同数据集上超过了 MultiRM-style 和 classical baseline。

## 10. m1A-m6A 邻近关系和 spacing permutation

### 10.1 为什么分析 m1A 和 m6A 的距离

模型创新不只在预测，还在“两个修饰位点之间是否存在空间关系”。如果 m1A 和 m6A 在同一转录本中经常以特定距离出现，就说明它们可能存在局部结构或调控关联。

### 10.2 spacing permutation 结果

{spacing_table}

解释：

- `fraction_le_25nt`、`fraction_le_50nt` 等指标衡量 m1A-m6A 是否靠得近。
- permutation 的作用是构造随机背景，看真实距离分布是否偏离随机。
- 目前结果显示真实 proximal 比例相对于随机背景并不是简单增强，因此论文中更稳妥的表述是：局部 proximal 组可用于后续 reader/external validation，而不能直接宣称全局 m1A-m6A 必然更靠近。

## 11. 序列和结构解释性分析

### 11.1 为什么做 sequence/structure comparison

如果 M4 的提升只是因为它学到了 DRACH 或 GGAC-like motif，那结构创新就不够强。为了证明结构有意义，需要比较 proximal 和 distal 组在 RNAfold 特征上的差异。

### 11.2 CDS proximal vs distal 的结构比较

{structure_table}

主要解释：

- CDS proximal 组在 MFE_per_nt、paired_fraction、central paired fraction 上与 distal 组存在明显差异。
- 这支持“局部结构环境参与 m1A-m6A proximal arrangement”的解释。
- 但这仍是计算证据，不能写成因果机制。

### 11.3 motif 对比

{sequence_table}

解释：

- motif 分析用于确认模型不是只依赖某个简单 motif。
- 当前 motif 结果整体不如结构特征稳定，因此论文中应突出“结构和上下文补充了 motif 无法完全解释的信号”。

{image('10_figures/main_figures_png/Figure3_sequence_structure_context.png', 'Figure 3：proximal m1A-m6A 的序列与 RNA 结构解释性分析')}

## 12. Reader enrichment 分析

### 12.1 为什么做 reader enrichment

m6A 的重要机制之一是 reader protein binding。YTHDF2 是典型 m6A reader。如果 proximal m1A-m6A 组更容易和 YTHDF2 overlap，说明这些位点可能处在更活跃的 post-transcriptional regulation 环境中。

### 12.2 reader enrichment 结果

{reader_table}

结论：

- m1A_3UTR_prox25 vs distal50：OR = `{fmt(reader[0]['odds_ratio'])}`，FDR = `{fmt(reader[0]['fdr_bh'])}`。
- m1A_CDS_prox25 vs distal50：OR = `{fmt(reader[1]['odds_ratio'])}`，FDR = `{fmt(reader[1]['fdr_bh'])}`。
- 与 m6A-only 同转录本背景相比，3UTR proximal 和 CDS proximal 的 enrichment 更强。

安全表述：proximal m1A-m6A arrangements 与 YTHDF2 reader-binding evidence 相关。不要写成“m1A 招募 YTHDF2”或“m6A 一定导致 m1A”。

## 13. 外部独立验证（hg19）

### 13.1 为什么做外部验证

只在自己数据集里发现 enrichment，可能是数据来源或处理流程造成的偏差。因此我们找了 hg19-compatible public m1A/m6A annotations 和 YTHDF2 reader 数据，做独立 overlap/enrichment。

### 13.2 外部验证结果

{external_table}

结论：

- 3UTR proximal vs distal：外部注释 OR = `{fmt(external[0]['odds_ratio'])}`，FDR = `{fmt(external[0]['fdr_bh'])}`。
- CDS proximal vs distal：外部注释 OR = `{fmt(external[1]['odds_ratio'])}`，FDR = `{fmt(external[1]['fdr_bh'])}`。
- 与 m6A-only 同转录本相比，外部支持也显著。

这个结果是你生物学补强里最重要的一环，因为这个外部独立验证说明 proximal 关系不是只在内部训练数据里成立。

{image('10_figures/main_figures_png/Figure4_reader_external_validation.png', 'Figure 4：reader enrichment 和 hg19 外部验证')}

## 14. GO/KEGG/Reactome 功能富集

### 14.1 为什么做功能富集

GO/KEGG/Reactome 不是证明位点机制的强证据，但可以说明这些 proximal groups 关联到什么类型的基因或通路，为论文 discussion 提供生物学背景。

### 14.2 GO top terms

{go_table}

### 14.3 KEGG/Reactome top terms

{pathway_table}

解释：

- 有一些 nominally enriched terms，如 cell fate、intracellular pH、plasma membrane organization、actin/cytoskeleton、metabolism 等。
- 但很多 FDR 不够强，因此论文里应该把这部分写成 exploratory biological context，而不是核心机制结论。

{image('10_figures/main_figures_png/Figure5_functional_enrichment.png', 'Figure 5：GO/KEGG/Reactome 探索性功能富集')}

## 15. 外部 MultiRM 模型实验

这个小实验包括两类：

1. official MultiRM zero-shot：直接用官方思路/模型在你的 101 nt 数据上评估。
2. MultiRM-style BiLSTM attention：按 MultiRM 的 attention-based multi-label 思路，在你的数据集上训练一个同风格 baseline。

结果显示 official zero-shot 下降明显，说明原始 MultiRM 的训练分布和你的 hg19 m1A/m6A 数据不一致。MultiRM-style retraining 明显更强，但仍低于 M4。这一组实验可以支撑论文中“不是只和弱 baseline 比”的说法。

## 16. 模型输出、阈值和预测文件

每个模型目录包含：

- `model.pt`：模型权重、配置、阈值、feature stats；
- `metrics.json`：训练历史、验证集/测试集指标；
- `training_config.json`：训练参数；
- `thresholds_validation.json`：验证集选择的阈值；
- `valid_predictions.tsv`：验证集预测；
- `test_predictions.tsv`：测试集预测。

这些文件位于 `08_models/M0_transformer_only` 到 `08_models/M4_structdualrm_full`。

## 17. 论文材料整理

目前已经生成：

- Nature/Springer LaTeX 模板稿；
- Word 草稿；
- Figure 1-6；
- source data；
- 模型架构图；
- 完整压缩包；
- `10_figures` 图件目录；
- `00_raw` 原始数据目录。

主要文件：

- `submission/nature_communications_structdualrm/sn_template_draft/StructDualRM_NatureCommunications_SN_Template_Draft.tex`
- `submission/nature_communications_structdualrm/word_draft/StructDualRM_NatureCommunications_SN_Template_Draft_Word_v2_architecture.docx`
- `submission/nature_communications_structdualrm_full_package.zip`

## 18. 现在已经完成了什么

你现在已经完成了一整套计算研究链：

1. 原始数据整理；
2. hg19 坐标过滤；
3. RefSeq transcript 映射；
4. 101 nt 窗口构建；
5. negative pool 构建；
6. RNAfold 结构特征；
7. M0-M4 消融实验；
8. 50 epoch 模型训练；
9. 同数据集 MultiRM benchmark；
10. reader enrichment；
11. sequence/structure 解释性分析；
12. spacing permutation；
13. GO/KEGG/Reactome；
14. hg19 外部公共注释验证；
15. 论文图、表、source data 和 Word/LaTeX 稿。

### 18.1 中间小实验总览

下面这些“小实验”不是零散附属内容，而是在论文逻辑中分别承担数据可信度、模型有效性、机制解释和外部可重复性的作用。

|小实验|想回答的问题|主要输入/输出|现在的结论|
|---|---|---|---|
|RNAfold 101 nt 抽样结构预测|局部二级结构是否可以作为模型输入和解释变量|`06_structure/rnafold_101nt.tsv`|完成 50,000 条 101 nt 窗口结构预测，为 M1/M3/M4 和结构解释提供输入|
|M0-M4 消融实验|结构、区域上下文、pair-aware graph 和 label gate 是否真的贡献性能|`09_results/metrics/model_comparison.tsv`|M4 的 macro AUPR `{fmt(m4['test_macro_AUPR'])}`、macro MCC `{fmt(m4['test_macro_MCC'])}` 最优，说明完整结构模块组合有效|
|同数据集 MultiRM-style benchmark|你的模型是否只是在和弱 baseline 比|`09_results/metrics/external_benchmark_comparison.tsv`|MultiRM-style retraining 仍低于 M4，支持 StructDualRM 在同数据条件下的优势|
|MultiRM official zero-shot|公开模型迁移到你的 hg19 m1A/m6A 数据时表现如何|`12_benchmarks`、`external/multirm` 相关结果|zero-shot 明显下降，说明训练分布和标签体系不同，不能直接替代同数据集 benchmark|
|spacing permutation|m1A-m6A 是否在转录本空间中表现出非随机邻近关系|`11_tables/biology_summary/spacing_permutation_results.tsv`|观察到的 proximal 关系需要和随机背景比较，作为“双位点关系”分析的统计支撑|
|sequence feature comparison|proximal/distal 组是否存在序列组成差异|`11_tables/biology_summary/sequence_feature_comparison.tsv`|用于说明模型和生物学分组不是完全脱离序列背景的黑箱结果|
|RNAfold structure comparison|proximal/distal 组是否存在结构差异|`11_tables/biology_summary/rnafold_structure_comparison.tsv`|为“结构作为创新点”提供解释性证据，但仍属于计算层面的机制提示|
|reader enrichment|proximal m1A-m6A 是否和 reader binding 证据相关|`11_tables/biology_summary/reader_enrichment.tsv`|3UTR 和 CDS proximal 组出现显著 enrichment，可作为生物学解释核心结果之一|
|外部独立验证（hg19）|内部 enrichment 是否能被外部公共注释支持|`11_tables/biology_summary/external_validation_summary.tsv`|hg19-compatible 外部注释支持 proximal 关系，增强可重复性|
|GO/KEGG/Reactome 富集|相关转录本集合是否落在可解释的功能通路上|`11_tables/biology_summary/go_enrichment.tsv`、`kegg_reactome_enrichment.tsv`|适合放在 Discussion 或补充结果中，用来解释潜在功能背景|
|论文图和 source data 整理|结果是否能被清楚复现和展示|`10_figures`、`submission/nature_communications_structdualrm/source_data`|已经整理 Figure 1-6、主图 PNG/SVG 和 source data，方便后续写论文|

## 19. 目前仍然需要注意的地方

如果目标是毕业论文或组会汇报，现在的计算链已经完整。  
如果目标是 Nature Communications，还需要补强：

1. 多随机种子重复训练，报告 mean ± SD；
2. transcript-level split 或 chromosome-level split，降低信息泄漏质疑；
3. 更严格的完全独立测试集预测验证；
4. bootstrap 或 paired permutation test 证明 M4 vs M0/MultiRM 的统计显著性；
5. gate/attention case study，展示具体位点解释；
6. 湿实验验证，例如若干候选位点的 RT-qPCR、MeRIP/m1A-IP、reader knockdown 或突变验证。

## 20. 最终结论

本项目的核心结论可以概括为：

StructDualRM 在 hg19 m1A/m6A 双标签预测任务中，通过序列、RNA 二级结构、pair-aware graph、转录本上下文和 label-specific gate 的整合，提高了模型预测性能，尤其改善了稀有 m1A 标签。消融实验、同数据集 MultiRM benchmark、reader enrichment 和 hg19 外部注释验证共同支持该模型和 proximal m1A-m6A 生物学分析框架。但目前的生物学结果仍主要是计算证据，后续如果冲击更高水平期刊，应重点补充多 seed 稳健性、严格独立验证和实验验证。
"""

    REPORT_ROOT.write_text(report, encoding="utf-8", newline="\n")
    REPORT_DOCS.write_text(report.replace("10_figures/", "../10_figures/"), encoding="utf-8", newline="\n")
    print({"markdown_root": str(REPORT_ROOT), "markdown_docs": str(REPORT_DOCS)})


if __name__ == "__main__":
    main()
