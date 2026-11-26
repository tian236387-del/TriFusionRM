from __future__ import annotations

import csv
import html
import json
import math
import os
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "submission" / "nature_communications_structdualrm"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def fmt(x: str | float, digits: int = 3) -> str:
    if x is None or x == "":
        return "NA"
    value = float(x)
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def sci(x: str | float) -> str:
    value = float(x)
    if value == 0:
        return "0"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def xml_text(value: object) -> str:
    return html.escape(str(value), quote=False)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933} .small{font-size:18px} .tiny{font-size:14px} .label{font-size:20px;font-weight:700} .title{font-size:26px;font-weight:700} .axis{stroke:#44515e;stroke-width:2} .grid{stroke:#d6dde5;stroke-width:1} .muted{fill:#667085}</style>',
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#44515e"/></marker>',
        "</defs>",
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
    ]


def save_svg(path: Path, body: list[str], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(path, "\n".join(svg_header(width, height) + body + ["</svg>\n"]))


def add_text(lines: list[str], x: float, y: float, text: str, cls: str = "small", anchor: str = "start") -> None:
    lines.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{xml_text(text)}</text>')


def add_wrapped(lines: list[str], x: float, y: float, text: str, width_chars: int, cls: str = "small") -> None:
    words = text.split()
    current: list[str] = []
    row = 0
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) > width_chars and current:
            add_text(lines, x, y + row * 22, " ".join(current), cls)
            row += 1
            current = [word]
        else:
            current.append(word)
    if current:
        add_text(lines, x, y + row * 22, " ".join(current), cls)


def make_figure1() -> None:
    width, height = 1800, 1120
    s = []
    add_text(s, 60, 70, "Fig. 1 | StructDualRM study design and model concept", "title")
    add_text(s, 70, 135, "A", "label")
    add_text(s, 930, 135, "B", "label")
    stages = [
        ("hg19 m1A/m6A maps", "30,636 hg19 m1A sites; 189,508 hg19 m6A sites"),
        ("RefSeq transcript space", "m1A=29,059; m6A=170,604 mapped sites"),
        ("101 nt sequence windows", "center-A filtering, positives and matched negatives"),
        ("RNAfold structure", "global and local secondary-structure descriptors"),
        ("Models and biology", "ablation, MultiRM benchmark, reader/external validation"),
    ]
    x0, y0, box_w, box_h, gap = 90, 180, 260, 140, 55
    colors = ["#e8f1fb", "#edf7ee", "#fff3da", "#f2edfb", "#e9f7f7"]
    for i, (title, subtitle) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        s.append(f'<rect x="{x}" y="{y0}" width="{box_w}" height="{box_h}" rx="10" fill="{colors[i]}" stroke="#8a9aaa" stroke-width="2"/>')
        add_wrapped(s, x + 22, y0 + 45, title, 22, "small")
        add_wrapped(s, x + 22, y0 + 88, subtitle, 29, "tiny")
        if i < len(stages) - 1:
            s.append(f'<line x1="{x + box_w + 8}" y1="{y0 + box_h / 2}" x2="{x + box_w + gap - 8}" y2="{y0 + box_h / 2}" class="axis" marker-end="url(#arrow)"/>')
    s.append('<rect x="90" y="410" width="760" height="500" rx="12" fill="#ffffff" stroke="#c7d0db" stroke-width="2"/>')
    add_text(s, 125, 465, "Training set composition", "label")
    rows = [
        ("joint positive sites", "62,079", "#2f7ebc"),
        ("m1A-positive", "4,675", "#b84a62"),
        ("m6A-positive", "57,418", "#4e9b55"),
        ("co-positive", "14", "#7b61b5"),
        ("negative pool", "186,237", "#777777"),
    ]
    for i, (lab, val, col) in enumerate(rows):
        y = 525 + i * 65
        s.append(f'<circle cx="145" cy="{y - 6}" r="11" fill="{col}"/>')
        add_text(s, 170, y, lab, "small")
        add_text(s, 730, y, val, "small", "end")
    s.append('<rect x="930" y="180" width="680" height="730" rx="12" fill="#ffffff" stroke="#c7d0db" stroke-width="2"/>')
    add_text(s, 970, 235, "StructDualRM architecture", "label")
    blocks = [
        (1010, 305, 250, 95, "#e8f1fb", "sequence transformer", "nucleotide context"),
        (1330, 305, 250, 95, "#f2edfb", "structure branch", "pairing and MFE context"),
        (1010, 485, 250, 95, "#fff3da", "graph/dual-site context", "local arrangement"),
        (1330, 485, 250, 95, "#edf7ee", "attention and gating", "feature arbitration"),
        (1160, 685, 290, 95, "#e9f7f7", "multi-label output", "m1A and m6A probabilities"),
    ]
    for x, y, w, h, color, title, sub in blocks:
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{color}" stroke="#8796a6" stroke-width="2"/>')
        add_text(s, x + 18, y + 38, title, "small")
        add_text(s, x + 18, y + 68, sub, "tiny")
    for x1, y1, x2, y2 in [(1260, 352, 1330, 352), (1135, 400, 1135, 485), (1455, 400, 1455, 485), (1260, 535, 1330, 535), (1135, 580, 1260, 685), (1455, 580, 1350, 685)]:
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="axis" marker-end="url(#arrow)"/>')
    add_wrapped(s, 970, 815, "The full model integrates primary sequence, RNA secondary structure and dual-site contextual signals; ablation models remove these components stepwise.", 72, "small")
    save_svg(PKG / "figures" / "Figure1_workflow_architecture.svg", s, width, height)


def make_figure2(model_rows: list[dict[str, str]], benchmark_rows: list[dict[str, str]]) -> None:
    width, height = 1800, 1120
    s = []
    add_text(s, 60, 70, "Fig. 2 | Predictive performance and external benchmark", "title")
    panels = [
        ("A", "Internal ablation: macro AUPR", model_rows, "variant", "test_macro_AUPR", 90, 180, 740, 410),
        ("B", "Internal ablation: macro MCC", model_rows, "variant", "test_macro_MCC", 930, 180, 740, 410),
        ("C", "Benchmark against MultiRM-style baselines", benchmark_rows, "variant", "test_macro_AUPR", 90, 680, 740, 360),
        ("D", "Benchmark macro MCC", benchmark_rows, "variant", "test_macro_MCC", 930, 680, 740, 360),
    ]
    palette = ["#789cc3", "#8dbb82", "#d4a04f", "#b77aa9", "#4d7599", "#7a7a7a", "#c5665d"]
    for letter, title, rows, name_key, val_key, x, y, w, h in panels:
        add_text(s, x - 35, y - 25, letter, "label")
        add_text(s, x, y - 25, title, "small")
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="#d7dee8"/>')
        vals = [float(r[val_key]) for r in rows]
        y_min = 0 if val_key.endswith("MCC") else max(0, min(vals) - 0.08)
        y_max = 1.0
        plot_x, plot_y = x + 85, y + 35
        plot_w, plot_h = w - 135, h - 110
        for tick in [y_min, (y_min + y_max) / 2, y_max]:
            ty = plot_y + plot_h - (tick - y_min) / (y_max - y_min) * plot_h
            s.append(f'<line x1="{plot_x}" y1="{ty}" x2="{plot_x + plot_w}" y2="{ty}" class="grid"/>')
            add_text(s, plot_x - 14, ty + 6, fmt(tick, 2), "tiny", "end")
        bar_gap = 12
        bar_w = max(32, (plot_w - bar_gap * (len(rows) - 1)) / len(rows))
        for i, r in enumerate(rows):
            val = float(r[val_key])
            bh = (val - y_min) / (y_max - y_min) * plot_h
            bx = plot_x + i * (bar_w + bar_gap)
            by = plot_y + plot_h - bh
            color = "#34495e" if r["variant"] == "M4" else palette[i % len(palette)]
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}"/>')
            add_text(s, bx + bar_w / 2, by - 8, fmt(val, 3), "tiny", "middle")
            label = r[name_key].replace("multirm_", "MultiRM_").replace("_", " ")
            short = {"M0": "M0", "M1": "M1", "M2": "M2", "M3": "M3", "M4": "M4"}.get(label, label[:18])
            add_text(s, bx + bar_w / 2, plot_y + plot_h + 30, short, "tiny", "middle")
        s.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" class="axis"/>')
        s.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" class="axis"/>')
    save_svg(PKG / "figures" / "Figure2_model_performance.svg", s, width, height)


def make_figure3(struct_rows: list[dict[str, str]], seq_rows: list[dict[str, str]], spacing_rows: list[dict[str, str]]) -> None:
    width, height = 1800, 1120
    s = []
    add_text(s, 60, 70, "Fig. 3 | Sequence and structural context of proximal dual modifications", "title")
    add_text(s, 80, 140, "A", "label")
    add_text(s, 940, 140, "B", "label")
    add_text(s, 80, 650, "C", "label")
    add_text(s, 940, 650, "D", "label")
    s.append('<rect x="105" y="170" width="760" height="420" fill="#ffffff" stroke="#d7dee8"/>')
    add_text(s, 135, 215, "CDS-proximal sites show less stable local folds than distal m1A sites", "small")
    features = ["gc_fraction_101nt", "MFE_per_nt", "paired_fraction_101nt", "central_21nt_paired_fraction"]
    rows = [r for r in struct_rows if r["comparison"] == "m1A_CDS_prox25 vs m1A_CDS_distal50" and r["feature"] in features]
    for i, r in enumerate(rows):
        y = 280 + i * 74
        add_text(s, 140, y, r["feature"].replace("_", " "), "tiny")
        x0 = 470
        left = float(r["left_median"])
        right = float(r["right_median"])
        scale_min = min(left, right, -0.35 if "MFE" in r["feature"] else 0)
        scale_max = max(left, right, 0.65)
        def px(v: float) -> float:
            return x0 + (v - scale_min) / (scale_max - scale_min) * 260
        s.append(f'<line x1="{x0}" y1="{y-8}" x2="{x0+260}" y2="{y-8}" stroke="#c8d1dc" stroke-width="5"/>')
        s.append(f'<circle cx="{px(right):.1f}" cy="{y-8}" r="12" fill="#9da8b5"/>')
        s.append(f'<circle cx="{px(left):.1f}" cy="{y-8}" r="12" fill="#b84a62"/>')
        add_text(s, 780, y, f"FDR={sci(r['fdr_bh'])}", "tiny")
    s.append('<circle cx="535" cy="545" r="10" fill="#b84a62"/><text x="555" y="551" class="tiny">prox25</text>')
    s.append('<circle cx="645" cy="545" r="10" fill="#9da8b5"/><text x="665" y="551" class="tiny">distal50</text>')
    s.append('<rect x="965" y="170" width="650" height="420" fill="#ffffff" stroke="#d7dee8"/>')
    add_text(s, 995, 215, "Sequence motifs are not sufficient to explain the proximal context", "small")
    motif_rows = [r for r in seq_rows if "m1A_CDS" in r["comparison"]]
    for i, r in enumerate(motif_rows):
        y = 290 + i * 95
        add_text(s, 1000, y, r["feature"].replace("_", " "), "small")
        add_text(s, 1280, y, f"OR={fmt(r['odds_ratio'], 2)}", "small")
        add_text(s, 1460, y, f"FDR={sci(r['fdr_bh'])}", "small")
    s.append('<rect x="105" y="680" width="760" height="330" fill="#ffffff" stroke="#d7dee8"/>')
    add_text(s, 135, 725, "Spacing is rarer than transcript-region-matched permutations", "small")
    rows = [r for r in spacing_rows if r["design"] == "region_stratified" and r["metric"].startswith("fraction")]
    for i, r in enumerate(rows):
        y = 785 + i * 48
        obs = float(r["observed"])
        null = float(r["null_mean"])
        add_text(s, 140, y, r["metric"].replace("fraction_", "").replace("_", " "), "tiny")
        s.append(f'<rect x="410" y="{y-18}" width="{obs*700:.1f}" height="18" fill="#b84a62"/>')
        s.append(f'<rect x="410" y="{y+4}" width="{null*700:.1f}" height="18" fill="#9da8b5"/>')
    s.append('<rect x="965" y="680" width="650" height="330" fill="#ffffff" stroke="#d7dee8"/>')
    add_text(s, 995, 725, "Interpretation", "small")
    add_wrapped(s, 995, 785, "Proximal m1A-m6A arrangements are not simply random co-occurrences. The strongest structural signal is observed in CDS-proximal sites, where local pairing and MFE shift relative to distal m1A controls.", 65, "small")
    add_wrapped(s, 995, 890, "The result supports structure-aware modeling as an innovation point, while keeping causal claims for future perturbation assays.", 65, "small")
    save_svg(PKG / "figures" / "Figure3_sequence_structure_context.svg", s, width, height)


def make_figure4(reader_rows: list[dict[str, str]], ext_rows: list[dict[str, str]]) -> None:
    width, height = 1800, 1120
    s = []
    add_text(s, 60, 70, "Fig. 4 | Reader enrichment and hg19 external validation", "title")
    panels = [("A", "YTHDF2 reader enrichment", reader_rows, 110), ("B", "Independent hg19 m1A/m6A annotation support", ext_rows, 930)]
    for letter, title, rows, x in panels:
        add_text(s, x - 40, 145, letter, "label")
        add_text(s, x, 145, title, "label")
        s.append(f'<rect x="{x}" y="180" width="690" height="760" fill="#ffffff" stroke="#d7dee8"/>')
        plot_x, plot_y, plot_w = x + 290, 265, 360
        s.append(f'<line x1="{plot_x}" y1="230" x2="{plot_x}" y2="840" stroke="#a8b2bd" stroke-width="2" stroke-dasharray="6,6"/>')
        for tick in [1, 1.5, 2, 2.5, 3]:
            tx = plot_x + (math.log(tick) / math.log(3.5)) * plot_w
            s.append(f'<line x1="{tx:.1f}" y1="840" x2="{tx:.1f}" y2="850" class="axis"/>')
            add_text(s, tx, 878, str(tick), "tiny", "middle")
        add_text(s, plot_x + plot_w / 2, 912, "odds ratio", "tiny", "middle")
        for i, r in enumerate(rows):
            y = plot_y + i * 140
            label = r["comparison"].replace("m1A_", "").replace("_same_transcript", "").replace("_", " ")
            add_wrapped(s, x + 30, y - 24, label, 28, "tiny")
            odds = float(r["odds_ratio"])
            cx = plot_x + (math.log(odds) / math.log(3.5)) * plot_w
            color = "#b84a62" if "3UTR" in r["comparison"] else "#4d7599"
            s.append(f'<circle cx="{cx:.1f}" cy="{y}" r="15" fill="{color}"/>')
            add_text(s, cx + 24, y + 6, f"OR {fmt(odds, 2)}", "tiny")
            add_text(s, x + 555, y + 6, f"FDR {sci(r['fdr_bh'])}", "tiny")
    save_svg(PKG / "figures" / "Figure4_reader_external_validation.svg", s, width, height)


def make_figure5(go_rows: list[dict[str, str]], pathway_rows: list[dict[str, str]]) -> None:
    width, height = 1800, 1120
    s = []
    add_text(s, 60, 70, "Fig. 5 | Exploratory functional enrichment", "title")
    panels = [("A", "GO Biological Process", go_rows[:8], 95), ("B", "KEGG/Reactome", pathway_rows[:8], 945)]
    for letter, title, rows, x in panels:
        add_text(s, x - 35, 145, letter, "label")
        add_text(s, x, 145, title, "label")
        s.append(f'<rect x="{x}" y="180" width="760" height="760" fill="#ffffff" stroke="#d7dee8"/>')
        maxv = max([-math.log10(max(float(r["p_value"]), 1e-300)) for r in rows] + [1])
        for i, r in enumerate(rows):
            y = 250 + i * 82
            score = -math.log10(max(float(r["p_value"]), 1e-300))
            w = 430 * score / maxv
            color = "#6f8fb3" if float(r["fdr_bh"]) >= 0.05 else "#b84a62"
            term = r["term"]
            add_wrapped(s, x + 24, y - 16, term, 42, "tiny")
            s.append(f'<rect x="{x+410}" y="{y-26}" width="{w:.1f}" height="28" fill="{color}"/>')
            add_text(s, x + 410 + w + 10, y - 4, f"p={sci(r['p_value'])}; FDR={sci(r['fdr_bh'])}", "tiny")
        add_text(s, x + 410, 900, "-log10 nominal p value", "tiny")
    add_wrapped(s, 105, 1010, "Functional enrichment is presented as exploratory biological context. The strongest mechanistic support in this draft comes from reader enrichment and independent hg19 annotation overlap.", 150, "small")
    save_svg(PKG / "figures" / "Figure5_functional_enrichment.svg", s, width, height)


def make_source_data() -> None:
    source = PKG / "source_data"
    mapping = {
        "Figure2_model_performance_source_data.tsv": ROOT / "09_results" / "metrics" / "model_comparison.tsv",
        "Figure2_external_benchmark_source_data.tsv": ROOT / "09_results" / "metrics" / "external_benchmark_comparison.tsv",
        "Figure3_sequence_structure_source_data.tsv": ROOT / "11_tables" / "biology_summary" / "rnafold_structure_comparison.tsv",
        "Figure3_spacing_source_data.tsv": ROOT / "11_tables" / "biology_summary" / "spacing_permutation_results.tsv",
        "Figure4_reader_enrichment_source_data.tsv": ROOT / "11_tables" / "biology_summary" / "reader_enrichment.tsv",
        "Figure4_external_validation_source_data.tsv": ROOT / "11_tables" / "biology_summary" / "external_validation_summary.tsv",
        "Figure5_GO_source_data.tsv": ROOT / "11_tables" / "biology_summary" / "go_enrichment.tsv",
        "Figure5_KEGG_Reactome_source_data.tsv": ROOT / "11_tables" / "biology_summary" / "kegg_reactome_enrichment.tsv",
    }
    source.mkdir(parents=True, exist_ok=True)
    for out_name, in_path in mapping.items():
        target = source / out_name
        target.write_bytes(in_path.read_bytes())


def latex_escape(text: str) -> str:
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def make_manuscripts(model_rows: list[dict[str, str]], benchmark_rows: list[dict[str, str]], reader_rows: list[dict[str, str]], ext_rows: list[dict[str, str]]) -> None:
    m4 = next(r for r in model_rows if r["variant"] == "M4")
    multirm = next(r for r in benchmark_rows if r["variant"] == "multirm_style_bilstm_attention")
    official = next(r for r in benchmark_rows if r["variant"] == "multirm_official_101nt_zero_shot")
    manuscript = f"""# StructDualRM: structure-aware integrated prediction and biological interpretation of m1A and m6A RNA modifications

**Draft target journal:** Nature Communications

**Author list:** [Add author names and affiliations]

**Corresponding author:** [Add name, email and ORCID]

## Abstract

RNA modifications shape transcript fate through sequence context, secondary structure and reader-protein interactions, yet most computational predictors treat related modification classes as independent sequence-labeling tasks. Here we present StructDualRM, a structure-aware multi-label framework for integrated prediction of m1A and m6A sites in hg19 transcript space. The workflow combines hg19 modification maps, RefSeq transcript projection, 101 nt sequence windows, RNAfold-derived local structure and dual-site contextual features. In held-out testing, the full StructDualRM model achieved a macro AUPR of {fmt(m4['test_macro_AUPR'], 4)} and macro MCC of {fmt(m4['test_macro_MCC'], 4)}, exceeding the sequence-only transformer and MultiRM-style baselines. Direct benchmarking on the same dataset showed that StructDualRM outperformed a MultiRM-style BiLSTM-attention model (macro AUPR {fmt(multirm['test_macro_AUPR'], 4)}, macro MCC {fmt(multirm['test_macro_MCC'], 4)}) and an official MultiRM zero-shot model (macro AUPR {fmt(official['test_macro_AUPR'], 4)}, macro MCC {fmt(official['test_macro_MCC'], 4)}). Biological analyses further showed that proximal m1A-m6A configurations were enriched for YTHDF2 reader overlap and independent hg19-compatible public m1A/m6A annotations. These results support RNA structure and dual-modification context as informative axes for joint epitranscriptomic modelling.

## Introduction

Post-transcriptional RNA modifications regulate transcript stability, localization, translation and cellular adaptation. N6-methyladenosine (m6A) is the most extensively studied internal mRNA modification, whereas N1-methyladenosine (m1A) has distinct chemical consequences and positional preferences. Their relationship is not only a question of site-level prediction: the spatial arrangement of modifications on the same transcript may influence reader binding, local RNA folding and downstream regulatory programs. Existing multi-label models such as MultiRM demonstrate that shared neural representations can predict multiple RNA modification types, but their emphasis remains primarily sequence-driven. A structure-aware design is therefore a natural route to improve both prediction and interpretation.

This study asks whether m1A and m6A can be predicted jointly while using RNA secondary structure and proximal dual-site context as explicit sources of biological signal. We built a hg19-based workflow that maps m1A and m6A sites into transcript coordinates, trains a series of ablation models, benchmarks against MultiRM-style baselines on the same dataset, and evaluates biological support using reader enrichment, gene-set enrichment and independent public hg19 annotations.

## Results

### A hg19 workflow for joint m1A/m6A modelling

The dataset was constructed from hg19 modification annotations and projected into RefSeq transcript coordinates. After filtering, the workflow retained 29,059 mapped m1A sites and 170,604 mapped m6A sites. Center-A filtering yielded 4,675 m1A-positive windows and 170,604 m6A-positive windows, with 14 co-positive sites and 186,237 negative candidates. The final supervised learning workflow used 34,972 training rows, 7,380 validation rows and 7,648 held-out test rows.

### StructDualRM improves held-out multi-label prediction

We trained five neural variants under the same split and 50-epoch setting: a sequence-only transformer (M0), a transformer with global structure (M1), a transformer-GNN variant (M2), a gated StructDualRM model (M3) and the full StructDualRM model (M4). M4 achieved the best held-out performance with macro AUPR {fmt(m4['test_macro_AUPR'], 4)}, macro MCC {fmt(m4['test_macro_MCC'], 4)} and average F1 {fmt(m4['test_average_F1'], 4)}. The gain was clearest for m1A AUPR ({fmt(m4['test_m1A_AUPR'], 4)}), suggesting that structure and dual-context features are particularly useful for the rarer label.

### Direct comparison with MultiRM-style baselines

To test whether the improvement reflected dataset-specific baselines rather than architecture, we evaluated classical k-mer models, a MultiRM-style BiLSTM-attention model and an official MultiRM zero-shot configuration on the same dataset. StructDualRM M4 outperformed the MultiRM-style model by {fmt(float(m4['test_macro_AUPR']) - float(multirm['test_macro_AUPR']), 4)} macro AUPR and {fmt(float(m4['test_macro_MCC']) - float(multirm['test_macro_MCC']), 4)} macro MCC. The official zero-shot setting performed substantially worse on this task, consistent with a distribution shift between the original MultiRM training domain and the present hg19 m1A/m6A dataset.

### Structure-aware modelling is supported by local sequence-structure analyses

The biological analyses suggest that the proximal m1A-m6A signal is not reducible to canonical motif presence alone. In CDS-proximal comparisons, local folding metrics shifted relative to distal m1A controls, including MFE per nucleotide, paired fraction and central paired fraction. These analyses support the structural branch as a mechanistic innovation point, while remaining correlative and computational.

### Proximal m1A-m6A sites are enriched for reader and external annotation support

YTHDF2 PAR-CLIP overlap was enriched in proximal m1A-m6A groups. For example, m1A 3UTR-proximal sites showed an odds ratio of {fmt(reader_rows[0]['odds_ratio'], 2)} versus distal m1A controls (FDR {sci(reader_rows[0]['fdr_bh'])}), and m1A CDS-proximal sites showed an odds ratio of {fmt(reader_rows[1]['odds_ratio'], 2)} (FDR {sci(reader_rows[1]['fdr_bh'])}). Independent hg19-compatible m1A/m6A public annotations showed concordant enrichment, with odds ratios of {fmt(ext_rows[0]['odds_ratio'], 2)} for 3UTR-proximal sites and {fmt(ext_rows[1]['odds_ratio'], 2)} for CDS-proximal sites against distal controls.

### Exploratory GO/KEGG/Reactome analyses

Gene-set analyses provided exploratory biological context for transcripts carrying proximal dual-modification arrangements. Several nominally enriched terms involved chromatin regulation, intracellular acidification, actin/cytoskeleton organization and vesicle-related processes. Because multiple-testing adjusted values were modest, these pathway results are treated as hypothesis-generating rather than primary mechanistic evidence.

## Discussion

StructDualRM advances integrated RNA modification prediction by bringing RNA secondary structure and proximal dual-modification context into a multi-label prediction framework. The strongest computational evidence is the held-out and direct benchmark performance. The strongest biological support is the concordant reader enrichment and external hg19 annotation overlap. Together, these analyses suggest that the spatial relationship between m1A and m6A sites can encode regulatory information that is missed by purely sequence-level models.

Several limitations remain. First, the biological analyses are computational and correlative; they do not establish causal reader recruitment or experimentally validate individual sites. Second, hg19-compatible public annotations differ in protocol, cell type and detection sensitivity. Third, the current GO/KEGG/Reactome analyses should be interpreted as exploratory. Future work should include cell-type matched perturbation assays, prospective validation on independent sequencing experiments and calibrated uncertainty estimates for site-level predictions.

## Methods

### Data processing

All primary processing was performed in hg19 coordinate space. Modification sites were parsed, filtered against the hg19 reference, mapped into RefSeq transcript coordinates and converted into centered 101 nt windows. Negative examples were sampled from transcript-compatible candidate adenines while avoiding known positive sites. RNA secondary structure was estimated with RNAfold on 101 nt windows and converted into local structural descriptors including MFE, MFE per nucleotide, paired fraction, center pairing and central-window paired fractions.

### Model training and evaluation

All neural models were trained for 50 epochs with the same train/validation/test split. Performance was evaluated using label-specific AUC, AUPR, F1, MCC, recall and specificity, and summarized with macro AUPR and macro MCC. Thresholds were selected on the validation set and applied to held-out testing.

### Benchmarking

Benchmark models included k-mer logistic regression, k-mer ExtraTrees, a MultiRM-style BiLSTM-attention architecture and an official MultiRM zero-shot configuration. All benchmark results reported here were evaluated on the same held-out split used for StructDualRM.

### Reader enrichment and external validation

Reader enrichment used public YTHDF2 PAR-CLIP hg19-compatible annotations. External validation used independent public m1A-miCLIP and m6A peak annotations prepared in hg19 coordinate space. Fisher exact tests compared proximal m1A-m6A groups with distal m1A controls or m6A-only sites on the same transcripts. P values were adjusted by Benjamini-Hochberg correction.

### Gene-set enrichment

GO Biological Process, KEGG and Reactome enrichment analyses were performed on transcript-level gene groups derived from proximal and comparison categories. Results are reported as exploratory because many adjusted P values do not pass a stringent false-discovery threshold.

## Data availability

The draft submission package includes source data underlying each generated figure, processed model-performance tables, biological enrichment tables and hg19-compatible external validation tables. Raw public data sources and final accession/DOI statements should be verified before journal submission.

## Code availability

Custom scripts used to generate the analyses and this draft package are included in the analysis archive. Before formal submission, the code should be deposited in a public repository or reproducible capsule and linked with a permanent DOI if possible.

## Acknowledgements

[Add funding and institutional acknowledgements.]

## Author contributions

[Add author contribution statements using Nature Communications format.]

## Competing interests

The authors declare no competing interests. [Edit if needed.]

## References

1. Meyer, K. D. et al. Comprehensive analysis of mRNA methylation reveals enrichment in 3' UTRs and near stop codons. *Cell* 149, 1635-1646 (2012).
2. Dominissini, D. et al. Topology of the human and mouse m6A RNA methylomes revealed by m6A-seq. *Nature* 485, 201-206 (2012).
3. Roundtree, I. A., Evans, M. E., Pan, T. & He, C. Dynamic RNA modifications in gene expression regulation. *Cell* 169, 1187-1200 (2017).
4. Shi, H., Wei, J. & He, C. Where, when, and how: context-dependent functions of RNA methylation writers, readers, and erasers. *Mol. Cell* 74, 640-650 (2019).
5. MultiRM original article: Attention-based multi-label neural networks for integrated prediction and interpretation of twelve widely occurring RNA modifications. [Verify full bibliographic details before submission.]
6. Lorenz, R. et al. ViennaRNA Package 2.0. *Algorithms Mol. Biol.* 6, 26 (2011).
"""

    write_text(PKG / "manuscript" / "StructDualRM_NatureCommunications_Draft.md", manuscript)

    body = latex_escape(manuscript)
    # Restore markdown headings into rough LaTeX after escaping.
    lines = []
    for raw in manuscript.splitlines():
        if raw.startswith("# "):
            continue
        if raw.startswith("## "):
            lines.append(r"\section*{" + latex_escape(raw[3:]) + "}")
        elif raw.startswith("### "):
            lines.append(r"\subsection*{" + latex_escape(raw[4:]) + "}")
        elif raw.startswith("**Draft target journal:**"):
            lines.append(r"\noindent\textbf{Draft target journal:} Nature Communications")
        elif raw.startswith("**Author list:**"):
            lines.append(r"\noindent\textbf{Author list:} [Add author names and affiliations]")
        elif raw.startswith("**Corresponding author:**"):
            lines.append(r"\noindent\textbf{Corresponding author:} [Add name, email and ORCID]")
        elif raw.strip() == "":
            lines.append("")
        else:
            lines.append(latex_escape(raw))
    figure_blocks = []
    legends = [
        ("Figure1_workflow_architecture.svg", "StructDualRM study design and model concept."),
        ("Figure2_model_performance.svg", "Predictive performance and external benchmark."),
        ("Figure3_sequence_structure_context.svg", "Sequence and structural context of proximal dual modifications."),
        ("Figure4_reader_external_validation.svg", "Reader enrichment and hg19 external validation."),
        ("Figure5_functional_enrichment.svg", "Exploratory functional enrichment."),
    ]
    for idx, (fname, caption) in enumerate(legends, 1):
        figure_blocks.append(r"\begin{figure}[p]")
        figure_blocks.append(r"\centering")
        figure_blocks.append(r"\fbox{\parbox{0.82\textwidth}{\centering Generated figure file: \texttt{" + latex_escape(fname) + r"}\\ Convert this SVG to PDF/TIFF for final submission if required.}}")
        figure_blocks.append(r"\caption{\textbf{" + latex_escape(f"Figure {idx}. ") + r"}" + latex_escape(caption) + r"}")
        figure_blocks.append(r"\end{figure}")
    refs = r"""
\begin{thebibliography}{9}
\bibitem{meyer2012} Meyer, K. D. et al. Comprehensive analysis of mRNA methylation reveals enrichment in 3' UTRs and near stop codons. \textit{Cell} \textbf{149}, 1635--1646 (2012).
\bibitem{dominissini2012} Dominissini, D. et al. Topology of the human and mouse m6A RNA methylomes revealed by m6A-seq. \textit{Nature} \textbf{485}, 201--206 (2012).
\bibitem{roundtree2017} Roundtree, I. A., Evans, M. E., Pan, T. \& He, C. Dynamic RNA modifications in gene expression regulation. \textit{Cell} \textbf{169}, 1187--1200 (2017).
\bibitem{shi2019} Shi, H., Wei, J. \& He, C. Where, when, and how: context-dependent functions of RNA methylation writers, readers, and erasers. \textit{Mol. Cell} \textbf{74}, 640--650 (2019).
\bibitem{multirm} MultiRM original article: Attention-based multi-label neural networks for integrated prediction and interpretation of twelve widely occurring RNA modifications. Verify full bibliographic details before submission.
\bibitem{viennarna} Lorenz, R. et al. ViennaRNA Package 2.0. \textit{Algorithms Mol. Biol.} \textbf{6}, 26 (2011).
\end{thebibliography}
"""
    filtered = []
    skip_refs = False
    for line in lines:
        if line == r"\section*{References}":
            skip_refs = True
            continue
        if skip_refs:
            continue
        filtered.append(line)
    tex = r"""\documentclass[12pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{setspace}
\usepackage{authblk}
\usepackage{hyperref}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\doublespacing
\title{StructDualRM: structure-aware integrated prediction and biological interpretation of m1A and m6A RNA modifications}
\author[1]{[First Author]}
\author[1]{[Second Author]}
\author[1]{[Corresponding Author]\thanks{Correspondence: [email]}}
\affil[1]{[Department, Institution, City, Country]}
\date{}
\begin{document}
\maketitle
""" + "\n".join(filtered) + "\n" + "\n".join(figure_blocks) + "\n" + refs + "\n\\end{document}\n"
    write_text(PKG / "manuscript" / "StructDualRM_NatureCommunications_Draft.tex", tex)


def make_readme() -> None:
    readme = """# Nature Communications draft package for StructDualRM

Generated on 2026-06-02 from the local hg19 m1A/m6A workflow.

## What is included

- `manuscript/StructDualRM_NatureCommunications_Draft.md`: English manuscript draft.
- `manuscript/StructDualRM_NatureCommunications_Draft.tex`: LaTeX draft using standard `article.cls`, as permitted by Nature Communications for TeX/LaTeX initial submissions.
- `figures/*.svg`: five generated multi-panel scientific figures.
- `source_data/*.tsv`: numerical source data underlying the generated figures.
- `template_sources/`: downloaded Springer Nature LaTeX author-support page and the December 2024 Springer Nature LaTeX template zip.
- `MANUSCRIPT_REQUIREMENTS_NOTE.md`: practical notes extracted from Nature Communications/Springer Nature official guidance.
- `included_file_manifest.tsv`: files included or referenced by the archive.

## Claim boundary

The draft can claim stronger prediction performance, same-dataset benchmarking against MultiRM-style baselines, reader enrichment and hg19 external annotation support. It should not claim causal reader recruitment or wet-lab validation until those experiments are performed.

## Before real submission

1. Replace author, affiliation, funding and ORCID placeholders.
2. Verify every reference, especially the MultiRM citation.
3. Convert SVG figures to journal-preferred final formats if requested.
4. Deposit code/data in a public repository or reproducible capsule and update Data/Code availability.
5. Ask a colleague outside the project to read the title, abstract and introduction for non-specialist clarity.
"""
    write_text(PKG / "README.md", readme)
    note = """# Nature Communications formatting notes used for this draft

This package follows official guidance checked on 2026-06-02:

- Nature Communications states that online submissions include a cover letter, manuscript text, individual figure files and optional Supplementary Information.
- For first submissions, manuscript text and figures may be incorporated into one Word, TeX/LaTeX or PDF file up to 30 MB, or figures may be grouped at the end.
- Nature Communications states that TeX/LaTeX submissions may use standard class files such as `article.cls`, `revtex.cls` or `amsart.cls`, should avoid non-standard fonts/packages and should use numerical references.
- Springer Nature provides a general LaTeX authoring template package and says it can be used for Springer Nature journals, while journal-level author instructions should still be checked.
- Source data should contain the numerical data underlying graphs and charts. This package therefore includes one source-data TSV per generated figure.

Official pages archived in `template_sources/`:

- Nature Communications, how to submit: https://www.nature.com/ncomms/submit/how-to-submit
- Springer Nature LaTeX author support: https://www.springernature.com/in/authors/campaigns/latex-author-support
"""
    write_text(PKG / "MANUSCRIPT_REQUIREMENTS_NOTE.md", note)


def make_cover_letter() -> None:
    text = """[Corresponding author name]
[Institution]
[Address]
[Email]

Dear Editors,

We are pleased to submit our manuscript entitled "StructDualRM: structure-aware integrated prediction and biological interpretation of m1A and m6A RNA modifications" for consideration as an Article in Nature Communications.

This work presents a hg19-based, structure-aware multi-label framework for integrated m1A/m6A prediction. The study combines model ablation, direct same-dataset benchmarking against MultiRM-style baselines, reader enrichment, GO/KEGG/Reactome context and independent hg19-compatible public annotation validation. The central advance is that RNA secondary structure and proximal dual-modification context improve prediction and provide biologically interpretable support for m1A-m6A arrangements.

We believe the manuscript will be of interest to the broad readership of Nature Communications because it connects computational epitranscriptomics, RNA structure and multi-modification biology in a reproducible framework.

All authors have approved the submission. The manuscript is not under consideration elsewhere and has no significant overlap with any submitted or published work from the authors. [Edit this statement as needed.]

Sincerely,

[Corresponding author name]
"""
    write_text(PKG / "manuscript" / "cover_letter_draft.txt", text)


def make_flat_submission_folder() -> None:
    flat = PKG / "flat_initial_submission"
    flat.mkdir(parents=True, exist_ok=True)
    for path in [
        PKG / "manuscript" / "StructDualRM_NatureCommunications_Draft.md",
        PKG / "manuscript" / "StructDualRM_NatureCommunications_Draft.tex",
        PKG / "manuscript" / "cover_letter_draft.txt",
    ]:
        shutil.copy2(path, flat / path.name)
    for path in sorted((PKG / "figures").glob("*.svg")):
        shutil.copy2(path, flat / path.name)
    note = """Flat initial-submission helper folder.

This folder keeps the manuscript, cover letter and generated SVG figure files in one directory for easier manual upload/reformatting. The LaTeX file intentionally uses standard article.cls and figure placeholders, because the final journal-ready figure format may need conversion from SVG to PDF/TIFF/PNG depending on submission-stage instructions.
"""
    write_text(flat / "README_flat_initial_submission.txt", note)


def make_inventory_and_zip() -> dict[str, object]:
    include_dirs = [
        "01_reference",
        "02_parsed_sites",
        "03_hg19_sites",
        "04_transcript_space",
        "05_datasets",
        "06_structure",
        "07_graphs",
        "08_models",
        "09_results",
        "11_tables",
        "12_benchmarks",
        "external",
        "scripts",
        "tests",
        "docs",
    ]
    rows: list[tuple[str, int, str]] = []
    for name in include_dirs:
        base = ROOT / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                rel = path.relative_to(ROOT).as_posix()
                rows.append((rel, path.stat().st_size, "analysis_data_and_results/" + rel))
    manifest = PKG / "included_file_manifest.tsv"
    lines = ["source_relative_path\tsize_bytes\tarchive_path"]
    lines.extend(f"{a}\t{b}\t{c}" for a, b, c in rows)
    write_text(manifest, "\n".join(lines) + "\n")
    zip_path = ROOT / "submission" / "nature_communications_structdualrm_full_package.zip"
    write_text(PKG / "ZIP_PACKAGE_PATH.txt", str(zip_path) + "\n")
    figure_count = len(list((PKG / "figures").glob("Figure*.svg")))
    summary = {
        "package_dir": str(PKG),
        "zip_path": str(zip_path),
        "figures": figure_count,
        "included_archive_files": len(rows),
    }
    write_text(PKG / "package_generation_summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in PKG.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(PKG.parent))
        for rel, _, arc in rows:
            path = ROOT / rel
            zf.write(path, "nature_communications_structdualrm/" + arc)
    return summary


def main() -> None:
    (PKG / "manuscript").mkdir(parents=True, exist_ok=True)
    (PKG / "figures").mkdir(parents=True, exist_ok=True)
    model_rows = read_tsv(ROOT / "09_results" / "metrics" / "model_comparison.tsv")
    benchmark_rows = read_tsv(ROOT / "09_results" / "metrics" / "external_benchmark_comparison.tsv")
    reader_rows = read_tsv(ROOT / "11_tables" / "biology_summary" / "reader_enrichment.tsv")
    ext_rows = read_tsv(ROOT / "11_tables" / "biology_summary" / "external_validation_summary.tsv")
    struct_rows = read_tsv(ROOT / "11_tables" / "biology_summary" / "rnafold_structure_comparison.tsv")
    seq_rows = read_tsv(ROOT / "11_tables" / "biology_summary" / "sequence_feature_comparison.tsv")
    spacing_rows = read_tsv(ROOT / "11_tables" / "biology_summary" / "spacing_permutation_results.tsv")
    go_rows = read_tsv(ROOT / "11_tables" / "biology_summary" / "go_enrichment.tsv")
    pathway_rows = read_tsv(ROOT / "11_tables" / "biology_summary" / "kegg_reactome_enrichment.tsv")
    make_figure1()
    make_figure2(model_rows, benchmark_rows)
    make_figure3(struct_rows, seq_rows, spacing_rows)
    make_figure4(reader_rows, ext_rows)
    make_figure5(go_rows, pathway_rows)
    make_source_data()
    make_manuscripts(model_rows, benchmark_rows, reader_rows, ext_rows)
    make_readme()
    make_cover_letter()
    make_flat_submission_folder()
    summary = make_inventory_and_zip()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
