from __future__ import annotations

import csv
from pathlib import Path
import shutil
import zipfile

from create_sn_template_manuscript import render_svg_to_png


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "submission" / "nature_communications_structdualrm"
SN_DRAFT = PKG / "sn_template_draft"
TEX = SN_DRAFT / "StructDualRM_NatureCommunications_SN_Template_Draft.tex"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:30px;font-weight:700}.label{font-size:22px;font-weight:700}.small{font-size:18px}.tiny{font-size:14px}.axis{stroke:#44515e;stroke-width:2}.muted{fill:#667085}</style>',
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#44515e"/></marker>',
        "</defs>",
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
    ]


def add_text(lines: list[str], x: float, y: float, text: str, cls: str = "small", anchor: str = "start") -> None:
    lines.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{text}</text>')


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


def box(lines: list[str], x: int, y: int, w: int, h: int, title: str, subtitle: str, fill: str, stroke: str = "#8796a6") -> None:
    lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    add_wrapped(lines, x + 22, y + 36, title, max(16, int(w / 12)), "small")
    if subtitle:
        add_wrapped(lines, x + 22, y + 72, subtitle, max(22, int(w / 10)), "tiny")


def arrow(lines: list[str], x1: int, y1: int, x2: int, y2: int) -> None:
    lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="axis" marker-end="url(#arrow)"/>')


def arrow_colored(lines: list[str], x1: int, y1: int, x2: int, y2: int, stroke: str, width: int = 4) -> None:
    lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}" marker-end="url(#arrow)"/>')


def panel(lines: list[str], x: int, y: int, w: int, h: int, label: str, fill: str = "#f7f9fc") -> None:
    lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="#d8e0ea" stroke-width="2"/>')
    add_text(lines, x + 24, y + 42, label, "label")


def mini_rna(lines: list[str], x: int, y: int, scale: int = 1) -> None:
    colors = ["#4db6ac", "#7e57c2", "#ef5350", "#66bb6a", "#42a5f5", "#ffa726", "#26a69a", "#ab47bc"]
    last = None
    for i in range(12):
        cx = x + i * 18 * scale
        cy = y + (8 if i % 2 else -8) * scale
        if last:
            lines.append(f'<line x1="{last[0]}" y1="{last[1]}" x2="{cx}" y2="{cy}" stroke="#9aa7b5" stroke-width="{2 * scale}"/>')
        lines.append(f'<circle cx="{cx}" cy="{cy}" r="{6 * scale}" fill="{colors[i % len(colors)]}" stroke="#ffffff" stroke-width="2"/>')
        last = (cx, cy)


def matrix_icon(lines: list[str], x: int, y: int, cols: int = 8, rows: int = 5, cell: int = 12) -> None:
    palette = ["#dbeafe", "#fde2e2", "#dcfce7", "#fef3c7"]
    for r in range(rows):
        for c in range(cols):
            fill = palette[(r + c) % len(palette)]
            lines.append(f'<rect x="{x + c * cell}" y="{y + r * cell}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff" stroke-width="1"/>')


def make_architecture_svg() -> Path:
    width, height = 2200, 1500
    s: list[str] = []
    add_text(s, 70, 70, "Fig. 6 | StructDualRM M4 architecture and interpretable dual-label prediction", "title")
    add_wrapped(
        s,
        70,
        110,
        "A Nature-style multi-panel schematic showing hg19 RNA modification windows, RNAfold structure construction, the StructDualRM forward model and label-specific interpretation routes.",
        150,
        "small",
    )

    forward = "#9aaef7"
    interpret = "#f2a0a0"
    dark = "#334155"

    # Panel a: input and encoding.
    panel(s, 70, 150, 2060, 350, "a", "#f5f7fb")
    add_text(s, 120, 220, "hg19 m1A/m6A candidate windows", "small")
    mini_rna(s, 135, 290, 1)
    add_text(s, 126, 350, "101 nt center-A window", "tiny")
    s.append('<rect x="335" y="205" width="170" height="125" rx="6" fill="#fff7ed" stroke="#f59e0b" stroke-width="2"/>')
    add_text(s, 362, 240, "m1A / m6A", "small")
    add_text(s, 365, 272, "site labels", "tiny")
    add_text(s, 365, 300, "0/1 per mark", "tiny")
    arrow_colored(s, 520, 270, 670, 270, forward, 8)
    box(s, 690, 205, 300, 128, "Sequence encoding", "base ids + position + center-distance embeddings", "#e8f1fb")
    matrix_icon(s, 1025, 220, 8, 6, 13)
    add_text(s, 1024, 330, "one-hot-like sequence tensor", "tiny")
    add_text(s, 1150, 220, "+", "label")
    box(s, 1210, 205, 330, 128, "RNAfold structure", "paired flags, base-pair partners and 6 scalar folding features", "#f2edfb")
    mini_rna(s, 1250, 360, 1)
    lines_y = [352, 360, 368]
    for i, yy in enumerate(lines_y):
        s.append(f'<line x1="{1260 + i * 30}" y1="{yy}" x2="{1390 - i * 15}" y2="{yy - 48}" stroke="#7c6aa6" stroke-width="2"/>')
    add_text(s, 1585, 220, "+", "label")
    box(s, 1640, 205, 350, 128, "Transcript context", "region id, relative transcript position and log10 transcript length", "#fff3da")
    arrow_colored(s, 990, 395, 990, 465, interpret, 7)
    add_text(s, 885, 450, "interpretation-ready inputs", "tiny")

    # Panel b: forward model.
    panel(s, 70, 530, 2060, 500, "b", "#fbfbff")
    add_text(s, 120, 600, "StructDualRM forward process", "small")
    box(s, 135, 650, 265, 106, "Embedding sum", "base + position + distance", "#e8f1fb")
    box(s, 450, 650, 285, 106, "Transformer encoder", "1 layer, 4 heads, d_model=64", "#dceaf7")
    box(s, 785, 625, 310, 112, "Center-aware pooling", "center state + local mean + attention", "#e8f1fb")
    box(s, 785, 775, 310, 112, "Pair-aware graph", "2 GraphLayer blocks + graph pooling", "#eadff8")
    box(s, 1145, 705, 250, 112, "Representation learning", "z_seq | z_graph | z_struct | z_context", "#e9f7f7")
    box(s, 1450, 625, 260, 112, "m1A gate + head", "g_m1A modulates graph and structure", "#f7e1e5")
    box(s, 1450, 785, 260, 112, "m6A gate + head", "g_m6A modulates graph and structure", "#e0f1df")
    box(s, 1780, 705, 240, 112, "Dual-label output", "P(m1A), P(m6A); calibrated thresholds", "#ffffff")
    arrow_colored(s, 400, 703, 450, 703, forward, 7)
    arrow_colored(s, 735, 703, 785, 680, forward, 7)
    arrow_colored(s, 735, 703, 785, 830, forward, 7)
    arrow_colored(s, 1095, 682, 1145, 750, forward, 7)
    arrow_colored(s, 1095, 830, 1145, 760, forward, 7)
    arrow_colored(s, 1395, 760, 1450, 680, forward, 7)
    arrow_colored(s, 1395, 760, 1450, 840, forward, 7)
    arrow_colored(s, 1710, 680, 1780, 746, forward, 7)
    arrow_colored(s, 1710, 840, 1780, 778, forward, 7)
    box(s, 135, 830, 265, 78, "Global structure MLP", "6 RNAfold features -> 32", "#f7eaf2")
    box(s, 450, 830, 285, 78, "Context branch", "region embedding + context MLP", "#fff3da")
    arrow_colored(s, 400, 869, 1145, 790, forward, 5)
    arrow_colored(s, 735, 869, 1145, 790, forward, 5)
    add_text(s, 1835, 645, "forward", "tiny")
    s.append(f'<rect x="1808" y="629" width="32" height="10" fill="{forward}"/>')
    add_text(s, 1940, 645, "gate/attribution", "tiny")
    s.append(f'<rect x="1905" y="629" width="32" height="10" fill="{interpret}"/>')
    arrow_colored(s, 1680, 625, 1250, 595, interpret, 6)
    arrow_colored(s, 1680, 897, 1250, 965, interpret, 6)
    add_text(s, 1275, 595, "label-specific gate signals", "tiny")
    add_text(s, 1275, 970, "gate_m1A / gate_m6A exported", "tiny")

    # Panel c: module details.
    panel(s, 70, 1060, 2060, 320, "c", "#fffdf4")
    box(s, 135, 1125, 540, 190, "GraphLayer message passing", "", "#f6eadf")
    add_text(s, 162, 1190, "self + backbone neighbours + paired partner", "tiny")
    s.append('<rect x="195" y="1210" width="38" height="38" fill="#9aaef7" stroke="#64748b"/>')
    s.append('<rect x="350" y="1190" width="38" height="38" fill="#9aaef7" stroke="#64748b"/>')
    s.append('<rect x="350" y="1250" width="38" height="38" fill="#d8b4fe" stroke="#64748b"/>')
    s.append('<rect x="535" y="1220" width="42" height="42" fill="#94a3b8" stroke="#64748b"/>')
    arrow(s, 233, 1229, 350, 1209)
    arrow(s, 233, 1229, 350, 1269)
    arrow(s, 388, 1209, 535, 1241)
    arrow(s, 388, 1269, 535, 1241)
    add_text(s, 170, 1300, "LayerNorm residual graph update", "tiny")

    box(s, 830, 1125, 500, 190, "Center-aware attention pooling", "", "#eef2ff")
    add_text(s, 858, 1190, "center token queries all token states", "tiny")
    matrix_icon(s, 875, 1180, 9, 6, 15)
    s.append('<circle cx="948" cy="1224" r="18" fill="#f87171" stroke="#ffffff" stroke-width="3"/>')
    arrow_colored(s, 1010, 1222, 1120, 1222, forward, 6)
    box(s, 1145, 1170, 135, 78, "z_seq", "3 x 64", "#ffffff")
    add_text(s, 880, 1290, "pooled central and local RNA context", "tiny")

    box(s, 1480, 1125, 560, 190, "Label-specific gating", "", "#eefaf8")
    add_text(s, 1512, 1190, "separate sigmoid gates weight graph and structure branches", "tiny")
    box(s, 1518, 1170, 120, 66, "concat z", "all branches", "#ffffff")
    box(s, 1685, 1135, 130, 66, "gate_m1A", "sigmoid", "#f7e1e5")
    box(s, 1685, 1220, 130, 66, "gate_m6A", "sigmoid", "#e0f1df")
    box(s, 1875, 1135, 120, 66, "m1A logit", "head", "#f7e1e5")
    box(s, 1875, 1220, 120, 66, "m6A logit", "head", "#e0f1df")
    arrow(s, 1638, 1203, 1685, 1168)
    arrow(s, 1638, 1203, 1685, 1253)
    arrow(s, 1815, 1168, 1875, 1168)
    arrow(s, 1815, 1253, 1875, 1253)

    # Legend.
    s.append('<rect x="70" y="1400" width="2060" height="70" rx="10" fill="#ffffff" stroke="#c7d0db" stroke-width="2"/>')
    add_text(s, 100, 1443, "Legend:", "label")
    legend = [
        ("#e8f1fb", "sequence"),
        ("#f2edfb", "pair-aware graph"),
        ("#f7eaf2", "global RNAfold structure"),
        ("#fff3da", "transcript context"),
        ("#d9f0ee", "label-specific gates"),
        ("#f7e1e5", "m1A output"),
        ("#e0f1df", "m6A output"),
    ]
    lx = 230
    for fill, label in legend:
        s.append(f'<rect x="{lx}" y="1422" width="28" height="28" rx="5" fill="{fill}" stroke="#8796a6"/>')
        add_text(s, lx + 38, 1444, label, "tiny")
        lx += 255 if label == "global RNAfold structure" else 210

    svg = "\n".join(svg_header(width, height) + s + ["</svg>\n"])
    out = PKG / "figures" / "Figure6_structdualrm_architecture.svg"
    write_text(out, svg)
    return out


def make_source_data() -> None:
    rows = [
        ("sequence_input", "101 nt centered sequence", "seq ids", "A/C/G/T/U encoded as 1-4, padding 0"),
        ("embedding", "base + position + center-distance embeddings", "d_model=64", "input to transformer"),
        ("transformer", "TransformerEncoder", "1 layer, 4 heads", "GELU feed-forward, dropout=0.2"),
        ("sequence_pooling", "center/local/attention pooling", "3 x 64", "z_seq"),
        ("graph_input", "paired flags + base-pair partner index", "101 nodes", "uses RNAfold base-pair list"),
        ("graph_layers", "GraphLayer x2", "self/backbone/pair messages", "LayerNorm residual graph update"),
        ("graph_pooling", "center/local/attention/pair pooling", "4 x 64", "z_graph"),
        ("structure_mlp", "RNAfold scalar features", "6 -> 32", "center paired, central paired fractions, paired fraction, MFE per nt, loop span"),
        ("context_branch", "region embedding + context MLP", "32", "region id, relative position, log10 transcript length"),
        ("gating", "label-specific gates", "g_m1A, g_m6A", "sigmoid MLP gates graph and structure features"),
        ("heads", "two prediction heads", "m1A and m6A logits", "MLP + sigmoid probabilities"),
        ("loss", "weighted BCE", "class weights [1.5, 1.0]", "validation thresholds selected by MCC"),
    ]
    out = PKG / "source_data" / "Figure6_model_architecture_source_data.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["module_id", "module_name", "dimension_or_setting", "description"])
        writer.writerows(rows)


def copy_png_to_template(svg_path: Path) -> Path:
    png = PKG / "figures_png" / "Figure6_structdualrm_architecture.png"
    render_svg_to_png(svg_path, png)
    shutil.copy2(png, SN_DRAFT / png.name)
    return png


def insert_tex_figure() -> None:
    text = TEX.read_text(encoding="utf-8")
    if "Figure6_structdualrm_architecture.png" in text:
        return
    figure_block = r"""

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Figure6_structdualrm_architecture.png}
\caption{\textbf{Detailed StructDualRM M4 model architecture.} The full model combines a center-aware sequence transformer, a pair-aware RNA structure graph branch, an RNAfold scalar-feature MLP and a transcript-context branch. Label-specific gates modulate graph and structure features separately for m1A and m6A before two output heads produce calibrated probabilities.}
\label{fig:model_architecture}
\end{figure*}
"""
    anchor = r"\subsection{Internal ablation identifies the full structure-aware model as the strongest variant}"
    if anchor not in text:
        text = text.replace(r"\section{Results}\label{sec:results}", r"\section{Results}\label{sec:results}" + figure_block)
    else:
        text = text.replace(anchor, figure_block + "\n" + anchor, 1)
    text = text.replace(
        "The full M4 model achieved the best overall held-out performance",
        r"The detailed M4 architecture is shown in Fig.~\ref{fig:model_architecture}. The full M4 model achieved the best overall held-out performance",
        1,
    )
    TEX.write_text(text, encoding="utf-8", newline="\n")


def refresh_template_zip() -> None:
    zip_path = ROOT / "submission" / "nature_communications_structdualrm_sn_template_draft.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in SN_DRAFT.rglob("*"):
            if path.is_file():
                zf.write(path, f"sn_template_draft/{path.relative_to(SN_DRAFT).as_posix()}")


def main() -> None:
    svg = make_architecture_svg()
    png = copy_png_to_template(svg)
    make_source_data()
    insert_tex_figure()
    refresh_template_zip()
    print({"svg": str(svg), "png": str(png), "tex": str(TEX)})


if __name__ == "__main__":
    main()
