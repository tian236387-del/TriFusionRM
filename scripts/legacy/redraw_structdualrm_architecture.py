from __future__ import annotations

import csv
import html
import shutil
from pathlib import Path

from create_sn_template_manuscript import render_svg_to_png


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "submission" / "nature_communications_structdualrm"
SN_DRAFT = PKG / "sn_template_draft"
SVG_NAME = "Figure6_structdualrm_architecture.svg"
PNG_NAME = "Figure6_structdualrm_architecture.png"


def esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def text(lines: list[str], x: float, y: float, value: str, cls: str = "small", anchor: str = "start") -> None:
    lines.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{esc(value)}</text>')


def rect(lines: list[str], x: int, y: int, w: int, h: int, fill: str, stroke: str = "#334155", rx: int = 8, sw: int = 2) -> None:
    lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def line(lines: list[str], x1: int, y1: int, x2: int, y2: int, color: str = "#111827", width: int = 3, arrow: bool = False, dashed: bool = False) -> None:
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash}{marker}/>')


def circle(lines: list[str], x: int, y: int, r: int, fill: str, stroke: str = "#334155", sw: int = 2) -> None:
    lines.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def panel(lines: list[str], x: int, y: int, w: int, h: int, title: str, fill: str, stroke: str) -> None:
    rect(lines, x, y, w, h, fill, stroke, 16, 3)
    text(lines, x + w / 2, y - 14, title, "label", "middle")


def matrix(lines: list[str], x: int, y: int, cols: int, rows: int, cell: int, palette: list[str]) -> None:
    for r in range(rows):
        for c in range(cols):
            rect(lines, x + c * cell, y + r * cell, cell - 1, cell - 1, palette[(r + c) % len(palette)], "#ffffff", 0, 1)


def color_bar(lines: list[str], x: int, y: int, colors: list[str], cell: int = 28, h: int = 34) -> None:
    for i, fill in enumerate(colors):
        rect(lines, x + i * cell, y, cell, h, fill, "#111827", 0, 1)


def rna_strip(lines: list[str], x: int, y: int, center: int, label: str, accent: str) -> None:
    bases = list("GUUAGCCUGGACUUGAAC")
    spacing = 28
    text(lines, x - 34, y + 7, "5'-", "tiny")
    for i, base in enumerate(bases):
        bx = x + i * spacing
        if i in {center - 2, center - 1, center, center + 1, center + 2}:
            rect(lines, bx - 12, y - 18, 24, 28, "#e8eef7", "#e8eef7", 6, 1)
        text(lines, bx, y + 5, base, "tiny", "middle")
        if i > 0:
            line(lines, bx - spacing + 10, y, bx - 10, y, "#64748b", 1, False)
    bx = x + center * spacing
    circle(lines, bx, y, 19, "#ffffff", accent, 3)
    text(lines, bx, y + 7, "A", "small", "middle")
    text(lines, x + len(bases) * spacing + 8, y + 7, "-3'", "tiny")
    line(lines, bx, y + 20, bx, y + 54, accent, 2, True, True)
    text(lines, x + 550, y + 34, "101 nt", "small")
    text(lines, bx, y + 91, label, "label", "middle")


def molecule_icon(lines: list[str], x: int, y: int, label: str, accent: str) -> None:
    pts = [(x, y), (x + 48, y - 28), (x + 92, y), (x + 72, y + 54), (x + 18, y + 54)]
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        line(lines, a[0], a[1], b[0], b[1], "#475569", 2)
    for cx, cy in pts:
        circle(lines, cx, cy, 6, "#ffffff", accent, 2)
    line(lines, x + 92, y, x + 140, y - 22, "#475569", 2)
    circle(lines, x + 140, y - 22, 7, "#ffffff", accent, 2)
    text(lines, x + 62, y + 96, label, "label", "middle")


def helix_icon(lines: list[str], x: int, y: int, accent: str) -> None:
    prev_top: tuple[int, int] | None = None
    prev_bot: tuple[int, int] | None = None
    for i in range(10):
        xx = x + i * 26
        top_y = y + int(28 * ((i % 4) - 1.5) / 1.5)
        bot_y = y - int(28 * ((i % 4) - 1.5) / 1.5) + 70
        if prev_top:
            line(lines, prev_top[0], prev_top[1], xx, top_y, accent, 5)
            line(lines, prev_bot[0], prev_bot[1], xx, bot_y, "#9ca3af", 4)
        line(lines, xx, top_y, xx, bot_y, "#cbd5e1", 2)
        prev_top, prev_bot = (xx, top_y), (xx, bot_y)


def graph_icon(lines: list[str], x: int, y: int, scale: float = 1.0) -> None:
    pts = [
        (x, y),
        (x + int(70 * scale), y - int(45 * scale)),
        (x + int(140 * scale), y - int(20 * scale)),
        (x + int(160 * scale), y + int(58 * scale)),
        (x + int(80 * scale), y + int(90 * scale)),
        (x - int(18 * scale), y + int(55 * scale)),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4), (2, 5)]
    for a, b in edges:
        line(lines, pts[a][0], pts[a][1], pts[b][0], pts[b][1], "#64748b", 2)
    labels = ["U", "G", "C", "A", "U", "G"]
    for i, (cx, cy) in enumerate(pts):
        circle(lines, cx, cy, int(15 * scale), "#bcd1ff", "#365899", 2)
        text(lines, cx, cy + 6, labels[i], "tiny", "middle")


def small_mlp(lines: list[str], x: int, y: int) -> None:
    left = [(x, y), (x, y + 42), (x, y + 84)]
    mid = [(x + 62, y - 8), (x + 62, y + 34), (x + 62, y + 76), (x + 62, y + 118)]
    right = [(x + 125, y + 20), (x + 125, y + 64), (x + 125, y + 108)]
    for a in left:
        for b in mid:
            line(lines, a[0], a[1], b[0], b[1], "#475569", 1)
    for a in mid:
        for b in right:
            line(lines, a[0], a[1], b[0], b[1], "#475569", 1)
    for group in (left, mid, right):
        for cx, cy in group:
            circle(lines, cx, cy, 11, "#f3a7b3", "#9f1239", 2)


def transformer_block(lines: list[str], x: int, y: int) -> None:
    rect(lines, x, y, 430, 330, "#f8fafc", "#737373", 22, 3)
    rect(lines, x + 45, y + 34, 330, 250, "#ffffff", "#111827", 0, 2)
    rect(lines, x + 90, y + 66, 230, 62, "#ffe5ea", "#be123c", 12, 3)
    text(lines, x + 205, y + 103, "Multi-Head Self-Attention", "label", "middle")
    rect(lines, x + 122, y + 151, 170, 48, "#fff2c7", "#b08900", 8, 3)
    text(lines, x + 207, y + 181, "Add & Norm", "label", "middle")
    rect(lines, x + 122, y + 221, 170, 48, "#d9fbf6", "#0f766e", 8, 3)
    text(lines, x + 207, y + 251, "Feed Forward", "label", "middle")
    rect(lines, x + 122, y + 291, 170, 48, "#fff2c7", "#b08900", 8, 3)
    text(lines, x + 207, y + 321, "Add & Norm", "label", "middle")
    line(lines, x + 207, y + 128, x + 207, y + 151, "#111827", 3, True)
    line(lines, x + 207, y + 199, x + 207, y + 221, "#111827", 3, True)
    line(lines, x + 207, y + 269, x + 207, y + 291, "#111827", 3, True)
    line(lines, x + 45, y + 34, x + 90, y + 92, "#111827", 2, True)
    line(lines, x + 375, y + 92, x + 320, y + 92, "#111827", 2, True)
    text(lines, x + 215, y + 380, "Transformer Encoder", "label", "middle")


def update_manifest() -> None:
    manifest = ROOT / "10_figures" / "figure_manifest.tsv"
    if not manifest.exists():
        return
    paths = {
        "svg": (
            Path("submission") / "nature_communications_structdualrm" / "figures" / SVG_NAME,
            Path("10_figures") / "main_figures_svg" / SVG_NAME,
        ),
        "png": (
            Path("submission") / "nature_communications_structdualrm" / "figures_png" / PNG_NAME,
            Path("10_figures") / "main_figures_png" / PNG_NAME,
        ),
    }
    rows = []
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        for row in reader:
            if row.get("figure") == "Figure6_structdualrm_architecture" and row.get("format") in paths:
                source, project = paths[row["format"]]
                row["source_path"] = str(source).replace("/", "\\")
                row["project_path"] = str(project).replace("/", "\\")
                source_abs = ROOT / source
                if source_abs.exists():
                    row["size_bytes"] = str(source_abs.stat().st_size)
            rows.append(row)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_source_data() -> None:
    rows = [
        ("sequence", "encoding", "101 nt center-A sequence -> nucleotide IDs -> base/position/center-distance embeddings"),
        ("sequence", "transformer", "Embedding tensor is passed through a center-aware Transformer encoder"),
        ("sequence", "pooling", "z_sequence concatenates center state, local mean and center-aware attention pooling"),
        ("rnafold_graph", "encoding", "dot-bracket output is converted to paired flags and base-pair partner graph"),
        ("rnafold_graph", "message_passing", "GraphLayer uses self, backbone-neighbor and base-pair partner messages"),
        ("rnafold_scalar", "features", "center_paired, central paired fractions, paired_fraction_101nt, MFE_per_nt and loop_span_around_center"),
        ("context", "features", "region embedding, relative transcript position and log10 transcript length"),
        ("fusion", "representation", "z=[z_sequence,z_graph,z_struct,z_context]"),
        ("gates", "outputs", "label-specific gates produce m1A and m6A representations before two sigmoid heads"),
    ]
    out = PKG / "source_data" / "Figure6_model_architecture_source_data.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["panel_or_module", "field", "description"])
        writer.writerows(rows)


def build_svg() -> str:
    width, height = 2200, 1720
    seq_col = ["#0f766e", "#2dd4bf", "#99f6e4", "#ccfbf1"]
    graph_col = ["#1e3a8a", "#3159a8", "#93b3e6", "#dbeafe"]
    struct_col = ["#8a1025", "#cf2747", "#f2a6b5", "#fee2e2"]
    context_col = ["#8a5a00", "#f0b429", "#f8dd8f", "#fff7cc"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.title{font-size:30px;font-weight:700}.label{font-size:20px;font-weight:700}.small{font-size:18px}.tiny{font-size:14px}.formula{font-size:18px;font-family:Cambria Math,Times New Roman,serif}</style>',
        '<marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,10 L13,5 z" fill="#111827"/></marker>',
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    text(lines, 40, 48, "StructDualRM complete model architecture", "title")
    text(lines, 40, 78, "Input sequence and RNAfold structure are encoded into sequence, graph, scalar-structure and transcript-context representations before gated dual-label prediction.", "small")

    # Top input and RNAfold panels.
    panel(lines, 70, 115, 720, 450, "Sequence", "#ffffff", "#0f9a94")
    rna_strip(lines, 120, 175, 6, "m6A", "#0ea5a7")
    molecule_icon(lines, 335, 235, "m6A", "#0ea5a7")
    rna_strip(lines, 120, 365, 6, "m1A", "#f97316")
    molecule_icon(lines, 335, 425, "m1A", "#f97316")

    panel(lines, 1025, 115, 430, 450, "Secondary structure", "#ffffff", "#0f9a94")
    line(lines, 790, 235, 1025, 235, "#111827", 3, True)
    text(lines, 875, 215, "RNAfold", "label")
    line(lines, 790, 430, 1025, 430, "#111827", 3, True)
    text(lines, 875, 410, "RNAfold", "label")
    helix_icon(lines, 1110, 190, "#1b7f36")
    helix_icon(lines, 1110, 388, "#1d4ed8")
    text(lines, 1110, 510, "dot-bracket structure", "small")
    text(lines, 1170, 538, "MFE", "small")

    # Sequence encoding branch.
    panel(lines, 55, 620, 755, 835, "Sequence branch", "#f8fff4", "#5b8f3a")
    text(lines, 92, 665, "A A U C G G A", "label")
    line(lines, 270, 659, 330, 659, "#111827", 3, True)
    for i, val in enumerate(["1", "1", "4", "2", "3", "3", "1"]):
        rect(lines, 335 + i * 40, 635, 36, 36, "#c8e9b7", "#111827", 0, 2)
        text(lines, 353 + i * 40, 660, val, "label", "middle")
    line(lines, 480, 675, 480, 735, "#111827", 3, True)
    rect(lines, 285, 735, 390, 95, "#ffffff", "#5b8f3a", 14, 2)
    text(lines, 480, 770, "Embedding sum", "label", "middle")
    text(lines, 480, 800, "base + position + center-distance", "small", "middle")
    line(lines, 480, 830, 480, 895, "#111827", 3, True)
    matrix(lines, 245, 895, 12, 3, 34, seq_col)
    text(lines, 132, 930, "101 x d_model", "small")
    line(lines, 480, 998, 480, 1045, "#111827", 3, True)
    transformer_block(lines, 270, 1045)
    rect(lines, 620, 960, 160, 70, "#ffffff", "#5b8f3a", 12, 2)
    text(lines, 700, 988, "center-aware", "small", "middle")
    text(lines, 700, 1015, "pooling", "small", "middle")
    line(lines, 620, 995, 552, 1045, "#111827", 3, True)
    color_bar(lines, 835, 1125, seq_col * 2, 26, 34)
    text(lines, 1055, 1150, "z_sequence", "formula")
    line(lines, 700, 995, 905, 1125, "#111827", 3, True)

    # RNAfold graph branch.
    panel(lines, 850, 620, 365, 360, "Pair-aware graph", "#f8fbff", "#2454a6")
    graph_icon(lines, 935, 715, 0.85)
    rect(lines, 1010, 840, 140, 55, "#e8eefc", "#2454a6", 10, 2)
    text(lines, 1080, 873, "GraphLayer x2", "small", "middle")
    line(lines, 1010, 795, 1080, 840, "#111827", 3, True)
    color_bar(lines, 925, 925, graph_col * 2, 26, 34)
    text(lines, 1135, 950, "z_graph", "formula")
    line(lines, 1080, 895, 1080, 925, "#111827", 3, True)
    line(lines, 1240, 565, 1025, 620, "#111827", 3, True)

    # RNAfold scalar branch.
    panel(lines, 1265, 620, 390, 360, "Structural scalar", "#fff8fb", "#be123c")
    text(lines, 1300, 710, "..((...))..", "formula")
    text(lines, 1300, 742, "MFE = -3.20", "small")
    text(lines, 1300, 790, "6 RNAfold features", "small")
    small_mlp(lines, 1445, 700)
    text(lines, 1508, 680, "MLP", "label", "middle")
    color_bar(lines, 1345, 925, struct_col * 2, 26, 34)
    text(lines, 1558, 950, "z_struct", "formula")
    line(lines, 1505, 825, 1505, 925, "#111827", 3, True)
    line(lines, 1240, 565, 1460, 620, "#111827", 3, True)

    # Context branch.
    panel(lines, 850, 1035, 805, 180, "Transcript-context branch", "#fffdf0", "#b7791f")
    rect(lines, 900, 1080, 150, 65, "#fff7cc", "#b7791f", 10, 2)
    text(lines, 975, 1108, "Region", "small", "middle")
    text(lines, 975, 1134, "embedding", "small", "middle")
    rect(lines, 1085, 1080, 205, 65, "#fff7cc", "#b7791f", 10, 2)
    text(lines, 1188, 1108, "relative position", "small", "middle")
    text(lines, 1188, 1134, "log10 tx length", "small", "middle")
    line(lines, 1290, 1113, 1380, 1113, "#111827", 3, True)
    color_bar(lines, 1395, 1095, context_col * 2, 24, 34)
    text(lines, 1568, 1120, "z_context", "formula")

    # Fusion and outputs.
    rect(lines, 1705, 705, 430, 135, "#f8fafc", "#737373", 14, 3)
    color_bar(lines, 1735, 745, seq_col + graph_col + struct_col + context_col, 28, 42)
    text(lines, 1920, 830, "Fusion: z = [z_sequence | z_graph | z_struct | z_context]", "label", "middle")
    line(lines, 1050, 1142, 1705, 780, "#111827", 3, True)
    line(lines, 1135, 950, 1705, 760, "#111827", 3, True)
    line(lines, 1558, 950, 1705, 780, "#111827", 3, True)
    line(lines, 1568, 1120, 1705, 805, "#111827", 3, True)

    rect(lines, 1720, 920, 215, 112, "#fee2e2", "#be123c", 14, 3)
    text(lines, 1828, 960, "m1A gate", "label", "middle")
    text(lines, 1828, 992, "graph + struct", "small", "middle")
    rect(lines, 1720, 1110, 215, 112, "#dcfce7", "#15803d", 14, 3)
    text(lines, 1828, 1150, "m6A gate", "label", "middle")
    text(lines, 1828, 1182, "graph + struct", "small", "middle")
    line(lines, 1920, 840, 1828, 920, "#111827", 3, True)
    line(lines, 1920, 840, 1828, 1110, "#111827", 3, True)

    rect(lines, 1980, 920, 145, 112, "#ffffff", "#be123c", 14, 3)
    text(lines, 2052, 960, "m1A head", "label", "middle")
    text(lines, 2052, 992, "Sigmoid", "small", "middle")
    rect(lines, 1980, 1110, 145, 112, "#ffffff", "#15803d", 14, 3)
    text(lines, 2052, 1150, "m6A head", "label", "middle")
    text(lines, 2052, 1182, "Sigmoid", "small", "middle")
    line(lines, 1935, 976, 1980, 976, "#111827", 3, True)
    line(lines, 1935, 1166, 1980, 1166, "#111827", 3, True)
    text(lines, 1998, 1070, "P(m1A)", "label")
    text(lines, 1998, 1260, "P(m6A)", "label")

    # Compact legend.
    rect(lines, 55, 1540, 2080, 110, "#ffffff", "#cbd5e1", 12, 2)
    text(lines, 85, 1580, "What was added to complete the sketch:", "label")
    text(lines, 85, 1615, "1. center-aware pooling after Transformer; 2. transcript-context branch; 3. gated fusion with separate m1A/m6A heads; 4. explicit final probabilities.", "small")
    color_bar(lines, 85, 1632, seq_col, 22, 22)
    text(lines, 190, 1650, "sequence", "tiny")
    color_bar(lines, 330, 1632, graph_col, 22, 22)
    text(lines, 435, 1650, "graph", "tiny")
    color_bar(lines, 570, 1632, struct_col, 22, 22)
    text(lines, 675, 1650, "structure scalar", "tiny")
    color_bar(lines, 855, 1632, context_col, 22, 22)
    text(lines, 960, 1650, "context", "tiny")

    lines.append("</svg>\n")
    return "\n".join(lines)


def main() -> None:
    svg_path = PKG / "figures" / SVG_NAME
    png_path = PKG / "figures_png" / PNG_NAME
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(build_svg(), encoding="utf-8", newline="\n")
    render_svg_to_png(svg_path, png_path)

    targets = [
        SN_DRAFT / PNG_NAME,
        SN_DRAFT / SVG_NAME,
        ROOT / "10_figures" / "main_figures_png" / PNG_NAME,
        ROOT / "10_figures" / "main_figures_svg" / SVG_NAME,
        PKG / "flat_initial_submission" / PNG_NAME,
        PKG / "flat_initial_submission" / SVG_NAME,
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png_path if target.suffix.lower() == ".png" else svg_path, target)

    write_source_data()
    update_manifest()
    print({"svg": str(svg_path), "png": str(png_path), "copied": len(targets)})


if __name__ == "__main__":
    main()
