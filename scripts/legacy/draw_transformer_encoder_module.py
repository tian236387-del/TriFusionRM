from __future__ import annotations

import csv
import html
import shutil
from pathlib import Path

from create_sn_template_manuscript import render_svg_to_png


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "submission" / "nature_communications_structdualrm"
SVG_NAME = "Transformer_center_aware_encoder.svg"
PNG_NAME = "Transformer_center_aware_encoder.png"


def esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def add_text(lines: list[str], x: float, y: float, text: str, cls: str = "small", anchor: str = "start") -> None:
    lines.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{esc(text)}</text>')


def rect(lines: list[str], x: int, y: int, w: int, h: int, fill: str, stroke: str = "#315a82", rx: int = 10, sw: int = 2) -> None:
    lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def line(lines: list[str], x1: int, y1: int, x2: int, y2: int, color: str = "#1f2937", width: int = 3, arrow: bool = False, dashed: bool = False) -> None:
    dash = ' stroke-dasharray="8 7"' if dashed else ""
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash}{marker}/>')


def circle(lines: list[str], x: int, y: int, r: int, fill: str, stroke: str = "#315a82", sw: int = 2) -> None:
    lines.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def matrix(lines: list[str], x: int, y: int, cols: int = 5, rows: int = 4, cell: int = 15) -> None:
    colors = ["#b7d8f2", "#cdecc5", "#fee2e2", "#fce7a4", "#d8c7f1"]
    for r in range(rows):
        for c in range(cols):
            fill = colors[(r + c) % len(colors)]
            rect(lines, x + c * cell, y + r * cell, cell - 1, cell - 1, fill, "#ffffff", 1, 1)


def tiny_sequence(lines: list[str], x: int, y: int) -> None:
    seq = "G G A C U"
    add_text(lines, x, y, seq, "mono")
    circle(lines, x + 55, y - 8, 17, "#ffffff", "#0ea5a7", 3)
    add_text(lines, x + 55, y - 1, "A", "small", "middle")
    line(lines, x + 55, y + 12, x + 55, y + 45, "#0ea5a7", 2, True, True)


def embedding_card(lines: list[str], x: int, y: int, title: str, subtitle: str, fill: str) -> None:
    rect(lines, x, y, 150, 115, fill, "#477299", 10, 2)
    matrix(lines, x + 26, y + 18, 5, 3, 17)
    add_text(lines, x + 75, y + 82, title, "small", "middle")
    add_text(lines, x + 75, y + 108, subtitle, "tiny", "middle")


def transformer_sublayer(lines: list[str], x: int, y: int, title: str, subtitle: str, fill: str) -> None:
    rect(lines, x, y, 205, 92, fill, "#315a82", 10, 2)
    add_text(lines, x + 102, y + 36, title, "small", "middle")
    add_text(lines, x + 102, y + 66, subtitle, "tiny", "middle")


def build_svg() -> str:
    width, height = 1320, 920
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.title{font-size:34px;font-weight:700}.subtitle{font-size:25px;font-weight:700}.small{font-size:22px}.tiny{font-size:16px}.micro{font-size:13px}.mono{font-family:Consolas,monospace;font-size:22px;font-weight:700}.formula{font-family:Cambria Math,Times New Roman,serif;font-size:22px}</style>',
        '<marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,10 L13,5 z" fill="#1f2937"/></marker>',
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    add_text(lines, 40, 55, "Center-aware Transformer Encoder", "title")
    add_text(lines, 40, 88, "Sequence branch used in StructDualRM", "small")
    add_text(lines, 1150, 88, "101 nt", "subtitle")

    # Outer module body.
    rect(lines, 45, 120, 1225, 785, "#dff0ff", "#315a82", 28, 5)

    # Input strip.
    rect(lines, 95, 165, 1080, 70, "#f8fbff", "#9ab8d5", 14, 2)
    add_text(lines, 122, 208, "5'-", "mono")
    tiny_sequence(lines, 190, 208)
    add_text(lines, 332, 208, "C U U G A A C C U G A ...", "mono")
    add_text(lines, 1110, 208, "-3'", "mono")

    # Embedding sum area.
    rect(lines, 95, 270, 1080, 220, "#eef7ff", "#7aa6ca", 22, 3)
    add_text(lines, 635, 315, "Embedding Sum", "subtitle", "middle")
    embedding_card(lines, 155, 345, "Base", "Emb", "#d8edf9")
    add_text(lines, 335, 412, "+", "title", "middle")
    embedding_card(lines, 380, 345, "Position", "Emb", "#fbe3df")
    add_text(lines, 560, 412, "+", "title", "middle")
    embedding_card(lines, 605, 345, "Center-Dist", "Emb", "#fde8bf")
    add_text(lines, 805, 412, "=", "title", "middle")
    rect(lines, 860, 345, 250, 115, "#e5f6df", "#79a96f", 10, 2)
    matrix(lines, 900, 370, 8, 4, 16)
    add_text(lines, 985, 448, "e_ij", "formula", "middle")
    add_text(lines, 635, 518, "e_ij = E_base(s_ij) + E_pos(j) + E_dist(|j - j0|),  j0 = 51", "formula", "middle")

    # Transformer block.
    rect(lines, 95, 565, 1080, 185, "#edf6ff", "#7aa6ca", 22, 3)
    add_text(lines, 635, 608, "Transformer Encoder Layer", "subtitle", "middle")
    transformer_sublayer(lines, 150, 635, "Multi-head", "self-attention", "#f6e2df")
    line(lines, 355, 681, 430, 681, "#1f2937", 3, True)
    transformer_sublayer(lines, 440, 635, "Add & Norm", "residual", "#e5f6df")
    line(lines, 645, 681, 720, 681, "#1f2937", 3, True)
    transformer_sublayer(lines, 730, 635, "Feed-forward", "GELU + dropout", "#fef3c7")
    line(lines, 935, 681, 1010, 681, "#1f2937", 3, True)
    transformer_sublayer(lines, 1020, 635, "Add & Norm", "residual", "#e5f6df")
    add_text(lines, 635, 778, "d_model = 64, heads = 4, layers = 1, dropout = 0.2", "small", "middle")

    # Downstream center-aware pooling.
    line(lines, 635, 750, 635, 805, "#1f2937", 4, True)
    rect(lines, 185, 805, 900, 82, "#ffffff", "#315a82", 18, 3)
    add_text(lines, 635, 838, "Center-aware pooling", "subtitle", "middle")
    add_text(lines, 635, 870, "z_seq = [h_center ; mean(h_center +/- 5 nt) ; sum_j a_ij h_ij]", "formula", "middle")

    # Internal arrows and center focus.
    line(lines, 985, 460, 985, 565, "#64748b", 3, True)
    line(lines, 250, 235, 250, 345, "#64748b", 2, True)
    line(lines, 455, 235, 455, 345, "#64748b", 2, True)
    line(lines, 680, 235, 680, 345, "#64748b", 2, True)
    add_text(lines, 100, 540, "Input tokens: A/C/G/T(U) -> integer IDs", "tiny")
    add_text(lines, 885, 540, "trainable embedding lookup", "tiny")
    add_text(lines, 1100, 835, "output to fusion", "tiny")
    line(lines, 1085, 845, 1190, 845, "#1f2937", 3, True)
    rect(lines, 1195, 816, 72, 58, "#e8f7f6", "#4d8d8a", 10, 2)
    add_text(lines, 1231, 852, "z_seq", "small", "middle")

    lines.append("</svg>\n")
    return "\n".join(lines)


def write_source_data() -> None:
    rows = [
        ("input", "length", "101 nt sequence window centered on candidate adenine"),
        ("embedding", "base", "A/C/G/T(U) token IDs are mapped by a trainable base embedding"),
        ("embedding", "position", "Absolute position j in the 101 nt window is represented by a trainable position embedding"),
        ("embedding", "center_distance", "Distance to the central adenine |j-j0| is represented by a trainable center-distance embedding"),
        ("transformer", "configuration", "d_model=64; heads=4; layers=1; dropout=0.2; GELU feed-forward"),
        ("pooling", "center", "Center hidden state h_center is retained"),
        ("pooling", "local", "Mean hidden state across center +/- 5 nt is retained"),
        ("pooling", "attention", "Center-aware attention computes sum_j a_ij h_ij"),
        ("output", "sequence_representation", "z_seq concatenates center, local mean and center-aware attention pooled vectors"),
    ]
    out = PKG / "source_data" / "Transformer_center_aware_encoder_source_data.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["module", "field", "description"])
        writer.writerows(rows)


def main() -> None:
    svg_path = ROOT / "10_figures" / "model_modules_svg" / SVG_NAME
    png_path = ROOT / "10_figures" / "model_modules_png" / PNG_NAME
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(build_svg(), encoding="utf-8", newline="\n")
    render_svg_to_png(svg_path, png_path)

    for target in [
        PKG / "figures" / SVG_NAME,
        PKG / "figures_png" / PNG_NAME,
        PKG / "sn_template_draft" / PNG_NAME,
        PKG / "sn_template_draft" / SVG_NAME,
        PKG / "flat_initial_submission" / PNG_NAME,
        PKG / "flat_initial_submission" / SVG_NAME,
    ]:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png_path if target.suffix.lower() == ".png" else svg_path, target)

    write_source_data()
    print({"svg": str(svg_path), "png": str(png_path)})


if __name__ == "__main__":
    main()
