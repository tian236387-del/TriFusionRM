from __future__ import annotations

import base64
import csv
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / "11_tables" / "biology_summary"
GO = BIO / "go_enrichment.tsv"
PATHWAY = BIO / "kegg_reactome_enrichment.tsv"
FIG_PNG = ROOT / "10_figures" / "biology_function_png"
FIG_SVG = ROOT / "10_figures" / "biology_function_svg"
PKG = ROOT / "submission" / "nature_communications_structdualrm"

RED = ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"]
TEXT = "#2b1111"
MUTED = "#6f3b3b"
GRID = "#efc0b6"
PANEL = "#fffafa"

GROUP_LABELS = {
    "m1A_3UTR_prox25": "3'UTR prox25",
    "m1A_3UTR_distal50": "3'UTR distal",
    "m1A_CDS_prox25": "CDS prox25",
    "m1A_CDS_distal50": "CDS distal",
    "m1A_exact_overlap_m6A": "exact overlap",
    "m6A_only_same_transcript": "m6A only same-tx",
}
GROUP_ORDER = [
    "m1A_3UTR_prox25",
    "m1A_3UTR_distal50",
    "m1A_CDS_prox25",
    "m1A_CDS_distal50",
    "m1A_exact_overlap_m6A",
    "m6A_only_same_transcript",
]
LIB_LABELS = {
    "GO_Biological_Process_2023": "GO BP",
    "KEGG_2021_Human": "KEGG",
    "Reactome_2022": "Reactome",
}
LIB_ORDER = ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"]

THEME_ORDER = [
    "RNA translation/processing",
    "Metabolism and acidification",
    "Membrane/trafficking/cytoskeleton",
    "Signalling and cell fate",
    "Immune/infection/stress",
    "Other",
]
THEME_COLORS = {
    "RNA translation/processing": "#99000d",
    "Metabolism and acidification": "#cb181d",
    "Membrane/trafficking/cytoskeleton": "#ef3b2c",
    "Signalling and cell fate": "#fb6a4a",
    "Immune/infection/stress": "#fc9272",
    "Other": "#fee0d2",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def fnum(value: object, default: float = float("nan")) -> float:
    try:
        value_text = str(value).strip()
        if not value_text:
            return default
        return float(value_text)
    except Exception:
        return default


def clean_term(term: str) -> str:
    term = re.sub(r"\s+R-HSA-\d+$", "", term)
    term = re.sub(r"\s+\(GO:\d+\)$", "", term)
    return term.strip()


def short_term(term: str, max_len: int = 56) -> str:
    term = clean_term(term)
    if len(term) <= max_len:
        return term
    return term[: max_len - 1].rstrip() + "..."


def classify_theme(term: str) -> str:
    t = clean_term(term).lower()
    if any(
        word in t
        for word in [
            "translation",
            "ribosome",
            "ribosomal",
            "peptide chain",
            "srp-dependent",
            "protein targeting",
            "nonsense mediated decay",
            "nmd",
            "transcription",
            "chromatin",
            "histone",
            "nucleic acid",
            "dna",
            "rna",
            "mrna",
            "gene expression",
            "polymerase",
            "telomere",
            "splicing",
            "sumoylation",
        ]
    ):
        return "RNA translation/processing"
    if any(word in t for word in ["metabolism", "glycolysis", "gluconeogenesis", "pyruvate", "tca", "thiamine", "folate", "fatty acid", "amino acid", "nitrogen", "digestion", "absorption", "acidification", "ph", "proton"]):
        return "Metabolism and acidification"
    if any(word in t for word in ["membrane", "vesicle", "golgi", "endocytosis", "phagosome", "lysosome", "trafficking", "actin", "cytoskeleton", "rho gtpase", "extracellular matrix", "collagen", "migration", "junction"]):
        return "Membrane/trafficking/cytoskeleton"
    if any(word in t for word in ["signaling", "signalling", "receptor", "notch", "wnt", "hippo", "pten", "cell fate", "differentiation", "muscle contraction", "hemostasis", "platelet", "hormone", "growth"]):
        return "Signalling and cell fate"
    if any(word in t for word in ["infection", "virus", "bacterial", "vibrio", "salmonella", "tuberculosis", "sars", "cancer", "carcinoma", "glioma", "immune", "inflammatory", "apoptosis", "autophagy", "stress"]):
        return "Immune/infection/stress"
    return "Other"


def score_from_fdr(fdr: float) -> float:
    if math.isnan(fdr):
        return 0.0
    return -math.log10(max(fdr, 1e-12))


def color_scale(value: float, vmin: float, vmax: float) -> str:
    if math.isnan(value):
        return "#f8f1f0"
    if vmax <= vmin:
        return RED[3]
    t = max(0.0, min(0.999, (value - vmin) / (vmax - vmin)))
    return RED[int(t * len(RED))]


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline="#efd7d2", width=2)
    draw.text((x0 + 22, y0 + 16), title, font=font(23, True), fill=TEXT)


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], max_chars: int, fill: str, size: int = 18) -> None:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font(size), fill=fill)
        y += size + 6


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in [GO, PATHWAY]:
        for row in read_tsv(path):
            fdr = fnum(row.get("fdr_bh"))
            p_value = fnum(row.get("p_value"))
            overlap = int(fnum(row.get("overlap_count"), 0))
            target_size = int(fnum(row.get("target_size"), 0))
            term_size = int(fnum(row.get("term_size_in_background"), 0))
            background = int(fnum(row.get("background_size"), 0))
            gene_ratio = overlap / target_size if target_size else 0.0
            bg_ratio = term_size / background if background else 0.0
            enrichment_ratio = gene_ratio / bg_ratio if bg_ratio else float("nan")
            rows.append(
                {
                    **row,
                    "term_clean": clean_term(row["term"]),
                    "term_short": short_term(row["term"]),
                    "library_label": LIB_LABELS.get(row["library"], row["library"]),
                    "group_label": GROUP_LABELS.get(row["group"], row["group"]),
                    "theme": classify_theme(row["term"]),
                    "p_value_float": p_value,
                    "fdr_float": fdr,
                    "score_neg_log10_fdr": score_from_fdr(fdr),
                    "overlap_count_int": overlap,
                    "target_size_int": target_size,
                    "term_size_int": term_size,
                    "background_size_int": background,
                    "gene_ratio": gene_ratio,
                    "background_gene_ratio": bg_ratio,
                    "enrichment_ratio": enrichment_ratio,
                }
            )
    return rows


def make_group_library_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["group"]), str(row["library"]))].append(row)
    out: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        for library in LIB_ORDER:
            items = by_key.get((group, library), [])
            if not items:
                out.append(
                    {
                        "group": group,
                        "group_label": GROUP_LABELS[group],
                        "library": library,
                        "library_label": LIB_LABELS[library],
                        "term_count": 0,
                        "nominal_p_lt_0_05_terms": 0,
                        "fdr_lt_0_25_terms": 0,
                        "fdr_lt_0_10_terms": 0,
                        "fdr_lt_0_05_terms": 0,
                        "best_fdr": "",
                        "best_score_neg_log10_fdr": 0,
                        "median_score_neg_log10_fdr": 0,
                        "top_term": "",
                        "top_theme": "",
                        "unique_overlap_gene_count": 0,
                    }
                )
                continue
            items = sorted(items, key=lambda r: (float(r["fdr_float"]), float(r["p_value_float"]), -int(r["overlap_count_int"])))
            scores = [float(r["score_neg_log10_fdr"]) for r in items]
            genes = set()
            for row in items:
                genes.update(g for g in str(row.get("overlap_genes", "")).split(";") if g)
            best = items[0]
            out.append(
                {
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "library": library,
                    "library_label": LIB_LABELS[library],
                    "term_count": len(items),
                    "nominal_p_lt_0_05_terms": sum(float(r["p_value_float"]) < 0.05 for r in items),
                    "fdr_lt_0_25_terms": sum(float(r["fdr_float"]) < 0.25 for r in items),
                    "fdr_lt_0_10_terms": sum(float(r["fdr_float"]) < 0.10 for r in items),
                    "fdr_lt_0_05_terms": sum(float(r["fdr_float"]) < 0.05 for r in items),
                    "best_fdr": best["fdr_float"],
                    "best_score_neg_log10_fdr": best["score_neg_log10_fdr"],
                    "median_score_neg_log10_fdr": median(scores),
                    "top_term": best["term_clean"],
                    "top_theme": best["theme"],
                    "unique_overlap_gene_count": len(genes),
                }
            )
    return out


def make_top_terms(rows: list[dict[str, object]], per_group_library: int = 8) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["group"]), str(row["library"]))].append(row)
    out: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        for library in LIB_ORDER:
            items = sorted(
                by_key.get((group, library), []),
                key=lambda r: (float(r["fdr_float"]), float(r["p_value_float"]), -int(r["overlap_count_int"]), str(r["term_clean"])),
            )
            for rank, row in enumerate(items[:per_group_library], start=1):
                out.append(
                    {
                        "group": row["group"],
                        "group_label": row["group_label"],
                        "library": row["library"],
                        "library_label": row["library_label"],
                        "rank_within_group_library": rank,
                        "term": row["term_clean"],
                        "term_short": row["term_short"],
                        "theme": row["theme"],
                        "overlap_count": row["overlap_count_int"],
                        "target_size": row["target_size_int"],
                        "term_size_in_background": row["term_size_int"],
                        "background_size": row["background_size_int"],
                        "gene_ratio": row["gene_ratio"],
                        "enrichment_ratio": row["enrichment_ratio"],
                        "p_value": row["p_value_float"],
                        "fdr_bh": row["fdr_float"],
                        "score_neg_log10_fdr": row["score_neg_log10_fdr"],
                        "overlap_genes": row.get("overlap_genes", ""),
                    }
                )
    return out


def make_theme_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["group"]), str(row["library"]), str(row["theme"]))].append(row)
    out: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        for library in LIB_ORDER:
            for theme in THEME_ORDER:
                items = by_key.get((group, library, theme), [])
                if not items:
                    out.append(
                        {
                            "group": group,
                            "group_label": GROUP_LABELS[group],
                            "library": library,
                            "library_label": LIB_LABELS[library],
                            "theme": theme,
                            "term_count": 0,
                            "fdr_lt_0_25_terms": 0,
                            "best_fdr": "",
                            "best_score_neg_log10_fdr": 0,
                            "median_score_neg_log10_fdr": 0,
                            "unique_overlap_gene_count": 0,
                            "representative_terms": "",
                        }
                    )
                    continue
                items = sorted(items, key=lambda r: (float(r["fdr_float"]), float(r["p_value_float"])))
                genes = set()
                for row in items:
                    genes.update(g for g in str(row.get("overlap_genes", "")).split(";") if g)
                scores = [float(r["score_neg_log10_fdr"]) for r in items]
                out.append(
                    {
                        "group": group,
                        "group_label": GROUP_LABELS[group],
                        "library": library,
                        "library_label": LIB_LABELS[library],
                        "theme": theme,
                        "term_count": len(items),
                        "fdr_lt_0_25_terms": sum(float(r["fdr_float"]) < 0.25 for r in items),
                        "best_fdr": items[0]["fdr_float"],
                        "best_score_neg_log10_fdr": items[0]["score_neg_log10_fdr"],
                        "median_score_neg_log10_fdr": median(scores),
                        "unique_overlap_gene_count": len(genes),
                        "representative_terms": "; ".join(str(r["term_clean"]) for r in items[:3]),
                    }
                )
    return out


def make_proximal_distal_contrasts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(str(row["group"]), str(row["library"]), str(row["term_clean"])): row for row in rows}
    pairs = [
        ("3UTR prox25 vs distal", "m1A_3UTR_prox25", "m1A_3UTR_distal50"),
        ("CDS prox25 vs distal", "m1A_CDS_prox25", "m1A_CDS_distal50"),
    ]
    out: list[dict[str, object]] = []
    for comparison, left_group, right_group in pairs:
        for library in LIB_ORDER:
            terms = {
                term
                for group in [left_group, right_group]
                for g, lib, term in by_key
                if g == group and lib == library
            }
            for term in sorted(terms):
                left = by_key.get((left_group, library, term))
                right = by_key.get((right_group, library, term))
                left_score = float(left["score_neg_log10_fdr"]) if left else 0.0
                right_score = float(right["score_neg_log10_fdr"]) if right else 0.0
                left_ratio = float(left["enrichment_ratio"]) if left else float("nan")
                right_ratio = float(right["enrichment_ratio"]) if right else float("nan")
                out.append(
                    {
                        "comparison": comparison,
                        "library": library,
                        "library_label": LIB_LABELS[library],
                        "term": term,
                        "term_short": short_term(term, 62),
                        "theme": classify_theme(term),
                        "left_group": left_group,
                        "right_group": right_group,
                        "left_label": GROUP_LABELS[left_group],
                        "right_label": GROUP_LABELS[right_group],
                        "left_overlap_count": int(left["overlap_count_int"]) if left else 0,
                        "right_overlap_count": int(right["overlap_count_int"]) if right else 0,
                        "left_fdr_bh": left["fdr_float"] if left else "",
                        "right_fdr_bh": right["fdr_float"] if right else "",
                        "left_score_neg_log10_fdr": left_score,
                        "right_score_neg_log10_fdr": right_score,
                        "score_delta_left_minus_right": left_score - right_score,
                        "left_enrichment_ratio": left_ratio if not math.isnan(left_ratio) else "",
                        "right_enrichment_ratio": right_ratio if not math.isnan(right_ratio) else "",
                        "enrichment_ratio_delta_left_minus_right": (left_ratio - right_ratio) if not math.isnan(left_ratio) and not math.isnan(right_ratio) else "",
                    }
                )
    out.sort(key=lambda r: (-float(r["score_delta_left_minus_right"]), str(r["comparison"]), str(r["library"]), str(r["term"])))
    return out


def make_representative_terms(top_terms: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_term: dict[str, dict[str, object]] = {}
    for row in top_terms:
        key = str(row["term"])
        score = float(row["score_neg_log10_fdr"])
        if key not in best_by_term or score > float(best_by_term[key]["score_neg_log10_fdr"]):
            best_by_term[key] = row
    selected = sorted(best_by_term.values(), key=lambda r: (-float(r["score_neg_log10_fdr"]), str(r["term"])))[:18]
    return selected


def make_heatmap_panel(draw: ImageDraw.ImageDraw, summary: list[dict[str, object]]) -> None:
    x0, y0, w, h = 45, 95, 860, 440
    draw_panel(draw, (x0, y0, x0 + w, y0 + h), "A  Library-level enrichment strength")
    data = {(r["group"], r["library"]): r for r in summary}
    max_score = max(float(r["best_score_neg_log10_fdr"]) for r in summary) if summary else 1.0
    left = x0 + 190
    top = y0 + 85
    cell_w = 190
    cell_h = 48
    for j, lib in enumerate(LIB_ORDER):
        draw.text((left + j * cell_w + 35, top - 38), LIB_LABELS[lib], font=font(18, True), fill=TEXT)
    for i, group in enumerate(GROUP_ORDER):
        y = top + i * cell_h
        draw.text((x0 + 25, y + 12), GROUP_LABELS[group], font=font(17), fill=TEXT)
        for j, lib in enumerate(LIB_ORDER):
            x = left + j * cell_w
            row = data.get((group, lib))
            score = float(row["best_score_neg_log10_fdr"]) if row else 0.0
            fill = color_scale(min(score, 8.0), 0.0, min(max_score, 8.0))
            draw.rounded_rectangle((x, y, x + cell_w - 12, y + cell_h - 8), radius=8, fill=fill, outline="#f0c9c1")
            count = int(row["fdr_lt_0_25_terms"]) if row else 0
            best = f"{float(row['best_fdr']):.1e}" if row and row["best_fdr"] != "" else "NA"
            draw.text((x + 12, y + 8), f"{count} terms", font=font(15, True), fill="#2b0b0b")
            draw.text((x + 96, y + 10), f"FDR {best}", font=font(13), fill="#3b1515")
    draw_wrapped(
        draw,
        "Cells show the strongest local ORA signal per group/library; counts indicate top-table terms with FDR < 0.25.",
        (x0 + 25, y0 + h - 55),
        92,
        MUTED,
        15,
    )


def make_bubble_panel(draw: ImageDraw.ImageDraw, selected_terms: list[dict[str, object]], top_terms: list[dict[str, object]]) -> None:
    x0, y0, w, h = 940, 95, 810, 440
    draw_panel(draw, (x0, y0, x0 + w, y0 + h), "B  Representative GO/KEGG/Reactome terms")
    selected = selected_terms[:12]
    term_order = [str(row["term"]) for row in selected]
    by_group_term: dict[tuple[str, str], dict[str, object]] = {}
    for row in top_terms:
        key = (str(row["group"]), str(row["term"]))
        if key not in by_group_term or float(row["score_neg_log10_fdr"]) > float(by_group_term[key]["score_neg_log10_fdr"]):
            by_group_term[key] = row
    left = x0 + 230
    top = y0 + 78
    col_w = 96
    row_h = 25
    for i, group in enumerate(GROUP_ORDER):
        label = GROUP_LABELS[group].replace(" ", "\n")
        for k, part in enumerate(label.split("\n")[:2]):
            draw.text((left + i * col_w + 5, top - 46 + k * 17), part, font=font(13, True), fill=TEXT)
    max_score = max((float(row["score_neg_log10_fdr"]) for row in top_terms), default=1.0)
    max_overlap = max((int(row["overlap_count"]) for row in top_terms), default=1)
    for j, term in enumerate(term_order):
        y = top + j * row_h
        draw.text((x0 + 24, y - 2), short_term(term, 30), font=font(13), fill=TEXT)
        for i, group in enumerate(GROUP_ORDER):
            row = by_group_term.get((group, term))
            cx = left + i * col_w + 42
            cy = y + 8
            draw.line((cx - 28, cy, cx + 28, cy), fill="#f4ddd8", width=1)
            if not row:
                continue
            score = min(float(row["score_neg_log10_fdr"]), 8.0)
            radius = 5 + 14 * math.sqrt(int(row["overlap_count"]) / max_overlap)
            fill = color_scale(score, 0.0, min(max_score, 8.0))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill, outline="#7f0c12", width=1)
    draw.text((x0 + 28, y0 + h - 58), "Bubble size = overlap genes; color = -log10(FDR)", font=font(15), fill=MUTED)
    for idx, r in enumerate([7, 13, 19]):
        cx = x0 + 455 + idx * 58
        cy = y0 + h - 49
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#fb6a4a", outline="#7f0c12")
        draw.text((cx - 12, cy + 22), str([3, 10, 25][idx]), font=font(12), fill=MUTED)


def make_theme_panel(draw: ImageDraw.ImageDraw, theme_summary: list[dict[str, object]]) -> None:
    x0, y0, w, h = 45, 575, 860, 470
    draw_panel(draw, (x0, y0, x0 + w, y0 + h), "C  Functional theme composition")
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in theme_summary:
        grouped[str(row["group"])][str(row["theme"])] += int(row["fdr_lt_0_25_terms"])
    left = x0 + 190
    top = y0 + 82
    bar_w = 600
    bar_h = 30
    for i, group in enumerate(GROUP_ORDER):
        y = top + i * 52
        draw.text((x0 + 25, y + 5), GROUP_LABELS[group], font=font(17), fill=TEXT)
        total = sum(grouped[group].values())
        if total == 0:
            draw.rounded_rectangle((left, y, left + bar_w, y + bar_h), radius=8, fill="#f8ebe8", outline="#f2d3cc")
            draw.text((left + 10, y + 6), "no FDR < 0.25 terms", font=font(14), fill=MUTED)
            continue
        x = left
        for theme in THEME_ORDER:
            count = grouped[group][theme]
            if not count:
                continue
            width = max(2, int(bar_w * count / total))
            draw.rectangle((x, y, x + width, y + bar_h), fill=THEME_COLORS[theme])
            if width > 42:
                draw.text((x + 5, y + 6), str(count), font=font(13, True), fill="white" if theme != "Other" else TEXT)
            x += width
        draw.rounded_rectangle((left, y, left + bar_w, y + bar_h), radius=8, outline="#7f0c12", width=1)
        draw.text((left + bar_w + 12, y + 6), f"n={total}", font=font(14), fill=MUTED)
    lx, ly = x0 + 25, y0 + h - 100
    for i, theme in enumerate(THEME_ORDER):
        col = i % 2
        row = i // 2
        px = lx + col * 410
        py = ly + row * 28
        draw.rectangle((px, py + 4, px + 18, py + 20), fill=THEME_COLORS[theme])
        draw.text((px + 26, py + 1), theme, font=font(14), fill=TEXT)


def make_contrast_panel(draw: ImageDraw.ImageDraw, contrasts: list[dict[str, object]]) -> None:
    x0, y0, w, h = 940, 575, 810, 470
    draw_panel(draw, (x0, y0, x0 + w, y0 + h), "D  Proximal versus distal term contrasts")
    candidates = [
        r
        for r in contrasts
        if float(r["score_delta_left_minus_right"]) > 0
        and float(r["left_score_neg_log10_fdr"]) >= 0.8
        and str(r["library"]) in {"GO_Biological_Process_2023", "Reactome_2022", "KEGG_2021_Human"}
    ]
    selected = sorted(candidates, key=lambda r: (-float(r["score_delta_left_minus_right"]), str(r["term"])))[:12]
    left = x0 + 300
    top = y0 + 80
    axis_w = 420
    row_h = 28
    max_delta = max((float(r["score_delta_left_minus_right"]) for r in selected), default=1.0)
    draw.line((left, top - 12, left, top + row_h * len(selected) + 5), fill=GRID, width=2)
    draw.line((left, top + row_h * len(selected) + 5, left + axis_w, top + row_h * len(selected) + 5), fill=GRID, width=2)
    for tick in [0, 0.5, 1.0]:
        x = left + int(axis_w * tick)
        draw.line((x, top + row_h * len(selected) + 2, x, top + row_h * len(selected) + 12), fill=GRID, width=2)
        draw.text((x - 10, top + row_h * len(selected) + 18), f"{tick * max_delta:.1f}", font=font(12), fill=MUTED)
    for i, row in enumerate(selected):
        y = top + i * row_h
        label = f"{row['comparison'].split()[0]} | {row['library_label']} | {short_term(str(row['term']), 28)}"
        draw.text((x0 + 25, y - 2), label, font=font(13), fill=TEXT)
        delta = float(row["score_delta_left_minus_right"])
        x = left + int(axis_w * delta / max_delta)
        draw.line((left, y + 9, x, y + 9), fill="#fc9272", width=5)
        draw.ellipse((x - 7, y + 2, x + 7, y + 16), fill="#99000d", outline="#5b0008")
        draw.text((x + 10, y), f"{delta:.2f}", font=font(12), fill=MUTED)
    draw.text((x0 + 25, y0 + h - 52), "Delta = proximal - distal in -log10(FDR); larger values indicate more proximal-skewed functional signal.", font=font(14), fill=MUTED)


def make_note_panel(draw: ImageDraw.ImageDraw) -> None:
    x0, y0, w, h = 45, 1075, 1705, 225
    draw_panel(draw, (x0, y0, x0 + w, y0 + h), "Interpretation boundary")
    text = (
        "This red-panel analysis refines the existing local GO/KEGG/Reactome ORA output. "
        "It uses transcript-level gene groups and previously generated local enrichment tables, "
        "so it should be read as hypothesis-generating functional context rather than primary causal evidence. "
        "The strongest manuscript claims should still rest on held-out model performance, same-dataset benchmarking, "
        "YTHDF2 reader concordance and hg19 external annotation support."
    )
    draw_wrapped(draw, text, (x0 + 25, y0 + 70), 165, MUTED, 18)


def make_png(
    summary: list[dict[str, object]],
    top_terms: list[dict[str, object]],
    theme_summary: list[dict[str, object]],
    contrasts: list[dict[str, object]],
    representative_terms: list[dict[str, object]],
    out: Path,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1800, 1330
    img = Image.new("RGB", (width, height), "#fffdfc")
    draw = ImageDraw.Draw(img)
    draw.text((45, 28), "Detailed GO/KEGG/Reactome enrichment in StructDualRM spatial gene groups", font=font(34, True), fill=TEXT)
    draw.text(
        (45, 66),
        "Red palette summary of transcript-level ORA terms from local GO Biological Process, KEGG and Reactome outputs",
        font=font(18),
        fill=MUTED,
    )
    make_heatmap_panel(draw, summary)
    make_bubble_panel(draw, representative_terms, top_terms)
    make_theme_panel(draw, theme_summary)
    make_contrast_panel(draw, contrasts)
    make_note_panel(draw)
    img.save(out, dpi=(300, 300))


def make_svg_wrapper(png: Path, svg: Path) -> None:
    svg.parent.mkdir(parents=True, exist_ok=True)
    encoded = base64.b64encode(png.read_bytes()).decode("ascii")
    with Image.open(png) as img:
        width, height = img.size
    svg.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<image width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>\n'
        "</svg>\n",
        encoding="utf-8",
    )


def update_figure_manifest(entries: list[tuple[str, str, Path, Path]]) -> None:
    manifest = ROOT / "10_figures" / "figure_manifest.tsv"
    existing: list[dict[str, str]] = []
    if manifest.exists():
        existing = read_tsv(manifest)
    key_to_row = {(row["figure"], row["format"]): row for row in existing if "figure" in row and "format" in row}
    for figure, fmt, source_path, project_path in entries:
        key_to_row[(figure, fmt)] = {
            "figure": figure,
            "format": fmt,
            "source_path": str(source_path.relative_to(ROOT)),
            "project_path": str(project_path.relative_to(ROOT)),
            "size_bytes": str(project_path.stat().st_size if project_path.exists() else ""),
        }
    write_tsv(manifest, list(key_to_row.values()), ["figure", "format", "source_path", "project_path", "size_bytes"])


def main() -> None:
    rows = load_rows()
    summary = make_group_library_summary(rows)
    top_terms = make_top_terms(rows)
    theme_summary = make_theme_summary(rows)
    contrasts = make_proximal_distal_contrasts(rows)
    representative_terms = make_representative_terms(top_terms)

    write_tsv(
        BIO / "gene_set_function_group_library_summary.tsv",
        summary,
        [
            "group",
            "group_label",
            "library",
            "library_label",
            "term_count",
            "nominal_p_lt_0_05_terms",
            "fdr_lt_0_25_terms",
            "fdr_lt_0_10_terms",
            "fdr_lt_0_05_terms",
            "best_fdr",
            "best_score_neg_log10_fdr",
            "median_score_neg_log10_fdr",
            "top_term",
            "top_theme",
            "unique_overlap_gene_count",
        ],
    )
    write_tsv(
        BIO / "gene_set_function_detailed_top_terms.tsv",
        top_terms,
        [
            "group",
            "group_label",
            "library",
            "library_label",
            "rank_within_group_library",
            "term",
            "term_short",
            "theme",
            "overlap_count",
            "target_size",
            "term_size_in_background",
            "background_size",
            "gene_ratio",
            "enrichment_ratio",
            "p_value",
            "fdr_bh",
            "score_neg_log10_fdr",
            "overlap_genes",
        ],
    )
    write_tsv(
        BIO / "gene_set_function_theme_summary.tsv",
        theme_summary,
        [
            "group",
            "group_label",
            "library",
            "library_label",
            "theme",
            "term_count",
            "fdr_lt_0_25_terms",
            "best_fdr",
            "best_score_neg_log10_fdr",
            "median_score_neg_log10_fdr",
            "unique_overlap_gene_count",
            "representative_terms",
        ],
    )
    write_tsv(
        BIO / "gene_set_function_proximal_distal_contrast.tsv",
        contrasts,
        [
            "comparison",
            "library",
            "library_label",
            "term",
            "term_short",
            "theme",
            "left_group",
            "right_group",
            "left_label",
            "right_label",
            "left_overlap_count",
            "right_overlap_count",
            "left_fdr_bh",
            "right_fdr_bh",
            "left_score_neg_log10_fdr",
            "right_score_neg_log10_fdr",
            "score_delta_left_minus_right",
            "left_enrichment_ratio",
            "right_enrichment_ratio",
            "enrichment_ratio_delta_left_minus_right",
        ],
    )
    write_tsv(
        BIO / "gene_set_function_representative_terms.tsv",
        representative_terms,
        [
            "group",
            "group_label",
            "library",
            "library_label",
            "rank_within_group_library",
            "term",
            "term_short",
            "theme",
            "overlap_count",
            "target_size",
            "term_size_in_background",
            "background_size",
            "gene_ratio",
            "enrichment_ratio",
            "p_value",
            "fdr_bh",
            "score_neg_log10_fdr",
            "overlap_genes",
        ],
    )

    png = FIG_PNG / "Figure_GO_KEGG_Reactome_detail_red.png"
    svg = FIG_SVG / "Figure_GO_KEGG_Reactome_detail_red.svg"
    make_png(summary, top_terms, theme_summary, contrasts, representative_terms, png)
    make_svg_wrapper(png, svg)

    copies = [
        (png, PKG / "figures_png" / png.name),
        (svg, PKG / "figures" / svg.name),
        (BIO / "gene_set_function_group_library_summary.tsv", PKG / "source_data" / "Figure_GO_KEGG_Reactome_group_library_summary.tsv"),
        (BIO / "gene_set_function_detailed_top_terms.tsv", PKG / "source_data" / "Figure_GO_KEGG_Reactome_detailed_top_terms.tsv"),
        (BIO / "gene_set_function_theme_summary.tsv", PKG / "source_data" / "Figure_GO_KEGG_Reactome_theme_summary.tsv"),
        (BIO / "gene_set_function_proximal_distal_contrast.tsv", PKG / "source_data" / "Figure_GO_KEGG_Reactome_proximal_distal_contrast.tsv"),
        (BIO / "gene_set_function_representative_terms.tsv", PKG / "source_data" / "Figure_GO_KEGG_Reactome_representative_terms.tsv"),
    ]
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    update_figure_manifest(
        [
            ("Figure_GO_KEGG_Reactome_detail_red", "svg", PKG / "figures" / svg.name, svg),
            ("Figure_GO_KEGG_Reactome_detail_red", "png", PKG / "figures_png" / png.name, png),
        ]
    )
    print(
        {
            "input_rows": len(rows),
            "group_library_rows": len(summary),
            "top_term_rows": len(top_terms),
            "theme_rows": len(theme_summary),
            "contrast_rows": len(contrasts),
            "png": str(png),
        }
    )


if __name__ == "__main__":
    main()
