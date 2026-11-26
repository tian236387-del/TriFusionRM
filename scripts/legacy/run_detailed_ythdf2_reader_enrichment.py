from __future__ import annotations

import base64
import csv
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / "11_tables" / "biology_summary"
SITE_GROUPS = BIO / "dual_site_groups.tsv"
MEMBERSHIP = BIO / "transcript_group_membership.tsv"
READER_ANNOTATION = ROOT / "external" / "hg19_public" / "gse49339_ythdf2_reader_hg19.tsv"
FIG_PNG = ROOT / "10_figures" / "biology_reader_png"
FIG_SVG = ROOT / "10_figures" / "biology_reader_svg"
PKG = ROOT / "submission" / "nature_communications_structdualrm"

GREEN = ["#f0fdf4", "#dcfce7", "#bbf7d0", "#86efac", "#4ade80", "#22c55e", "#16a34a", "#15803d"]
TEXT = "#102016"
MUTED = "#4b6354"
GRID = "#b7e4c7"
PANEL = "#fbfffc"

SPATIAL_GROUPS = [
    ("m1A_CDS_prox25", "CDS", "prox25", "CDS prox25"),
    ("m1A_CDS_intermediate26_50", "CDS", "intermediate26_50", "CDS 26-50"),
    ("m1A_CDS_distal50", "CDS", "distal50", "CDS distal"),
    ("m1A_3UTR_prox25", "3UTR", "prox25", "3UTR prox25"),
    ("m1A_3UTR_intermediate26_50", "3UTR", "intermediate26_50", "3UTR 26-50"),
    ("m1A_3UTR_distal50", "3UTR", "distal50", "3UTR distal"),
    ("m1A_other_region_prox25", "5UTR/other", "prox25", "5UTR/other prox25"),
    ("m1A_other_region_intermediate26_50", "5UTR/other", "intermediate26_50", "5UTR/other 26-50"),
    ("m1A_other_region_distal50", "5UTR/other", "distal50", "5UTR/other distal"),
]
CONTEXT_GROUPS = [
    ("m1A_exact_overlap_m6A", "exact m1A-m6A overlap"),
    ("m1A_no_m6A_same_transcript", "m1A without same-tx m6A"),
    ("m6A_only_same_transcript", "m6A only same-tx"),
    ("matched_unmodified_A", "matched unmodified A"),
]
GROUP_LABELS = {group: label for group, _, _, label in SPATIAL_GROUPS}
GROUP_LABELS.update(CONTEXT_GROUPS)
DISTANCE_ORDER = [
    ("prox25", "<=25 nt"),
    ("intermediate26_50", "26-50 nt"),
    ("distal50", ">50 nt"),
]
REGION_ORDER = ["CDS", "3UTR", "5UTR/other"]
VIRTUAL_GROUPS = {
    "m1A_prox25_all": ["m1A_CDS_prox25", "m1A_3UTR_prox25", "m1A_other_region_prox25"],
    "m1A_intermediate26_50_all": [
        "m1A_CDS_intermediate26_50",
        "m1A_3UTR_intermediate26_50",
        "m1A_other_region_intermediate26_50",
    ],
    "m1A_distal50_all": ["m1A_CDS_distal50", "m1A_3UTR_distal50", "m1A_other_region_distal50"],
    "m1A_prox_or_intermediate_all": [
        "m1A_CDS_prox25",
        "m1A_3UTR_prox25",
        "m1A_other_region_prox25",
        "m1A_CDS_intermediate26_50",
        "m1A_3UTR_intermediate26_50",
        "m1A_other_region_intermediate26_50",
    ],
}
VIRTUAL_LABELS = {
    "m1A_prox25_all": "all m1A prox25",
    "m1A_intermediate26_50_all": "all m1A 26-50",
    "m1A_distal50_all": "all m1A distal",
    "m1A_prox_or_intermediate_all": "all m1A <=50 nt",
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


def normalize_symbol(value: str) -> str:
    return value.strip().upper()


def truthy(value: object) -> bool:
    return str(value).strip() in {"1", "true", "True", "yes", "Y"}


def row_value(row: dict[str, str], candidates: Iterable[str]) -> str:
    lower_map = {key.lower(): key for key in row}
    for key in candidates:
        if key in row and row[key]:
            return str(row[key])
        mapped = lower_map.get(key.lower())
        if mapped and row.get(mapped):
            return str(row[mapped])
    return ""


def is_ythdf2_m6a(row: dict[str, str]) -> bool:
    rbp = row_value(row, ["rbpName", "RBP", "rbp", "protein"]).upper().replace("-", "")
    mod = row_value(row, ["RNAmodType", "mod_type", "modification"]).upper().replace("-", "")
    return "YTHDF2" in rbp and "M6A" in mod


def load_reader_summary() -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for row in read_tsv(READER_ANNOTATION):
        if not is_ythdf2_m6a(row):
            continue
        gene = normalize_symbol(row_value(row, ["gene_name", "geneName", "Gene", "gene"]))
        if not gene:
            continue
        current = summary.setdefault(
            gene,
            {
                "reader_site_count": 0,
                "peak_regions": Counter(),
                "sources": set(),
                "chromosomes": set(),
            },
        )
        current["reader_site_count"] = int(current["reader_site_count"]) + 1
        region = row_value(row, ["Region", "region", "region_type"]) or "unknown"
        current["peak_regions"][region] += 1
        source = row_value(row, ["source", "Source"])
        if source:
            current["sources"].add(source)
        chrom = row_value(row, ["chr", "chrom", "chromosome"])
        if chrom:
            current["chromosomes"].add(chrom)
    return summary


def collapse_memberships(rows: list[dict[str, str]], key_field: str = "gene_name") -> list[dict[str, str]]:
    collapsed: dict[str, dict[str, str]] = {}
    for row in rows:
        key = normalize_symbol(row.get(key_field, ""))
        if not key:
            continue
        out = collapsed.setdefault(key, {key_field: key})
        for col, value in row.items():
            if col == key_field:
                continue
            if truthy(value):
                out[col] = "1"
            elif col not in out:
                out[col] = value
    return list(collapsed.values())


def build_entity_sets(membership_rows: list[dict[str, str]]) -> dict[str, set[str]]:
    collapsed = collapse_memberships(membership_rows)
    columns = [col for col in membership_rows[0].keys() if col not in {"transcript_id", "gene_id", "gene_name"}]
    sets: dict[str, set[str]] = {}
    for col in columns:
        sets[col] = {row["gene_name"] for row in collapsed if truthy(row.get(col))}
    for virtual, members in VIRTUAL_GROUPS.items():
        genes: set[str] = set()
        for member in members:
            genes.update(sets.get(member, set()))
        sets[virtual] = genes
    return sets


def fisher_exact(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    try:
        from scipy.stats import fisher_exact as scipy_fisher

        odds, p_value = scipy_fisher([[a, b], [c, d]])
        return float(odds), float(p_value)
    except Exception:
        n = a + b + c + d
        row1 = a + b
        col1 = a + c

        def log_choose(n_value: int, k_value: int) -> float:
            if k_value < 0 or k_value > n_value:
                return float("-inf")
            return math.lgamma(n_value + 1) - math.lgamma(k_value + 1) - math.lgamma(n_value - k_value + 1)

        def hyperprob(x: int) -> float:
            return math.exp(log_choose(col1, x) + log_choose(n - col1, row1 - x) - log_choose(n, row1))

        lo = max(0, row1 - (n - col1))
        hi = min(row1, col1)
        obs = hyperprob(a)
        p_value = sum(hyperprob(x) for x in range(lo, hi + 1) if hyperprob(x) <= obs + 1e-15)
        odds = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        return odds, min(1.0, p_value)


def bh_adjust(rows: list[dict[str, object]], key: str = "p_value") -> None:
    valid = [row for row in rows if not math.isnan(float(row[key]))]
    ranked = sorted(valid, key=lambda row: float(row[key]))
    total = len(ranked)
    running = 1.0
    for rank, row in reversed(list(enumerate(ranked, start=1))):
        running = min(running, float(row[key]) * total / rank)
        row["fdr_bh"] = running


def star(p_value: float) -> str:
    if math.isnan(p_value):
        return "NA"
    if p_value < 0.001:
        return "***"
    if p_value < 0.05:
        return "**"
    return "ns"


def color_scale(value: float, vmin: float, vmax: float) -> str:
    if math.isnan(value):
        return "#f1f5f9"
    if vmax <= vmin:
        return GREEN[3]
    t = max(0.0, min(0.999, (value - vmin) / (vmax - vmin)))
    return GREEN[int(t * len(GREEN))]


def fmt_fraction(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value * 100:.1f}%"


def fmt_float(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def make_group_summary(site_rows: list[dict[str, str]], reader_summary: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    group_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in site_rows:
        group_rows[row["site_group"]].append(row)

    meta = {group: ("spatial", region, distance, label) for group, region, distance, label in SPATIAL_GROUPS}
    for group, label in CONTEXT_GROUPS:
        meta[group] = ("context", "", "", label)
    reader_genes = set(reader_summary)
    rows: list[dict[str, object]] = []
    for group, rows_in_group in sorted(group_rows.items()):
        genes = {normalize_symbol(row.get("gene_name", "")) for row in rows_in_group if row.get("gene_name")}
        transcripts = {row.get("transcript_id", "") for row in rows_in_group if row.get("transcript_id")}
        positive_genes = genes & reader_genes
        reader_sites = [row for row in rows_in_group if normalize_symbol(row.get("gene_name", "")) in reader_genes]
        peak_counts = [int(reader_summary[gene]["reader_site_count"]) for gene in positive_genes]
        site_regions = Counter(row.get("region_type", "unknown") or "unknown" for row in rows_in_group)
        group_type, region, distance, label = meta.get(group, ("other", "", "", group))
        rows.append(
            {
                "group": group,
                "label": label,
                "group_type": group_type,
                "region": region,
                "distance_class": distance,
                "site_count": len(rows_in_group),
                "transcript_count": len(transcripts),
                "gene_count": len(genes),
                "reader_positive_gene_count": len(positive_genes),
                "reader_positive_gene_fraction": len(positive_genes) / len(genes) if genes else 0.0,
                "reader_supported_site_count_gene_level": len(reader_sites),
                "reader_supported_site_fraction_gene_level": len(reader_sites) / len(rows_in_group) if rows_in_group else 0.0,
                "reader_peak_count_total_linked_genes": sum(peak_counts),
                "median_reader_peak_count_positive_gene": median(peak_counts) if peak_counts else 0,
                "site_region_breakdown": ";".join(f"{key}:{value}" for key, value in sorted(site_regions.items())),
            }
        )
    return rows


def make_region_distance_summary(group_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted = {group for group, _, _, _ in SPATIAL_GROUPS}
    rows = [row for row in group_summary if row["group"] in wanted]
    rows.sort(key=lambda row: (REGION_ORDER.index(str(row["region"])), [d for d, _ in DISTANCE_ORDER].index(str(row["distance_class"]))))
    return rows


def comparison_pairs() -> list[tuple[str, str, str, str]]:
    return [
        ("m1A_3UTR_prox25", "m1A_3UTR_distal50", "3UTR prox25 vs distal", "regional_distance"),
        ("m1A_3UTR_intermediate26_50", "m1A_3UTR_distal50", "3UTR 26-50 vs distal", "regional_distance"),
        ("m1A_CDS_prox25", "m1A_CDS_distal50", "CDS prox25 vs distal", "regional_distance"),
        ("m1A_CDS_intermediate26_50", "m1A_CDS_distal50", "CDS 26-50 vs distal", "regional_distance"),
        ("m1A_other_region_prox25", "m1A_other_region_distal50", "5UTR/other prox25 vs distal", "regional_distance"),
        ("m1A_other_region_intermediate26_50", "m1A_other_region_distal50", "5UTR/other 26-50 vs distal", "regional_distance"),
        ("m1A_prox25_all", "m1A_distal50_all", "all m1A prox25 vs distal", "pooled_distance"),
        ("m1A_prox_or_intermediate_all", "m1A_distal50_all", "all m1A <=50 nt vs distal", "pooled_distance"),
        ("m1A_exact_overlap_m6A", "m1A_no_m6A_same_transcript", "exact overlap vs m1A no same-tx m6A", "dual_context"),
        ("m1A_exact_overlap_m6A", "m6A_only_same_transcript", "exact overlap vs m6A only same-tx", "dual_context"),
        ("m1A_3UTR_prox25", "m6A_only_same_transcript", "3UTR prox25 vs m6A only same-tx", "cross_context"),
        ("m1A_CDS_prox25", "m6A_only_same_transcript", "CDS prox25 vs m6A only same-tx", "cross_context"),
        ("m1A_no_m6A_same_transcript", "m6A_only_same_transcript", "m1A no same-tx m6A vs m6A only", "cross_context"),
    ]


def make_enrichment_rows(entity_sets: dict[str, set[str]], reader_summary: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    reader_positive = set(reader_summary)
    labels = dict(GROUP_LABELS)
    labels.update(VIRTUAL_LABELS)
    rows: list[dict[str, object]] = []
    for left, right, comparison, comparison_type in comparison_pairs():
        left_entities = set(entity_sets.get(left, set()))
        right_entities = set(entity_sets.get(right, set())) - left_entities
        left_pos = len(left_entities & reader_positive)
        right_pos = len(right_entities & reader_positive)
        left_total = len(left_entities)
        right_total = len(right_entities)
        left_neg = left_total - left_pos
        right_neg = right_total - right_pos
        odds, p_value = fisher_exact(left_pos, left_neg, right_pos, right_neg)
        left_fraction = left_pos / left_total if left_total else 0.0
        right_fraction = right_pos / right_total if right_total else 0.0
        rows.append(
            {
                "comparison": comparison,
                "comparison_type": comparison_type,
                "left_group": left,
                "right_group": right,
                "left_label": labels.get(left, left),
                "right_label": labels.get(right, right),
                "left_total_genes": left_total,
                "right_total_genes_nonoverlap": right_total,
                "left_reader_positive_genes": left_pos,
                "right_reader_positive_genes": right_pos,
                "left_reader_fraction": left_fraction,
                "right_reader_fraction": right_fraction,
                "risk_difference": left_fraction - right_fraction,
                "odds_ratio": odds,
                "log2_odds_ratio": math.log2(odds) if odds > 0 and not math.isinf(odds) else float("nan"),
                "p_value": p_value,
                "fdr_bh": 1.0,
                "significance": star(p_value),
            }
        )
    bh_adjust(rows)
    for row in rows:
        row["significance"] = star(float(row["fdr_bh"]))
    return rows


def make_peak_region_rows(group_summary: list[dict[str, object]], reader_summary: dict[str, dict[str, object]], site_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups = {str(row["group"]) for row in group_summary}
    genes_by_group: dict[str, set[str]] = defaultdict(set)
    for row in site_rows:
        group = row["site_group"]
        if group not in groups:
            continue
        gene = normalize_symbol(row.get("gene_name", ""))
        if gene:
            genes_by_group[group].add(gene)

    out: list[dict[str, object]] = []
    for group, genes in sorted(genes_by_group.items()):
        totals: Counter[str] = Counter()
        for gene in genes:
            summary = reader_summary.get(gene)
            if not summary:
                continue
            totals.update(summary["peak_regions"])
        total_peaks = sum(totals.values())
        for region in sorted(totals):
            out.append(
                {
                    "group": group,
                    "label": GROUP_LABELS.get(group, group),
                    "reader_peak_region": region,
                    "reader_peak_count": totals[region],
                    "total_linked_reader_peaks": total_peaks,
                    "fraction": totals[region] / total_peaks if total_peaks else 0.0,
                }
            )
    return out


def make_gene_overlap_rows(site_rows: list[dict[str, str]], reader_summary: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    groups_by_gene: dict[str, set[str]] = defaultdict(set)
    sites_by_gene: Counter[str] = Counter()
    transcripts_by_gene: dict[str, set[str]] = defaultdict(set)
    for row in site_rows:
        gene = normalize_symbol(row.get("gene_name", ""))
        if not gene:
            continue
        groups_by_gene[gene].add(row["site_group"])
        sites_by_gene[gene] += 1
        transcript = row.get("transcript_id", "")
        if transcript:
            transcripts_by_gene[gene].add(transcript)

    rows: list[dict[str, object]] = []
    for gene, summary in sorted(reader_summary.items()):
        local_groups = groups_by_gene.get(gene, set())
        if not local_groups:
            continue
        region_counter: Counter[str] = summary["peak_regions"]
        rows.append(
            {
                "gene_name": gene,
                "reader_site_count": summary["reader_site_count"],
                "reader_peak_regions": ";".join(f"{key}:{value}" for key, value in sorted(region_counter.items())),
                "local_site_count": sites_by_gene[gene],
                "local_transcript_count": len(transcripts_by_gene[gene]),
                "positive_group_count": len(local_groups),
                "positive_groups": ";".join(sorted(local_groups)),
            }
        )
    rows.sort(key=lambda row: (-int(row["reader_site_count"]), -int(row["positive_group_count"]), str(row["gene_name"])))
    return rows


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=14, fill=PANEL, outline=GRID, width=2)
    x1, y1, _, _ = box
    draw.text((x1 + 24, y1 + 16), title, font=font(24, True), fill=TEXT)


def draw_wrapped_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, fnt: ImageFont.ImageFont, fill: str) -> None:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if text_size(draw, candidate, fnt)[0] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += 23


def make_heatmap(draw: ImageDraw.ImageDraw, region_rows: list[dict[str, object]]) -> None:
    x0, y0, w, h = 48, 122, 690, 430
    draw_panel(draw, (x0, y0, x0 + w, y0 + h), "A  YTHDF2-positive fraction by region and distance")
    by_key = {(str(row["region"]), str(row["distance_class"])): row for row in region_rows}
    cell_w, cell_h = 162, 84
    bx, by = x0 + 160, y0 + 100
    values = [float(row["reader_positive_gene_fraction"]) for row in region_rows]
    vmin, vmax = min(values), max(values)
    for c, (distance, label) in enumerate(DISTANCE_ORDER):
        draw.text((bx + c * cell_w + 20, by - 38), label, font=font(15, True), fill=MUTED)
    for r, region in enumerate(REGION_ORDER):
        draw.text((x0 + 30, by + r * cell_h + 28), region, font=font(17, True), fill=TEXT)
        for c, (distance, _) in enumerate(DISTANCE_ORDER):
            row = by_key.get((region, distance))
            value = float(row["reader_positive_gene_fraction"]) if row else float("nan")
            x = bx + c * cell_w
            y = by + r * cell_h
            draw.rounded_rectangle([x, y, x + cell_w - 12, y + cell_h - 12], radius=8, fill=color_scale(value, vmin, vmax), outline="#74c69d", width=1)
            if row:
                draw.text((x + 34, y + 18), fmt_fraction(value), font=font(22, True), fill=TEXT)
                draw.text((x + 22, y + 51), f"{row['reader_positive_gene_count']}/{row['gene_count']} genes", font=font(12), fill=MUTED)
    draw.text((x0 + 30, y0 + h - 42), "Values use gene-level YTHDF2 PAR-CLIP support linked to local m1A/m6A groups.", font=font(13), fill=MUTED)


def make_or_panel(draw: ImageDraw.ImageDraw, enrichment_rows: list[dict[str, object]]) -> None:
    x0, y0, w, h = 770, 122, 982, 590
    draw_panel(draw, (x0, y0, x0 + w, y0 + h), "B  Gene-level YTHDF2 enrichment contrasts")
    selected_names = [
        "3UTR prox25 vs distal",
        "3UTR 26-50 vs distal",
        "CDS prox25 vs distal",
        "CDS 26-50 vs distal",
        "5UTR/other prox25 vs distal",
        "all m1A prox25 vs distal",
        "all m1A <=50 nt vs distal",
        "exact overlap vs m1A no same-tx m6A",
        "3UTR prox25 vs m6A only same-tx",
        "CDS prox25 vs m6A only same-tx",
    ]
    rows_by_name = {str(row["comparison"]): row for row in enrichment_rows}
    rows = [rows_by_name[name] for name in selected_names if name in rows_by_name]
    plot_x, plot_y = x0 + 352, y0 + 78
    plot_w, plot_h = 520, 430
    min_log, max_log = -0.55, 2.1

    def sx(log_value: float) -> int:
        return plot_x + int((max(min_log, min(max_log, log_value)) - min_log) / (max_log - min_log) * plot_w)

    for tick in [-0.5, 0, 0.5, 1.0, 1.5, 2.0]:
        tx = sx(tick)
        draw.line([tx, plot_y - 12, tx, plot_y + plot_h], fill="#d8f3dc", width=1)
        label = "1" if tick == 0 else f"{2 ** tick:.1f}"
        draw.text((tx - 12, plot_y + plot_h + 10), label, font=font(12), fill=MUTED)
    draw.line([sx(0), plot_y - 18, sx(0), plot_y + plot_h], fill="#40916c", width=2)
    for idx, row in enumerate(rows):
        y = plot_y + 18 + idx * 40
        odds = float(row["odds_ratio"])
        log_value = math.log2(odds) if odds > 0 and not math.isinf(odds) else max_log
        x = sx(log_value)
        draw.text((x0 + 26, y - 10), str(row["comparison"])[:43], font=font(13), fill=TEXT)
        draw.line([sx(0), y, x, y], fill="#52b788", width=8)
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill="#1b9e77", outline="#0b3d2e", width=1)
        draw.text((plot_x + plot_w + 16, y - 12), f"OR {odds:.2f} {row['significance']}", font=font(13, True), fill=TEXT)
    draw.text((plot_x + 190, y0 + h - 42), "odds ratio, log2-scaled axis", font=font(14, True), fill=MUTED)
    draw.text((x0 + 26, y0 + h - 42), "*** FDR < 0.001; ** FDR < 0.05", font=font(13), fill=MUTED)


def make_bar_panel(draw: ImageDraw.ImageDraw, group_summary: list[dict[str, object]]) -> None:
    x0, y0, w, h = 48, 590, 690, 545
    draw_panel(draw, (x0, y0, x0 + w, y0 + h), "C  Reader-supported burden in key local groups")
    rows_by_group = {str(row["group"]): row for row in group_summary}
    groups = [
        "m1A_exact_overlap_m6A",
        "m1A_3UTR_prox25",
        "m1A_CDS_prox25",
        "m1A_other_region_prox25",
        "m1A_no_m6A_same_transcript",
        "m6A_only_same_transcript",
        "matched_unmodified_A",
    ]
    bar_x, bar_y, bar_w = x0 + 255, y0 + 88, 360
    for idx, group in enumerate(groups):
        row = rows_by_group.get(group)
        if not row:
            continue
        y = bar_y + idx * 58
        gene_frac = float(row["reader_positive_gene_fraction"])
        site_frac = float(row["reader_supported_site_fraction_gene_level"])
        draw.text((x0 + 26, y - 4), str(row["label"])[:29], font=font(13), fill=TEXT)
        draw.rounded_rectangle([bar_x, y, bar_x + bar_w, y + 18], radius=5, fill="#eafaf1", outline="#d8f3dc")
        draw.rounded_rectangle([bar_x, y, bar_x + int(bar_w * min(gene_frac, 0.45) / 0.45), y + 18], radius=5, fill="#2d6a4f")
        draw.rounded_rectangle([bar_x, y + 24, bar_x + bar_w, y + 42], radius=5, fill="#eafaf1", outline="#d8f3dc")
        draw.rounded_rectangle([bar_x, y + 24, bar_x + int(bar_w * min(site_frac, 0.45) / 0.45), y + 42], radius=5, fill="#74c69d")
        draw.text((bar_x + bar_w + 12, y - 2), fmt_fraction(gene_frac), font=font(12, True), fill=TEXT)
        draw.text((bar_x + bar_w + 12, y + 22), fmt_fraction(site_frac), font=font(12), fill=MUTED)
    draw.rectangle([x0 + 30, y0 + h - 74, x0 + 46, y0 + h - 58], fill="#2d6a4f")
    draw.text((x0 + 54, y0 + h - 78), "positive genes", font=font(12), fill=TEXT)
    draw.rectangle([x0 + 190, y0 + h - 74, x0 + 206, y0 + h - 58], fill="#74c69d")
    draw.text((x0 + 214, y0 + h - 78), "sites inherited from gene-level support", font=font(12), fill=TEXT)


def make_peak_region_panel(draw: ImageDraw.ImageDraw, peak_rows: list[dict[str, object]]) -> None:
    x0, y0, w, h = 770, 750, 982, 385
    draw_panel(draw, (x0, y0, x0 + w, y0 + h), "D  YTHDF2 PAR-CLIP peak-region composition")
    selected = [
        "m1A_3UTR_prox25",
        "m1A_CDS_prox25",
        "m1A_other_region_prox25",
        "m1A_exact_overlap_m6A",
        "m6A_only_same_transcript",
    ]
    rows_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in peak_rows:
        rows_by_group[str(row["group"])].append(row)
    colors = {
        "3UTR": "#1b9e77",
        "CDS": "#74c69d",
        "5UTR": "#b7e4c7",
        "unknown": "#d8f3dc",
        "Intron": "#95d5b2",
        "Other": "#40916c",
    }
    bar_x, bar_y, bar_w = x0 + 260, y0 + 78, 590
    legend_items: list[str] = []
    for idx, group in enumerate(selected):
        rows = rows_by_group.get(group, [])
        if not rows:
            continue
        y = bar_y + idx * 47
        draw.text((x0 + 28, y + 5), GROUP_LABELS.get(group, group)[:31], font=font(13), fill=TEXT)
        x = bar_x
        total = sum(int(row["reader_peak_count"]) for row in rows)
        for row in sorted(rows, key=lambda item: str(item["reader_peak_region"])):
            region = str(row["reader_peak_region"])
            frac = float(row["fraction"])
            seg_w = int(bar_w * frac)
            color = colors.get(region, "#52b788")
            draw.rectangle([x, y, x + seg_w, y + 26], fill=color, outline="#f8fffb")
            x += seg_w
            if region not in legend_items:
                legend_items.append(region)
        draw.rectangle([bar_x, y, bar_x + bar_w, y + 26], outline="#2d6a4f", width=1)
        draw.text((bar_x + bar_w + 12, y + 3), f"{total} peaks", font=font(12, True), fill=MUTED)
    lx, ly = x0 + 28, y0 + h - 38
    for idx, region in enumerate(legend_items[:6]):
        x = lx + idx * 130
        draw.rectangle([x, ly, x + 16, ly + 16], fill=colors.get(region, "#52b788"))
        draw.text((x + 22, ly - 2), region, font=font(12), fill=TEXT)


def make_note_panel(draw: ImageDraw.ImageDraw) -> None:
    x0, y0, w, h = 48, 1172, 1704, 158
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=14, fill="#f6fff8", outline=GRID, width=2)
    draw.text((x0 + 26, y0 + 22), "Interpretation boundary", font=font(22, True), fill=TEXT)
    note = (
        "YTHDF2 evidence comes from the local hg19-compatible GSE49339 PAR-CLIP annotation table. "
        "Because this prepared public table is linked by gene name, enrichment tests are gene-level Fisher exact tests; "
        "site-level bars inherit reader support from the annotated gene and should be interpreted as reader-supported burden, not base-resolution binding."
    )
    draw_wrapped_text(draw, (x0 + 26, y0 + 60), note, 1600, font(16), MUTED)


def make_png(
    group_summary: list[dict[str, object]],
    region_rows: list[dict[str, object]],
    enrichment_rows: list[dict[str, object]],
    peak_rows: list[dict[str, object]],
    out: Path,
) -> None:
    width, height = 1800, 1380
    canvas = Image.new("RGBA", (width, height), "#f0fdf4")
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 30), "Detailed YTHDF2 reader enrichment in m1A-m6A spatial groups", font=font(34, True), fill=TEXT)
    draw.text(
        (48, 74),
        "Green palette; hg19 YTHDF2 PAR-CLIP support is evaluated against local StructDualRM m1A/m6A group definitions.",
        font=font(18),
        fill=MUTED,
    )
    make_heatmap(draw, region_rows)
    make_or_panel(draw, enrichment_rows)
    make_bar_panel(draw, group_summary)
    make_peak_region_panel(draw, peak_rows)
    make_note_panel(draw)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out)


def make_svg_wrapper(png_path: Path, svg_path: Path) -> None:
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1380" viewBox="0 0 1800 1380">\n'
        f'<image href="data:image/png;base64,{data}" x="0" y="0" width="1800" height="1380"/>\n'
        "</svg>\n"
    )
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8", newline="\n")


def update_figure_manifest(entries: list[tuple[str, str, Path, Path]]) -> None:
    manifest = ROOT / "10_figures" / "figure_manifest.tsv"
    fields = ["figure", "format", "source_path", "project_path", "size_bytes"]
    existing: list[dict[str, str]] = []
    if manifest.exists():
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle, delimiter="\t"))
    remove = {(figure, fmt) for figure, fmt, _, _ in entries}
    kept = [row for row in existing if (row.get("figure", ""), row.get("format", "")) not in remove]
    for figure, fmt, submission_path, project_path in entries:
        kept.append(
            {
                "figure": figure,
                "format": fmt,
                "source_path": str(submission_path.relative_to(ROOT)),
                "project_path": str(project_path.relative_to(ROOT)),
                "size_bytes": project_path.stat().st_size if project_path.exists() else 0,
            }
        )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)


def main() -> None:
    site_rows = read_tsv(SITE_GROUPS)
    membership_rows = read_tsv(MEMBERSHIP)
    reader_summary = load_reader_summary()
    entity_sets = build_entity_sets(membership_rows)

    group_summary = make_group_summary(site_rows, reader_summary)
    region_distance = make_region_distance_summary(group_summary)
    enrichment = make_enrichment_rows(entity_sets, reader_summary)
    peak_regions = make_peak_region_rows(group_summary, reader_summary, site_rows)
    gene_overlap = make_gene_overlap_rows(site_rows, reader_summary)

    write_tsv(
        BIO / "ythdf2_reader_group_summary.tsv",
        group_summary,
        [
            "group",
            "label",
            "group_type",
            "region",
            "distance_class",
            "site_count",
            "transcript_count",
            "gene_count",
            "reader_positive_gene_count",
            "reader_positive_gene_fraction",
            "reader_supported_site_count_gene_level",
            "reader_supported_site_fraction_gene_level",
            "reader_peak_count_total_linked_genes",
            "median_reader_peak_count_positive_gene",
            "site_region_breakdown",
        ],
    )
    write_tsv(
        BIO / "ythdf2_reader_region_distance_summary.tsv",
        region_distance,
        [
            "group",
            "label",
            "group_type",
            "region",
            "distance_class",
            "site_count",
            "transcript_count",
            "gene_count",
            "reader_positive_gene_count",
            "reader_positive_gene_fraction",
            "reader_supported_site_count_gene_level",
            "reader_supported_site_fraction_gene_level",
            "reader_peak_count_total_linked_genes",
            "median_reader_peak_count_positive_gene",
            "site_region_breakdown",
        ],
    )
    write_tsv(
        BIO / "ythdf2_reader_detailed_enrichment.tsv",
        enrichment,
        [
            "comparison",
            "comparison_type",
            "left_group",
            "right_group",
            "left_label",
            "right_label",
            "left_total_genes",
            "right_total_genes_nonoverlap",
            "left_reader_positive_genes",
            "right_reader_positive_genes",
            "left_reader_fraction",
            "right_reader_fraction",
            "risk_difference",
            "odds_ratio",
            "log2_odds_ratio",
            "p_value",
            "fdr_bh",
            "significance",
        ],
    )
    write_tsv(
        BIO / "ythdf2_reader_peak_region_by_group.tsv",
        peak_regions,
        ["group", "label", "reader_peak_region", "reader_peak_count", "total_linked_reader_peaks", "fraction"],
    )
    write_tsv(
        BIO / "ythdf2_reader_gene_overlap_summary.tsv",
        gene_overlap,
        ["gene_name", "reader_site_count", "reader_peak_regions", "local_site_count", "local_transcript_count", "positive_group_count", "positive_groups"],
    )

    png = FIG_PNG / "Figure_ythdf2_reader_enrichment_green.png"
    svg = FIG_SVG / "Figure_ythdf2_reader_enrichment_green.svg"
    make_png(group_summary, region_distance, enrichment, peak_regions, png)
    make_svg_wrapper(png, svg)

    copies = [
        (png, PKG / "figures_png" / png.name),
        (svg, PKG / "figures" / svg.name),
        (BIO / "ythdf2_reader_group_summary.tsv", PKG / "source_data" / "Figure_ythdf2_reader_group_summary.tsv"),
        (BIO / "ythdf2_reader_region_distance_summary.tsv", PKG / "source_data" / "Figure_ythdf2_reader_region_distance_summary.tsv"),
        (BIO / "ythdf2_reader_detailed_enrichment.tsv", PKG / "source_data" / "Figure_ythdf2_reader_detailed_enrichment.tsv"),
        (BIO / "ythdf2_reader_peak_region_by_group.tsv", PKG / "source_data" / "Figure_ythdf2_reader_peak_region_by_group.tsv"),
        (BIO / "ythdf2_reader_gene_overlap_summary.tsv", PKG / "source_data" / "Figure_ythdf2_reader_gene_overlap_summary.tsv"),
    ]
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    update_figure_manifest(
        [
            ("Figure_ythdf2_reader_enrichment_green", "svg", PKG / "figures" / svg.name, svg),
            ("Figure_ythdf2_reader_enrichment_green", "png", PKG / "figures_png" / png.name, png),
        ]
    )
    print(
        {
            "ythdf2_annotated_genes": len(reader_summary),
            "groups": len(group_summary),
            "comparisons": len(enrichment),
            "gene_overlap_rows": len(gene_overlap),
            "png": str(png),
        }
    )


if __name__ == "__main__":
    main()
