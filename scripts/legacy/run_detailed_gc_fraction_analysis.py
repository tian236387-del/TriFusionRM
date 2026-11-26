from __future__ import annotations

import base64
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / "11_tables" / "biology_summary"
FEATURES = BIO / "sequence_structure_site_features.tsv"
FIG_PNG = ROOT / "10_figures" / "biology_gc_png"
FIG_SVG = ROOT / "10_figures" / "biology_gc_svg"
PKG = ROOT / "submission" / "nature_communications_structdualrm"

REGIONS = ["CDS", "3UTR", "5UTR"]
DISTANCES = [
    ("prox25", "proximal <=25 nt"),
    ("intermediate26_50", "26-50 nt"),
    ("distal50", "distal >50 nt"),
]
SPATIAL_GROUPS = [
    ("m1A_CDS_prox25", "CDS", "prox25", "CDS prox25"),
    ("m1A_CDS_intermediate26_50", "CDS", "intermediate26_50", "CDS 26-50"),
    ("m1A_CDS_distal50", "CDS", "distal50", "CDS distal"),
    ("m1A_3UTR_prox25", "3UTR", "prox25", "3UTR prox25"),
    ("m1A_3UTR_intermediate26_50", "3UTR", "intermediate26_50", "3UTR 26-50"),
    ("m1A_3UTR_distal50", "3UTR", "distal50", "3UTR distal"),
    ("m1A_other_region_prox25", "5UTR", "prox25", "5UTR prox25"),
    ("m1A_other_region_intermediate26_50", "5UTR", "intermediate26_50", "5UTR 26-50"),
    ("m1A_other_region_distal50", "5UTR", "distal50", "5UTR distal"),
]
CONTEXT_GROUPS = [
    ("m1A_exact_overlap_m6A", "exact overlap"),
    ("m1A_no_m6A_same_transcript", "m1A no same-tx m6A"),
    ("m6A_only_same_transcript", "m6A only same-tx"),
    ("matched_unmodified_A", "matched unmodified A"),
]
GC_BINS = [
    ("low", 0.0, 0.45),
    ("mid", 0.45, 0.60),
    ("high", 0.60, 0.75),
    ("very_high", 0.75, 1.01),
]
YELLOW = ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#d95f0e"]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
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


def safe_float(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except ValueError:
        return None


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def sem(values: list[float]) -> float:
    return sd(values) / math.sqrt(len(values)) if values else 0.0


def mann_whitney(values_a: list[float], values_b: list[float]) -> tuple[float, float, float]:
    if not values_a or not values_b:
        return float("nan"), float("nan"), float("nan")
    labelled = [(x, 0) for x in values_a] + [(x, 1) for x in values_b]
    labelled.sort(key=lambda item: item[0])
    ranks = [0.0] * len(labelled)
    i = 0
    while i < len(labelled):
        j = i + 1
        while j < len(labelled) and labelled[j][0] == labelled[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    rank_a = sum(rank for rank, item in zip(ranks, labelled) if item[1] == 0)
    n1, n2 = len(values_a), len(values_b)
    u = rank_a - n1 * (n1 + 1) / 2.0
    delta = 2.0 * u / (n1 * n2) - 1.0
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    p = math.erfc(abs((u - mu) / sigma) / math.sqrt(2.0)) if sigma else 1.0
    return u, delta, p


def bh_adjust(rows: list[dict[str, object]], key: str = "p_value") -> None:
    valid = [row for row in rows if not math.isnan(float(row[key]))]
    ranked = sorted(valid, key=lambda row: float(row[key]))
    m = len(ranked)
    running = 1.0
    for rank, row in reversed(list(enumerate(ranked, start=1))):
        running = min(running, float(row[key]) * m / rank)
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
        return YELLOW[2]
    t = max(0.0, min(0.999, (value - vmin) / (vmax - vmin)))
    idx = int(t * len(YELLOW))
    return YELLOW[idx]


def gc_bin(value: float) -> str:
    for label, lo, hi in GC_BINS:
        if lo <= value < hi:
            return label
    return "very_high"


def load_gc_values() -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in read_tsv(FEATURES):
        gc = safe_float(row.get("gc_fraction_101nt", ""))
        if gc is None:
            continue
        values[row["site_group"]].append(gc)
    return values


def make_summary(values_by_group: dict[str, list[float]]) -> list[dict[str, object]]:
    labels = {group: label for group, _, _, label in SPATIAL_GROUPS}
    labels.update(CONTEXT_GROUPS)
    rows = []
    for group, values in values_by_group.items():
        if not values:
            continue
        label = labels.get(group, group)
        rows.append(
            {
                "group": group,
                "label": label,
                "n": len(values),
                "mean_gc": mean(values),
                "median_gc": median(values),
                "sd_gc": sd(values),
                "sem_gc": sem(values),
                "q1_gc": quantile(values, 0.25),
                "q3_gc": quantile(values, 0.75),
                "p10_gc": quantile(values, 0.10),
                "p90_gc": quantile(values, 0.90),
                "high_gc_fraction_ge_0_60": sum(1 for x in values if x >= 0.60) / len(values),
                "very_high_gc_fraction_ge_0_75": sum(1 for x in values if x >= 0.75) / len(values),
            }
        )
    rows.sort(key=lambda row: (str(row["group"])))
    return rows


def make_region_distance_summary(values_by_group: dict[str, list[float]]) -> list[dict[str, object]]:
    rows = []
    for group, region, distance, label in SPATIAL_GROUPS:
        values = values_by_group.get(group, [])
        if not values:
            continue
        rows.append(
            {
                "group": group,
                "region": region,
                "distance_class": distance,
                "label": label,
                "n": len(values),
                "mean_gc": mean(values),
                "median_gc": median(values),
                "q1_gc": quantile(values, 0.25),
                "q3_gc": quantile(values, 0.75),
                "high_gc_fraction_ge_0_60": sum(1 for x in values if x >= 0.60) / len(values),
            }
        )
    return rows


def make_bin_rows(values_by_group: dict[str, list[float]]) -> list[dict[str, object]]:
    rows = []
    for group, values in sorted(values_by_group.items()):
        total = len(values)
        if not total:
            continue
        counts = defaultdict(int)
        for value in values:
            counts[gc_bin(value)] += 1
        for label, lo, hi in GC_BINS:
            rows.append(
                {
                    "group": group,
                    "gc_bin": label,
                    "gc_range": f"[{lo:.2f},{hi:.2f})",
                    "count": counts[label],
                    "total": total,
                    "fraction": counts[label] / total,
                }
            )
    return rows


def make_comparisons(values_by_group: dict[str, list[float]]) -> list[dict[str, object]]:
    pairs = [
        ("m1A_CDS_prox25", "m1A_CDS_distal50", "CDS prox25 vs distal"),
        ("m1A_CDS_intermediate26_50", "m1A_CDS_distal50", "CDS 26-50 vs distal"),
        ("m1A_3UTR_prox25", "m1A_3UTR_distal50", "3UTR prox25 vs distal"),
        ("m1A_3UTR_intermediate26_50", "m1A_3UTR_distal50", "3UTR 26-50 vs distal"),
        ("m1A_other_region_prox25", "m1A_other_region_distal50", "5UTR prox25 vs distal"),
        ("m1A_other_region_intermediate26_50", "m1A_other_region_distal50", "5UTR 26-50 vs distal"),
        ("m1A_exact_overlap_m6A", "m1A_no_m6A_same_transcript", "exact overlap vs m1A no same-tx m6A"),
        ("m1A_exact_overlap_m6A", "matched_unmodified_A", "exact overlap vs matched unmodified A"),
        ("m6A_only_same_transcript", "matched_unmodified_A", "m6A only same-tx vs matched unmodified A"),
    ]
    rows = []
    for left, right, comparison in pairs:
        x = values_by_group.get(left, [])
        y = values_by_group.get(right, [])
        u, delta, p = mann_whitney(x, y)
        rows.append(
            {
                "comparison": comparison,
                "left_group": left,
                "right_group": right,
                "left_n": len(x),
                "right_n": len(y),
                "left_median_gc": median(x) if x else float("nan"),
                "right_median_gc": median(y) if y else float("nan"),
                "median_difference": (median(x) - median(y)) if x and y else float("nan"),
                "cliffs_delta": delta,
                "mann_whitney_u": u,
                "p_value": p,
                "fdr_bh": 1.0,
                "significance": star(p),
            }
        )
    bh_adjust(rows)
    for row in rows:
        row["significance"] = star(float(row["fdr_bh"]))
    return rows


def draw_boxplot(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, values: list[float], label: str, fill: str) -> None:
    axis_min, axis_max = 0.30, 0.85

    def sx(value: float) -> int:
        return x + int((max(axis_min, min(axis_max, value)) - axis_min) / (axis_max - axis_min) * w)

    q10, q1, med, q3, q90 = [quantile(values, q) for q in (0.10, 0.25, 0.50, 0.75, 0.90)]
    cy = y + h // 2
    draw.line([sx(q10), cy, sx(q90), cy], fill="#7c4a03", width=3)
    draw.rectangle([sx(q1), cy - 18, sx(q3), cy + 18], fill=fill, outline="#7c4a03", width=2)
    draw.line([sx(med), cy - 22, sx(med), cy + 22], fill="#111827", width=3)
    draw.text((x - 155, y + 7), f"{label} (n={len(values)})", font=font(12), fill="#4b5563")


def make_png(values_by_group: dict[str, list[float]], summary_rows: list[dict[str, object]], comparison_rows: list[dict[str, object]], out: Path) -> None:
    width, height = 1800, 1600
    canvas = Image.new("RGBA", (width, height), "#fffaf0")
    draw = ImageDraw.Draw(canvas)
    draw.text((44, 30), "Detailed GC fraction landscape of m1A-m6A spatial groups", font=font(34, True), fill="#23180a")
    draw.text((44, 72), "GC fraction is computed from local 101 nt windows; yellow intensity denotes higher median GC or high-GC proportion.", font=font(18), fill="#5c3b00")

    # Panel A: region-distance boxplots.
    x0, y0, panel_w, panel_h = 44, 120, 1050, 520
    draw.rounded_rectangle([x0, y0, x0 + panel_w, y0 + panel_h], radius=12, fill="#fffdf5", outline="#facc15", width=2)
    draw.text((x0 + 20, y0 + 14), "A  Regional m1A proximity distributions", font=font(22, True), fill="#23180a")
    axis_x, axis_y = x0 + 280, y0 + 78
    axis_w = 690
    for tick in [0.30, 0.45, 0.60, 0.75, 0.85]:
        tx = axis_x + int((tick - 0.30) / (0.85 - 0.30) * axis_w)
        draw.line([tx, axis_y, tx, axis_y + 360], fill="#fde68a", width=1)
        draw.text((tx - 14, axis_y + 370), f"{tick:.2f}", font=font(11), fill="#6b4e16")
    box_idx = 0
    palette = {"prox25": "#facc15", "intermediate26_50": "#fbbf24", "distal50": "#f59e0b"}
    for region_idx, region in enumerate(REGIONS):
        draw.text((x0 + 22, axis_y + region_idx * 118 + 26), region, font=font(16, True), fill="#23180a")
        for dist_idx, (dist, dist_label) in enumerate(DISTANCES):
            group = next(g for g, r, d, _ in SPATIAL_GROUPS if r == region and d == dist)
            values = values_by_group.get(group, [])
            if not values:
                continue
            y = axis_y + region_idx * 118 + dist_idx * 36
            draw_boxplot(draw, axis_x, y, axis_w, 34, values, dist_label.replace(" nt", ""), palette[dist])
            box_idx += 1
    draw.text((axis_x + 355, y0 + panel_h - 34), "GC fraction", font=font(14, True), fill="#5c3b00")

    # Panel B: heatmap of medians.
    hx, hy, hw, hh = 1130, 120, 626, 520
    draw.rounded_rectangle([hx, hy, hx + hw, hy + hh], radius=12, fill="#fffdf5", outline="#facc15", width=2)
    draw.text((hx + 20, hy + 14), "B  Median GC fraction matrix", font=font(22, True), fill="#23180a")
    medians = {row["group"]: float(row["median_gc"]) for row in summary_rows}
    cell_w, cell_h = 142, 74
    base_x, base_y = hx + 110, hy + 92
    for c, (_, label) in enumerate(DISTANCES):
        draw.text((base_x + c * cell_w + 8, base_y - 30), label.split()[0], font=font(12, True), fill="#5c3b00")
    for r, region in enumerate(REGIONS):
        draw.text((hx + 24, base_y + r * cell_h + 24), region, font=font(15, True), fill="#23180a")
        for c, (dist, _) in enumerate(DISTANCES):
            group = next(g for g, rr, d, _ in SPATIAL_GROUPS if rr == region and d == dist)
            value = medians.get(group, float("nan"))
            color = color_scale(value, 0.45, 0.75)
            x = base_x + c * cell_w
            y = base_y + r * cell_h
            draw.rectangle([x, y, x + cell_w - 8, y + cell_h - 8], fill=color, outline="#92400e", width=1)
            draw.text((x + 36, y + 20), f"{value:.3f}", font=font(18, True), fill="#23180a")
    context_y = base_y + 3 * cell_h + 24
    for idx, (group, label) in enumerate(CONTEXT_GROUPS):
        value = medians.get(group, float("nan"))
        color = color_scale(value, 0.45, 0.75)
        x = hx + 28 + (idx % 2) * 285
        y = context_y + (idx // 2) * 72
        draw.rectangle([x, y, x + 250, y + 50], fill=color, outline="#92400e", width=1)
        draw.text((x + 10, y + 7), label[:22], font=font(12, True), fill="#23180a")
        draw.text((x + 184, y + 26), f"{value:.3f}", font=font(14, True), fill="#23180a")

    # Panel C: high-GC fractions.
    cx, cy, cw, ch = 44, 680, 820, 520
    draw.rounded_rectangle([cx, cy, cx + cw, cy + ch], radius=12, fill="#fffdf5", outline="#facc15", width=2)
    draw.text((cx + 20, cy + 14), "C  High-GC window fraction", font=font(22, True), fill="#23180a")
    selected_context = SPATIAL_GROUPS + [(g, "", "", label) for g, label in CONTEXT_GROUPS]
    bar_x, bar_y = cx + 255, cy + 70
    bar_w = 460
    for idx, item in enumerate(selected_context):
        group = item[0]
        label = item[3]
        values = values_by_group.get(group, [])
        if not values:
            continue
        frac = sum(1 for v in values if v >= 0.60) / len(values)
        y = bar_y + idx * 25
        draw.text((cx + 20, y - 2), label[:28], font=font(12), fill="#23180a")
        draw.rectangle([bar_x, y, bar_x + bar_w, y + 14], fill="#fef3c7", outline="#fde68a")
        draw.rectangle([bar_x, y, bar_x + int(bar_w * frac), y + 14], fill=color_scale(frac, 0.30, 0.80))
        draw.text((bar_x + bar_w + 12, y - 2), f"{frac*100:.1f}%", font=font(12, True), fill="#5c3b00")
    draw.text((bar_x + 145, cy + ch - 32), "fraction with GC >= 0.60", font=font(14, True), fill="#5c3b00")

    # Panel D: comparison effect sizes.
    dx, dy, dw, dh = 910, 680, 846, 520
    draw.rounded_rectangle([dx, dy, dx + dw, dy + dh], radius=12, fill="#fffdf5", outline="#facc15", width=2)
    draw.text((dx + 20, dy + 14), "D  GC fraction shifts in key contrasts", font=font(22, True), fill="#23180a")
    zero_x = dx + 520
    scale = 520
    draw.line([zero_x, dy + 70, zero_x, dy + 440], fill="#92400e", width=2)
    for idx, row in enumerate(comparison_rows):
        y = dy + 78 + idx * 40
        diff = float(row["median_difference"])
        x2 = zero_x + int(max(-0.18, min(0.18, diff)) / 0.18 * (scale / 2))
        color = "#f59e0b" if diff >= 0 else "#fde68a"
        draw.text((dx + 22, y - 7), str(row["comparison"])[:42], font=font(12), fill="#23180a")
        draw.line([zero_x, y, x2, y], fill=color, width=10)
        draw.ellipse([x2 - 6, y - 6, x2 + 6, y + 6], fill="#d97706", outline="#7c2d12")
        draw.text((dx + 700, y - 9), f"{diff:+.3f} {row['significance']}", font=font(12, True), fill="#5c3b00")
    draw.text((zero_x - 80, dy + dh - 34), "median GC difference", font=font(14, True), fill="#5c3b00")

    note_y = 1240
    draw.rounded_rectangle([44, note_y, width - 44, note_y + 110], radius=12, fill="#fffdf5", outline="#facc15", width=2)
    draw.text((70, note_y + 22), "Interpretation note", font=font(19, True), fill="#23180a")
    draw.text((70, note_y + 50), "The CDS-proximal m1A group shows lower local GC than CDS-distal controls; GC differences are evaluated alongside RNAfold structure.", font=font(15), fill="#5c3b00")
    draw.text((70, note_y + 76), "Small 3UTR proximal groups should be interpreted cautiously because distribution estimates are sample-size sensitive.", font=font(15), fill="#5c3b00")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out)


def make_svg_wrapper(png_path: Path, svg_path: Path) -> None:
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1600" viewBox="0 0 1800 1600">\n'
        f'<image href="data:image/png;base64,{data}" x="0" y="0" width="1800" height="1600"/>\n'
        "</svg>\n"
    )
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8", newline="\n")


def main() -> None:
    values = load_gc_values()
    summary = make_summary(values)
    region_distance = make_region_distance_summary(values)
    bins = make_bin_rows(values)
    comparisons = make_comparisons(values)

    write_tsv(
        BIO / "gc_fraction_group_summary.tsv",
        summary,
        ["group", "label", "n", "mean_gc", "median_gc", "sd_gc", "sem_gc", "q1_gc", "q3_gc", "p10_gc", "p90_gc", "high_gc_fraction_ge_0_60", "very_high_gc_fraction_ge_0_75"],
    )
    write_tsv(
        BIO / "gc_fraction_region_distance_summary.tsv",
        region_distance,
        ["group", "region", "distance_class", "label", "n", "mean_gc", "median_gc", "q1_gc", "q3_gc", "high_gc_fraction_ge_0_60"],
    )
    write_tsv(BIO / "gc_fraction_bin_distribution.tsv", bins, ["group", "gc_bin", "gc_range", "count", "total", "fraction"])
    write_tsv(
        BIO / "gc_fraction_comparison_tests.tsv",
        comparisons,
        ["comparison", "left_group", "right_group", "left_n", "right_n", "left_median_gc", "right_median_gc", "median_difference", "cliffs_delta", "mann_whitney_u", "p_value", "fdr_bh", "significance"],
    )

    png = FIG_PNG / "Figure_gc_fraction_detail_yellow.png"
    svg = FIG_SVG / "Figure_gc_fraction_detail_yellow.svg"
    make_png(values, summary, comparisons, png)
    make_svg_wrapper(png, svg)

    copies = [
        (png, PKG / "figures_png" / png.name),
        (svg, PKG / "figures" / svg.name),
        (BIO / "gc_fraction_group_summary.tsv", PKG / "source_data" / "Figure_gc_fraction_group_summary.tsv"),
        (BIO / "gc_fraction_region_distance_summary.tsv", PKG / "source_data" / "Figure_gc_fraction_region_distance_summary.tsv"),
        (BIO / "gc_fraction_bin_distribution.tsv", PKG / "source_data" / "Figure_gc_fraction_bin_distribution.tsv"),
        (BIO / "gc_fraction_comparison_tests.tsv", PKG / "source_data" / "Figure_gc_fraction_comparison_tests.tsv"),
    ]
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print({"groups": len(summary), "comparisons": len(comparisons), "png": str(png)})


if __name__ == "__main__":
    main()
