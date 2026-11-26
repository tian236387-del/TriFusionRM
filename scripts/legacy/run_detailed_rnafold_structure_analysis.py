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
FIG_PNG = ROOT / "10_figures" / "biology_structure_png"
FIG_SVG = ROOT / "10_figures" / "biology_structure_svg"
PKG = ROOT / "submission" / "nature_communications_structdualrm"

FEATURE_DEFS = [
    ("MFE_per_nt", "MFE per nt", "more positive = less stable"),
    ("paired_fraction_101nt", "paired fraction", "lower = more accessible"),
    ("center_paired", "center paired", "binary centre pairing"),
    ("central_11nt_paired_fraction", "central 11 nt paired", "local pairing"),
    ("central_21nt_paired_fraction", "central 21 nt paired", "local pairing"),
    ("loop_span_around_center", "loop span", "larger = longer local loop"),
]
REGIONS = ["CDS", "3UTR", "5UTR"]
DISTANCES = [
    ("prox25", "proximal"),
    ("intermediate26_50", "26-50"),
    ("distal50", "distal"),
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
]
LIGHT_SCALE = ["#eff6ff", "#dbeafe", "#bfdbfe", "#bae6fd", "#a7f3d0", "#fde68a"]


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


def star(p: float) -> str:
    if math.isnan(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.05:
        return "**"
    return "ns"


def color_scale(value: float, vmin: float, vmax: float) -> str:
    if math.isnan(value):
        return "#f1f5f9"
    if vmax <= vmin:
        return LIGHT_SCALE[2]
    t = max(0.0, min(0.999, (value - vmin) / (vmax - vmin)))
    return LIGHT_SCALE[int(t * len(LIGHT_SCALE))]


def load_values() -> dict[str, dict[str, list[float]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in read_tsv(FEATURES):
        if row.get("has_rnafold") != "1":
            continue
        group = row["site_group"]
        for feature, _, _ in FEATURE_DEFS:
            value = safe_float(row.get(feature, ""))
            if value is not None:
                values[group][feature].append(value)
    return values


def make_summary(values: dict[str, dict[str, list[float]]]) -> list[dict[str, object]]:
    labels = {g: label for g, _, _, label in SPATIAL_GROUPS}
    labels.update(CONTEXT_GROUPS)
    rows = []
    for group, feat_values in values.items():
        for feature, feature_label, direction in FEATURE_DEFS:
            xs = feat_values.get(feature, [])
            if not xs:
                continue
            rows.append(
                {
                    "group": group,
                    "label": labels.get(group, group),
                    "feature": feature,
                    "feature_label": feature_label,
                    "direction_hint": direction,
                    "n": len(xs),
                    "mean": mean(xs),
                    "median": median(xs),
                    "sd": sd(xs),
                    "q1": quantile(xs, 0.25),
                    "q3": quantile(xs, 0.75),
                    "p10": quantile(xs, 0.10),
                    "p90": quantile(xs, 0.90),
                }
            )
    rows.sort(key=lambda row: (str(row["group"]), str(row["feature"])))
    return rows


def make_state_rows(values: dict[str, dict[str, list[float]]]) -> list[dict[str, object]]:
    labels = {g: label for g, _, _, label in SPATIAL_GROUPS}
    labels.update(CONTEXT_GROUPS)
    rows = []
    for group, feat_values in sorted(values.items()):
        mfe = feat_values.get("MFE_per_nt", [])
        paired = feat_values.get("paired_fraction_101nt", [])
        center = feat_values.get("center_paired", [])
        central21 = feat_values.get("central_21nt_paired_fraction", [])
        n = max(len(mfe), len(paired), len(center), len(central21))
        if n == 0:
            continue
        metrics = {
            "less_stable_MFE_per_nt_gt_-0_25": sum(1 for x in mfe if x > -0.25) / len(mfe) if mfe else float("nan"),
            "low_paired_fraction_lt_0_50": sum(1 for x in paired if x < 0.50) / len(paired) if paired else float("nan"),
            "center_unpaired_fraction": sum(1 for x in center if x < 0.5) / len(center) if center else float("nan"),
            "central21_low_paired_lt_0_50": sum(1 for x in central21 if x < 0.50) / len(central21) if central21 else float("nan"),
        }
        for state, fraction in metrics.items():
            rows.append({"group": group, "label": labels.get(group, group), "state_metric": state, "n": n, "fraction": fraction})
    return rows


def make_comparisons(values: dict[str, dict[str, list[float]]]) -> list[dict[str, object]]:
    pairs = [
        ("m1A_CDS_prox25", "m1A_CDS_distal50", "CDS prox25 vs distal"),
        ("m1A_CDS_intermediate26_50", "m1A_CDS_distal50", "CDS 26-50 vs distal"),
        ("m1A_3UTR_prox25", "m1A_3UTR_distal50", "3UTR prox25 vs distal"),
        ("m1A_3UTR_intermediate26_50", "m1A_3UTR_distal50", "3UTR 26-50 vs distal"),
        ("m1A_other_region_prox25", "m1A_other_region_distal50", "5UTR prox25 vs distal"),
        ("m1A_other_region_intermediate26_50", "m1A_other_region_distal50", "5UTR 26-50 vs distal"),
        ("m1A_exact_overlap_m6A", "m1A_no_m6A_same_transcript", "exact overlap vs m1A no same-tx m6A"),
        ("m6A_only_same_transcript", "m1A_no_m6A_same_transcript", "m6A only same-tx vs m1A no same-tx m6A"),
    ]
    rows = []
    for left, right, comparison in pairs:
        for feature, feature_label, direction in FEATURE_DEFS:
            x = values.get(left, {}).get(feature, [])
            y = values.get(right, {}).get(feature, [])
            u, delta, p = mann_whitney(x, y)
            rows.append(
                {
                    "comparison": comparison,
                    "left_group": left,
                    "right_group": right,
                    "feature": feature,
                    "feature_label": feature_label,
                    "direction_hint": direction,
                    "left_n": len(x),
                    "right_n": len(y),
                    "left_median": median(x) if x else float("nan"),
                    "right_median": median(y) if y else float("nan"),
                    "median_difference": median(x) - median(y) if x and y else float("nan"),
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


def summary_lookup(summary_rows: list[dict[str, object]], feature: str) -> dict[str, float]:
    return {str(row["group"]): float(row["median"]) for row in summary_rows if row["feature"] == feature}


def draw_heatmap(draw: ImageDraw.ImageDraw, x: int, y: int, title: str, med: dict[str, float], vmin: float, vmax: float, fmt: str) -> None:
    draw.text((x, y), title, font=font(18, True), fill="#0f172a")
    cell_w, cell_h = 142, 64
    base_x, base_y = x + 104, y + 58
    for c, (_, label) in enumerate(DISTANCES):
        draw.text((base_x + c * cell_w + 30, base_y - 26), label, font=font(12, True), fill="#475569")
    for r, region in enumerate(REGIONS):
        draw.text((x + 8, base_y + r * cell_h + 20), region, font=font(14, True), fill="#0f172a")
        for c, (dist, _) in enumerate(DISTANCES):
            group = next(g for g, rr, d, _ in SPATIAL_GROUPS if rr == region and d == dist)
            value = med.get(group, float("nan"))
            px = base_x + c * cell_w
            py = base_y + r * cell_h
            draw.rectangle([px, py, px + cell_w - 8, py + cell_h - 8], fill=color_scale(value, vmin, vmax), outline="#93c5fd", width=1)
            draw.text((px + 38, py + 18), format(value, fmt), font=font(17, True), fill="#0f172a")


def make_png(values: dict[str, dict[str, list[float]]], summary_rows: list[dict[str, object]], state_rows: list[dict[str, object]], comparison_rows: list[dict[str, object]], out: Path) -> None:
    width, height = 1800, 1600
    canvas = Image.new("RGBA", (width, height), "#f8fbff")
    draw = ImageDraw.Draw(canvas)
    draw.text((44, 30), "Detailed RNAfold structural differences across m1A-m6A groups", font=font(34, True), fill="#0f172a")
    draw.text((44, 74), "Pastel panels summarize local folding stability, pairedness and centre accessibility from 101 nt RNAfold windows.", font=font(18), fill="#475569")

    # Panel A: two median heatmaps.
    x0, y0, w0, h0 = 44, 120, 840, 520
    draw.rounded_rectangle([x0, y0, x0 + w0, y0 + h0], radius=12, fill="#ffffff", outline="#bfdbfe", width=2)
    draw.text((x0 + 20, y0 + 14), "A  Region-distance RNAfold medians", font=font(22, True), fill="#0f172a")
    draw_heatmap(draw, x0 + 28, y0 + 60, "MFE per nt", summary_lookup(summary_rows, "MFE_per_nt"), -0.45, -0.15, ".3f")
    draw_heatmap(draw, x0 + 28, y0 + 286, "Paired fraction", summary_lookup(summary_rows, "paired_fraction_101nt"), 0.35, 0.70, ".3f")

    # Panel B: context medians.
    bx, by, bw, bh = 930, 120, 826, 520
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=12, fill="#ffffff", outline="#bfdbfe", width=2)
    draw.text((bx + 20, by + 14), "B  Context groups across six structure features", font=font(22, True), fill="#0f172a")
    feature_labels = [label for _, label, _ in FEATURE_DEFS]
    for c, label in enumerate(feature_labels):
        draw.text((bx + 170 + c * 94, by + 82), label.replace(" ", "\n")[:14], font=font(10, True), fill="#475569")
    for r, (group, label) in enumerate(CONTEXT_GROUPS):
        draw.text((bx + 24, by + 128 + r * 86), label[:22], font=font(13, True), fill="#0f172a")
        for c, (feature, _, _) in enumerate(FEATURE_DEFS):
            vals = values.get(group, {}).get(feature, [])
            value = median(vals) if vals else float("nan")
            if feature == "MFE_per_nt":
                color = color_scale(value, -0.45, -0.15)
                text = f"{value:.2f}"
            elif feature == "loop_span_around_center":
                color = color_scale(value, 0, 8)
                text = f"{value:.1f}"
            else:
                color = color_scale(value, 0.0, 0.8)
                text = f"{value:.2f}"
            px = bx + 168 + c * 94
            py = by + 112 + r * 86
            draw.rectangle([px, py, px + 78, py + 54], fill=color, outline="#a5b4fc", width=1)
            draw.text((px + 15, py + 18), text, font=font(13, True), fill="#0f172a")

    # Panel C: accessibility fractions.
    cx, cy, cw, ch = 44, 680, 800, 520
    draw.rounded_rectangle([cx, cy, cx + cw, cy + ch], radius=12, fill="#ffffff", outline="#bfdbfe", width=2)
    draw.text((cx + 20, cy + 14), "C  Accessibility-state fractions", font=font(22, True), fill="#0f172a")
    states = [
        ("less_stable_MFE_per_nt_gt_-0_25", "less stable"),
        ("low_paired_fraction_lt_0_50", "low paired"),
        ("center_unpaired_fraction", "center unpaired"),
        ("central21_low_paired_lt_0_50", "central21 low"),
    ]
    groups = [g for g, _, _, _ in SPATIAL_GROUPS[:6]] + [g for g, _ in CONTEXT_GROUPS]
    labels = {g: label for g, _, _, label in SPATIAL_GROUPS}
    labels.update(CONTEXT_GROUPS)
    state_map = {(row["group"], row["state_metric"]): float(row["fraction"]) for row in state_rows}
    bar_x, bar_y, bar_w = cx + 240, cy + 92, 70
    state_colors = ["#dbeafe", "#dcfce7", "#ede9fe", "#fef3c7"]
    for i, (_, label) in enumerate(states):
        draw.text((bar_x + i * 125 - 6, cy + 64), label, font=font(10, True), fill="#475569")
    for idx, group in enumerate(groups):
        y = bar_y + idx * 36
        draw.text((cx + 22, y - 4), labels[group][:22], font=font(12), fill="#0f172a")
        for state_idx, (state, _state_label) in enumerate(states):
            frac = state_map.get((group, state), 0.0)
            px = bar_x + state_idx * 125
            draw.rectangle([px, y, px + bar_w, y + 14], fill="#f8fafc", outline="#cbd5e1")
            draw.rectangle([px, y, px + int(bar_w * frac), y + 14], fill=state_colors[state_idx], outline="#cbd5e1")
            draw.text((px + bar_w + 6, y - 2), f"{frac*100:.0f}", font=font(10), fill="#475569")
    for i, (_, label) in enumerate(states):
        draw.rectangle([cx + 80 + i * 160, cy + ch - 44, cx + 102 + i * 160, cy + ch - 22], fill=state_colors[i], outline="#cbd5e1")
        draw.text((cx + 108 + i * 160, cy + ch - 42), label, font=font(11), fill="#475569")

    # Panel D: key effect sizes.
    dx, dy, dw, dh = 890, 680, 866, 520
    draw.rounded_rectangle([dx, dy, dx + dw, dy + dh], radius=12, fill="#ffffff", outline="#bfdbfe", width=2)
    draw.text((dx + 20, dy + 14), "D  CDS proximal structure shifts vs distal", font=font(22, True), fill="#0f172a")
    cds_rows = [row for row in comparison_rows if row["comparison"] == "CDS prox25 vs distal" and row["feature"] in {"MFE_per_nt", "paired_fraction_101nt", "central_11nt_paired_fraction", "central_21nt_paired_fraction", "loop_span_around_center"}]
    zero_x = dx + 486
    draw.line([zero_x, dy + 88, zero_x, dy + 380], fill="#64748b", width=2)
    for idx, row in enumerate(cds_rows):
        y = dy + 104 + idx * 58
        delta = float(row["cliffs_delta"])
        x2 = zero_x + int(max(-0.45, min(0.45, delta)) / 0.45 * 250)
        color = "#bfdbfe" if delta < 0 else "#a7f3d0"
        draw.text((dx + 32, y - 10), str(row["feature_label"]), font=font(14, True), fill="#0f172a")
        draw.line([zero_x, y, x2, y], fill=color, width=14)
        draw.ellipse([x2 - 7, y - 7, x2 + 7, y + 7], fill="#60a5fa", outline="#1e3a8a")
        draw.text((dx + 700, y - 12), f"delta={delta:+.3f} {row['significance']}", font=font(13, True), fill="#334155")
        draw.text((dx + 32, y + 12), f"{float(row['left_median']):.3f} vs {float(row['right_median']):.3f}", font=font(11), fill="#64748b")
    draw.text((zero_x - 70, dy + dh - 38), "Cliff's delta", font=font(14, True), fill="#334155")

    note_y = 1236
    draw.rounded_rectangle([44, note_y, width - 44, note_y + 118], radius=12, fill="#ffffff", outline="#bfdbfe", width=2)
    draw.text((70, note_y + 22), "Interpretation note", font=font(19, True), fill="#0f172a")
    draw.text((70, note_y + 50), "CDS-proximal m1A sites are less strongly folded than CDS-distal controls: MFE per nt is less negative and paired fractions are lower.", font=font(15), fill="#475569")
    draw.text((70, note_y + 76), "Matched unmodified A controls are not used for structure contrasts because RNAfold coverage is absent for those rows in the current local tables.", font=font(15), fill="#475569")
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
    values = load_values()
    summary = make_summary(values)
    states = make_state_rows(values)
    comparisons = make_comparisons(values)

    write_tsv(
        BIO / "rnafold_structure_group_summary.tsv",
        summary,
        ["group", "label", "feature", "feature_label", "direction_hint", "n", "mean", "median", "sd", "q1", "q3", "p10", "p90"],
    )
    write_tsv(BIO / "rnafold_structure_accessibility_states.tsv", states, ["group", "label", "state_metric", "n", "fraction"])
    write_tsv(
        BIO / "rnafold_structure_detailed_comparisons.tsv",
        comparisons,
        ["comparison", "left_group", "right_group", "feature", "feature_label", "direction_hint", "left_n", "right_n", "left_median", "right_median", "median_difference", "cliffs_delta", "mann_whitney_u", "p_value", "fdr_bh", "significance"],
    )

    png = FIG_PNG / "Figure_rnafold_structure_detail_pastel.png"
    svg = FIG_SVG / "Figure_rnafold_structure_detail_pastel.svg"
    make_png(values, summary, states, comparisons, png)
    make_svg_wrapper(png, svg)

    copies = [
        (png, PKG / "figures_png" / png.name),
        (svg, PKG / "figures" / svg.name),
        (BIO / "rnafold_structure_group_summary.tsv", PKG / "source_data" / "Figure_rnafold_structure_group_summary.tsv"),
        (BIO / "rnafold_structure_accessibility_states.tsv", PKG / "source_data" / "Figure_rnafold_structure_accessibility_states.tsv"),
        (BIO / "rnafold_structure_detailed_comparisons.tsv", PKG / "source_data" / "Figure_rnafold_structure_detailed_comparisons.tsv"),
    ]
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print({"groups": len(values), "features": len(FEATURE_DEFS), "comparisons": len(comparisons), "png": str(png)})


if __name__ == "__main__":
    main()
