from __future__ import annotations

import csv
import html
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from create_sn_template_manuscript import render_svg_to_png


ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / "11_tables" / "biology_summary"
FIG_SVG = ROOT / "10_figures" / "biology_spatial_svg"
FIG_PNG = ROOT / "10_figures" / "biology_spatial_png"
PKG = ROOT / "submission" / "nature_communications_structdualrm"

DISTANCE_BINS = [
    ("exact", 0, 0),
    ("1-10 nt", 1, 10),
    ("11-25 nt", 11, 25),
    ("26-50 nt", 26, 50),
    ("51-100 nt", 51, 100),
    ("101-200 nt", 101, 200),
    ("201-500 nt", 201, 500),
    (">500 nt", 501, math.inf),
    ("no m6A same transcript", None, None),
]
THRESHOLDS = [0, 10, 25, 50, 100, 200, 500]
REGIONS = ["5UTR", "CDS", "3UTR"]
BLUE = ["#eff6ff", "#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8", "#1e3a8a"]


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_int(value: object) -> int | None:
    try:
        if value in {"", None}:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def distance_bin(distance: int | None) -> str:
    if distance is None:
        return "no m6A same transcript"
    for label, lo, hi in DISTANCE_BINS:
        if lo is None:
            continue
        if lo <= distance <= hi:
            return label
    return ">500 nt"


def signed_direction(tx_pos: int | None, partner_pos: int | None) -> tuple[int | None, str]:
    if tx_pos is None or partner_pos is None:
        return None, "no_partner"
    signed = partner_pos - tx_pos
    if signed < 0:
        return signed, "m6A_upstream"
    if signed > 0:
        return signed, "m6A_downstream"
    return signed, "exact_overlap"


def median(values: list[int]) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2


def pct(num: float, den: float) -> float:
    return num / den if den else 0.0


def prepare_spatial_tables() -> dict[str, list[dict[str, object]]]:
    rows = read_tsv(BIO / "dual_site_groups.tsv")
    sites: list[dict[str, object]] = []
    for row in rows:
        if row.get("mod_type") != "m1A":
            continue
        region = row.get("region_type") or "unknown"
        distance = parse_int(row.get("nearest_partner_distance"))
        tx_pos = parse_int(row.get("tx_pos"))
        partner_pos = parse_int(row.get("nearest_partner_tx_pos"))
        signed, direction = signed_direction(tx_pos, partner_pos)
        sites.append(
            {
                "site_id": row.get("site_id", ""),
                "site_group": row.get("site_group", ""),
                "transcript_id": row.get("transcript_id", ""),
                "gene_name": row.get("gene_name", ""),
                "region_type": region,
                "tx_pos": tx_pos if tx_pos is not None else "",
                "nearest_m6A_tx_pos": partner_pos if partner_pos is not None else "",
                "nearest_distance_nt": distance if distance is not None else "",
                "signed_distance_nt": signed if signed is not None else "",
                "direction": direction,
                "distance_bin": distance_bin(distance),
            }
        )

    totals = Counter(site["region_type"] for site in sites)
    bins: list[dict[str, object]] = []
    for region in REGIONS:
        region_sites = [site for site in sites if site["region_type"] == region]
        total = len(region_sites)
        for label, _lo, _hi in DISTANCE_BINS:
            count = sum(1 for site in region_sites if site["distance_bin"] == label)
            bins.append(
                {
                    "region_type": region,
                    "distance_bin": label,
                    "n_sites": count,
                    "region_total_m1A": total,
                    "fraction_of_region_m1A": pct(count, total),
                }
            )

    thresholds: list[dict[str, object]] = []
    for region in REGIONS:
        region_sites = [site for site in sites if site["region_type"] == region]
        total = len(region_sites)
        distances = [int(site["nearest_distance_nt"]) for site in region_sites if site["nearest_distance_nt"] != ""]
        for threshold in THRESHOLDS:
            count = sum(1 for d in distances if d <= threshold)
            thresholds.append(
                {
                    "region_type": region,
                    "threshold_nt": threshold,
                    "n_sites_within_threshold": count,
                    "region_total_m1A": total,
                    "fraction_of_region_m1A": pct(count, total),
                }
            )

    directions: list[dict[str, object]] = []
    for region in REGIONS:
        region_sites = [site for site in sites if site["region_type"] == region and site["direction"] != "no_partner"]
        total = len(region_sites)
        for direction in ["m6A_upstream", "exact_overlap", "m6A_downstream"]:
            count = sum(1 for site in region_sites if site["direction"] == direction)
            directions.append(
                {
                    "region_type": region,
                    "direction": direction,
                    "n_sites": count,
                    "m1A_with_m6A_same_transcript": total,
                    "fraction": pct(count, total),
                }
            )

    summaries: list[dict[str, object]] = []
    for region in REGIONS:
        region_sites = [site for site in sites if site["region_type"] == region]
        distances = [int(site["nearest_distance_nt"]) for site in region_sites if site["nearest_distance_nt"] != ""]
        total = len(region_sites)
        summaries.append(
            {
                "region_type": region,
                "m1A_sites": total,
                "m1A_with_m6A_same_transcript": len(distances),
                "fraction_with_m6A_same_transcript": pct(len(distances), total),
                "median_nearest_distance_nt": median(distances),
                "fraction_exact": pct(sum(1 for d in distances if d == 0), total),
                "fraction_le_25nt": pct(sum(1 for d in distances if d <= 25), total),
                "fraction_le_50nt": pct(sum(1 for d in distances if d <= 50), total),
                "fraction_le_200nt": pct(sum(1 for d in distances if d <= 200), total),
                "fraction_no_m6A_same_transcript": pct(totals[region] - len(distances), total),
            }
        )

    write_tsv(
        BIO / "m1A_m6A_spatial_site_level.tsv",
        sites,
        [
            "site_id",
            "site_group",
            "transcript_id",
            "gene_name",
            "region_type",
            "tx_pos",
            "nearest_m6A_tx_pos",
            "nearest_distance_nt",
            "signed_distance_nt",
            "direction",
            "distance_bin",
        ],
    )
    write_tsv(BIO / "m1A_m6A_spatial_distance_bins.tsv", bins, ["region_type", "distance_bin", "n_sites", "region_total_m1A", "fraction_of_region_m1A"])
    write_tsv(
        BIO / "m1A_m6A_spatial_threshold_curve.tsv",
        thresholds,
        ["region_type", "threshold_nt", "n_sites_within_threshold", "region_total_m1A", "fraction_of_region_m1A"],
    )
    write_tsv(BIO / "m1A_m6A_spatial_direction_summary.tsv", directions, ["region_type", "direction", "n_sites", "m1A_with_m6A_same_transcript", "fraction"])
    write_tsv(
        BIO / "m1A_m6A_spatial_region_summary.tsv",
        summaries,
        [
            "region_type",
            "m1A_sites",
            "m1A_with_m6A_same_transcript",
            "fraction_with_m6A_same_transcript",
            "median_nearest_distance_nt",
            "fraction_exact",
            "fraction_le_25nt",
            "fraction_le_50nt",
            "fraction_le_200nt",
            "fraction_no_m6A_same_transcript",
        ],
    )
    return {"sites": sites, "bins": bins, "thresholds": thresholds, "directions": directions, "summaries": summaries}


def svg_text(lines: list[str], x: float, y: float, value: str, cls: str = "small", anchor: str = "start") -> None:
    lines.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{esc(value)}</text>')


def svg_rect(lines: list[str], x: int, y: int, w: int, h: int, fill: str, stroke: str = "#1e3a8a", rx: int = 5, sw: int = 1) -> None:
    lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def svg_line(lines: list[str], x1: int, y1: int, x2: int, y2: int, color: str = "#1e3a8a", width: int = 2) -> None:
    lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')


def shade(value: float, max_value: float) -> str:
    if max_value <= 0:
        return BLUE[0]
    idx = min(len(BLUE) - 1, max(0, int(round((value / max_value) * (len(BLUE) - 1)))))
    return BLUE[idx]


def draw_stacked_bars(lines: list[str], bins: list[dict[str, object]], x: int, y: int) -> None:
    svg_text(lines, x, y - 22, "A. Fine distance-bin composition by transcript region", "label")
    bin_labels = [label for label, _lo, _hi in DISTANCE_BINS]
    colors = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef", "#deebf7", "#f7fbff"]
    bar_w = 620
    bar_h = 30
    row_gap = 52
    by_region_bin = {(row["region_type"], row["distance_bin"]): row for row in bins}
    for r_idx, region in enumerate(REGIONS):
        yy = y + r_idx * row_gap
        svg_text(lines, x, yy + 22, region, "small", "end")
        cur = x + 20
        total = float(next(row["region_total_m1A"] for row in bins if row["region_type"] == region))
        for b_idx, label in enumerate(bin_labels):
            row = by_region_bin[(region, label)]
            w = int(round(bar_w * float(row["n_sites"]) / total)) if total else 0
            if w > 0:
                svg_rect(lines, cur, yy, w, bar_h, colors[b_idx], "#ffffff", 0, 1)
            cur += w
        svg_text(lines, x + 655, yy + 22, f"n={int(total)}", "tiny")
    lx, ly = x + 20, y + 185
    for i, label in enumerate(bin_labels):
        svg_rect(lines, lx + (i % 5) * 150, ly + (i // 5) * 30, 18, 18, colors[i], "#ffffff", 0, 1)
        svg_text(lines, lx + (i % 5) * 150 + 25, ly + (i // 5) * 30 + 15, label, "tiny")


def draw_heatmap(lines: list[str], bins: list[dict[str, object]], x: int, y: int) -> None:
    svg_text(lines, x, y - 22, "B. Region x nearest-m6A distance heatmap", "label")
    labels = [label for label, _lo, _hi in DISTANCE_BINS[:-1]]
    cell_w, cell_h = 88, 42
    values = {(row["region_type"], row["distance_bin"]): float(row["fraction_of_region_m1A"]) for row in bins}
    max_value = max(values.values()) if values else 0.0
    for i, label in enumerate(labels):
        svg_text(lines, x + 105 + i * cell_w + cell_w / 2, y - 4, label.replace(" nt", ""), "tiny", "middle")
    for r_idx, region in enumerate(REGIONS):
        yy = y + r_idx * cell_h
        svg_text(lines, x + 90, yy + 27, region, "small", "end")
        for c_idx, label in enumerate(labels):
            value = values.get((region, label), 0.0)
            xx = x + 105 + c_idx * cell_w
            svg_rect(lines, xx, yy, cell_w - 2, cell_h - 2, shade(value, max_value), "#ffffff", 0, 1)
            svg_text(lines, xx + cell_w / 2, yy + 27, f"{value*100:.1f}", "tiny", "middle")
    svg_text(lines, x + 105, y + 152, "Cell value = % of all m1A sites in that region", "tiny")


def draw_direction(lines: list[str], directions: list[dict[str, object]], x: int, y: int) -> None:
    svg_text(lines, x, y - 22, "C. Direction of the nearest m6A relative to m1A", "label")
    colors = {"m6A_upstream": "#1e40af", "exact_overlap": "#93c5fd", "m6A_downstream": "#0ea5e9"}
    names = {"m6A_upstream": "upstream", "exact_overlap": "exact", "m6A_downstream": "downstream"}
    by = {(row["region_type"], row["direction"]): row for row in directions}
    bar_w = 520
    for r_idx, region in enumerate(REGIONS):
        yy = y + r_idx * 54
        svg_text(lines, x, yy + 22, region, "small", "end")
        cur = x + 20
        total = float(by.get((region, "m6A_upstream"), {}).get("m1A_with_m6A_same_transcript", 0))
        for direction in ["m6A_upstream", "exact_overlap", "m6A_downstream"]:
            frac = float(by.get((region, direction), {}).get("fraction", 0.0))
            w = int(round(bar_w * frac))
            if w > 0:
                svg_rect(lines, cur, yy, w, 32, colors[direction], "#ffffff", 0, 1)
            cur += w
        svg_text(lines, x + 560, yy + 23, f"with partner n={int(total)}", "tiny")
    lx = x + 20
    for i, direction in enumerate(["m6A_upstream", "exact_overlap", "m6A_downstream"]):
        svg_rect(lines, lx + i * 140, y + 185, 20, 20, colors[direction], "#ffffff", 0, 1)
        svg_text(lines, lx + i * 140 + 28, y + 202, names[direction], "tiny")


def draw_curve(lines: list[str], thresholds: list[dict[str, object]], x: int, y: int) -> None:
    svg_text(lines, x, y - 22, "D. Cumulative proximity curve", "label")
    plot_w, plot_h = 560, 235
    svg_line(lines, x + 50, y + plot_h, x + 50 + plot_w, y + plot_h, "#334155", 2)
    svg_line(lines, x + 50, y, x + 50, y + plot_h, "#334155", 2)
    max_thr = max(THRESHOLDS)
    max_frac = max(float(row["fraction_of_region_m1A"]) for row in thresholds) or 1.0
    colors = {"5UTR": "#0c4a6e", "CDS": "#2563eb", "3UTR": "#38bdf8"}
    by_region = defaultdict(list)
    for row in thresholds:
        by_region[row["region_type"]].append(row)
    for region in REGIONS:
        pts = []
        for row in sorted(by_region[region], key=lambda r: int(r["threshold_nt"])):
            tx = int(row["threshold_nt"])
            frac = float(row["fraction_of_region_m1A"])
            px = x + 50 + int(plot_w * tx / max_thr)
            py = y + plot_h - int(plot_h * frac / max_frac)
            pts.append((px, py, tx, frac))
        for a, b in zip(pts, pts[1:]):
            svg_line(lines, a[0], a[1], b[0], b[1], colors[region], 4)
        for px, py, _tx, _frac in pts:
            lines.append(f'<circle cx="{px}" cy="{py}" r="5" fill="{colors[region]}" stroke="#ffffff" stroke-width="1"/>')
    for tick in [0, 100, 200, 500]:
        px = x + 50 + int(plot_w * tick / max_thr)
        svg_line(lines, px, y + plot_h, px, y + plot_h + 6, "#334155", 1)
        svg_text(lines, px, y + plot_h + 24, str(tick), "tiny", "middle")
    svg_text(lines, x + 330, y + plot_h + 50, "nearest m6A threshold (nt)", "small", "middle")
    svg_text(lines, x + 4, y + 12, f"max {max_frac*100:.1f}%", "tiny")
    for i, region in enumerate(REGIONS):
        lines.append(f'<circle cx="{x + 420 + i*85}" cy="{y + 18}" r="6" fill="{colors[region]}"/>')
        svg_text(lines, x + 432 + i * 85, y + 24, region, "tiny")


def draw_summary(lines: list[str], summaries: list[dict[str, object]], x: int, y: int) -> None:
    svg_text(lines, x, y - 22, "E. Region-specific summary", "label")
    headers = ["region", "m1A", "with m6A", "median d", "<=25 nt", "no m6A"]
    widths = [85, 75, 95, 95, 85, 85]
    cur = x
    for header, w in zip(headers, widths):
        svg_rect(lines, cur, y, w, 34, "#1e3a8a", "#ffffff", 0, 1)
        svg_text(lines, cur + w / 2, y + 23, header, "tinywhite", "middle")
        cur += w
    for idx, row in enumerate(summaries):
        cur = x
        yy = y + 34 * (idx + 1)
        fill = "#eff6ff" if idx % 2 == 0 else "#dbeafe"
        values = [
            row["region_type"],
            int(row["m1A_sites"]),
            f"{float(row['fraction_with_m6A_same_transcript'])*100:.1f}%",
            f"{float(row['median_nearest_distance_nt']):.0f}",
            f"{float(row['fraction_le_25nt'])*100:.1f}%",
            f"{float(row['fraction_no_m6A_same_transcript'])*100:.1f}%",
        ]
        for value, w in zip(values, widths):
            svg_rect(lines, cur, yy, w, 34, fill, "#ffffff", 0, 1)
            svg_text(lines, cur + w / 2, yy + 23, str(value), "tiny", "middle")
            cur += w
    svg_text(lines, x, y + 165, "Denominator: all m1A sites in each region.", "tiny")


def build_svg(tables: dict[str, list[dict[str, object]]]) -> str:
    width, height = 1800, 1200
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#0f172a}.title{font-size:30px;font-weight:700}.label{font-size:21px;font-weight:700}.small{font-size:17px}.tiny{font-size:13px}.tinywhite{font-size:13px;fill:#ffffff;font-weight:700}</style>',
        "</defs>",
        '<rect width="100%" height="100%" fill="#f8fbff"/>',
    ]
    svg_text(lines, 46, 55, "Detailed m1A-m6A spatial relationship in hg19 transcript space", "title")
    svg_text(lines, 46, 83, "Blue-scale summary of nearest m6A distances, region stratification and orientation for m1A sites.", "small")
    svg_rect(lines, 35, 105, 1730, 1045, "#ffffff", "#bfdbfe", 18, 2)
    draw_stacked_bars(lines, tables["bins"], 120, 155)
    draw_heatmap(lines, tables["bins"], 930, 155)
    draw_direction(lines, tables["directions"], 120, 520)
    draw_curve(lines, tables["thresholds"], 900, 510)
    draw_summary(lines, tables["summaries"], 1180, 860)
    svg_text(lines, 70, 1110, "Interpretation: this figure refines the biology analysis by separating exact/proximal/intermediate/distal distances, transcript regions and nearest-partner orientation.", "tiny")
    svg_text(lines, 70, 1130, "It should be interpreted together with the permutation result: m1A-m6A spacing is region-dependent, not a blanket global proximal-enrichment claim.", "tiny")
    lines.append("</svg>\n")
    return "\n".join(lines)


def main() -> None:
    tables = prepare_spatial_tables()
    svg_path = FIG_SVG / "Figure_spatial_m1A_m6A_blue.svg"
    png_path = FIG_PNG / "Figure_spatial_m1A_m6A_blue.png"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(build_svg(tables), encoding="utf-8", newline="\n")
    render_svg_to_png(svg_path, png_path)

    for target in [
        PKG / "figures" / "Figure_spatial_m1A_m6A_blue.svg",
        PKG / "figures_png" / "Figure_spatial_m1A_m6A_blue.png",
        PKG / "source_data" / "Figure_spatial_m1A_m6A_blue_distance_bins.tsv",
        PKG / "source_data" / "Figure_spatial_m1A_m6A_blue_threshold_curve.tsv",
        PKG / "source_data" / "Figure_spatial_m1A_m6A_blue_direction_summary.tsv",
        PKG / "source_data" / "Figure_spatial_m1A_m6A_blue_region_summary.tsv",
    ]:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".svg":
            shutil.copy2(svg_path, target)
        elif target.suffix == ".png":
            shutil.copy2(png_path, target)
        elif "distance_bins" in target.name:
            shutil.copy2(BIO / "m1A_m6A_spatial_distance_bins.tsv", target)
        elif "threshold_curve" in target.name:
            shutil.copy2(BIO / "m1A_m6A_spatial_threshold_curve.tsv", target)
        elif "direction_summary" in target.name:
            shutil.copy2(BIO / "m1A_m6A_spatial_direction_summary.tsv", target)
        elif "region_summary" in target.name:
            shutil.copy2(BIO / "m1A_m6A_spatial_region_summary.tsv", target)

    print({"svg": str(svg_path), "png": str(png_path), "m1A_sites": len(tables["sites"])})


if __name__ == "__main__":
    main()
