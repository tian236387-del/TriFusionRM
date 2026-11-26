from __future__ import annotations

import base64
import csv
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / "11_tables" / "biology_summary"
FIG_PNG = ROOT / "10_figures" / "biology_motif_png"
FIG_SVG = ROOT / "10_figures" / "biology_motif_svg"
PKG = ROOT / "submission" / "nature_communications_structdualrm"

DATASET = ROOT / "05_datasets" / "dataset_joint_m1A_m6A_101nt.tsv"
FEATURES = BIO / "sequence_structure_site_features.tsv"

REL_POSITIONS = list(range(-5, 6))
BASES = ["A", "C", "G", "T"]
DISPLAY_BASE = {"A": "A", "C": "C", "G": "G", "T": "U"}
BASE_COLORS = {
    "A": "#dc2626",
    "C": "#2563eb",
    "G": "#f59e0b",
    "T": "#16a34a",
}
PANEL_GROUPS = [
    ("m1A_CDS_prox25", "m1A CDS prox25"),
    ("m1A_CDS_intermediate26_50", "m1A CDS 26-50"),
    ("m1A_CDS_distal50", "m1A CDS distal"),
    ("m1A_3UTR_prox25", "m1A 3UTR prox25"),
    ("m1A_3UTR_intermediate26_50", "m1A 3UTR 26-50"),
    ("m1A_3UTR_distal50", "m1A 3UTR distal"),
    ("m1A_other_region_prox25", "m1A 5UTR prox25"),
    ("m1A_other_region_intermediate26_50", "m1A 5UTR 26-50"),
    ("m1A_other_region_distal50", "m1A 5UTR distal"),
    ("m1A_exact_overlap_m6A", "m1A exact overlap"),
    ("m1A_no_m6A_same_transcript", "m1A no same-tx m6A"),
    ("m6A_only_same_transcript", "m6A only same-tx"),
    ("all_m1A", "all m1A"),
    ("all_m6A", "all m6A"),
    ("matched_unmodified_A", "matched unmodified A"),
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_seq(seq: str) -> str:
    return seq.upper().replace("U", "T")


def centered_window(seq: str, flank: int = 5) -> str:
    seq = normalize_seq(seq)
    center = len(seq) // 2
    return seq[center - flank : center + flank + 1]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_group_sequences() -> dict[str, list[str]]:
    feature_rows = read_tsv(FEATURES)
    site_groups: dict[str, list[str]] = defaultdict(list)
    seen_by_group: dict[str, set[str]] = defaultdict(set)
    for row in feature_rows:
        group = row["site_group"]
        site_id = row["site_id"]
        if site_id not in seen_by_group[group]:
            site_groups[group].append(site_id)
            seen_by_group[group].add(site_id)

    all_target_ids = set().union(*(set(ids) for ids in site_groups.values())) if site_groups else set()
    seq_by_id: dict[str, str] = {}
    group_sequences: dict[str, list[str]] = defaultdict(list)

    with DATASET.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sid = row["site_id"]
            seq = normalize_seq(row.get("sequence_101nt", ""))
            if len(seq) != 101 or set(seq) - set("ACGT"):
                continue
            if sid in all_target_ids:
                seq_by_id[sid] = seq
            if row.get("label_m1A") == "1":
                group_sequences["all_m1A"].append(seq)
            if row.get("label_m6A") == "1":
                group_sequences["all_m6A"].append(seq)

    for group, ids in site_groups.items():
        for sid in sorted(ids):
            seq = seq_by_id.get(sid)
            if seq:
                group_sequences[group].append(seq)

    return {group: seqs for group, seqs in group_sequences.items() if seqs}


def compute_pwm(seqs: list[str], flank: int = 5) -> tuple[list[dict[str, float]], list[dict[str, int]]]:
    counts: list[Counter[str]] = [Counter() for _ in range(flank * 2 + 1)]
    for raw_seq in seqs:
        window = centered_window(raw_seq, flank)
        if len(window) != flank * 2 + 1:
            continue
        for idx, base in enumerate(window):
            if base in BASES:
                counts[idx][base] += 1
    freqs: list[dict[str, float]] = []
    int_counts: list[dict[str, int]] = []
    for pos_counts in counts:
        total = sum(pos_counts.get(base, 0) for base in BASES)
        int_counts.append({base: pos_counts.get(base, 0) for base in BASES})
        denom = total + 0.5 * len(BASES)
        freqs.append({base: (pos_counts.get(base, 0) + 0.5) / denom if denom else 0.25 for base in BASES})
    return freqs, int_counts


def information_heights(freqs: list[dict[str, float]]) -> list[dict[str, float]]:
    heights: list[dict[str, float]] = []
    for pos_freq in freqs:
        entropy = -sum(p * math.log2(p) for p in pos_freq.values() if p > 0)
        info = max(0.0, 2.0 - entropy)
        heights.append({base: pos_freq[base] * info for base in BASES})
    return heights


def centered_kmers(seqs: list[str], width: int = 7) -> Counter[str]:
    out: Counter[str] = Counter()
    half = width // 2
    for seq in seqs:
        seq = normalize_seq(seq)
        center = len(seq) // 2
        kmer = seq[center - half : center + half + 1]
        if len(kmer) == width and not (set(kmer) - set("ACGT")):
            out[kmer] += 1
    return out


def sorted_kmers(counter: Counter[str], limit: int) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def log2_odds_ratio(a_count: int, a_total: int, b_count: int, b_total: int, pseudocount: float = 0.5) -> float:
    a_other = max(0, a_total - a_count)
    b_other = max(0, b_total - b_count)
    a_odds = (a_count + pseudocount) / (a_other + pseudocount)
    b_odds = (b_count + pseudocount) / (b_other + pseudocount)
    return math.log2(a_odds / b_odds)


def two_proportion_p_value(a_count: int, a_total: int, b_count: int, b_total: int) -> float:
    if a_total <= 0 or b_total <= 0:
        return 1.0
    pooled = (a_count + b_count) / (a_total + b_total)
    if pooled <= 0.0 or pooled >= 1.0:
        return 1.0
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / a_total + 1.0 / b_total))
    if se == 0.0:
        return 1.0
    z = ((a_count / a_total) - (b_count / b_total)) / se
    return math.erfc(abs(z) / math.sqrt(2.0))


def add_bh_fdr(rows: list[dict[str, object]], family_field: str = "comparison_family") -> None:
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_family[str(row[family_field])].append(row)
    for family_rows in by_family.values():
        ranked = sorted(family_rows, key=lambda row: float(row["p_value"]))
        m = len(ranked)
        previous = 1.0
        for idx in range(m, 0, -1):
            row = ranked[idx - 1]
            adjusted = min(previous, float(row["p_value"]) * m / idx)
            row["fdr_bh"] = adjusted
            previous = adjusted


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def render_letter(letter: str, color: str, width: int, height: int, letter_font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> Image.Image:
    if height <= 1 or width <= 1:
        return Image.new("RGBA", (max(width, 1), max(height, 1)), (255, 255, 255, 0))
    tmp = Image.new("RGBA", (120, 120), (255, 255, 255, 0))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), letter, font=letter_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((120 - tw) / 2 - bbox[0], (120 - th) / 2 - bbox[1]), letter, font=letter_font, fill=hex_to_rgb(color) + (255,))
    alpha_bbox = tmp.getbbox()
    if alpha_bbox:
        tmp = tmp.crop(alpha_bbox)
    return tmp.resize((width, max(height, 1)), Image.Resampling.LANCZOS)


def draw_logo(draw: ImageDraw.ImageDraw, canvas: Image.Image, x: int, y: int, w: int, h: int, title: str, seqs: list[str]) -> dict[str, object]:
    title_font = font(22, True)
    tiny_font = font(13)
    axis_font = font(12)
    letter_font = font(74, True)
    freqs, counts = compute_pwm(seqs)
    heights = information_heights(freqs)
    top7 = sorted_kmers(centered_kmers(seqs, 7), 1)
    top_kmer = top7[0][0].replace("T", "U") if top7 else ""
    top_freq = top7[0][1] / len(seqs) if top7 and seqs else 0.0
    total_ic = sum(sum(pos.values()) for pos in heights)

    draw.rounded_rectangle([x, y, x + w, y + h], radius=12, fill="#ffffff", outline="#dbeafe", width=2)
    draw.text((x + 12, y + 8), title, font=title_font, fill="#0f172a")
    draw.text((x + w - 112, y + 12), f"n={len(seqs)}", font=tiny_font, fill="#475569")
    draw.text((x + 12, y + 36), f"top7: {top_kmer} ({top_freq*100:.1f}%)", font=tiny_font, fill="#475569")

    logo_x, logo_y = x + 52, y + 72
    logo_w, logo_h = w - 78, h - 118
    draw.line([logo_x, logo_y + logo_h, logo_x + logo_w, logo_y + logo_h], fill="#334155", width=2)
    draw.line([logo_x, logo_y, logo_x, logo_y + logo_h], fill="#334155", width=2)
    draw.text((logo_x - 34, logo_y - 2), "2", font=axis_font, fill="#334155")
    draw.text((logo_x - 34, logo_y + logo_h - 10), "0", font=axis_font, fill="#334155")
    draw.text((logo_x - 48, logo_y + 35), "bits", font=axis_font, fill="#334155")

    col_w = max(18, logo_w // len(REL_POSITIONS))
    for idx, rel in enumerate(REL_POSITIONS):
        stack = sorted(heights[idx].items(), key=lambda item: item[1])
        cur_y = logo_y + logo_h
        for base, bits in stack:
            letter_h = int(round((bits / 2.0) * logo_h))
            if letter_h < 2:
                continue
            img = render_letter(DISPLAY_BASE[base], BASE_COLORS[base], col_w - 2, letter_h, letter_font)
            cur_y -= letter_h
            canvas.alpha_composite(img, (logo_x + idx * col_w + 1, cur_y))
        if rel in {-5, 0, 5}:
            label = "A" if rel == 0 else str(rel)
            draw.text((logo_x + idx * col_w + col_w / 2 - 5, logo_y + logo_h + 7), label, font=axis_font, fill="#334155")
    draw.text((logo_x + logo_w / 2 - 42, y + h - 22), "relative position to center A", font=tiny_font, fill="#475569")

    return {"n": len(seqs), "top7": top_kmer, "top7_fraction": top_freq, "total_information_bits": total_ic, "freqs": freqs, "counts": counts, "heights": heights}


def build_tables(group_sequences: dict[str, list[str]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    pwm_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for group, label in PANEL_GROUPS:
        seqs = group_sequences.get(group, [])
        if not seqs:
            continue
        freqs, counts = compute_pwm(seqs)
        heights = information_heights(freqs)
        for idx, rel in enumerate(REL_POSITIONS):
            for base in BASES:
                pwm_rows.append(
                    {
                        "group": group,
                        "label": label,
                        "n_sequences": len(seqs),
                        "relative_position": rel,
                        "base": DISPLAY_BASE[base],
                        "count": counts[idx][base],
                        "frequency": freqs[idx][base],
                        "information_height_bits": heights[idx][base],
                    }
                )
        kmers = centered_kmers(seqs, 7)
        top_kmers = sorted_kmers(kmers, 10)
        for rank, (kmer, count) in enumerate(top_kmers, start=1):
            top_rows.append(
                {
                    "group": group,
                    "label": label,
                    "rank": rank,
                    "centered_7mer": kmer.replace("T", "U"),
                    "count": count,
                    "fraction": count / len(seqs),
                }
            )
        summary_rows.append(
            {
                "group": group,
                "label": label,
                "n_sequences": len(seqs),
                "total_information_bits_center_pm5": sum(sum(pos.values()) for pos in heights),
                "top_centered_7mer": top_kmers[0][0].replace("T", "U") if top_kmers else "",
                "top_centered_7mer_fraction": top_kmers[0][1] / len(seqs) if top_kmers else 0.0,
            }
        )
    return pwm_rows, summary_rows, top_rows


def build_enrichment_table(group_sequences: dict[str, list[str]]) -> list[dict[str, object]]:
    labels = dict(PANEL_GROUPS)
    cache: dict[str, tuple[list[dict[str, float]], list[dict[str, int]]]] = {}

    def pwm_for(group: str) -> tuple[list[dict[str, float]], list[dict[str, int]]]:
        if group not in cache:
            cache[group] = compute_pwm(group_sequences.get(group, []))
        return cache[group]

    rows: list[dict[str, object]] = []
    for group, label in PANEL_GROUPS:
        seqs = group_sequences.get(group, [])
        if not seqs:
            continue
        comparisons = [("matched_unmodified_A", "matched unmodified A")]
        if group.startswith("m1A_") and group != "all_m1A":
            comparisons.append(("all_m1A", "all m1A"))
        if group == "m6A_only_same_transcript":
            comparisons.append(("all_m6A", "all m6A"))

        group_freqs, group_counts = pwm_for(group)
        group_total = len(seqs)
        for background_group, background_label in comparisons:
            background_seqs = group_sequences.get(background_group, [])
            if not background_seqs:
                continue
            background_freqs, background_counts = pwm_for(background_group)
            background_total = len(background_seqs)
            family = f"{group}_vs_{background_group}"
            for idx, rel in enumerate(REL_POSITIONS):
                for base in BASES:
                    group_count = group_counts[idx][base]
                    background_count = background_counts[idx][base]
                    rows.append(
                        {
                            "comparison_family": family,
                            "group": group,
                            "label": label,
                            "background_group": background_group,
                            "background_label": labels.get(background_group, background_label),
                            "relative_position": rel,
                            "base": DISPLAY_BASE[base],
                            "group_count": group_count,
                            "group_total": group_total,
                            "group_frequency": group_freqs[idx][base],
                            "background_count": background_count,
                            "background_total": background_total,
                            "background_frequency": background_freqs[idx][base],
                            "log2_odds_ratio": log2_odds_ratio(group_count, group_total, background_count, background_total),
                            "p_value": two_proportion_p_value(group_count, group_total, background_count, background_total),
                            "fdr_bh": 1.0,
                        }
                    )
    add_bh_fdr(rows)
    return rows


def make_png(group_sequences: dict[str, list[str]], out: Path) -> None:
    width, height = 1680, 2050
    canvas = Image.new("RGBA", (width, height), "#f8fbff")
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 34), "Detailed center-A sequence motifs for m1A-m6A spatial groups", font=font(34, True), fill="#0f172a")
    draw.text(
        (42, 76),
        "Information-content sequence logos use local 101 nt windows; panels show center +/-5 nt and are descriptive unless tested separately.",
        font=font(18),
        fill="#334155",
    )
    panel_w, panel_h = 520, 330
    x0, y0 = 42, 125
    gap_x, gap_y = 24, 32
    for idx, (group, label) in enumerate(PANEL_GROUPS):
        row, col = divmod(idx, 3)
        x = x0 + col * (panel_w + gap_x)
        y = y0 + row * (panel_h + gap_y)
        seqs = group_sequences.get(group, [])
        if seqs:
            draw_logo(draw, canvas, x, y, panel_w, panel_h, label, seqs)

    note_y = y0 + 5 * (panel_h + gap_y) + 8
    draw.rounded_rectangle([42, note_y, width - 42, note_y + 105], radius=12, fill="#ffffff", outline="#bfdbfe", width=2)
    draw.text((68, note_y + 28), "Interpretation note", font=font(20, True), fill="#0f172a")
    draw.text(
        (68, note_y + 58),
        "Small groups such as m1A 3UTR prox25 should be read cautiously. Existing DRACH/GGAC tests were not strongly FDR-significant,",
        font=font(16),
        fill="#334155",
    )
    draw.text(
        (68, note_y + 82),
        "so this figure refines motif description rather than claiming a new motif mechanism.",
        font=font(16),
        fill="#334155",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out)


def make_svg_wrapper(png_path: Path, svg_path: Path) -> None:
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1680" height="2050" viewBox="0 0 1680 2050">\n'
        f'<image href="data:image/png;base64,{data}" x="0" y="0" width="1680" height="2050"/>\n'
        "</svg>\n"
    )
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8", newline="\n")


def main() -> None:
    group_sequences = load_group_sequences()
    pwm_rows, summary_rows, top_rows = build_tables(group_sequences)
    enrichment_rows = build_enrichment_table(group_sequences)

    write_tsv(
        BIO / "sequence_motif_pwm_center_pm5.tsv",
        pwm_rows,
        ["group", "label", "n_sequences", "relative_position", "base", "count", "frequency", "information_height_bits"],
    )
    write_tsv(
        BIO / "sequence_motif_group_summary.tsv",
        summary_rows,
        ["group", "label", "n_sequences", "total_information_bits_center_pm5", "top_centered_7mer", "top_centered_7mer_fraction"],
    )
    write_tsv(BIO / "sequence_motif_top_centered_7mers.tsv", top_rows, ["group", "label", "rank", "centered_7mer", "count", "fraction"])
    write_tsv(
        BIO / "sequence_motif_position_base_enrichment.tsv",
        enrichment_rows,
        [
            "comparison_family",
            "group",
            "label",
            "background_group",
            "background_label",
            "relative_position",
            "base",
            "group_count",
            "group_total",
            "group_frequency",
            "background_count",
            "background_total",
            "background_frequency",
            "log2_odds_ratio",
            "p_value",
            "fdr_bh",
        ],
    )

    png_path = FIG_PNG / "Figure_sequence_motif_detail.png"
    svg_path = FIG_SVG / "Figure_sequence_motif_detail.svg"
    make_png(group_sequences, png_path)
    make_svg_wrapper(png_path, svg_path)

    for target in [
        PKG / "figures_png" / "Figure_sequence_motif_detail.png",
        PKG / "figures" / "Figure_sequence_motif_detail.svg",
        PKG / "source_data" / "Figure_sequence_motif_pwm_center_pm5.tsv",
        PKG / "source_data" / "Figure_sequence_motif_group_summary.tsv",
        PKG / "source_data" / "Figure_sequence_motif_top_centered_7mers.tsv",
        PKG / "source_data" / "Figure_sequence_motif_position_base_enrichment.tsv",
    ]:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".png":
            shutil.copy2(png_path, target)
        elif target.suffix == ".svg":
            shutil.copy2(svg_path, target)
        elif "pwm" in target.name:
            shutil.copy2(BIO / "sequence_motif_pwm_center_pm5.tsv", target)
        elif "group_summary" in target.name:
            shutil.copy2(BIO / "sequence_motif_group_summary.tsv", target)
        elif "top_centered" in target.name:
            shutil.copy2(BIO / "sequence_motif_top_centered_7mers.tsv", target)
        elif "position_base_enrichment" in target.name:
            shutil.copy2(BIO / "sequence_motif_position_base_enrichment.tsv", target)

    print({"png": str(png_path), "svg": str(svg_path), "groups": len(summary_rows)})


if __name__ == "__main__":
    main()
