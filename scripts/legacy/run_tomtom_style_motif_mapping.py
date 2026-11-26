from __future__ import annotations

import base64
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "05_datasets" / "dataset_joint_m1A_m6A_101nt.tsv"
PRED = ROOT / "08_models" / "M4_structdualrm_full" / "test_predictions.tsv"
ATTRACT_DB = ROOT / "external" / "attract" / "ATtRACT" / "ATtRACT_db.txt"
ATTRACT_PWM = ROOT / "external" / "attract" / "ATtRACT" / "pwm.txt"

OUT = ROOT / "11_tables" / "biology_summary"
MOTIF_DIR = ROOT / "07_motif" / "tomtom_style"
FIG_PNG = ROOT / "10_figures" / "biology_motif_png"
FIG_SVG = ROOT / "10_figures" / "biology_motif_svg"
PKG = ROOT / "submission" / "nature_communications_structdualrm"

BASES = ["A", "C", "G", "U"]
BASE_COLORS = {"A": "#dc2626", "C": "#2563eb", "G": "#f59e0b", "U": "#16a34a"}
QUERY_WIDTH = 7
SEARCH_FLANK = 12
MIN_OVERLAP = 5

QUERY_DEFS = [
    {"id": "Q1_m1A_CDS", "label": "m1A CDS model region", "task": "m1A", "region": "CDS", "limit": 220},
    {"id": "Q2_m1A_5UTR", "label": "m1A 5UTR model region", "task": "m1A", "region": "5UTR", "limit": 160},
    {"id": "Q3_m1A_3UTR", "label": "m1A 3UTR model region", "task": "m1A", "region": "3UTR", "limit": 120},
    {"id": "Q4_m6A_CDS", "label": "m6A CDS model region", "task": "m6A", "region": "CDS", "limit": 900},
    {"id": "Q5_m6A_3UTR", "label": "m6A 3UTR model region", "task": "m6A", "region": "3UTR", "limit": 900},
    {"id": "Q6_dual_high", "label": "dual-score model region", "task": "dual", "region": "any", "limit": 180},
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
    return seq.upper().replace("T", "U")


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


def load_joined_rows() -> list[dict[str, str]]:
    data = {row["site_id"]: row for row in read_tsv(DATASET)}
    rows = []
    for pred in read_tsv(PRED):
        row = data.get(pred["site_id"])
        if not row:
            continue
        merged = dict(row)
        merged.update(pred)
        seq = normalize_seq(merged.get("sequence_101nt", ""))
        if len(seq) != 101 or set(seq) - set(BASES):
            continue
        merged["sequence_101nt_rna"] = seq
        rows.append(merged)
    return rows


def load_valid_dataset_rows() -> list[dict[str, str]]:
    rows = []
    for row in read_tsv(DATASET):
        seq = normalize_seq(row.get("sequence_101nt", ""))
        if len(seq) != 101 or set(seq) - set(BASES):
            continue
        row["sequence_101nt_rna"] = seq
        rows.append(row)
    return rows


def model_score(row: dict[str, str], task: str) -> float:
    if task == "m1A":
        return float(row["pred_m1A_prob"])
    if task == "m6A":
        return float(row["pred_m6A_prob"])
    return math.sqrt(max(0.0, float(row["pred_m1A_prob"])) * max(0.0, float(row["pred_m6A_prob"])))


def select_model_rows(rows: list[dict[str, str]], query_def: dict[str, object]) -> list[dict[str, str]]:
    task = str(query_def["task"])
    region = str(query_def["region"])
    limit = int(query_def["limit"])
    candidates = [row for row in rows if region == "any" or row.get("region_type") == region]
    if task == "m1A":
        positives = [row for row in candidates if row.get("calibrated_m1A_label") == "1"]
    elif task == "m6A":
        positives = [row for row in candidates if row.get("calibrated_m6A_label") == "1"]
    else:
        positives = candidates
    positives.sort(key=lambda row: model_score(row, task), reverse=True)
    selected = positives[:limit]
    if len(selected) < min(30, limit):
        candidates.sort(key=lambda row: model_score(row, task), reverse=True)
        selected = candidates[:limit]
    return selected


def select_background_rows(rows: list[dict[str, str]], query_def: dict[str, object], max_rows: int = 4000) -> list[dict[str, str]]:
    region = str(query_def["region"])
    background = [row for row in rows if row.get("label_m1A") == "0" and row.get("label_m6A") == "0"]
    if region != "any":
        regional = [row for row in background if row.get("region_type") == region]
        if len(regional) >= 100:
            background = regional
    return background[:max_rows]


def select_score_reference_rows(rows: list[dict[str, str]], query_def: dict[str, object], selected_ids: set[str]) -> list[dict[str, str]]:
    region = str(query_def["region"])
    reference = [row for row in rows if row["site_id"] not in selected_ids and (region == "any" or row.get("region_type") == region)]
    if len(reference) < 30:
        reference = [row for row in rows if row["site_id"] not in selected_ids]
    return reference


def pwm_from_sequences(seqs: list[str], start: int, width: int, pseudocount: float = 0.5) -> list[dict[str, float]]:
    pwm = []
    for offset in range(width):
        counts = {base: pseudocount for base in BASES}
        for seq in seqs:
            base = seq[start + offset]
            if base in counts:
                counts[base] += 1.0
        total = sum(counts.values())
        pwm.append({base: counts[base] / total for base in BASES})
    return pwm


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    return sum(p[base] * math.log2(p[base] / max(q[base], 1e-9)) for base in BASES if p[base] > 0)


def choose_query_window(selected: list[dict[str, str]], background: list[dict[str, str]]) -> tuple[int, int, float]:
    seqs = [row["sequence_101nt_rna"] for row in selected]
    bg = [row["sequence_101nt_rna"] for row in background]
    center = 50
    best_start = center - QUERY_WIDTH // 2
    best_score = -1.0
    for rel_start in range(-SEARCH_FLANK, SEARCH_FLANK - QUERY_WIDTH + 2):
        start = center + rel_start
        query_pwm = pwm_from_sequences(seqs, start, QUERY_WIDTH)
        bg_pwm = pwm_from_sequences(bg, start, QUERY_WIDTH)
        score = sum(kl_divergence(query_pwm[i], bg_pwm[i]) for i in range(QUERY_WIDTH))
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start - center, best_score


def parse_attract_pwm() -> dict[str, list[dict[str, float]]]:
    motifs: dict[str, list[dict[str, float]]] = {}
    lines = [line.strip() for line in ATTRACT_PWM.read_text(encoding="utf-8").splitlines() if line.strip()]
    idx = 0
    while idx < len(lines):
        if not lines[idx].startswith(">"):
            idx += 1
            continue
        parts = lines[idx][1:].split()
        matrix_id = parts[0]
        length = int(parts[1])
        idx += 1
        pwm = []
        for _ in range(length):
            values = [float(x) for x in lines[idx].split()]
            total = sum(values)
            pwm.append({base: values[pos] / total for pos, base in enumerate(BASES)})
            idx += 1
        motifs[matrix_id] = pwm
    return motifs


def parse_attract_metadata() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with ATTRACT_DB.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 13:
                continue
            matrix_id = row[11]
            if row[3] != "Homo_sapiens":
                continue
            if matrix_id not in metadata:
                metadata[matrix_id] = {
                    "gene_name": row[0],
                    "gene_id": row[1],
                    "organism": row[3],
                    "known_motif": row[4].replace("T", "U"),
                    "database": row[7],
                    "family": row[10],
                    "matrix_id": matrix_id,
                    "attract_score": row[12],
                }
    return metadata


def reverse_complement_pwm(pwm: list[dict[str, float]]) -> list[dict[str, float]]:
    comp = {"A": "U", "U": "A", "C": "G", "G": "C"}
    return [{base: col[comp[base]] for base in BASES} for col in reversed(pwm)]


def column_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    avec = [a[base] for base in BASES]
    bvec = [b[base] for base in BASES]
    am = mean(avec)
    bm = mean(bvec)
    num = sum((x - am) * (y - bm) for x, y in zip(avec, bvec))
    den_a = math.sqrt(sum((x - am) ** 2 for x in avec))
    den_b = math.sqrt(sum((y - bm) ** 2 for y in bvec))
    if den_a == 0.0 or den_b == 0.0:
        return 0.0
    return num / (den_a * den_b)


def align_similarity(query_pwm: list[dict[str, float]], target_pwm: list[dict[str, float]]) -> tuple[float, int, int]:
    q_len = len(query_pwm)
    t_len = len(target_pwm)
    best = (-2.0, 0, 0)
    for offset in range(-t_len + MIN_OVERLAP, q_len - MIN_OVERLAP + 1):
        q_start = max(0, offset)
        t_start = max(0, -offset)
        overlap = min(q_len - q_start, t_len - t_start)
        if overlap < MIN_OVERLAP:
            continue
        score = sum(column_similarity(query_pwm[q_start + i], target_pwm[t_start + i]) for i in range(overlap)) / overlap
        if score > best[0]:
            best = (score, offset, overlap)
    return best


def bh_fdr(rows: list[dict[str, object]], key: str = "empirical_p_value") -> None:
    ranked = sorted(rows, key=lambda row: float(row[key]))
    m = len(ranked)
    running = 1.0
    for rank, row in reversed(list(enumerate(ranked, start=1))):
        running = min(running, float(row[key]) * m / rank)
        row["q_value_bh"] = running


def mann_whitney_p_value(x: list[float], y: list[float]) -> float:
    if not x or not y:
        return 1.0
    labelled = [(value, 0) for value in x] + [(value, 1) for value in y]
    labelled.sort(key=lambda item: item[0])
    ranks = [0.0] * len(labelled)
    i = 0
    while i < len(labelled):
        j = i + 1
        while j < len(labelled) and labelled[j][0] == labelled[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_x = sum(rank for rank, item in zip(ranks, labelled) if item[1] == 0)
    n1 = len(x)
    n2 = len(y)
    u1 = rank_x - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0
    sd_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sd_u == 0.0:
        return 1.0
    z = (u1 - mean_u) / sd_u
    return math.erfc(abs(z) / math.sqrt(2.0))


def star(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.05:
        return "**"
    return "ns"


def information_heights(pwm: list[dict[str, float]]) -> list[dict[str, float]]:
    heights = []
    for col in pwm:
        entropy = -sum(col[base] * math.log2(col[base]) for base in BASES if col[base] > 0)
        info = max(0.0, 2.0 - entropy)
        heights.append({base: col[base] * info for base in BASES})
    return heights


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


def draw_logo(canvas: Image.Image, draw: ImageDraw.ImageDraw, pwm: list[dict[str, float]], x: int, y: int, w: int, h: int) -> None:
    heights = information_heights(pwm)
    axis_font = font(11)
    letter_font = font(70, True)
    draw.line([x, y + h, x + w, y + h], fill="#334155", width=2)
    col_w = max(16, w // len(pwm))
    for idx, col in enumerate(heights):
        stack = sorted(col.items(), key=lambda item: item[1])
        cur_y = y + h
        for base, bits in stack:
            letter_h = int(round((bits / 2.0) * h))
            if letter_h < 2:
                continue
            img = render_letter(base, BASE_COLORS[base], col_w - 2, letter_h, letter_font)
            cur_y -= letter_h
            canvas.alpha_composite(img, (x + idx * col_w + 1, cur_y))
    draw.text((x - 22, y - 2), "2", font=axis_font, fill="#475569")
    draw.text((x - 22, y + h - 9), "0", font=axis_font, fill="#475569")


def make_png(matches: list[dict[str, object]], out: Path) -> None:
    width, height = 1800, 1550
    canvas = Image.new("RGBA", (width, height), "#f8fbff")
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 30), "TomTom-style motif mapping of StructDualRM model-identified regions", font=font(34, True), fill="#0f172a")
    draw.text((42, 72), "Top: known ATtRACT RBP motif. Bottom: model-selected query motif from M4 predictions. Stars mark recognition-score separation.", font=font(18), fill="#334155")
    panel_w, panel_h = 545, 440
    x0, y0 = 42, 130
    gap_x, gap_y = 35, 42
    for idx, row in enumerate(matches):
        r, c = divmod(idx, 3)
        x = x0 + c * (panel_w + gap_x)
        y = y0 + r * (panel_h + gap_y)
        draw.rounded_rectangle([x, y, x + panel_w, y + panel_h], radius=12, fill="#ffffff", outline="#bfdbfe", width=2)
        draw.text((x + 16, y + 12), str(row["query_label"]), font=font(20, True), fill="#0f172a")
        draw.text((x + panel_w - 78, y + 12), str(row["recognition_significance"]), font=font(26, True), fill="#1d4ed8")
        known = f"{row['target_gene']} / {row['known_motif']} ({row['target_database']})"
        draw.text((x + 16, y + 44), f"Known: {known[:48]}", font=font(13), fill="#475569")
        draw.text((x + 16, y + 64), f"TomTom-style empirical p={float(row['empirical_p_value']):.3g}; sim={float(row['similarity']):.3f}", font=font(13), fill="#475569")
        draw.text((x + 16, y + 94), "Known motif", font=font(14, True), fill="#0f172a")
        draw_logo(canvas, draw, row["target_pwm"], x + 70, y + 112, panel_w - 110, 105)
        draw.text((x + 16, y + 248), "Model motif", font=font(14, True), fill="#0f172a")
        draw_logo(canvas, draw, row["query_pwm"], x + 70, y + 266, panel_w - 110, 105)
        draw.text((x + 16, y + 397), f"n={row['n_selected']}; window={row['window_relative_start']}..{int(row['window_relative_start']) + QUERY_WIDTH - 1}", font=font(13), fill="#475569")
    note_y = y0 + 2 * (panel_h + gap_y) + 8
    draw.rounded_rectangle([42, note_y, width - 42, note_y + 104], radius=12, fill="#ffffff", outline="#bfdbfe", width=2)
    draw.text((68, note_y + 24), "Method note", font=font(19, True), fill="#0f172a")
    draw.text((68, note_y + 52), "Official MEME Suite TomTom was not available in this Windows environment; query/target MEME files are exported for official rerun.", font=font(15), fill="#334155")
    draw.text((68, note_y + 76), "Here, motif matching uses local TomTom-compatible PWM alignment against downloaded ATtRACT human RBP motifs.", font=font(15), fill="#334155")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out)


def make_svg_wrapper(png_path: Path, svg_path: Path) -> None:
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1550" viewBox="0 0 1800 1550">\n'
        f'<image href="data:image/png;base64,{data}" x="0" y="0" width="1800" height="1550"/>\n'
        "</svg>\n"
    )
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8", newline="\n")


def write_meme(path: Path, motifs: list[tuple[str, str, list[dict[str, float]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["MEME version 4", "", "ALPHABET= ACGU", "", "strands: + -", "", "Background letter frequencies", "A 0.25 C 0.25 G 0.25 U 0.25", ""]
    for motif_id, label, pwm in motifs:
        safe_label = label.replace(" ", "_").replace("/", "_")
        lines.append(f"MOTIF {motif_id} {safe_label}")
        lines.append(f"letter-probability matrix: alength= 4 w= {len(pwm)} nsites= 20 E= 0")
        for col in pwm:
            lines.append(" ".join(f"{col[base]:.6f}" for base in BASES))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    rows = load_joined_rows()
    dataset_rows = load_valid_dataset_rows()
    target_pwms_all = parse_attract_pwm()
    target_meta = parse_attract_metadata()
    target_pwms = {mid: pwm for mid, pwm in target_pwms_all.items() if mid in target_meta and len(pwm) >= MIN_OVERLAP}

    query_rows: list[dict[str, object]] = []
    recognition_rows: list[dict[str, object]] = []
    query_motifs: list[tuple[str, str, list[dict[str, float]]]] = []
    selected_by_query: dict[str, list[dict[str, str]]] = {}
    background_by_query: dict[str, list[dict[str, str]]] = {}

    for qdef in QUERY_DEFS:
        selected = select_model_rows(rows, qdef)
        background = select_background_rows(dataset_rows, qdef)
        if not selected or not background:
            continue
        start, rel_start, window_score = choose_query_window(selected, background)
        seqs = [row["sequence_101nt_rna"] for row in selected]
        query_pwm = pwm_from_sequences(seqs, start, QUERY_WIDTH)
        query_id = str(qdef["id"])
        query_label = str(qdef["label"])
        query_motifs.append((query_id, query_label, query_pwm))
        selected_by_query[query_id] = selected
        background_by_query[query_id] = background
        task = str(qdef["task"])
        selected_scores = [model_score(row, task) for row in selected]
        score_reference = select_score_reference_rows(rows, qdef, {row["site_id"] for row in selected})
        reference_scores = [model_score(row, task) for row in score_reference]
        p_score = mann_whitney_p_value(selected_scores, reference_scores)
        recognition_rows.append(
            {
                "query_id": query_id,
                "query_label": query_label,
                "task": task,
                "region": qdef["region"],
                "n_selected": len(selected),
                "n_motif_background": len(background),
                "n_score_reference": len(score_reference),
                "mean_selected_score": mean(selected_scores),
                "mean_reference_score": mean(reference_scores) if reference_scores else 0.0,
                "recognition_p_value": p_score,
                "recognition_significance": star(p_score),
                "window_relative_start": rel_start,
                "window_relative_end": rel_start + QUERY_WIDTH - 1,
                "window_kl_score": window_score,
            }
        )
        for pos, col in enumerate(query_pwm):
            for base in BASES:
                query_rows.append(
                    {
                        "query_id": query_id,
                        "query_label": query_label,
                        "relative_position": rel_start + pos,
                        "base": base,
                        "probability": col[base],
                    }
                )

    all_match_rows: list[dict[str, object]] = []
    figure_rows: list[dict[str, object]] = []
    rec_by_query = {row["query_id"]: row for row in recognition_rows}
    for query_id, query_label, query_pwm in query_motifs:
        raw_scores = []
        for matrix_id, target_pwm in target_pwms.items():
            f_score, f_offset, f_overlap = align_similarity(query_pwm, target_pwm)
            rc_pwm = reverse_complement_pwm(target_pwm)
            r_score, r_offset, r_overlap = align_similarity(query_pwm, rc_pwm)
            if r_score > f_score:
                score, offset, overlap, orientation, display_pwm = r_score, r_offset, r_overlap, "-", rc_pwm
            else:
                score, offset, overlap, orientation, display_pwm = f_score, f_offset, f_overlap, "+", target_pwm
            raw_scores.append((matrix_id, score, offset, overlap, orientation, display_pwm))
        raw_scores.sort(key=lambda item: item[1], reverse=True)
        total_targets = len(raw_scores)
        family_rows: list[dict[str, object]] = []
        for rank, (matrix_id, score, offset, overlap, orientation, display_pwm) in enumerate(raw_scores, start=1):
            meta = target_meta[matrix_id]
            empirical = rank / total_targets
            row = {
                "query_id": query_id,
                "query_label": query_label,
                "target_rank": rank,
                "target_matrix_id": matrix_id,
                "target_gene": meta["gene_name"],
                "target_gene_id": meta["gene_id"],
                "target_organism": meta["organism"],
                "known_motif": meta["known_motif"],
                "target_database": meta["database"],
                "target_family": meta["family"],
                "target_attract_score": meta["attract_score"],
                "orientation": orientation,
                "offset": offset,
                "overlap": overlap,
                "similarity": score,
                "empirical_p_value": empirical,
                "q_value_bh": 1.0,
            }
            family_rows.append(row)
        bh_fdr(family_rows)
        all_match_rows.extend(family_rows[:20])
        best = dict(family_rows[0])
        rec = rec_by_query[query_id]
        best.update(
            {
                "query_pwm": query_pwm,
                "target_pwm": raw_scores[0][5],
                "n_selected": rec["n_selected"],
                "recognition_p_value": rec["recognition_p_value"],
                "recognition_significance": rec["recognition_significance"],
                "window_relative_start": rec["window_relative_start"],
            }
        )
        figure_rows.append(best)

    write_tsv(
        OUT / "tomtom_style_query_motif_pwm.tsv",
        query_rows,
        ["query_id", "query_label", "relative_position", "base", "probability"],
    )
    write_tsv(
        OUT / "tomtom_style_recognition_scores.tsv",
        recognition_rows,
        [
            "query_id",
            "query_label",
            "task",
            "region",
            "n_selected",
            "n_motif_background",
            "n_score_reference",
            "mean_selected_score",
            "mean_reference_score",
            "recognition_p_value",
            "recognition_significance",
            "window_relative_start",
            "window_relative_end",
            "window_kl_score",
        ],
    )
    write_tsv(
        OUT / "tomtom_style_attract_matches.tsv",
        all_match_rows,
        [
            "query_id",
            "query_label",
            "target_rank",
            "target_matrix_id",
            "target_gene",
            "target_gene_id",
            "target_organism",
            "known_motif",
            "target_database",
            "target_family",
            "target_attract_score",
            "orientation",
            "offset",
            "overlap",
            "similarity",
            "empirical_p_value",
            "q_value_bh",
        ],
    )

    write_meme(MOTIF_DIR / "query_motifs.meme", query_motifs)
    target_motifs = [
        (matrix_id, f"{target_meta[matrix_id]['gene_name']}_{target_meta[matrix_id]['known_motif']}", pwm)
        for matrix_id, pwm in sorted(target_pwms.items(), key=lambda item: item[0])
    ]
    write_meme(MOTIF_DIR / "attract_human_rbp_targets.meme", target_motifs)
    (MOTIF_DIR / "README.md").write_text(
        "# TomTom-style motif mapping\n\n"
        "This directory contains MEME-format query motifs derived from StructDualRM M4 model-selected regions and ATtRACT human RBP target motifs.\n"
        "Official MEME Suite TomTom was not available in the current Windows environment. When tomtom is installed, rerun:\n\n"
        "```bash\n"
        "tomtom -oc official_tomtom query_motifs.meme attract_human_rbp_targets.meme\n"
        "```\n",
        encoding="utf-8",
        newline="\n",
    )

    png = FIG_PNG / "Figure_tomtom_style_motif_mapping.png"
    svg = FIG_SVG / "Figure_tomtom_style_motif_mapping.svg"
    make_png(figure_rows, png)
    make_svg_wrapper(png, svg)

    copies = [
        (png, PKG / "figures_png" / png.name),
        (svg, PKG / "figures" / svg.name),
        (OUT / "tomtom_style_query_motif_pwm.tsv", PKG / "source_data" / "Figure_tomtom_style_query_motif_pwm.tsv"),
        (OUT / "tomtom_style_recognition_scores.tsv", PKG / "source_data" / "Figure_tomtom_style_recognition_scores.tsv"),
        (OUT / "tomtom_style_attract_matches.tsv", PKG / "source_data" / "Figure_tomtom_style_attract_matches.tsv"),
    ]
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    print({"queries": len(query_motifs), "targets": len(target_pwms), "png": str(png), "matches": str(OUT / "tomtom_style_attract_matches.tsv")})


if __name__ == "__main__":
    main()
