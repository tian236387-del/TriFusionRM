#!/usr/bin/env python
"""Draw a publication-style RNA modification metric table as SVG.

The script reads per-class metrics from model ``metrics.json`` files and writes
a dark table similar to the EvoRMD/MultiRM/TransRNAm comparison figure.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CLASS_ORDER = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
MODEL_SPECS = [
    ("EvoRMD", "evormd_adapted"),
    ("MultiRM", "multirm_official"),
    ("TransRNAm", "transrnam_official"),
]


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: object) -> Optional[float]:
    if value in {"", None}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def metric_key(metric: str) -> Tuple[str, str, str]:
    normalized = metric.strip().lower()
    if normalized in {"auc", "auc_ovr", "macro_auc_ovr"}:
        return "AUC", "AUC_ovr", "macro_AUC_ovr"
    if normalized in {"aupr", "aupr_ovr", "macro_aupr_ovr"}:
        return "AUPR", "AUPR_ovr", "macro_AUPR_ovr"
    raise ValueError("--metric must be AUC or AUPR")


def get_per_class(metrics: Mapping[str, object], mod_type: str, key: str) -> Optional[float]:
    test = metrics.get("test", {})
    if not isinstance(test, Mapping):
        return None
    per_class = test.get("per_class", {})
    if not isinstance(per_class, Mapping):
        return None
    values = per_class.get(mod_type, {})
    if not isinstance(values, Mapping):
        return None
    return as_float(values.get(key))


def get_macro(metrics: Mapping[str, object], macro_key: str, per_class_key: str, classes: Sequence[str]) -> Optional[float]:
    test = metrics.get("test", {})
    if isinstance(test, Mapping):
        macro_value = as_float(test.get(macro_key))
        if macro_value is not None:
            return macro_value
    values = [get_per_class(metrics, mod, per_class_key) for mod in classes]
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4f}"


def model_paths(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    baseline_root = Path(args.baseline_root)
    paths = [(display, baseline_root / dirname / "metrics.json") for display, dirname in MODEL_SPECS]
    if args.final_model_dir:
        paths.append((args.final_model_name, Path(args.final_model_dir) / "metrics.json"))
    return paths


def collect_table(
    paths: Sequence[Tuple[str, Path]],
    classes: Sequence[str],
    per_class_key: str,
    macro_key: str,
) -> Tuple[List[str], List[List[Optional[float]]], List[Optional[float]], Dict[str, str]]:
    names = [name for name, _ in paths]
    metrics_by_name = {name: load_json(path) for name, path in paths}
    missing = {name: str(path) for name, path in paths if not metrics_by_name.get(name)}
    rows: List[List[Optional[float]]] = []
    for mod_type in classes:
        rows.append([get_per_class(metrics_by_name[name], mod_type, per_class_key) for name in names])
    macro = [get_macro(metrics_by_name[name], macro_key, per_class_key, classes) for name in names]
    return names, rows, macro, missing


def best_mask(values: Sequence[Optional[float]], eps: float = 1e-12) -> List[bool]:
    valid = [value for value in values if value is not None]
    if not valid:
        return [False for _ in values]
    best = max(valid)
    return [value is not None and abs(value - best) <= eps for value in values]


def write_tsv(
    path: Path,
    classes: Sequence[str],
    names: Sequence[str],
    rows: Sequence[Sequence[Optional[float]]],
    macro: Sequence[Optional[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["Modification", *names])
        for mod_type, values in zip(classes, rows):
            writer.writerow([mod_type, *[fmt(value) for value in values]])
        writer.writerow(["Avg.", *[fmt(value) for value in macro]])


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int,
    weight: str = "400",
    anchor: str = "start",
    fill: str = "#f5f1ea",
    family: str = "Georgia, 'Times New Roman', serif",
    opacity: float = 1.0,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}" opacity="{opacity:.3f}">'
        f"{html.escape(text)}</text>"
    )


def draw_svg(
    path: Path,
    title: str,
    classes: Sequence[str],
    names: Sequence[str],
    rows: Sequence[Sequence[Optional[float]]],
    macro: Sequence[Optional[float]],
    *,
    watermark: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    left = 28
    right = 28
    top = 28
    title_h = 74
    header_h = 46
    row_h = 33
    avg_h = 39
    bottom = 24
    first_col = 300
    model_col = 250
    width = left + first_col + model_col * len(names) + right
    height = top + title_h + header_h + row_h * len(classes) + avg_h + bottom
    bg = "#2d3442"
    avg_bg = "#414956"
    line = "#d9dde4"
    text = "#f6f1ec"
    muted = "#cfd5df"
    best = "#ffffff"

    y_table = top + title_h
    y_header_mid = y_table + 31
    y_header_bottom = y_table + header_h
    y_avg = y_header_bottom + row_h * len(classes)
    y_bottom = y_avg + avg_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{bg}"/>',
    ]
    if watermark:
        parts.append(
            f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" font-family="Arial, sans-serif" '
            f'font-size="140" font-weight="700" text-anchor="middle" fill="#798292" opacity="0.16" '
            f'transform="rotate(-23 {width / 2:.1f} {height / 2:.1f})">{html.escape(watermark)}</text>'
        )

    parts.append(svg_text(left, top + 24, title, size=28, weight="700"))
    parts.append(f'<line x1="{left}" y1="{y_table:.1f}" x2="{width - right}" y2="{y_table:.1f}" stroke="{line}" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{y_header_bottom:.1f}" x2="{width - right}" y2="{y_header_bottom:.1f}" stroke="{line}" stroke-width="1.3" opacity="0.82"/>')
    parts.append(f'<rect x="{left}" y="{y_avg:.1f}" width="{width - left - right}" height="{avg_h}" fill="{avg_bg}" opacity="0.96"/>')
    parts.append(f'<line x1="{left}" y1="{y_bottom:.1f}" x2="{width - right}" y2="{y_bottom:.1f}" stroke="{line}" stroke-width="2"/>')

    parts.append(svg_text(left + 22, y_header_mid, "Modification", size=26, weight="700"))
    for idx, name in enumerate(names):
        x = left + first_col + model_col * idx + model_col / 2
        parts.append(svg_text(x, y_header_mid, name, size=26, weight="700", anchor="middle"))

    for row_idx, (mod_type, values) in enumerate(zip(classes, rows)):
        y = y_header_bottom + row_h * row_idx + 24
        mask = best_mask(values)
        parts.append(svg_text(left + 22, y, mod_type, size=24, weight="500"))
        for idx, value in enumerate(values):
            x = left + first_col + model_col * idx + model_col / 2
            parts.append(
                svg_text(
                    x,
                    y,
                    fmt(value),
                    size=24,
                    weight="700" if mask[idx] else "400",
                    anchor="middle",
                    fill=best if mask[idx] else text,
                )
            )

    macro_mask = best_mask(macro)
    y = y_avg + 27
    parts.append(svg_text(left + 22, y, "Avg.", size=25, weight="700"))
    for idx, value in enumerate(macro):
        x = left + first_col + model_col * idx + model_col / 2
        parts.append(
            svg_text(
                x,
                y,
                fmt(value),
                size=25,
                weight="700" if macro_mask[idx] else "500",
                anchor="middle",
                fill=best if macro_mask[idx] else muted,
            )
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw RNA modification AUC/AUPR comparison table.")
    parser.add_argument("--metric", default="AUC", help="AUC or AUPR.")
    parser.add_argument(
        "--baseline-root",
        default="rmbase11_full_work/official_reproduction_benchmark/cdhit80_strict_split",
    )
    parser.add_argument(
        "--final-model-dir",
        default="rmbase11_full_work/models/encode_top30_rbp_hparam_sweep/rbp_s010_l0000_initm15_drop04_lr3e4/no_position_rbp_pos",
        help="Directory containing the final model metrics.json. Set empty to omit the proposed model.",
    )
    parser.add_argument("--final-model-name", default="StructDualRM-RBP")
    parser.add_argument("--classes", default=",".join(CLASS_ORDER))
    parser.add_argument("--out", default="", help="Output SVG path. Defaults to rmbase11_full_work/figures/final_model_<metric>_table.svg")
    parser.add_argument("--tsv-out", default="", help="Optional TSV output path. Defaults next to SVG.")
    parser.add_argument("--watermark", default="", help="Optional diagonal watermark text, e.g. PREPRINT.")
    parser.add_argument("--title", default="", help="Optional custom title.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    metric_label, per_class_key, macro_key = metric_key(args.metric)
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    paths = model_paths(args)
    names, rows, macro, missing = collect_table(paths, classes, per_class_key, macro_key)
    suffix = metric_label.lower()
    out = Path(args.out or f"rmbase11_full_work/figures/final_model_{suffix}_table.svg")
    tsv_out = Path(args.tsv_out or out.with_suffix(".tsv"))
    title = args.title or (
        f"Table: {metric_label} comparison of RNA modifications across baseline and StructDualRM-RBP models."
    )
    write_tsv(tsv_out, classes, names, rows, macro)
    draw_svg(out, title, classes, names, rows, macro, watermark=args.watermark)
    if missing:
        print("Missing metrics.json for:")
        for name, path in missing.items():
            print(f"  {name}: {path}")
    print(f"Wrote {out}")
    print(f"Wrote {tsv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
