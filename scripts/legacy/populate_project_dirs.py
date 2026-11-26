from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "00_raw"
FIG_DIR = ROOT / "10_figures"
M1A_SRC = Path(r"D:\dataset\m1A_gg")
M6A_SRC = Path(r"D:\dataset\m6A_gg")
FIG_SRC = ROOT / "submission" / "nature_communications_structdualrm"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def all_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file()])


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return "exists"
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def populate_raw() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    primary_sources: list[tuple[str, Path]] = []
    for label, src_root in [("m1A", M1A_SRC), ("m6A", M6A_SRC)]:
        for path in sorted(src_root.iterdir()) if src_root.exists() else []:
            if path.is_file():
                primary_sources.append((label, path))

    linked_rows = []
    for label, src in primary_sources:
        dst = RAW_DIR / label / src.name
        method = link_or_copy(src, dst)
        linked_rows.append(
            {
                "mod_type": label,
                "source_path": str(src),
                "project_path": str(dst.relative_to(ROOT)),
                "size_bytes": src.stat().st_size,
                "materialization": method,
            }
        )

    with (RAW_DIR / "primary_raw_files.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["mod_type", "source_path", "project_path", "size_bytes", "materialization"]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(linked_rows)

    manifest_rows = []
    for label, src_root in [("m1A", M1A_SRC), ("m6A", M6A_SRC)]:
        for path in all_files(src_root):
            rel = path.relative_to(src_root)
            manifest_rows.append(
                {
                    "mod_type": label,
                    "source_root": str(src_root),
                    "relative_path": str(rel),
                    "absolute_path": str(path),
                    "size_bytes": path.stat().st_size,
                    "included_in_00_raw": "yes" if path in [src for _, src in primary_sources] else "manifest_only",
                }
            )

    with (RAW_DIR / "raw_input_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["mod_type", "source_root", "relative_path", "absolute_path", "size_bytes", "included_in_00_raw"]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    readme = """# 00_raw

This directory records the original m1A/m6A input layer used by the StructDualRM workflow.

Contents:

- `m1A/`: top-level original m1A files from `D:\\dataset\\m1A_gg`.
- `m6A/`: top-level original m6A files from `D:\\dataset\\m6A_gg`.
- `primary_raw_files.tsv`: files materialized inside this project. Files are hardlinks when possible to avoid duplicating several GB of raw archives.
- `raw_input_manifest.tsv`: complete recursive manifest of the original external raw-data folders, including extracted files that were not copied here.

Why not copy every recursive file?

The external raw folders contain about 22 GB of archives plus extracted duplicates. This project keeps the top-level raw downloads and records a full manifest so the workflow remains traceable without unnecessary duplication.
"""
    write_text(RAW_DIR / "README.md", readme)


def populate_figures() -> None:
    svg_src = FIG_SRC / "figures"
    png_src = FIG_SRC / "figures_png"
    svg_dst = FIG_DIR / "main_figures_svg"
    png_dst = FIG_DIR / "main_figures_png"
    svg_dst.mkdir(parents=True, exist_ok=True)
    png_dst.mkdir(parents=True, exist_ok=True)

    rows = []
    for fmt, src_dir, dst_dir, pattern in [
        ("svg", svg_src, svg_dst, "Figure*.svg"),
        ("png", png_src, png_dst, "Figure*.png"),
    ]:
        for src in sorted(src_dir.glob(pattern)):
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            rows.append(
                {
                    "figure": src.stem,
                    "format": fmt,
                    "source_path": str(src.relative_to(ROOT)),
                    "project_path": str(dst.relative_to(ROOT)),
                    "size_bytes": dst.stat().st_size,
                }
            )

    with (FIG_DIR / "figure_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["figure", "format", "source_path", "project_path", "size_bytes"]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    readme = """# 10_figures

This directory contains the final manuscript figures generated for the StructDualRM draft.

Contents:

- `main_figures_svg/`: vector SVG versions of Figures 1-6.
- `main_figures_png/`: PNG versions used by Word/LaTeX conversion.
- `figure_manifest.tsv`: figure file manifest and source paths.

The figure source data tables are stored in `submission/nature_communications_structdualrm/source_data/`.
"""
    write_text(FIG_DIR / "README.md", readme)


def main() -> None:
    populate_raw()
    populate_figures()
    raw_files = len(list(RAW_DIR.rglob("*")))
    fig_files = len(list(FIG_DIR.rglob("*")))
    print({"00_raw_entries": raw_files, "10_figures_entries": fig_files})


if __name__ == "__main__":
    main()
