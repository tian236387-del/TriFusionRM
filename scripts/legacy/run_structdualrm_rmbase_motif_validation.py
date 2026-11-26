#!/usr/bin/env python
"""Validate frozen StructDualRM motifs against strict-train class references."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_structdualrm_clustered_tomtom import benjamini_hochberg  # noqa: E402
from run_structdualrm_official_tomtom import (  # noqa: E402
    BASES,
    draw_logo,
    parse_meme_for_plot,
    parse_tomtom,
    pwm_consensus,
    pwm_information,
    read_tsv,
    run_tomtom,
    write_meme,
    write_tsv,
)


DISPLAY_NAMES = {"AtoI": "A-to-I"}
BASE_TO_INDEX = {base: index for index, base in enumerate(BASES)}
INK = "#243241"
GOLD = "#C9942E"
BLUE = "#2D6A8A"
GRAY = "#B4BDC6"


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def normalize_sequence(sequence: str) -> str:
    return sequence.strip().upper().replace("T", "U")


def build_train_references(
    dataset: Path,
    classes: Sequence[str],
    width: int,
    pseudocount: float,
) -> tuple[list[tuple[str, str, np.ndarray]], dict[str, int], list[dict[str, object]]]:
    if width % 2 != 1:
        raise ValueError("Reference width must be odd")
    class_set = set(classes)
    half_width = width // 2
    counts = {name: np.full((width, 4), pseudocount, dtype=np.float64) for name in classes}
    raw_counts = defaultdict(int)
    unique_hashes = {name: set() for name in classes}
    unique_counts = defaultdict(int)
    with dataset.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            mod_type = row.get("mod_type", "")
            if mod_type not in class_set or row.get("split_cdhit80_cluster") != "train":
                continue
            sequence = normalize_sequence(row.get("sequence_201nt", ""))
            if len(sequence) != 201:
                continue
            raw_counts[mod_type] += 1
            digest = hashlib.blake2b(sequence.encode("ascii"), digest_size=16).digest()
            if digest in unique_hashes[mod_type]:
                continue
            unique_hashes[mod_type].add(digest)
            center = len(sequence) // 2
            region = sequence[center - half_width : center + half_width + 1]
            if len(region) != width or any(base not in BASE_TO_INDEX for base in region):
                continue
            unique_counts[mod_type] += 1
            for position, base in enumerate(region):
                counts[mod_type][position, BASE_TO_INDEX[base]] += 1.0

    motifs: list[tuple[str, str, np.ndarray]] = []
    support: dict[str, int] = {}
    summary: list[dict[str, object]] = []
    for mod_type in classes:
        matrix = counts[mod_type]
        matrix /= matrix.sum(axis=1, keepdims=True)
        motif_id = f"RMBASE_TRAIN_{mod_type}"
        motifs.append((motif_id, f"{mod_type}_strict_train_reference", matrix))
        support[motif_id] = unique_counts[mod_type]
        summary.append(
            {
                "mod_type": mod_type,
                "reference_width": width,
                "train_raw_support": raw_counts[mod_type],
                "train_unique_201nt": unique_counts[mod_type],
                "consensus": pwm_consensus(matrix),
                "information_bits": pwm_information(matrix),
            }
        )
    return motifs, support, summary


def add_query_and_target_metadata(
    matches: list[dict[str, object]],
    component_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    component_by_query = {row["query_id"]: row for row in component_rows}
    enriched = []
    for row in matches:
        query_id = str(row["query_id"])
        component = component_by_query[query_id]
        target_mod = str(row["target_id"]).removeprefix("RMBASE_TRAIN_")
        enriched.append(
            {
                **row,
                "mod_type": component["mod_type"],
                "component": component["component"],
                "selected_width": component["selected_width"],
                "test_raw_support": component["test_raw_support"],
                "test_unique_201nt": component["test_unique_201nt"],
                "validation_test_pwm_pearson": component["validation_test_pwm_pearson"],
                "validation_bootstrap_q05_r": component["validation_bootstrap_q05_r"],
                "target_mod_type": target_mod,
                "same_class": int(component["mod_type"] == target_mod),
            }
        )
    return enriched


def representative_same_class_matches(
    matches: Sequence[Mapping[str, object]],
    class_order: Sequence[str],
    target_count: int,
    minimum_raw_support: int,
    minimum_unique_support: int,
    minimum_consistency: float,
    minimum_bootstrap_q05: float,
) -> list[dict[str, object]]:
    rows = []
    for mod_type in class_order:
        query_id = f"{mod_type}__C1_W"
        query_rows = [
            row
            for row in matches
            if str(row["query_id"]).startswith(query_id)
        ]
        if not query_rows:
            continue
        same_class = next(row for row in query_rows if row["target_mod_type"] == mod_type)
        best = min(query_rows, key=lambda row: (float(row["p_value"]), float(row["q_value"])))
        consistency = float(same_class["validation_test_pwm_pearson"] or "nan")
        bootstrap_q05 = float(same_class["validation_bootstrap_q05_r"] or "nan")
        raw_support = int(same_class["test_raw_support"])
        unique_support = int(same_class["test_unique_201nt"])
        stable = (
            raw_support >= minimum_raw_support
            and unique_support >= minimum_unique_support
            and math.isfinite(consistency)
            and consistency >= minimum_consistency
            and math.isfinite(bootstrap_q05)
            and bootstrap_q05 >= minimum_bootstrap_q05
        )
        rows.append(
            {
                **same_class,
                "best_target_mod_type": best["target_mod_type"],
                "same_class_is_best": int(best["target_mod_type"] == mod_type),
                "query_target_bonferroni_p": min(1.0, float(same_class["p_value"]) * target_count),
                "motif_stable": int(stable),
            }
        )
    adjusted = benjamini_hochberg([float(row["query_target_bonferroni_p"]) for row in rows])
    for row, q_value in zip(rows, adjusted):
        row["across_class_hierarchical_q"] = q_value
        row["main_figure_eligible"] = int(bool(row["motif_stable"]) and q_value < 0.05)
    return rows


def alignment_ranges(query_width: int, target_width: int, offset: int) -> tuple[tuple[int, int], tuple[int, int]]:
    query_start = max(0, -offset)
    query_end = min(query_width, target_width - offset)
    target_start = max(0, offset)
    target_end = min(target_width, query_width + offset)
    return (query_start, query_end), (target_start, target_end)


def add_alignment_box(ax: plt.Axes, start: int, end: int, color: str) -> None:
    if end <= start:
        return
    ax.add_patch(
        Rectangle(
            (start + 0.03, 0.03),
            end - start - 0.06,
            1.97,
            fill=False,
            edgecolor=color,
            linewidth=1.4,
            linestyle=(0, (4, 2)),
            zorder=20,
        )
    )


def save_figure(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 360} if extension == "png" else {}
        figure.savefig(base.with_suffix(f".{extension}"), facecolor="white", bbox_inches="tight", **kwargs)
    plt.close(figure)


def plot_motif_pairs(
    rows: Sequence[Mapping[str, object]],
    query_pwms: Mapping[str, np.ndarray],
    target_pwms: Mapping[str, np.ndarray],
    output_base: Path,
) -> None:
    columns = 2
    outer_rows = math.ceil(len(rows) / columns)
    figure = plt.figure(figsize=(11.2, 2.15 * outer_rows + 0.7), constrained_layout=True)
    outer = figure.add_gridspec(outer_rows, columns, hspace=0.38, wspace=0.2)
    for index, row in enumerate(rows):
        outer_row, outer_column = divmod(index, columns)
        subgrid = outer[outer_row, outer_column].subgridspec(1, 2, wspace=0.08)
        query_ax = figure.add_subplot(subgrid[0])
        target_ax = figure.add_subplot(subgrid[1])
        query_id = str(row["query_id"])
        target_id = str(row["target_id"])
        query_pwm = query_pwms[query_id]
        target_pwm = target_pwms[target_id]
        draw_logo(query_ax, query_pwm)
        draw_logo(target_ax, target_pwm)
        if int(row["main_figure_eligible"]):
            color = GOLD
        elif float(row["q_value"]) < 0.05:
            color = BLUE
        else:
            color = GRAY
        query_range, target_range = alignment_ranges(
            query_pwm.shape[0],
            target_pwm.shape[0],
            int(row["optimal_offset"]),
        )
        add_alignment_box(query_ax, *query_range, color)
        add_alignment_box(target_ax, *target_range, color)
        display_name = DISPLAY_NAMES.get(str(row["mod_type"]), str(row["mod_type"]))
        query_ax.set_title(f"{display_name} | Model C1", loc="left", fontsize=9.5, weight="bold", color=INK)
        target_ax.set_title("Strict-train reference", loc="left", fontsize=9.5, weight="bold", color=INK)
        target_ax.text(
            0.99,
            0.98,
            f"q={float(row['q_value']):.3g} | rank={row['rank_within_query']}/{len(rows)}",
            transform=target_ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color=INK,
        )
        query_ax.set_ylabel("Bits", fontsize=7)
        target_ax.set_ylabel("")
        target_ax.set_yticklabels([])
        query_ax.set_xlabel("Model motif", fontsize=7.5)
        target_ax.set_xlabel("RMBase/MultiRM-derived motif", fontsize=7.5)
    for index in range(len(rows), outer_rows * columns):
        outer_row, outer_column = divmod(index, columns)
        empty_ax = figure.add_subplot(outer[outer_row, outer_column])
        empty_ax.axis("off")
    figure.suptitle(
        "Frozen StructDualRM Motifs versus Strict-Train Modification References",
        x=0.01,
        ha="left",
        fontsize=13,
        weight="bold",
        color=INK,
    )
    save_figure(figure, output_base)


def plot_similarity_heatmap(
    matches: Sequence[Mapping[str, object]],
    class_order: Sequence[str],
    output_base: Path,
) -> None:
    matrix = np.zeros((len(class_order), len(class_order)), dtype=np.float64)
    by_key = {
        (str(row["mod_type"]), str(row["target_mod_type"])): row
        for row in matches
        if int(row["component"]) == 1
    }
    for row_index, query_class in enumerate(class_order):
        for column_index, target_class in enumerate(class_order):
            q_value = float(by_key[(query_class, target_class)]["q_value"])
            matrix[row_index, column_index] = -math.log10(max(q_value, 1e-6))
    labels = [DISPLAY_NAMES.get(name, name) for name in class_order]
    figure, ax = plt.subplots(figsize=(8.2, 7.2))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=max(2.0, float(matrix.max())), aspect="equal")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Strict-train reference modification")
    ax.set_ylabel("Frozen model motif (component 1)")
    ax.set_title("Cross-Modification Tomtom Similarity", loc="left", fontsize=12, weight="bold")
    for index in range(len(labels)):
        ax.add_patch(Rectangle((index - 0.48, index - 0.48), 0.96, 0.96, fill=False, edgecolor="white", linewidth=1.2))
    colorbar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("-log10(Tomtom q-value)")
    figure.tight_layout()
    save_figure(figure, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv")
    parser.add_argument("--query-meme", default="tomtom_structdualrm_clustered_20260803/motifs/structdualrm_test_validation_frozen_components.meme")
    parser.add_argument("--component-summary", default="tomtom_structdualrm_clustered_20260803/tables/motif_component_summary.tsv")
    parser.add_argument("--out-dir", default="tomtom_structdualrm_rmbase_validation_20260803")
    parser.add_argument("--reference-width", type=int, default=11)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    parser.add_argument("--tomtom-command", default="tomtom")
    parser.add_argument("--min-overlap", type=int, default=5)
    parser.add_argument("--minimum-raw-support", type=int, default=20)
    parser.add_argument("--minimum-unique-support", type=int, default=10)
    parser.add_argument("--minimum-consistency", type=float, default=0.5)
    parser.add_argument("--minimum-bootstrap-q05", type=float, default=0.5)
    args = parser.parse_args()

    dataset = resolve(args.dataset)
    query_meme = resolve(args.query_meme)
    component_summary_path = resolve(args.component_summary)
    output = resolve(args.out_dir)
    tables = output / "tables"
    motifs = output / "motifs"
    figures = output / "figures"
    official = output / "official_tomtom"
    output.mkdir(parents=True, exist_ok=True)

    component_rows = read_tsv(component_summary_path)
    class_order = []
    for row in component_rows:
        if row["mod_type"] not in class_order:
            class_order.append(row["mod_type"])
    references, reference_support, reference_summary = build_train_references(
        dataset,
        class_order,
        args.reference_width,
        args.pseudocount,
    )
    print(
        f"Built {len(references)} strict-train reference motifs from "
        f"{sum(reference_support.values())} unique sequences",
        flush=True,
    )
    target_meme = motifs / "strict_train_rmbase_multirm_references.meme"
    write_meme(target_meme, references, reference_support)
    write_tsv(tables / "strict_train_reference_summary.tsv", reference_summary, list(reference_summary[0]))

    print("Running official Tomtom", flush=True)
    run_tomtom(args.tomtom_command, query_meme, target_meme, official, args.min_overlap)
    target_meta = {
        motif_id: {
            "target_gene": mod_type,
            "target_consensus": pwm_consensus(pwm),
            "target_matrix_id": motif_id,
            "target_database": "strict_train_RMBase_MultiRM",
        }
        for motif_id, _, pwm in references
        for mod_type in [motif_id.removeprefix("RMBASE_TRAIN_")]
    }
    matches = parse_tomtom(official / "tomtom.tsv", target_meta)
    matches = add_query_and_target_metadata(matches, component_rows)
    write_tsv(tables / "official_tomtom_all_matches.tsv", matches, list(matches[0]))
    representatives = representative_same_class_matches(
        matches,
        class_order,
        len(references),
        args.minimum_raw_support,
        args.minimum_unique_support,
        args.minimum_consistency,
        args.minimum_bootstrap_q05,
    )
    write_tsv(tables / "representative_same_class_matches.tsv", representatives, list(representatives[0]))

    query_pwms = {motif_id: pwm for motif_id, _, pwm in parse_meme_for_plot(query_meme)}
    target_pwms = {motif_id: pwm for motif_id, _, pwm in references}
    plot_motif_pairs(
        representatives,
        query_pwms,
        target_pwms,
        figures / "figure_rmbase_modification_motif_pairs",
    )
    plot_similarity_heatmap(
        matches,
        class_order,
        figures / "figure_rmbase_cross_modification_heatmap",
    )
    manifest = {
        "purpose": "EvoRMD-style validation of learned modification motifs against split-independent class references",
        "reference_source": "unique 201-nt positive sequences from split_cdhit80_cluster=train",
        "reference_window": f"centered {args.reference_width} nt",
        "query_source": "validation-selected frozen StructDualRM test motif components",
        "representative_query": "component 1, ordered by validation mixture weight",
        "tomtom": {"distance": "pearson", "minimum_overlap": args.min_overlap, "norc": True},
        "interpretation_boundary": "internal split-concordance, not an independent external database validation",
        "classes": class_order,
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"classes": len(class_order), "matches": len(matches), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
