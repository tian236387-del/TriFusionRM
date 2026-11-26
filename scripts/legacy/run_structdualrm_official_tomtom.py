#!/usr/bin/env python
"""Extract sequence-branch motifs and run official MEME Suite Tomtom.

The extraction uses in-silico single-nucleotide mutagenesis on frozen
StructDualRM seed models.  For each correctly classified positive sample, the
highest-attribution fixed-width region is retained.  Regions are aggregated
into class-specific PWMs and compared with human ATtRACT RBP motifs by the
official ``tomtom`` executable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from train_motif_struct_fusion_v4 import MotifStructFusionV4, MotifStructV4Config  # noqa: E402
from train_paired_gat_rbp_fusion import configure_classes, encode_sequence  # noqa: E402


BASES = ("A", "C", "G", "U")
BASE_TO_ID = {"A": 1, "C": 2, "G": 3, "U": 4, "T": 4}
BASE_COLORS = {"A": "#d62728", "C": "#1f77b4", "G": "#f2a900", "U": "#2ca02c"}
DEFAULT_CLASSES = ("Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "Y", "AtoI")
DEFAULT_CHECKPOINTS = (
    "rmbase12_work/models/final_modality_factorial_ablation_seed42_20260726/full_no_pair_edges/checkpoint.pt",
    "rmbase12_work/models/no_pair_edges_multiseed_20260730/seed17/checkpoint.pt",
    "rmbase12_work/models/no_pair_edges_multiseed_20260730/seed73/checkpoint.pt",
)
DEFAULT_WEIGHTS = (0.2, 0.4, 0.4)


@dataclass(frozen=True)
class SelectedSample:
    partition: str
    site_id: str
    mod_type: str
    score: float


@dataclass
class FrozenSeedModel:
    name: str
    weight: float
    model: MotifStructFusionV4


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def normalize_sequence(sequence: str) -> str:
    return sequence.upper().replace("T", "U")


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_float_csv(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in parse_csv(value))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def select_prediction_samples(
    prediction_path: Path,
    partition: str,
    class_names: Sequence[str],
    max_samples: int,
) -> list[SelectedSample]:
    rows = read_tsv(prediction_path)
    selected: list[SelectedSample] = []
    for mod_type in class_names:
        probability_field = f"prob_{mod_type}"
        candidates = [
            row
            for row in rows
            if row.get("true_mod_type") == mod_type
            and row.get("pred_mod_type") == mod_type
            and probability_field in row
        ]
        candidates.sort(key=lambda row: float(row[probability_field]), reverse=True)
        for row in candidates[:max_samples]:
            selected.append(
                SelectedSample(
                    partition=partition,
                    site_id=row["site_id"],
                    mod_type=mod_type,
                    score=float(row[probability_field]),
                )
            )
    return selected


def load_selected_sequences(dataset_path: Path, selected: Sequence[SelectedSample]) -> dict[str, dict[str, str]]:
    wanted = {sample.site_id for sample in selected}
    records: dict[str, dict[str, str]] = {}
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"site_id", "sequence_41nt", "sequence_101nt", "sequence_201nt"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")
        for row in reader:
            site_id = row["site_id"]
            if site_id not in wanted:
                continue
            records[site_id] = {
                "sequence_41nt": normalize_sequence(row["sequence_41nt"]),
                "sequence_101nt": normalize_sequence(row["sequence_101nt"]),
                "sequence_201nt": normalize_sequence(row["sequence_201nt"]),
            }
            if len(records) == len(wanted):
                break
    missing_ids = wanted.difference(records)
    if missing_ids:
        example = sorted(missing_ids)[0]
        raise ValueError(f"Missing {len(missing_ids)} selected sites in dataset; first={example}")
    for site_id, row in records.items():
        for length in (41, 101, 201):
            sequence = row[f"sequence_{length}nt"]
            if len(sequence) != length or set(sequence).difference(BASES):
                raise ValueError(f"Invalid {length}-nt sequence for {site_id}")
    return records


def load_seed_models(
    checkpoint_paths: Sequence[Path],
    weights: Sequence[float],
    class_names: Sequence[str],
    device: torch.device,
) -> list[FrozenSeedModel]:
    if len(checkpoint_paths) != len(weights):
        raise ValueError("The number of checkpoints and ensemble weights must match.")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Ensemble weights must have a positive sum.")
    configure_classes(list(class_names))
    models: list[FrozenSeedModel] = []
    for checkpoint_path, raw_weight in zip(checkpoint_paths, weights):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = dict(checkpoint["config"])
        config["num_classes"] = len(class_names)
        model = MotifStructFusionV4(MotifStructV4Config(**config))
        state = checkpoint.get("best_state_dict") or checkpoint["model_state_dict"]
        model.load_state_dict(state)
        model.eval().to(device)
        models.append(
            FrozenSeedModel(
                name=checkpoint_path.parent.name,
                weight=float(raw_weight) / total_weight,
                model=model,
            )
        )
    return models


def build_mutants(sequences: Mapping[int, str]) -> tuple[dict[int, torch.Tensor], np.ndarray]:
    max_length = max(sequences)
    mutations: list[tuple[int, int]] = []
    original = sequences[max_length]
    for global_position, original_base in enumerate(original):
        for alternate in BASES:
            if alternate != original_base:
                mutations.append((global_position, BASE_TO_ID[alternate]))
    variant_count = len(mutations)
    tensors: dict[int, torch.Tensor] = {}
    for length, sequence in sequences.items():
        base_tensor = encode_sequence(sequence, length).long()
        variants = base_tensor.unsqueeze(0).repeat(variant_count, 1)
        global_start = (max_length - length) // 2
        global_end = global_start + length
        for row_index, (global_position, alternate_id) in enumerate(mutations):
            if global_start <= global_position < global_end:
                variants[row_index, global_position - global_start] = alternate_id
        tensors[length] = variants
    positions = np.asarray([position for position, _ in mutations], dtype=np.int64)
    return tensors, positions


def class_margin(logits: torch.Tensor, class_index: int) -> torch.Tensor:
    target = logits[:, class_index]
    mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
    mask[class_index] = False
    return target - torch.logsumexp(logits[:, mask], dim=-1)


@torch.inference_mode()
def ensemble_sequence_margin(
    models: Sequence[FrozenSeedModel],
    sequences: Mapping[int, torch.Tensor],
    class_index: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows = next(iter(sequences.values())).shape[0]
    scores: list[np.ndarray] = []
    for start in range(0, rows, batch_size):
        end = min(rows, start + batch_size)
        seq_multi = {f"seq_len_{length}": tensor[start:end].to(device) for length, tensor in sequences.items()}
        seq41 = seq_multi["seq_len_41"]
        ensemble_margin = torch.zeros(end - start, device=device)
        for frozen in models:
            output = frozen.model.seq_branch(seq41, seq_multi)  # type: ignore[operator]
            ensemble_margin += frozen.weight * class_margin(output["logits"], class_index)
        scores.append(ensemble_margin.cpu().numpy())
    return np.concatenate(scores)


def sample_attribution(
    models: Sequence[FrozenSeedModel],
    sequence_row: Mapping[str, str],
    class_index: int,
    device: torch.device,
    batch_size: int,
) -> tuple[float, np.ndarray]:
    raw_sequences = {length: sequence_row[f"sequence_{length}nt"] for length in (41, 101, 201)}
    original_tensors = {length: encode_sequence(sequence, length).long().unsqueeze(0) for length, sequence in raw_sequences.items()}
    original_margin = float(
        ensemble_sequence_margin(models, original_tensors, class_index, device, batch_size)[0]
    )
    mutant_tensors, mutant_positions = build_mutants(raw_sequences)
    mutant_margins = ensemble_sequence_margin(models, mutant_tensors, class_index, device, batch_size)
    effects = np.zeros(201, dtype=np.float64)
    for position in range(201):
        position_scores = mutant_margins[mutant_positions == position]
        effects[position] = original_margin - float(position_scores.mean())
    return original_margin, effects


def best_window(effects: np.ndarray, width: int) -> tuple[int, float]:
    positive = np.maximum(effects, 0.0)
    scores = np.convolve(positive, np.ones(width, dtype=np.float64), mode="valid")
    center = (len(effects) - 1) / 2.0
    starts = np.arange(len(scores))
    distances = np.abs(starts + (width - 1) / 2.0 - center)
    best_score = float(scores.max())
    tied = np.flatnonzero(np.isclose(scores, best_score, rtol=1e-10, atol=1e-12))
    if tied.size == 0:
        return int(np.argmin(distances)), 0.0
    best_start = int(tied[np.argmin(distances[tied])])
    return best_start, best_score


def pwm_from_regions(regions: Sequence[str], pseudocount: float) -> np.ndarray:
    if not regions:
        raise ValueError("Cannot build a PWM without attributed regions.")
    width = len(regions[0])
    counts = np.full((width, len(BASES)), float(pseudocount), dtype=np.float64)
    base_index = {base: index for index, base in enumerate(BASES)}
    for region in regions:
        if len(region) != width:
            raise ValueError("Attributed regions have inconsistent widths.")
        for position, base in enumerate(region):
            counts[position, base_index[base]] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def pwm_consensus(pwm: np.ndarray) -> str:
    return "".join(BASES[int(index)] for index in np.argmax(pwm, axis=1))


def pwm_information(pwm: np.ndarray) -> float:
    entropy = -(pwm * np.log2(np.clip(pwm, 1e-12, 1.0))).sum(axis=1)
    return float(np.maximum(0.0, 2.0 - entropy).sum())


def extract_partition(
    partition: str,
    samples: Sequence[SelectedSample],
    sequence_rows: Mapping[str, Mapping[str, str]],
    models: Sequence[FrozenSeedModel],
    class_names: Sequence[str],
    motif_width: int,
    pseudocount: float,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], list[dict[str, object]]]:
    class_index = {name: index for index, name in enumerate(class_names)}
    regions_by_class: dict[str, list[str]] = defaultdict(list)
    region_rows: list[dict[str, object]] = []
    total = len(samples)
    for sample_number, sample in enumerate(samples, start=1):
        sequence_row = sequence_rows[sample.site_id]
        margin, effects = sample_attribution(
            models,
            sequence_row,
            class_index[sample.mod_type],
            device,
            batch_size,
        )
        start, window_score = best_window(effects, motif_width)
        end = start + motif_width
        region = sequence_row["sequence_201nt"][start:end]
        regions_by_class[sample.mod_type].append(region)
        relative_start = start - 100
        region_rows.append(
            {
                "partition": partition,
                "site_id": sample.site_id,
                "mod_type": sample.mod_type,
                "ensemble_probability": sample.score,
                "original_sequence_margin": margin,
                "region_start_0based_201nt": start,
                "region_end_exclusive_201nt": end,
                "relative_start": relative_start,
                "relative_end": relative_start + motif_width - 1,
                "attributed_region": region,
                "positive_attribution_sum": window_score,
                "mean_position_attribution": float(effects[start:end].mean()),
                "position_attribution_csv": ",".join(f"{value:.8g}" for value in effects),
            }
        )
        print(
            f"[{partition}] {sample_number}/{total} {sample.mod_type} {sample.site_id} "
            f"region={region} attribution={window_score:.4g}",
            flush=True,
        )
    pwm_by_class = {
        mod_type: pwm_from_regions(regions, pseudocount)
        for mod_type, regions in regions_by_class.items()
        if regions
    }
    summary_rows: list[dict[str, object]] = []
    for mod_type in class_names:
        regions = regions_by_class.get(mod_type, [])
        pwm = pwm_by_class.get(mod_type)
        summary_rows.append(
            {
                "partition": partition,
                "mod_type": mod_type,
                "n_attributed_regions": len(regions),
                "evidence_tier": "primary" if len(regions) >= 20 else ("exploratory" if regions else "unavailable"),
                "consensus": pwm_consensus(pwm) if pwm is not None else "",
                "information_content_bits": pwm_information(pwm) if pwm is not None else "",
                "median_positive_attribution_sum": float(
                    np.median(
                        [float(row["positive_attribution_sum"]) for row in region_rows if row["mod_type"] == mod_type]
                    )
                )
                if regions
                else "",
            }
        )
    return region_rows, pwm_by_class, summary_rows


def write_meme(path: Path, motifs: Sequence[tuple[str, str, np.ndarray]], nsites: Mapping[str, int] | None = None) -> None:
    lines = [
        "MEME version 4",
        "",
        "ALPHABET= ACGU",
        "",
        "strands: +",
        "",
        "Background letter frequencies",
        "A 0.25 C 0.25 G 0.25 U 0.25",
        "",
    ]
    for motif_id, label, pwm in motifs:
        safe_label = label.replace(" ", "_").replace("/", "_")
        site_count = int(nsites.get(motif_id, 20)) if nsites else 20
        lines.extend(
            [
                f"MOTIF {motif_id} {safe_label}",
                f"letter-probability matrix: alength= 4 w= {pwm.shape[0]} nsites= {max(1, site_count)} E= 0",
            ]
        )
        lines.extend(" ".join(f"{value:.8f}" for value in row) for row in pwm)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_attract(
    database_path: Path,
    pwm_path: Path,
) -> tuple[list[tuple[str, str, np.ndarray]], dict[str, dict[str, str]]]:
    metadata_rows = read_tsv(database_path)
    human_by_matrix: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata_rows:
        if row.get("Organism", "").lower() == "homo_sapiens":
            human_by_matrix[row["Matrix_id"]].append(row)
    matrices: dict[str, np.ndarray] = {}
    current_id = ""
    current_rows: list[list[float]] = []

    def save_current() -> None:
        if current_id and current_rows and current_id in human_by_matrix:
            matrix = np.asarray(current_rows, dtype=np.float64)
            if matrix.ndim == 2 and matrix.shape[1] == 4:
                matrices[current_id] = matrix / matrix.sum(axis=1, keepdims=True)

    with pwm_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                save_current()
                current_id = line[1:].split()[0]
                current_rows = []
            else:
                values = [float(value) for value in line.split()]
                if len(values) == 4:
                    current_rows.append(values)
    save_current()

    motifs: list[tuple[str, str, np.ndarray]] = []
    target_meta: dict[str, dict[str, str]] = {}
    for matrix_id in sorted(matrices):
        rows = sorted(
            human_by_matrix[matrix_id],
            key=lambda row: ("**" not in row.get("Score", ""), row.get("Gene_name", ""), row.get("Motif", "")),
        )
        representative = rows[0]
        motif_id = f"AT_{matrix_id}"
        label = f"{representative['Gene_name']}|{representative['Motif']}"
        motifs.append((motif_id, label, matrices[matrix_id]))
        target_meta[motif_id] = {
            "target_gene": representative.get("Gene_name", ""),
            "target_consensus": representative.get("Motif", ""),
            "target_matrix_id": matrix_id,
            "target_database": representative.get("Database", ""),
            "target_score": representative.get("Score", ""),
        }
    return motifs, target_meta


def run_tomtom(tomtom_command: str, query_meme: Path, target_meme: Path, output_dir: Path, min_overlap: int) -> None:
    executable = shutil.which(tomtom_command) if not Path(tomtom_command).exists() else tomtom_command
    if not executable:
        raise FileNotFoundError(f"Official Tomtom executable not found: {tomtom_command}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    command = [
        str(executable),
        "-oc",
        str(output_dir),
        "-dist",
        "pearson",
        "-thresh",
        "1",
        "-min-overlap",
        str(min_overlap),
        "-norc",
        str(query_meme),
        str(target_meme),
    ]
    subprocess.run(command, check=True)


def parse_tomtom(
    tomtom_tsv: Path,
    target_meta: Mapping[str, Mapping[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_row in read_tsv(tomtom_tsv):
        row = {str(key).lstrip("#"): value for key, value in raw_row.items() if key is not None}
        if not row or not row.get("Query_ID") or row.get("Query_ID", "").startswith("#"):
            continue
        metadata = target_meta.get(row["Target_ID"], {})
        rows.append(
            {
                "query_id": row["Query_ID"],
                "target_id": row["Target_ID"],
                "target_gene": metadata.get("target_gene", ""),
                "target_consensus": metadata.get("target_consensus", ""),
                "target_matrix_id": metadata.get("target_matrix_id", ""),
                "target_database": metadata.get("target_database", ""),
                "optimal_offset": row.get("Optimal_offset", ""),
                "p_value": float(row["p-value"]),
                "e_value": float(row["E-value"]),
                "q_value": float(row["q-value"]),
                "overlap": int(row.get("Overlap", 0)),
                "orientation": row.get("Orientation", "+"),
                "query_consensus": row.get("Query_consensus", ""),
                "tomtom_target_consensus": row.get("Target_consensus", ""),
            }
        )
    rows.sort(key=lambda row: (float(row["q_value"]), float(row["p_value"])))
    rank_by_query: dict[str, int] = defaultdict(int)
    for row in rows:
        query_id = str(row["query_id"])
        rank_by_query[query_id] += 1
        row["rank_within_query"] = rank_by_query[query_id]
        row["significant_q_lt_0_05"] = int(float(row["q_value"]) < 0.05)
    return rows


def motif_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return float("nan")
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    if np.std(first_flat) == 0 or np.std(second_flat) == 0:
        return 0.0
    return float(np.corrcoef(first_flat, second_flat)[0, 1])


def draw_logo(ax: plt.Axes, pwm: np.ndarray) -> None:
    information = np.maximum(0.0, 2.0 + (pwm * np.log2(np.clip(pwm, 1e-12, 1.0))).sum(axis=1))
    heights = pwm * information[:, None]
    font = FontProperties(family="DejaVu Sans", weight="bold")
    for position, column in enumerate(heights):
        baseline = 0.0
        for base_index in np.argsort(column):
            height = float(column[base_index])
            if height <= 1e-4:
                continue
            base = BASES[int(base_index)]
            path = TextPath((0, 0), base, size=1, prop=font)
            bounds = path.get_extents()
            transform = (
                Affine2D()
                .translate(-bounds.x0, -bounds.y0)
                .scale(0.82 / max(bounds.width, 1e-9), height / max(bounds.height, 1e-9))
                .translate(position + 0.09, baseline)
            )
            ax.add_patch(
                PathPatch(
                    path,
                    transform=transform + ax.transData,
                    facecolor=BASE_COLORS[base],
                    edgecolor="none",
                )
            )
            baseline += height
    ax.set_xlim(0, pwm.shape[0])
    ax.set_ylim(0, 2.05)
    ax.set_xticks(np.arange(pwm.shape[0]) + 0.5)
    ax.set_xticklabels(np.arange(1, pwm.shape[0] + 1), fontsize=7)
    ax.set_yticks((0, 1, 2))
    ax.tick_params(axis="both", length=2, width=0.7, labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.7)


def make_figure(
    path_base: Path,
    matches: Sequence[Mapping[str, object]],
    query_pwms: Mapping[str, np.ndarray],
    target_pwms: Mapping[str, np.ndarray],
    support_by_class: Mapping[str, int],
    consistency_by_class: Mapping[str, float],
    panel_count: int,
    minimum_support: int,
    minimum_consistency: float,
    maximum_q_value: float,
) -> None:
    best_by_query: dict[str, Mapping[str, object]] = {}
    for row in matches:
        query_id = str(row["query_id"])
        if query_id not in best_by_query:
            best_by_query[query_id] = row
    eligible = [
        row
        for query_id, row in best_by_query.items()
        if support_by_class.get(query_id, 0) >= minimum_support
        and consistency_by_class.get(query_id, float("-inf")) >= minimum_consistency
        and float(row["q_value"]) < maximum_q_value
    ]
    selected = sorted(eligible, key=lambda row: (float(row["q_value"]), float(row["p_value"])))[:panel_count]
    if not selected:
        return
    columns = min(3, len(selected))
    rows = math.ceil(len(selected) / columns)
    figure = plt.figure(figsize=(4.25 * columns, 3.35 * rows), constrained_layout=True)
    outer = figure.add_gridspec(rows, columns)
    for panel_index, match in enumerate(selected):
        row_index, column_index = divmod(panel_index, columns)
        subgrid = outer[row_index, column_index].subgridspec(3, 1, height_ratios=(0.24, 1, 1), hspace=0.17)
        title_ax = figure.add_subplot(subgrid[0])
        known_ax = figure.add_subplot(subgrid[1])
        model_ax = figure.add_subplot(subgrid[2])
        query_id = str(match["query_id"])
        target_id = str(match["target_id"])
        gene = str(match["target_gene"])
        q_value = float(match["q_value"])
        title_ax.axis("off")
        title_ax.text(0.0, 0.68, f"{query_id}  |  {gene}", fontsize=11, weight="bold", va="center")
        title_ax.text(
            1.0,
            0.68,
            f"q={q_value:.2g}  n={support_by_class.get(query_id, 0)}",
            fontsize=9,
            ha="right",
            va="center",
            color="#334155",
        )
        draw_logo(known_ax, target_pwms[target_id])
        known_ax.set_ylabel("Known RBP\n(bits)", fontsize=8)
        draw_logo(model_ax, query_pwms[query_id])
        model_ax.set_ylabel("Model region\n(bits)", fontsize=8)
        model_ax.set_xlabel("Position", fontsize=8)
    for panel_index in range(len(selected), rows * columns):
        row_index, column_index = divmod(panel_index, columns)
        empty_ax = figure.add_subplot(outer[row_index, column_index])
        empty_ax.axis("off")
    path_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_base.with_suffix(".png"), dpi=360, facecolor="white", bbox_inches="tight")
    figure.savefig(path_base.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    figure.savefig(path_base.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv",
    )
    parser.add_argument(
        "--valid-predictions",
        default="rmbase12_work/models/structdualrm_final_ensemble_20260730/valid_predictions.tsv",
    )
    parser.add_argument(
        "--test-predictions",
        default="rmbase12_work/models/structdualrm_final_ensemble_20260730/test_predictions.tsv",
    )
    parser.add_argument("--checkpoints", default=",".join(DEFAULT_CHECKPOINTS))
    parser.add_argument("--ensemble-weights", default=",".join(str(value) for value in DEFAULT_WEIGHTS))
    parser.add_argument("--class-names", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--attract-db", default="external/attract/ATtRACT/ATtRACT_db.txt")
    parser.add_argument("--attract-pwm", default="external/attract/ATtRACT/pwm.txt")
    parser.add_argument("--out-dir", default="tomtom_structdualrm_final_20260803")
    parser.add_argument("--max-samples-per-class", type=int, default=100)
    parser.add_argument("--motif-width", type=int, default=7)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tomtom-command", default="tomtom")
    parser.add_argument("--min-overlap", type=int, default=5)
    parser.add_argument("--figure-panels", type=int, default=6)
    parser.add_argument("--figure-minimum-support", type=int, default=20)
    parser.add_argument("--figure-minimum-consistency", type=float, default=0.5)
    parser.add_argument("--figure-maximum-q-value", type=float, default=0.05)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--tomtom-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    class_names = parse_csv(args.class_names)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    motif_dir = out_dir / "motifs"
    table_dir = out_dir / "tables"
    figure_dir = out_dir / "figures"
    query_meme = motif_dir / "structdualrm_test_attributed_motifs.meme"
    target_meme = motif_dir / "attract_human_rbp_targets.meme"
    tomtom_dir = out_dir / "official_tomtom"

    target_motifs, target_meta = parse_attract(resolve(args.attract_db), resolve(args.attract_pwm))
    write_meme(target_meme, target_motifs)
    target_pwms = {motif_id: pwm for motif_id, _, pwm in target_motifs}

    if not args.tomtom_only and not args.plot_only:
        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        checkpoint_paths = tuple(resolve(path) for path in parse_csv(args.checkpoints))
        models = load_seed_models(
            checkpoint_paths,
            parse_float_csv(args.ensemble_weights),
            class_names,
            device,
        )
        valid_samples = select_prediction_samples(
            resolve(args.valid_predictions), "validation", class_names, args.max_samples_per_class
        )
        test_samples = select_prediction_samples(
            resolve(args.test_predictions), "test", class_names, args.max_samples_per_class
        )
        all_samples = valid_samples + test_samples
        sequence_rows = load_selected_sequences(resolve(args.dataset), all_samples)
        all_region_rows: list[dict[str, object]] = []
        all_summary_rows: list[dict[str, object]] = []
        partition_pwms: dict[str, dict[str, np.ndarray]] = {}
        for partition, samples in (("validation", valid_samples), ("test", test_samples)):
            region_rows, pwms, summary_rows = extract_partition(
                partition,
                samples,
                sequence_rows,
                models,
                class_names,
                args.motif_width,
                args.pseudocount,
                device,
                args.batch_size,
            )
            all_region_rows.extend(region_rows)
            all_summary_rows.extend(summary_rows)
            partition_pwms[partition] = pwms
            support = {mod_type: sum(1 for sample in samples if sample.mod_type == mod_type) for mod_type in class_names}
            write_meme(
                motif_dir / f"structdualrm_{partition}_attributed_motifs.meme",
                [(mod_type, f"{mod_type}_{partition}_attributed_region", pwm) for mod_type, pwm in pwms.items()],
                support,
            )
        write_tsv(
            table_dir / "attributed_regions.tsv",
            all_region_rows,
            [
                "partition",
                "site_id",
                "mod_type",
                "ensemble_probability",
                "original_sequence_margin",
                "region_start_0based_201nt",
                "region_end_exclusive_201nt",
                "relative_start",
                "relative_end",
                "attributed_region",
                "positive_attribution_sum",
                "mean_position_attribution",
                "position_attribution_csv",
            ],
        )
        consistency_rows: list[dict[str, object]] = []
        for mod_type in class_names:
            validation_pwm = partition_pwms["validation"].get(mod_type)
            test_pwm = partition_pwms["test"].get(mod_type)
            consistency_rows.append(
                {
                    "mod_type": mod_type,
                    "validation_test_pwm_pearson": motif_similarity(validation_pwm, test_pwm)
                    if validation_pwm is not None and test_pwm is not None
                    else "",
                }
            )
        write_tsv(
            table_dir / "motif_extraction_summary.tsv",
            [
                {
                    **row,
                    "validation_test_pwm_pearson": next(
                        item["validation_test_pwm_pearson"]
                        for item in consistency_rows
                        if item["mod_type"] == row["mod_type"]
                    ),
                }
                for row in all_summary_rows
            ],
            [
                "partition",
                "mod_type",
                "n_attributed_regions",
                "evidence_tier",
                "consensus",
                "information_content_bits",
                "median_positive_attribution_sum",
                "validation_test_pwm_pearson",
            ],
        )
        test_support = {
            mod_type: sum(1 for sample in test_samples if sample.mod_type == mod_type)
            for mod_type in class_names
        }
        write_meme(
            query_meme,
            [
                (mod_type, f"{mod_type}_test_attributed_region", pwm)
                for mod_type, pwm in partition_pwms["test"].items()
            ],
            test_support,
        )
        manifest = {
            "method": "sequence-branch in-silico single-nucleotide mutagenesis",
            "dataset": str(resolve(args.dataset)),
            "split": "split_cdhit80_cluster",
            "selection_rule": "correctly classified true positives, descending frozen-ensemble class probability",
            "max_samples_per_class": args.max_samples_per_class,
            "motif_width": args.motif_width,
            "attribution_score": "original one-vs-rest sequence logit margin minus mean margin over three alternate bases",
            "window_rule": "maximum sum of positive per-position attribution; center-proximal tie break",
            "checkpoints": [str(path) for path in checkpoint_paths],
            "ensemble_weights": list(parse_float_csv(args.ensemble_weights)),
            "target_database": "ATtRACT human RBP PWMs",
            "target_motif_count": len(target_motifs),
            "device": str(device),
            "limitations": [
                "Tomtom motif similarity does not establish physical RBP binding or causality.",
                "The extracted motifs explain the sequence branch; structure and eCLIP branches require separate evidence.",
                "Classes with fewer than 20 attributed test regions are exploratory.",
            ],
        }
        (out_dir / "experiment_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if args.extract_only:
        print(json.dumps({"query_meme": str(query_meme), "target_meme": str(target_meme)}, indent=2))
        return

    if not args.plot_only:
        run_tomtom(args.tomtom_command, query_meme, target_meme, tomtom_dir, args.min_overlap)
    matches = parse_tomtom(tomtom_dir / "tomtom.tsv", target_meta)
    write_tsv(
        table_dir / "official_tomtom_matches.tsv",
        matches,
        [
            "query_id",
            "rank_within_query",
            "target_id",
            "target_gene",
            "target_consensus",
            "target_matrix_id",
            "target_database",
            "optimal_offset",
            "overlap",
            "orientation",
            "p_value",
            "e_value",
            "q_value",
            "significant_q_lt_0_05",
            "query_consensus",
            "tomtom_target_consensus",
        ],
    )
    query_pwms = {
        motif_id: pwm
        for motif_id, _, pwm in parse_meme_for_plot(query_meme)
    }
    summary_rows = read_tsv(table_dir / "motif_extraction_summary.tsv")
    support_by_class = {
        row["mod_type"]: int(row["n_attributed_regions"])
        for row in summary_rows
        if row["partition"] == "test"
    }
    consistency_by_class = {
        row["mod_type"]: float(row["validation_test_pwm_pearson"])
        for row in summary_rows
        if row["partition"] == "test" and row["validation_test_pwm_pearson"]
    }
    best_by_query: dict[str, dict[str, object]] = {}
    for row in matches:
        query_id = str(row["query_id"])
        if query_id not in best_by_query:
            best_by_query[query_id] = dict(row)
    best_rows: list[dict[str, object]] = []
    for mod_type in class_names:
        row = best_by_query.get(mod_type)
        if row is None:
            continue
        support = support_by_class.get(mod_type, 0)
        consistency = consistency_by_class.get(mod_type, float("nan"))
        q_value = float(row["q_value"])
        stable = bool(support >= args.figure_minimum_support and consistency >= args.figure_minimum_consistency)
        significant = bool(q_value < args.figure_maximum_q_value)
        best_rows.append(
            {
                "mod_type": mod_type,
                "n_attributed_regions": support,
                "validation_test_pwm_pearson": consistency if math.isfinite(consistency) else "",
                "target_gene": row["target_gene"],
                "target_consensus": row["target_consensus"],
                "target_matrix_id": row["target_matrix_id"],
                "p_value": row["p_value"],
                "e_value": row["e_value"],
                "q_value": q_value,
                "motif_stable": int(stable),
                "tomtom_significant_q_lt_0_05": int(significant),
                "main_figure_eligible": int(stable and significant),
            }
        )
    write_tsv(
        table_dir / "official_tomtom_per_class_best.tsv",
        best_rows,
        [
            "mod_type",
            "n_attributed_regions",
            "validation_test_pwm_pearson",
            "target_gene",
            "target_consensus",
            "target_matrix_id",
            "p_value",
            "e_value",
            "q_value",
            "motif_stable",
            "tomtom_significant_q_lt_0_05",
            "main_figure_eligible",
        ],
    )
    make_figure(
        figure_dir / "figure_official_tomtom_significant_rbp_matches",
        matches,
        query_pwms,
        target_pwms,
        support_by_class,
        consistency_by_class,
        args.figure_panels,
        args.figure_minimum_support,
        args.figure_minimum_consistency,
        args.figure_maximum_q_value,
    )
    make_figure(
        figure_dir / "figure_official_tomtom_all_class_top_matches",
        matches,
        query_pwms,
        target_pwms,
        support_by_class,
        consistency_by_class,
        len(query_pwms),
        1,
        -1.0,
        1.01,
    )
    significant = sum(1 for row in matches if int(row["significant_q_lt_0_05"]) == 1)
    print(
        json.dumps(
            {
                "queries": len(query_pwms),
                "targets": len(target_motifs),
                "matches": len(matches),
                "significant_matches_q_lt_0_05": significant,
                "output_dir": str(out_dir),
            },
            indent=2,
        )
    )


def parse_meme_for_plot(path: Path) -> list[tuple[str, str, np.ndarray]]:
    motifs: list[tuple[str, str, np.ndarray]] = []
    current_id = ""
    current_label = ""
    current_rows: list[list[float]] = []

    def save_current() -> None:
        if current_id and current_rows:
            motifs.append((current_id, current_label, np.asarray(current_rows, dtype=np.float64)))

    matrix_mode = False
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("MOTIF "):
                save_current()
                parts = line.split(maxsplit=2)
                current_id = parts[1]
                current_label = parts[2] if len(parts) > 2 else parts[1]
                current_rows = []
                matrix_mode = False
            elif line.startswith("letter-probability matrix:"):
                matrix_mode = True
            elif matrix_mode and line:
                parts = line.split()
                if len(parts) == 4:
                    try:
                        current_rows.append([float(value) for value in parts])
                    except ValueError:
                        matrix_mode = False
                else:
                    matrix_mode = False
            elif not line:
                matrix_mode = False
    save_current()
    return motifs


if __name__ == "__main__":
    main()
