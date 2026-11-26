#!/usr/bin/env python
"""Validation-frozen multi-motif Tomtom analysis for StructDualRM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from run_structdualrm_official_tomtom import (
    BASES,
    best_window,
    make_figure,
    motif_similarity,
    parse_attract,
    parse_meme_for_plot,
    parse_tomtom,
    pwm_consensus,
    pwm_from_regions,
    pwm_information,
    read_tsv,
    run_tomtom,
    write_meme,
    write_tsv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSES = ("Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "Y", "AtoI")
BASE_TO_INDEX = {base: index for index, base in enumerate(BASES)}


@dataclass(frozen=True)
class AttributionRecord:
    partition: str
    site_id: str
    mod_type: str
    ensemble_probability: float
    effects: np.ndarray


@dataclass(frozen=True)
class MotifInstance:
    partition: str
    site_id: str
    mod_type: str
    width: int
    full_sequence_hash: str
    region: str
    start: int
    attribution_sum: float
    ensemble_probability: float


@dataclass
class PwmMixture:
    weights: np.ndarray
    pwms: np.ndarray


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def parse_integer_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_class_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def load_attributions(path: Path) -> list[AttributionRecord]:
    records: list[AttributionRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            effects = np.fromstring(row["position_attribution_csv"], sep=",")
            if effects.shape != (201,):
                raise ValueError(f"Expected 201 attribution values for {row['site_id']}; found {effects.size}")
            records.append(
                AttributionRecord(
                    partition=row["partition"],
                    site_id=row["site_id"],
                    mod_type=row["mod_type"],
                    ensemble_probability=float(row["ensemble_probability"]),
                    effects=effects,
                )
            )
    return records


def load_sequences(path: Path, site_ids: set[str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"site_id", "sequence_201nt"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing fields: {sorted(missing)}")
        for row in reader:
            site_id = row["site_id"]
            if site_id not in site_ids:
                continue
            sequence = row["sequence_201nt"].upper().replace("T", "U")
            if len(sequence) != 201 or set(sequence).difference(BASES):
                raise ValueError(f"Invalid 201-nt sequence for {site_id}")
            sequences[site_id] = sequence
            if len(sequences) == len(site_ids):
                break
    missing_ids = site_ids.difference(sequences)
    if missing_ids:
        raise ValueError(f"Missing {len(missing_ids)} selected sequences; first={sorted(missing_ids)[0]}")
    return sequences


def build_instances(
    records: Sequence[AttributionRecord],
    sequences: Mapping[str, str],
    widths: Sequence[int],
) -> dict[tuple[str, str, int], list[MotifInstance]]:
    output: dict[tuple[str, str, int], list[MotifInstance]] = defaultdict(list)
    for record in records:
        sequence = sequences[record.site_id]
        sequence_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()[:16]
        for width in widths:
            start, score = best_window(record.effects, width)
            output[(record.partition, record.mod_type, width)].append(
                MotifInstance(
                    partition=record.partition,
                    site_id=record.site_id,
                    mod_type=record.mod_type,
                    width=width,
                    full_sequence_hash=sequence_hash,
                    region=sequence[start : start + width],
                    start=start,
                    attribution_sum=score,
                    ensemble_probability=record.ensemble_probability,
                )
            )
    return output


def collapse_exact_sequences(instances: Sequence[MotifInstance]) -> list[MotifInstance]:
    best_by_sequence: dict[str, MotifInstance] = {}
    for instance in instances:
        current = best_by_sequence.get(instance.full_sequence_hash)
        if current is None or (instance.ensemble_probability, instance.attribution_sum, instance.site_id) > (
            current.ensemble_probability,
            current.attribution_sum,
            current.site_id,
        ):
            best_by_sequence[instance.full_sequence_hash] = instance
    return sorted(best_by_sequence.values(), key=lambda item: item.full_sequence_hash)


def encode_regions(regions: Sequence[str]) -> np.ndarray:
    return np.asarray([[BASE_TO_INDEX[base] for base in region] for region in regions], dtype=np.int64)


def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(np.exp(values - maximum).sum(axis=axis))


def mixture_log_prob(encoded: np.ndarray, model: PwmMixture) -> np.ndarray:
    rows = np.arange(encoded.shape[1])
    component_scores = []
    for component in range(model.pwms.shape[0]):
        probabilities = model.pwms[component, rows[None, :], encoded]
        component_scores.append(np.log(model.weights[component]) + np.log(np.clip(probabilities, 1e-12, 1)).sum(axis=1))
    return logsumexp(np.stack(component_scores, axis=1), axis=1)


def mixture_posteriors(encoded: np.ndarray, model: PwmMixture) -> np.ndarray:
    rows = np.arange(encoded.shape[1])
    scores = []
    for component in range(model.pwms.shape[0]):
        probabilities = model.pwms[component, rows[None, :], encoded]
        scores.append(np.log(model.weights[component]) + np.log(np.clip(probabilities, 1e-12, 1)).sum(axis=1))
    matrix = np.stack(scores, axis=1)
    normalizer = logsumexp(matrix, axis=1)
    return np.exp(matrix - normalizer[:, None])


def fit_pwm_mixture(
    regions: Sequence[str],
    components: int,
    pseudocount: float,
    random_seed: int,
    max_iterations: int = 200,
) -> PwmMixture:
    encoded = encode_regions(regions)
    sample_count, width = encoded.shape
    if components < 1 or components > sample_count:
        raise ValueError("Invalid component count")
    if components == 1:
        return PwmMixture(np.ones(1), np.asarray([pwm_from_regions(regions, pseudocount)]))
    if len(set(regions)) < components:
        raise ValueError("Fewer distinct regions than requested mixture components")
    generator = np.random.default_rng(random_seed)
    center_indices = [int(generator.integers(0, sample_count))]
    while len(center_indices) < components:
        distances = np.stack(
            [(encoded != encoded[index]).sum(axis=1) for index in center_indices],
            axis=1,
        ).min(axis=1)
        distances[center_indices] = -1
        next_index = int(np.argmax(distances))
        if distances[next_index] <= 0:
            raise ValueError("Unable to initialize distinct mixture components")
        center_indices.append(next_index)
    centers = encoded[center_indices]
    labels = np.argmin(
        np.stack([(encoded != center).sum(axis=1) for center in centers], axis=1),
        axis=1,
    )
    responsibilities = np.eye(components, dtype=np.float64)[labels]
    previous = float("-inf")
    for _ in range(max_iterations):
        effective = responsibilities.sum(axis=0)
        weights = (effective + 1.0) / (sample_count + components)
        pwms = np.full((components, width, 4), pseudocount, dtype=np.float64)
        for component in range(components):
            for position in range(width):
                np.add.at(pwms[component, position], encoded[:, position], responsibilities[:, component])
        pwms /= pwms.sum(axis=2, keepdims=True)
        model = PwmMixture(weights, pwms)
        updated = mixture_posteriors(encoded, model)
        likelihood = float(mixture_log_prob(encoded, model).sum())
        responsibilities = updated
        if abs(likelihood - previous) < 1e-8:
            break
        previous = likelihood
    order = sorted(range(components), key=lambda index: (-model.weights[index], pwm_consensus(model.pwms[index])))
    return PwmMixture(model.weights[order], model.pwms[order])


def hard_assign(regions: Sequence[str], model: PwmMixture) -> np.ndarray:
    return np.argmax(mixture_posteriors(encode_regions(regions), model), axis=1)


def cross_validated_candidate(
    regions: Sequence[str],
    components: int,
    folds: int,
    pseudocount: float,
    random_seed: int,
    minimum_component_sequences: int,
) -> tuple[np.ndarray, bool]:
    sample_count = len(regions)
    fold_count = min(folds, sample_count)
    if fold_count < 2:
        return np.asarray([], dtype=np.float64), components == 1
    scores = np.full(sample_count, np.nan, dtype=np.float64)
    indices = np.arange(sample_count)
    generator = np.random.default_rng(random_seed)
    heldout_folds = np.array_split(generator.permutation(indices), fold_count)
    valid = True
    for fold_index, heldout_indices in enumerate(heldout_folds):
        train_indices = np.setdiff1d(indices, heldout_indices, assume_unique=True)
        if len(train_indices) < components * max(2, minimum_component_sequences // 2):
            valid = False
            break
        train_regions = [regions[index] for index in train_indices]
        if len(set(train_regions)) < components:
            valid = False
            break
        model = fit_pwm_mixture(train_regions, components, pseudocount, random_seed + fold_index)
        assignments = hard_assign(train_regions, model)
        counts = np.bincount(assignments, minlength=components)
        if counts.min() < max(2, minimum_component_sequences // 2):
            valid = False
            break
        heldout_encoded = encode_regions([regions[index] for index in heldout_indices])
        per_base_gain = mixture_log_prob(heldout_encoded, model) / heldout_encoded.shape[1] + math.log(4.0)
        scores[heldout_indices] = per_base_gain
    return scores[np.isfinite(scores)], valid and np.isfinite(scores).all()


def select_validation_models(
    instances: Mapping[tuple[str, str, int], Sequence[MotifInstance]],
    class_names: Sequence[str],
    widths: Sequence[int],
    maximum_components: int,
    folds: int,
    pseudocount: float,
    random_seed: int,
    minimum_component_sequences: int,
) -> tuple[dict[str, tuple[int, PwmMixture]], list[dict[str, object]]]:
    selected: dict[str, tuple[int, PwmMixture]] = {}
    audit_rows: list[dict[str, object]] = []
    for class_offset, mod_type in enumerate(class_names):
        candidates: list[dict[str, object]] = []
        for width in widths:
            collapsed = collapse_exact_sequences(instances.get(("validation", mod_type, width), []))
            regions = [item.region for item in collapsed]
            if not regions:
                continue
            max_components = min(maximum_components, max(1, len(regions) // minimum_component_sequences))
            for components in range(1, max_components + 1):
                scores, valid = cross_validated_candidate(
                    regions,
                    components,
                    folds,
                    pseudocount,
                    random_seed + class_offset * 100 + width * 10 + components,
                    minimum_component_sequences,
                )
                mean = float(scores.mean()) if scores.size else float("nan")
                standard_error = float(scores.std(ddof=1) / math.sqrt(scores.size)) if scores.size > 1 else float("inf")
                candidates.append(
                    {
                        "mod_type": mod_type,
                        "width": width,
                        "components": components,
                        "validation_unique_201nt": len(regions),
                        "cv_information_gain_nats_per_nt": mean,
                        "cv_standard_error": standard_error,
                        "candidate_valid": int(valid),
                        "selected": 0,
                    }
                )
        valid_candidates = [row for row in candidates if row["candidate_valid"] and math.isfinite(float(row["cv_information_gain_nats_per_nt"]))]
        if not valid_candidates:
            audit_rows.extend(candidates)
            continue
        best = max(valid_candidates, key=lambda row: float(row["cv_information_gain_nats_per_nt"]))
        threshold = float(best["cv_information_gain_nats_per_nt"]) - float(best["cv_standard_error"])
        eligible = [row for row in valid_candidates if float(row["cv_information_gain_nats_per_nt"]) >= threshold]
        chosen = min(eligible, key=lambda row: (int(row["components"]), int(row["width"])))
        chosen["selected"] = 1
        width = int(chosen["width"])
        collapsed = collapse_exact_sequences(instances[("validation", mod_type, width)])
        regions = [item.region for item in collapsed]
        model = fit_pwm_mixture(
            regions,
            int(chosen["components"]),
            pseudocount,
            random_seed + class_offset * 1000,
        )
        selected[mod_type] = (width, model)
        audit_rows.extend(candidates)
    return selected, audit_rows


def bootstrap_pwm_stability(regions: Sequence[str], pwm: np.ndarray, replicates: int, seed: int, pseudocount: float) -> tuple[float, float]:
    if len(regions) < 2:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    correlations = []
    for _ in range(replicates):
        indices = generator.integers(0, len(regions), size=len(regions))
        replicate = pwm_from_regions([regions[index] for index in indices], pseudocount)
        correlations.append(motif_similarity(pwm, replicate))
    return float(np.median(correlations)), float(np.quantile(correlations, 0.05))


def component_outputs(
    instances: Mapping[tuple[str, str, int], Sequence[MotifInstance]],
    selected_models: Mapping[str, tuple[int, PwmMixture]],
    pseudocount: float,
    bootstrap_replicates: int,
    random_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray], dict[str, int], dict[str, float]]:
    assignment_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    test_pwms: dict[str, np.ndarray] = {}
    test_support: dict[str, int] = {}
    consistency: dict[str, float] = {}
    for class_offset, (mod_type, (width, model)) in enumerate(selected_models.items()):
        partition_data: dict[str, dict[str, object]] = {}
        for partition in ("validation", "test"):
            raw_instances = list(instances.get((partition, mod_type, width), []))
            unique_instances = collapse_exact_sequences(raw_instances)
            unique_regions = [item.region for item in unique_instances]
            if unique_regions:
                unique_assignments = hard_assign(unique_regions, model)
            else:
                unique_assignments = np.asarray([], dtype=np.int64)
            assignment_by_hash = {
                item.full_sequence_hash: int(unique_assignments[index])
                for index, item in enumerate(unique_instances)
            }
            raw_assignments = np.asarray([assignment_by_hash[item.full_sequence_hash] for item in raw_instances], dtype=np.int64) if raw_instances else np.asarray([], dtype=np.int64)
            partition_data[partition] = {
                "raw": raw_instances,
                "unique": unique_instances,
                "unique_regions": unique_regions,
                "unique_assignments": unique_assignments,
                "raw_assignments": raw_assignments,
            }
            for index, item in enumerate(raw_instances):
                component = int(raw_assignments[index]) + 1
                assignment_rows.append(
                    {
                        "partition": partition,
                        "site_id": item.site_id,
                        "mod_type": mod_type,
                        "selected_width": width,
                        "component": component,
                        "full_sequence_hash": item.full_sequence_hash,
                        "region": item.region,
                        "region_start_0based_201nt": item.start,
                        "relative_start": item.start - 100,
                        "positive_attribution_sum": item.attribution_sum,
                        "ensemble_probability": item.ensemble_probability,
                    }
                )
        validation_data = partition_data["validation"]
        test_data = partition_data["test"]
        for component in range(model.pwms.shape[0]):
            query_id = f"{mod_type}__C{component + 1}_W{width}"
            validation_regions = [
                region
                for region, assignment in zip(validation_data["unique_regions"], validation_data["unique_assignments"])
                if int(assignment) == component
            ]
            test_regions = [
                region
                for region, assignment in zip(test_data["unique_regions"], test_data["unique_assignments"])
                if int(assignment) == component
            ]
            validation_pwm = pwm_from_regions(validation_regions, pseudocount) if validation_regions else model.pwms[component]
            test_pwm = pwm_from_regions(test_regions, pseudocount) if test_regions else None
            raw_validation = int(np.sum(validation_data["raw_assignments"] == component))
            raw_test = int(np.sum(test_data["raw_assignments"] == component))
            bootstrap_median, bootstrap_q05 = bootstrap_pwm_stability(
                validation_regions,
                validation_pwm,
                bootstrap_replicates,
                random_seed + class_offset * 100 + component,
                pseudocount,
            )
            correlation = motif_similarity(validation_pwm, test_pwm) if test_pwm is not None else float("nan")
            summary_rows.append(
                {
                    "query_id": query_id,
                    "mod_type": mod_type,
                    "selected_width": width,
                    "selected_components": model.pwms.shape[0],
                    "component": component + 1,
                    "validation_raw_support": raw_validation,
                    "validation_unique_201nt": len(validation_regions),
                    "test_raw_support": raw_test,
                    "test_unique_201nt": len(test_regions),
                    "validation_consensus": pwm_consensus(validation_pwm),
                    "test_consensus": pwm_consensus(test_pwm) if test_pwm is not None else "",
                    "validation_information_bits": pwm_information(validation_pwm),
                    "test_information_bits": pwm_information(test_pwm) if test_pwm is not None else "",
                    "validation_bootstrap_median_r": bootstrap_median if math.isfinite(bootstrap_median) else "",
                    "validation_bootstrap_q05_r": bootstrap_q05 if math.isfinite(bootstrap_q05) else "",
                    "validation_test_pwm_pearson": correlation if math.isfinite(correlation) else "",
                }
            )
            if test_pwm is not None:
                test_pwms[query_id] = test_pwm
                test_support[query_id] = raw_test
                consistency[query_id] = correlation
    return assignment_rows, summary_rows, test_pwms, test_support, consistency


def summarize_matches(
    matches: Sequence[Mapping[str, object]],
    component_rows: Sequence[Mapping[str, object]],
    old_results_path: Path,
    minimum_raw_support: int,
    minimum_unique_support: int,
    minimum_consistency: float,
    minimum_bootstrap_q05: float,
    target_motif_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    component_by_query = {str(row["query_id"]): row for row in component_rows}
    best_by_query: dict[str, Mapping[str, object]] = {}
    for row in matches:
        best_by_query.setdefault(str(row["query_id"]), row)
    component_best: list[dict[str, object]] = []
    for query_id, component in component_by_query.items():
        match = best_by_query.get(query_id)
        if match is None:
            continue
        support = int(component["test_raw_support"])
        unique_support = int(component["test_unique_201nt"])
        consistency = float(component["validation_test_pwm_pearson"]) if component["validation_test_pwm_pearson"] != "" else float("nan")
        bootstrap_q05 = float(component["validation_bootstrap_q05_r"]) if component["validation_bootstrap_q05_r"] != "" else float("nan")
        stable = (
            support >= minimum_raw_support
            and unique_support >= minimum_unique_support
            and math.isfinite(consistency)
            and consistency >= minimum_consistency
            and math.isfinite(bootstrap_q05)
            and bootstrap_q05 >= minimum_bootstrap_q05
        )
        component_best.append(
            {
                **component,
                "target_gene": match["target_gene"],
                "target_consensus": match["target_consensus"],
                "target_matrix_id": match["target_matrix_id"],
                "p_value": match["p_value"],
                "e_value": match["e_value"],
                "q_value": match["q_value"],
                "query_target_bonferroni_p": min(1.0, float(match["p_value"]) * target_motif_count),
                "within_class_component_bonferroni_p": "",
                "across_class_hierarchical_q": "",
                "motif_stable": int(stable),
                "tomtom_significant_q_lt_0_05": int(float(match["q_value"]) < 0.05),
                "main_figure_eligible": int(stable and float(match["q_value"]) < 0.05),
            }
        )
    old_by_class = {row["mod_type"]: row for row in read_tsv(old_results_path)} if old_results_path.is_file() else {}
    for mod_type in sorted({str(row["mod_type"]) for row in component_best}):
        rows = [row for row in component_best if row["mod_type"] == mod_type]
        for row in rows:
            row["within_class_component_bonferroni_p"] = min(
                1.0,
                float(row["query_target_bonferroni_p"]) * len(rows),
            )
    best_hierarchical_by_class = {
        mod_type: min(
            float(row["within_class_component_bonferroni_p"])
            for row in component_best
            if row["mod_type"] == mod_type
        )
        for mod_type in sorted({str(row["mod_type"]) for row in component_best})
    }
    class_names = list(best_hierarchical_by_class)
    class_adjusted = benjamini_hochberg([best_hierarchical_by_class[name] for name in class_names])
    across_class_q = dict(zip(class_names, class_adjusted))
    best_component_p_by_class = {
        mod_type: min(
            float(row["within_class_component_bonferroni_p"])
            for row in component_best
            if row["mod_type"] == mod_type
        )
        for mod_type in class_names
    }
    for row in component_best:
        row["across_class_hierarchical_q"] = across_class_q[str(row["mod_type"])]
        row["component_is_class_best"] = int(
            math.isclose(
                float(row["within_class_component_bonferroni_p"]),
                best_component_p_by_class[str(row["mod_type"])],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )
        row["main_figure_eligible"] = int(
            bool(row["motif_stable"])
            and bool(row["component_is_class_best"])
            and float(row["across_class_hierarchical_q"]) < 0.05
        )
    class_rows: list[dict[str, object]] = []
    classes = sorted({str(row["mod_type"]) for row in component_best})
    for mod_type in classes:
        rows = [row for row in component_best if row["mod_type"] == mod_type]
        best = min(rows, key=lambda row: (float(row["within_class_component_bonferroni_p"]), float(row["p_value"])))
        old = old_by_class.get(mod_type, {})
        class_rows.append(
            {
                "mod_type": mod_type,
                "selected_width": best["selected_width"],
                "selected_components": best["selected_components"],
                "best_query_id": best["query_id"],
                "test_raw_support": best["test_raw_support"],
                "test_unique_201nt": best["test_unique_201nt"],
                "validation_test_pwm_pearson": best["validation_test_pwm_pearson"],
                "validation_bootstrap_q05_r": best["validation_bootstrap_q05_r"],
                "target_gene": best["target_gene"],
                "target_consensus": best["target_consensus"],
                "target_matrix_id": best["target_matrix_id"],
                "p_value": best["p_value"],
                "q_value": best["q_value"],
                "query_target_bonferroni_p": best["query_target_bonferroni_p"],
                "within_class_component_bonferroni_p": best["within_class_component_bonferroni_p"],
                "across_class_hierarchical_q": best["across_class_hierarchical_q"],
                "main_figure_eligible": best["main_figure_eligible"],
                "original_target_gene": old.get("target_gene", ""),
                "original_q_value": old.get("q_value", ""),
            }
        )
    return component_best, class_rows


def benjamini_hochberg(values: Sequence[float]) -> list[float]:
    count = len(values)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: values[index])
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        index = order[rank_index]
        rank = rank_index + 1
        running = min(running, float(values[index]) * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def make_selected_figure(
    output_base: Path,
    selected_rows: Sequence[Mapping[str, object]],
    query_pwms: Mapping[str, np.ndarray],
    target_pwms: Mapping[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt
    from run_structdualrm_official_tomtom import draw_logo

    selected = list(selected_rows)
    if not selected:
        return
    figure, axes = plt.subplots(len(selected), 2, figsize=(8.6, 2.65 * len(selected)), squeeze=False)
    for row_index, row in enumerate(selected):
        query_id = str(row["query_id"])
        target_id = str(row["target_id"])
        draw_logo(axes[row_index, 0], query_pwms[query_id])
        draw_logo(axes[row_index, 1], target_pwms[target_id])
        axes[row_index, 0].set_title(
            f"{row['mod_type']} component {row['component']} | model PWM",
            loc="left",
            fontsize=10,
            weight="bold",
        )
        axes[row_index, 1].set_title(
            f"{row['target_gene']} | hierarchical q={float(row['across_class_hierarchical_q']):.3g}",
            loc="left",
            fontsize=10,
            weight="bold",
        )
        axes[row_index, 0].set_ylabel("Information (bits)", fontsize=8)
        axes[row_index, 1].set_ylabel("Information (bits)", fontsize=8)
    figure.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".png"), dpi=360, facecolor="white", bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv")
    parser.add_argument("--attributed-regions", default="tomtom_structdualrm_final_20260803/tables/attributed_regions.tsv")
    parser.add_argument("--old-results", default="tomtom_structdualrm_final_20260803/tables/official_tomtom_per_class_best.tsv")
    parser.add_argument("--attract-db", default="external/attract/ATtRACT/ATtRACT_db.txt")
    parser.add_argument("--attract-pwm", default="external/attract/ATtRACT/pwm.txt")
    parser.add_argument("--out-dir", default="tomtom_structdualrm_clustered_20260803")
    parser.add_argument("--class-names", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--motif-widths", default="6,7,8,9,10")
    parser.add_argument("--maximum-components", type=int, default=3)
    parser.add_argument("--validation-folds", type=int, default=5)
    parser.add_argument("--minimum-component-sequences", type=int, default=10)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=20260803)
    parser.add_argument("--tomtom-command", default="tomtom")
    parser.add_argument("--min-overlap", type=int, default=5)
    parser.add_argument("--minimum-raw-support", type=int, default=20)
    parser.add_argument("--minimum-unique-support", type=int, default=10)
    parser.add_argument("--minimum-consistency", type=float, default=0.5)
    parser.add_argument("--minimum-bootstrap-q05", type=float, default=0.5)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    class_names = parse_class_csv(args.class_names)
    widths = parse_integer_csv(args.motif_widths)
    out_dir = resolve(args.out_dir)
    motif_dir = out_dir / "motifs"
    table_dir = out_dir / "tables"
    figure_dir = out_dir / "figures"
    tomtom_dir = out_dir / "official_tomtom"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_attributions(resolve(args.attributed_regions))
    records = [record for record in records if record.mod_type in class_names]
    print(f"Loaded {len(records)} cached attribution records", flush=True)
    sequences = load_sequences(resolve(args.dataset), {record.site_id for record in records})
    print(f"Loaded {len(sequences)} aligned 201-nt sequences", flush=True)
    instances = build_instances(records, sequences, widths)
    selected_models, selection_rows = select_validation_models(
        instances,
        class_names,
        widths,
        args.maximum_components,
        args.validation_folds,
        args.pseudocount,
        args.random_seed,
        args.minimum_component_sequences,
    )
    for mod_type, (width, model) in selected_models.items():
        print(f"Selected {mod_type}: width={width}, components={model.pwms.shape[0]}", flush=True)
    assignment_rows, component_rows, test_pwms, test_support, consistency = component_outputs(
        instances,
        selected_models,
        args.pseudocount,
        args.bootstrap_replicates,
        args.random_seed,
    )

    write_tsv(
        table_dir / "validation_model_selection.tsv",
        selection_rows,
        [
            "mod_type", "width", "components", "validation_unique_201nt",
            "cv_information_gain_nats_per_nt", "cv_standard_error", "candidate_valid", "selected",
        ],
    )
    write_tsv(
        table_dir / "motif_component_summary.tsv",
        component_rows,
        list(component_rows[0].keys()) if component_rows else [],
    )
    write_tsv(
        table_dir / "motif_component_assignments.tsv",
        assignment_rows,
        list(assignment_rows[0].keys()) if assignment_rows else [],
    )

    query_meme = motif_dir / "structdualrm_test_validation_frozen_components.meme"
    target_meme = motif_dir / "attract_human_rbp_targets.meme"
    write_meme(query_meme, [(query_id, query_id, pwm) for query_id, pwm in test_pwms.items()], test_support)
    target_motifs, target_meta = parse_attract(resolve(args.attract_db), resolve(args.attract_pwm))
    write_meme(target_meme, target_motifs)
    target_pwms = {motif_id: pwm for motif_id, _, pwm in target_motifs}
    run_tomtom(args.tomtom_command, query_meme, target_meme, tomtom_dir, args.min_overlap)
    matches = parse_tomtom(tomtom_dir / "tomtom.tsv", target_meta)
    write_tsv(table_dir / "official_tomtom_matches.tsv", matches, list(matches[0].keys()) if matches else [])
    component_best, class_rows = summarize_matches(
        matches,
        component_rows,
        resolve(args.old_results),
        args.minimum_raw_support,
        args.minimum_unique_support,
        args.minimum_consistency,
        args.minimum_bootstrap_q05,
        len(target_motifs),
    )
    write_tsv(table_dir / "official_tomtom_per_component_best.tsv", component_best, list(component_best[0].keys()) if component_best else [])
    write_tsv(table_dir / "official_tomtom_per_class_best.tsv", class_rows, list(class_rows[0].keys()) if class_rows else [])

    query_pwms = {motif_id: pwm for motif_id, _, pwm in parse_meme_for_plot(query_meme)}
    match_by_query = {}
    for row in matches:
        match_by_query.setdefault(str(row["query_id"]), row)
    selected_figure_rows = []
    for row in component_best:
        if not int(row["main_figure_eligible"]):
            continue
        match = match_by_query[str(row["query_id"])]
        selected_figure_rows.append({**row, "target_id": match["target_id"]})
    if not args.skip_figures:
        make_selected_figure(
            figure_dir / "figure_clustered_tomtom_hierarchical_significant_matches",
            selected_figure_rows,
            query_pwms,
            target_pwms,
        )

    manifest = {
        "method": "validation-frozen categorical PWM mixture over cached in-silico mutagenesis attribution",
        "split": "split_cdhit80_cluster",
        "candidate_widths": list(widths),
        "candidate_components": list(range(1, args.maximum_components + 1)),
        "selection": "five-fold validation information gain with one-standard-error simplicity rule",
        "exact_duplicate_control": "collapse identical 201-nt input sequences within each partition",
        "test_usage": "test partition used once after freezing width, component count, and validation mixture",
        "tomtom": {"version": "5.5.7", "distance": "pearson", "minimum_overlap": args.min_overlap, "norc": True},
        "multiple_testing": {
            "within_query": f"Bonferroni over {len(target_motifs)} ATtRACT matrices",
            "within_class": "Bonferroni over validation-selected motif components",
            "across_classes": "Benjamini-Hochberg FDR over evaluable modification classes",
        },
        "main_figure_filters": {
            "hierarchical_q_value": 0.05,
            "raw_test_support": args.minimum_raw_support,
            "unique_201nt_test_support": args.minimum_unique_support,
            "validation_test_pwm_pearson": args.minimum_consistency,
            "validation_bootstrap_q05_r": args.minimum_bootstrap_q05,
        },
        "selected_models": {
            mod_type: {"width": width, "components": int(model.pwms.shape[0])}
            for mod_type, (width, model) in selected_models.items()
        },
        "limitations": [
            "Tomtom motif similarity does not establish physical RBP binding or causality.",
            "Subtypes are statistical motif components and are not assumed to be biological RBP identities.",
            "Classes with limited independent 201-nt sequences remain exploratory.",
        ],
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_classes": len(selected_models),
                "test_query_components": len(test_pwms),
                "hierarchically_significant_stable_components": sum(int(row["main_figure_eligible"]) for row in component_best),
                "output_dir": str(out_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
