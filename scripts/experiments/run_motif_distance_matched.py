#!/usr/bin/env python3
"""Experiment 1: motif-directed and matched-random sequence perturbation.

The script evaluates the frozen no-pair StructDualRM checkpoint.  It selects
one test site per CD-HIT80 cluster for each independently replicated motif,
applies synchronized edits to the 41/101/201 nt windows, and compares target
class logits for motif disruption against a distance-matched random region.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_motif_struct_fusion_v4 import (
    CLASS_NAMES,
    NODE_FEATURE_FIELDS,
    STRUCT_FIELDS,
    MotifStructFusionV4,
    MotifStructV4Config,
    RMBaseFusionMultiLengthPositionalDataset,
    collate_batch_v4,
    configure_classes,
    configure_structure_inputs,
    load_rbp_positional_features,
    load_rnafold_filtered,
    read_tsv,
)
from train_paired_gat_rbp_fusion import normalize_matrix, parse_float


REPLICATED_MOTIFS: Mapping[str, tuple[str, ...]] = {
    "Am": ("GAGAGU", "AGAGUU"),
    "Um": ("GACU", "ACUG", "CUGA"),
    "m1A": ("UCGA", "GUUCG", "GGUUC"),
    "m5C": ("GGGC", "GGCC", "CCCG"),
    "m6A": ("GGACC", "UGGAC", "GGACA"),
    "Y": ("UUUU", "UUUC", "AUUU"),
    "AtoI": ("UAGGC", "UAGCU", "UAGGU"),
}

DISPLAY_NAMES = {"Y": "Psi", "AtoI": "I"}
BASES = "ACGT"
CENTER_41 = 20
CENTRAL_START_41 = 10
CENTRAL_END_41 = 31


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_csv(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def stable_rng(*parts: object, seed: int) -> random.Random:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    value = int.from_bytes(digest[:8], "big") ^ int(seed)
    return random.Random(value)


def canonical_sequence(seq: str) -> str:
    return str(seq or "").upper().replace("U", "T")


def find_motif_start(seq41: str, motif: str) -> int | None:
    sequence = canonical_sequence(seq41)
    target = canonical_sequence(motif)
    starts = [
        start
        for start in range(CENTRAL_START_41, CENTRAL_END_41 - len(target) + 1)
        if sequence[start : start + len(target)] == target
    ]
    if not starts:
        return None
    return min(starts, key=lambda start: (abs((start + len(target) / 2.0) - CENTER_41), start))


def choose_random_region(
    motif_start: int,
    motif_length: int,
    site_id: str,
    motif: str,
    seed: int,
) -> tuple[int, tuple[int, ...], int]:
    motif_positions = set(range(motif_start, motif_start + motif_length))
    candidates: list[int] = []
    for start in range(CENTRAL_START_41, CENTRAL_END_41 - motif_length + 1):
        positions = set(range(start, start + motif_length))
        if CENTER_41 in positions or positions & motif_positions:
            continue
        candidates.append(start)
    if not candidates:
        raise ValueError(
            f"No matched-random region for site={site_id}, motif={motif}, "
            f"start={motif_start}, length={motif_length}."
        )
    motif_distance = abs(motif_start + motif_length / 2.0 - CENTER_41)
    distance_error = lambda start: abs(abs(start + motif_length / 2.0 - CENTER_41) - motif_distance)
    min_distance = min(distance_error(start) for start in candidates)
    nearest = [start for start in candidates if distance_error(start) == min_distance]
    rng = stable_rng(site_id, motif, "random_region", seed=seed)
    random_start = rng.choice(nearest)
    target_positions = set(range(motif_start, motif_start + motif_length)) - {CENTER_41}
    editable_count = len(target_positions)
    random_positions = list(range(random_start, random_start + motif_length))
    if editable_count < motif_length:
        drop = stable_rng(site_id, motif, "random_edit_subset", seed=seed).choice(random_positions)
        random_positions.remove(drop)
    if len(random_positions) != editable_count:
        raise AssertionError("matched random edit count does not match motif edit count")
    return random_start, tuple(random_positions), len(candidates)


def build_manifest(rows: Sequence[dict[str, str]], seed: int) -> list[dict[str, str]]:
    candidates: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        source_class = str(row.get("mod_type", ""))
        motifs = REPLICATED_MOTIFS.get(source_class, ())
        if not motifs:
            continue
        cluster_id = str(row.get("cluster_id", ""))
        if not cluster_id:
            raise ValueError(f"missing cluster_id for {row.get('site_id')}")
        for motif in motifs:
            start = find_motif_start(row["sequence_41nt"], motif)
            if start is None:
                continue
            key = (source_class, motif, cluster_id)
            item = {
                "modification": DISPLAY_NAMES.get(source_class, source_class),
                "source_class": source_class,
                "motif": motif,
                "motif_t": canonical_sequence(motif),
                "site_id": str(row["site_id"]),
                "cluster_id": cluster_id,
                "mod_type": source_class,
                "class_index": str(row.get("y_true", row.get("class_index", ""))),
                "motif_start_41_0based": str(start),
                "motif_end_41_exclusive": str(start + len(motif)),
                "motif_length": str(len(motif)),
                "center_index_41": str(CENTER_41),
            }
            if key not in candidates or item["site_id"] < candidates[key]["site_id"]:
                candidates[key] = item

    manifest: list[dict[str, str]] = []
    for item in sorted(candidates.values(), key=lambda x: (x["source_class"], x["motif"], x["cluster_id"], x["site_id"])):
        start = int(item["motif_start_41_0based"])
        length = int(item["motif_length"])
        motif_positions = tuple(pos for pos in range(start, start + length) if pos != CENTER_41)
        random_start, random_positions, n_candidates = choose_random_region(
            start, length, item["site_id"], item["motif"], seed
        )
        item["motif_edit_positions_41_0based"] = ",".join(map(str, motif_positions))
        item["random_start_41_0based"] = str(random_start)
        item["random_end_41_exclusive"] = str(random_start + length)
        item["random_edit_positions_41_0based"] = ",".join(map(str, random_positions))
        item["editable_position_count"] = str(len(motif_positions))
        item["random_region_candidate_count"] = str(n_candidates)
        manifest.append(item)
    return manifest


def mutate_sequences(
    row: Mapping[str, str],
    positions_41: Sequence[int],
    replicate_seed: int,
) -> dict[str, str]:
    sequence_values = {
        "sequence_41nt": str(row["sequence_41nt"]),
        "sequence_101nt": str(row["sequence_101nt"]),
        "sequence_201nt": str(row["sequence_201nt"]),
    }
    rng = random.Random(replicate_seed)
    replacement_by_offset: dict[int, str] = {}
    for position in positions_41:
        if position == CENTER_41:
            raise ValueError("center base cannot be mutated")
        original = canonical_sequence(sequence_values["sequence_41nt"])[position]
        choices = [base for base in BASES if base != original]
        replacement_by_offset[position - CENTER_41] = rng.choice(choices)
    result = dict(row)
    for field, sequence in sequence_values.items():
        values = list(sequence.upper())
        center = len(values) // 2
        use_uracil = "U" in sequence.upper() and "T" not in sequence.upper()
        for offset, replacement in replacement_by_offset.items():
            index = center + offset
            if not 0 <= index < len(values):
                raise ValueError(f"mapped edit index out of range: {field}, {index}")
            values[index] = "U" if use_uracil and replacement == "T" else replacement
        result[field] = "".join(values)
    return result


def compute_train_struct_stats(
    common_dataset: Path,
    structure_path: Path,
    fields: Sequence[str],
) -> dict[str, list[float]]:
    train_ids: set[str] = set()
    with common_dataset.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("split_cdhit80_cluster") == "train":
                train_ids.add(str(row["site_id"]))
    values: list[list[float]] = []
    with structure_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("site_id") in train_ids:
                values.append([parse_float(row.get(field, 0.0)) for field in fields])
    if not values:
        raise RuntimeError("No training structure rows were found for normalization.")
    _, stats = normalize_matrix(np.asarray(values, dtype=np.float32))
    return stats


def make_model(config: Mapping[str, object], rbp_shape: tuple[int, int, int], device: torch.device) -> torch.nn.Module:
    class_names = parse_csv(config["class_names"])
    configure_classes(class_names)
    configure_structure_inputs(str(config.get("extra_struct_fields", "")), str(config.get("node_feature_fields", "paired_flags")))
    afamily_classes = tuple(name for name in parse_csv(config.get("afamily_classes", "")) if name in CLASS_NAMES)
    afamily_indices = tuple(CLASS_NAMES.index(name) for name in afamily_classes)
    multi_lengths = tuple(int(value) for value in parse_csv(config["multi_length_lengths"] or "41,101,201"))
    multi_columns = parse_csv(config["multi_length_columns"])
    cfg = MotifStructV4Config(
        input_length=int(config["input_length"]),
        num_classes=len(CLASS_NAMES),
        d_model=int(config["d_model"]),
        gat_layers=int(config["gat_layers"]),
        dropout=float(config["dropout"]),
        rbp_dim=0,
        rbp_pos_shape=rbp_shape,
        use_structure=not parse_bool(config["disable_structure"]),
        use_rbp=parse_bool(config["use_rbp"]),
        use_rbp_positional=parse_bool(config["use_rbp_positional"]),
        use_position=not parse_bool(config["disable_position"]),
        structure_uses_sequence=parse_bool(config["structure_uses_sequence"]),
        structure_scale=float(config["structure_scale"]),
        rbp_scale=float(config["rbp_scale"]),
        rbp_pos_scale=float(config["rbp_pos_scale"]),
        gate_init_logit=float(config["gate_init_logit"]),
        rbp_pos_gate_init_logit=float(config["rbp_pos_gate_init_logit"]),
        modality_dropout_struct=float(config["modality_dropout_struct"]),
        modality_dropout_rbp=float(config["modality_dropout_rbp"]),
        struct_fields=tuple(STRUCT_FIELDS),
        node_feature_fields=tuple(NODE_FEATURE_FIELDS),
        seq_encoder=str(config["seq_encoder"]),
        seq_residual_encoder=str(config["seq_residual_encoder"]),
        seq_residual_focus_indices=(),
        seq_residual_scale=float(config["seq_residual_scale"]),
        seq_residual_gate_init_logit=float(config["seq_residual_gate_init_logit"]),
        multi_length_columns=tuple(multi_columns),
        multi_length_lengths=multi_lengths,
        multi_length_base_encoder=str(config["multi_length_base_encoder"]),
        multi_length_share_encoder=parse_bool(config["multi_length_share_encoder"]),
        m1a_aux_weight=float(config["m1a_aux_weight"]),
        m1a_aux_pos_weight=1.0,
        afamily_classes=afamily_classes,
        afamily_indices=afamily_indices,
        afamily_aux_weight=float(config["afamily_aux_weight"]),
        afamily_residual_scale=float(config["afamily_residual_scale"]),
        afamily_fusion_mode=str(config["afamily_fusion_mode"]),
        afamily_m1a_expert_scale=float(config["afamily_m1a_expert_scale"]),
        transformer_layers=int(config["transformer_layers"]),
        transformer_heads=int(config["transformer_heads"]),
        transformer_ff_mult=int(config["transformer_ff_mult"]),
    )
    model = MotifStructFusionV4(cfg).to(device)
    state = torch.load(str(config["model_path"]), map_location=device)
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def softmax(values: np.ndarray) -> np.ndarray:
    values = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=-1, keepdims=True)


def perturbation_rows(
    base_row: Mapping[str, str],
    item: Mapping[str, str],
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    rows: list[dict[str, str]] = []
    meta: list[dict[str, object]] = []
    rows.append(dict(base_row))
    meta.append({"condition": "original", "replicate": 0})
    motif_positions = tuple(int(x) for x in item["motif_edit_positions_41_0based"].split(",") if x)
    random_positions = tuple(int(x) for x in item["random_edit_positions_41_0based"].split(",") if x)
    for replicate in range(1, 11):
        rows.append(mutate_sequences(base_row, motif_positions, seed + replicate * 1009))
        meta.append({"condition": "motif_disrupted", "replicate": replicate})
    for replicate in range(1, 11):
        rows.append(mutate_sequences(base_row, random_positions, seed + 20000 + replicate * 1009))
        meta.append({"condition": "matched_random_disrupted", "replicate": replicate})
    return rows, meta


def bh_fdr(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    order = np.argsort(np.asarray(values, dtype=float))
    q = np.empty(len(values), dtype=float)
    running = 1.0
    for rank in range(len(values) - 1, -1, -1):
        idx = int(order[rank])
        value = float(values[idx])
        running = min(running, value * len(values) / (rank + 1))
        q[idx] = running
    return q.tolist()


def permutation_p(values: np.ndarray, n_permutations: int, rng: np.random.Generator) -> float:
    observed = float(values.mean())
    if len(values) == 0:
        return float("nan")
    exceed = 0
    batch_size = 512
    for start in range(0, n_permutations, batch_size):
        size = min(batch_size, n_permutations - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(values)))
        perm_means = (signs * values[None, :]).mean(axis=1)
        exceed += int(np.sum(perm_means >= observed))
    return (exceed + 1.0) / (n_permutations + 1.0)


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--common-dataset", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--rbp-positional", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--site-chunk", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["model_path"] = str(args.model)
    rows = read_tsv(args.test_dataset)
    if any(row.get("split", "") != "test" for row in rows):
        raise ValueError("test dataset contains non-test rows")
    manifest = build_manifest(rows, args.seed)
    if not manifest:
        raise RuntimeError("No replicated motif-positive test sites were found.")
    manifest_path = args.out_dir / "motif_attribution_site_manifest.tsv"
    write_tsv(manifest_path, manifest)
    row_by_id = {row["site_id"]: row for row in rows}
    selected_site_ids = {item["site_id"] for item in manifest}
    fold_by_id = load_rnafold_filtered(args.structure, selected_site_ids)
    missing_structure = selected_site_ids - set(fold_by_id)
    if missing_structure:
        raise RuntimeError(f"missing RNAfold rows for {len(missing_structure)} selected sites")
    rbp_by_id, rbp_shape, rbp_metadata = load_rbp_positional_features(str(args.rbp_positional), selected_site_ids)
    missing_rbp = selected_site_ids - set(rbp_by_id)
    if missing_rbp:
        raise RuntimeError(f"missing RBP rows for {len(missing_rbp)} selected sites")
    struct_stats = compute_train_struct_stats(args.common_dataset, args.structure, STRUCT_FIELDS)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = make_model(config, rbp_shape, device)
    target_index = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    long_path = args.out_dir / "motif_attribution_perturbation_predictions.tsv"
    site_path = args.out_dir / "motif_attribution_per_site.tsv"
    long_fields = [
        "modification", "source_class", "motif", "site_id", "cluster_id", "condition", "replicate",
        "target_class", "target_index", "target_logit", "target_probability", "predicted_class",
        "predicted_index", "target_rank", "motif_start_41_0based", "motif_edit_positions_41_0based",
        "random_start_41_0based", "random_edit_positions_41_0based",
    ]
    site_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    with long_path.open("w", encoding="utf-8", newline="") as long_handle:
        long_writer = csv.DictWriter(long_handle, fieldnames=long_fields, delimiter="\t", lineterminator="\n")
        long_writer.writeheader()
        for chunk_start in range(0, len(manifest), args.site_chunk):
            chunk = manifest[chunk_start : chunk_start + args.site_chunk]
            variant_rows: list[dict[str, str]] = []
            variant_meta: list[dict[str, object]] = []
            for local_idx, item in enumerate(chunk):
                item_rows, item_meta = perturbation_rows(
                    row_by_id[item["site_id"]], item, args.seed + chunk_start + local_idx
                )
                variant_rows.extend(item_rows)
                for meta in item_meta:
                    meta = dict(meta)
                    meta["local_idx"] = local_idx
                    variant_meta.append(meta)
            dataset = RMBaseFusionMultiLengthPositionalDataset(
                variant_rows,
                fold_by_id,
                int(config["input_length"]),
                rbp_by_id=None,
                rbp_dim=0,
                struct_stats=struct_stats,
                rbp_pos_by_id=rbp_by_id,
                rbp_pos_shape=rbp_shape,
                shuffle_structure=False,
                shuffle_rbp=False,
                disable_pair_edges=parse_bool(config["disable_pair_edges"]),
                seed=int(config["seed"]),
                sequence_column=str(config["sequence_column"]),
                multi_length_columns=parse_csv(config["multi_length_columns"]),
                multi_length_lengths=tuple(int(x) for x in parse_csv(config["multi_length_lengths"])),
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch_v4, num_workers=0)
            logits_parts: list[np.ndarray] = []
            with torch.inference_mode():
                for batch in loader:
                    seq = batch["seq"].to(device)
                    seq_multi = {
                        key: batch[key].to(device)
                        for key in ("seq_len_41", "seq_len_101", "seq_len_201")
                        if key in batch
                    }
                    output = model(
                        seq,
                        batch["paired_flags"].to(device),
                        batch["pair_partner"].to(device),
                        batch["struct"].to(device),
                        batch["rbp"].to(device),
                        rbp_pos=batch["rbp_pos"].to(device),
                        seq_multi=seq_multi,
                    )
                    logits_parts.append(output["logits"].detach().float().cpu().numpy())
            logits = np.concatenate(logits_parts, axis=0)
            probs = softmax(logits)
            for local_idx, item in enumerate(chunk):
                start = local_idx * 21
                end = start + 21
                item_logits = logits[start:end]
                item_probs = probs[start:end]
                target = target_index[item["source_class"]]
                target_logits = item_logits[:, target]
                target_probs = item_probs[:, target]
                predicted = np.argmax(item_logits, axis=1)
                ranks = 1 + (item_logits > item_logits[:, [target]]).sum(axis=1)
                original_logit = float(target_logits[0])
                original_prob = float(target_probs[0])
                motif_logits = target_logits[1:11]
                random_logits = target_logits[11:21]
                motif_probs = target_probs[1:11]
                random_probs = target_probs[11:21]
                motif_delta = original_logit - motif_logits
                random_delta = original_logit - random_logits
                contrast = motif_delta - random_delta
                site_result = {
                    "modification": item["modification"],
                    "source_class": item["source_class"],
                    "motif": item["motif"],
                    "site_id": item["site_id"],
                    "cluster_id": item["cluster_id"],
                    "target_class": item["modification"],
                    "target_index": target,
                    "original_target_logit": original_logit,
                    "original_target_probability": original_prob,
                    "motif_delta_logit_mean": float(motif_delta.mean()),
                    "random_delta_logit_mean": float(random_delta.mean()),
                    "delta_logit_contrast_mean": float(contrast.mean()),
                    "motif_delta_logit_sd": float(motif_delta.std(ddof=1)),
                    "random_delta_logit_sd": float(random_delta.std(ddof=1)),
                    "motif_probability_drop_mean": float(original_prob - motif_probs.mean()),
                    "random_probability_drop_mean": float(original_prob - random_probs.mean()),
                    "motif_top1_flip_rate": float(np.mean(predicted[1:11] != predicted[0])),
                    "random_top1_flip_rate": float(np.mean(predicted[11:21] != predicted[0])),
                    "motif_rank_change_mean": float(ranks[1:11].mean() - ranks[0]),
                    "random_rank_change_mean": float(ranks[11:21].mean() - ranks[0]),
                    "motif_delta_positive_rate": float(np.mean(motif_delta > 0)),
                    "random_delta_positive_rate": float(np.mean(random_delta > 0)),
                    "motif_edit_positions_41_0based": item["motif_edit_positions_41_0based"],
                    "random_edit_positions_41_0based": item["random_edit_positions_41_0based"],
                }
                site_rows.append(site_result)
                grouped[(item["source_class"], item["motif"])].append(site_result)
                for variant_idx, meta in enumerate(variant_meta[local_idx * 21 : (local_idx + 1) * 21]):
                    condition = str(meta["condition"])
                    long_writer.writerow(
                        {
                            "modification": item["modification"],
                            "source_class": item["source_class"],
                            "motif": item["motif"],
                            "site_id": item["site_id"],
                            "cluster_id": item["cluster_id"],
                            "condition": condition,
                            "replicate": meta["replicate"],
                            "target_class": item["modification"],
                            "target_index": target,
                            "target_logit": float(target_logits[variant_idx]),
                            "target_probability": float(target_probs[variant_idx]),
                            "predicted_class": DISPLAY_NAMES.get(CLASS_NAMES[int(predicted[variant_idx])], CLASS_NAMES[int(predicted[variant_idx])]),
                            "predicted_index": int(predicted[variant_idx]),
                            "target_rank": int(ranks[variant_idx]),
                            "motif_start_41_0based": item["motif_start_41_0based"],
                            "motif_edit_positions_41_0based": item["motif_edit_positions_41_0based"],
                            "random_start_41_0based": item["random_start_41_0based"],
                            "random_edit_positions_41_0based": item["random_edit_positions_41_0based"],
                        }
                    )
            print(f"processed {min(chunk_start + len(chunk), len(manifest))}/{len(manifest)} sites", flush=True)

    write_tsv(site_path, site_rows)
    rng = np.random.default_rng(args.seed + 9001)
    motif_summary: list[dict[str, object]] = []
    for (source_class, motif), records in sorted(grouped.items()):
        contrasts = np.asarray([float(row["delta_logit_contrast_mean"]) for row in records], dtype=float)
        p_perm = permutation_p(contrasts, args.permutations, rng)
        ci_low, ci_high = bootstrap_ci(contrasts, args.bootstrap, rng)
        motif_summary.append(
            {
                "modification": DISPLAY_NAMES.get(source_class, source_class),
                "source_class": source_class,
                "motif": motif,
                "n_sites": len(records),
                "n_clusters": len({str(row["cluster_id"]) for row in records}),
                "mean_delta_logit_motif": float(np.mean([row["motif_delta_logit_mean"] for row in records])),
                "mean_delta_logit_random": float(np.mean([row["random_delta_logit_mean"] for row in records])),
                "mean_delta_logit_contrast": float(contrasts.mean()),
                "median_delta_logit_contrast": float(np.median(contrasts)),
                "contrast_ci95_low": ci_low,
                "contrast_ci95_high": ci_high,
                "sign_permutation_p": p_perm,
                "mean_motif_probability_drop": float(np.mean([row["motif_probability_drop_mean"] for row in records])),
                "mean_random_probability_drop": float(np.mean([row["random_probability_drop_mean"] for row in records])),
                "mean_motif_top1_flip_rate": float(np.mean([row["motif_top1_flip_rate"] for row in records])),
                "mean_random_top1_flip_rate": float(np.mean([row["random_top1_flip_rate"] for row in records])),
                "mean_motif_rank_change": float(np.mean([row["motif_rank_change_mean"] for row in records])),
                "mean_random_rank_change": float(np.mean([row["random_rank_change_mean"] for row in records])),
                "motif_delta_positive_site_fraction": float(np.mean([row["motif_delta_logit_mean"] > 0 for row in records])),
                "random_delta_positive_site_fraction": float(np.mean([row["random_delta_logit_mean"] > 0 for row in records])),
            }
        )
    q_values = bh_fdr([float(row["sign_permutation_p"]) for row in motif_summary])
    for row, q_value in zip(motif_summary, q_values):
        row["sign_permutation_FDR"] = q_value
        row["contrast_significant_FDR05"] = int(q_value < 0.05 and float(row["contrast_ci95_low"]) > 0)
    write_tsv(args.out_dir / "motif_attribution_summary.tsv", motif_summary)

    modification_summary: list[dict[str, object]] = []
    for source_class in sorted({str(row["source_class"]) for row in site_rows}):
        records = [row for row in site_rows if row["source_class"] == source_class]
        by_cluster = defaultdict(list)
        for row in records:
            by_cluster[str(row['cluster_id'])].append(float(row['delta_logit_contrast_mean']))
        values = np.asarray([np.mean(v) for v in by_cluster.values()], dtype=float)
        ci_low, ci_high = bootstrap_ci(values, args.bootstrap, rng)
        p_perm = permutation_p(values, args.permutations, rng)
        modification_summary.append(
            {
                "modification": DISPLAY_NAMES.get(source_class, source_class),
                "source_class": source_class,
                "n_sites": len(records),
                "n_independent_clusters": len(by_cluster),
                "n_motifs": len({str(row["motif"]) for row in records}),
                "mean_delta_logit_contrast": float(values.mean()),
                "contrast_ci95_low": ci_low,
                "contrast_ci95_high": ci_high,
                "sign_permutation_p": p_perm,
                "mean_motif_probability_drop": float(np.mean([row["motif_probability_drop_mean"] for row in records])),
                "mean_random_probability_drop": float(np.mean([row["random_probability_drop_mean"] for row in records])),
                "mean_motif_top1_flip_rate": float(np.mean([row["motif_top1_flip_rate"] for row in records])),
                "mean_random_top1_flip_rate": float(np.mean([row["random_top1_flip_rate"] for row in records])),
                "mean_motif_rank_change": float(np.mean([row["motif_rank_change_mean"] for row in records])),
                "mean_random_rank_change": float(np.mean([row["random_rank_change_mean"] for row in records])),
            }
        )
    q_values = bh_fdr([float(row["sign_permutation_p"]) for row in modification_summary])
    for row, q_value in zip(modification_summary, q_values):
        row["sign_permutation_FDR"] = q_value
    write_tsv(args.out_dir / "modification_attribution_summary.tsv", modification_summary)

    audit = {
        "experiment": "motif-directed perturbation vs matched random disruption",
        "checkpoint": str(args.model),
        "config": str(args.config),
        "test_dataset": str(args.test_dataset),
        "structure": str(args.structure),
        "rbp_positional": str(args.rbp_positional),
        "device": str(device),
        "seed": args.seed,
        "replicates_per_condition": 10,
        "bootstrap_replicates": args.bootstrap,
        "sign_permutation_replicates": args.permutations,
        "selection": "one motif-positive site per source_class/motif/CD-HIT80 cluster, lexicographically smallest site_id",
        "motif_window": "central 21 nt of sequence_41nt; occurrence closest to center selected",
        "center_index_41_0based": CENTER_41,
        "center_base_mutation": "forbidden",
        "synchronized_offsets": "41/101/201 nt windows use the same offset relative to their center",
        "random_region": "same motif length, central 21 nt, no overlap with motif, excludes center; minimize difference between motif and control absolute distance to center",
        "modification_statistics": "Average motif-site contrasts within each cluster first; bootstrap and sign-permute independent cluster means. Secondary descriptive metrics retain motif-site weighting.",
        "target_classes": [DISPLAY_NAMES.get(name, name) for name in REPLICATED_MOTIFS],
        "source_target_classes": list(REPLICATED_MOTIFS),
        "manifest_rows": len(manifest),
        "unique_sites": len(selected_site_ids),
        "summary_rows": len(motif_summary),
        "output_files": {
            "manifest": manifest_path.name,
            "long_predictions": long_path.name,
            "per_site": site_path.name,
            "motif_summary": "motif_attribution_summary.tsv",
            "modification_summary": "modification_attribution_summary.tsv",
        },
        "class_names": list(CLASS_NAMES),
        "class_aliases": DISPLAY_NAMES,
        "rbp_shape": list(rbp_shape),
        "rbp_metadata": rbp_metadata,
    }
    (args.out_dir / "experiment1_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest_rows": len(manifest),
        "unique_sites": len(selected_site_ids),
        "motif_summary": str(args.out_dir / "motif_attribution_summary.tsv"),
        "device": str(device),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
