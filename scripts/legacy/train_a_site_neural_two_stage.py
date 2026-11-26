#!/usr/bin/env python
"""Train a neural m1A/m6A specialist and calibrate two-stage fusion.

This script keeps the trained 12-class main model fixed.  It trains a small
PyTorch binary classifier only on m1A and m6A training rows, then uses the
validation set to decide how much of the main model's m1A/m6A probability mass
should be redistributed by the specialist.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_a_site_two_stage_reranker import (
    adjust_probability_mass,
    evaluate,
    infer_class_names,
    load_structure,
    parse_csv,
    per_class_rows,
    prediction_matrix,
    read_tsv,
    safe_float,
    split_rows,
    summarize_comparison,
    write_predictions,
    write_tsv,
)


BASE_TO_ID = {"A": 1, "C": 2, "G": 3, "U": 4, "T": 4}
FOCUS_CLASSES = ["m1A", "m6A"]
DEFAULT_STRUCT_FIELDS = [
    "MFE",
    "center_paired",
    "central_11nt_paired_fraction",
    "central_21nt_paired_fraction",
    "paired_fraction_window",
    "MFE_per_nt",
    "loop_span_around_center",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_sequence(seq: str) -> str:
    return str(seq or "").upper().replace("T", "U")


def encode_sequence(seq: str, length: int) -> torch.Tensor:
    seq = normalize_sequence(seq)
    if len(seq) >= length:
        start = max(0, (len(seq) - length) // 2)
        seq = seq[start : start + length]
    else:
        left = (length - len(seq)) // 2
        seq = "N" * left + seq + "N" * (length - len(seq) - left)
    return torch.tensor([BASE_TO_ID.get(base, 0) for base in seq], dtype=torch.long)


def parse_paired_flags(value: str, length: int) -> List[float]:
    raw = [item.strip() for item in str(value or "").split(",") if item.strip()]
    vals = [safe_float(item, 0.0) for item in raw]
    if len(vals) >= length:
        start = max(0, (len(vals) - length) // 2)
        vals = vals[start : start + length]
    else:
        left = (length - len(vals)) // 2
        vals = [0.0] * left + vals + [0.0] * (length - len(vals) - left)
    return vals


def dot_values(value: str, length: int) -> List[float]:
    mapping = {
        ".": 0.0,
        "(": 1.0,
        "[": 1.0,
        "{": 1.0,
        "<": 1.0,
        ")": -1.0,
        "]": -1.0,
        "}": -1.0,
        ">": -1.0,
    }
    chars = list(str(value or ""))
    if len(chars) >= length:
        start = max(0, (len(chars) - length) // 2)
        chars = chars[start : start + length]
    else:
        left = (length - len(chars)) // 2
        chars = ["."] * left + chars + ["."] * (length - len(chars) - left)
    return [mapping.get(ch, 0.0) for ch in chars]


def structure_vector(
    row: Mapping[str, str],
    struct_by_id: Mapping[str, Mapping[str, str]],
    struct_fields: Sequence[str],
    struct_window: int,
) -> np.ndarray:
    struct = struct_by_id.get(str(row.get("site_id", "")), {})
    values: List[float] = [safe_float(struct.get(field), 0.0) for field in struct_fields]
    values.extend(parse_paired_flags(str(struct.get("paired_flags", "")), struct_window))
    values.extend(dot_values(str(struct.get("dot_bracket", "")), struct_window))
    return np.asarray(values, dtype=np.float32)


def cap_focus_train_rows(
    rows: Sequence[Dict[str, str]],
    max_ratio_to_minority: int,
    max_per_class: int,
    seed: int,
) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        mod = str(row.get("mod_type", ""))
        if mod in FOCUS_CLASSES:
            grouped[mod].append(row)
    counts = [len(grouped[name]) for name in FOCUS_CLASSES if grouped.get(name)]
    if len(counts) < len(FOCUS_CLASSES):
        raise ValueError(f"Missing focus class in training data: counts={dict((k, len(v)) for k, v in grouped.items())}")
    minority = max(1, min(counts))
    limit = max(1, minority * max_ratio_to_minority)
    if max_per_class > 0:
        limit = min(limit, max_per_class)
    rng = random.Random(seed)
    capped: List[Dict[str, str]] = []
    for name in FOCUS_CLASSES:
        group = list(grouped[name])
        rng.shuffle(group)
        capped.extend(group[: min(len(group), limit)])
    rng.shuffle(capped)
    return capped


def rows_from_predictions(
    pred_rows: Sequence[Mapping[str, str]],
    rows_by_id: Mapping[str, Dict[str, str]],
    only_focus: bool,
) -> Tuple[List[Dict[str, str]], np.ndarray]:
    rows: List[Dict[str, str]] = []
    labels: List[int] = []
    for pred in pred_rows:
        site_id = str(pred.get("site_id", ""))
        if site_id not in rows_by_id:
            continue
        mod = str(pred.get("true_mod_type", ""))
        if only_focus and mod not in FOCUS_CLASSES:
            continue
        rows.append(rows_by_id[site_id])
        labels.append(1 if mod == "m1A" else 0)
    return rows, np.asarray(labels, dtype=np.float32)


class ASiteDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, str]],
        labels: Optional[Sequence[float]],
        sequence_columns: Sequence[str],
        sequence_lengths: Sequence[int],
        struct_by_id: Mapping[str, Mapping[str, str]],
        struct_fields: Sequence[str],
        struct_window: int,
        struct_mean: np.ndarray,
        struct_std: np.ndarray,
    ):
        self.rows = list(rows)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.sequence_columns = list(sequence_columns)
        self.sequence_lengths = list(sequence_lengths)
        self.struct_by_id = struct_by_id
        self.struct_fields = list(struct_fields)
        self.struct_window = int(struct_window)
        self.struct_mean = struct_mean.astype(np.float32)
        self.struct_std = np.maximum(struct_std.astype(np.float32), 1e-6)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        seqs = [
            encode_sequence(str(row.get(column, "")), length)
            for column, length in zip(self.sequence_columns, self.sequence_lengths)
        ]
        struct = structure_vector(row, self.struct_by_id, self.struct_fields, self.struct_window)
        struct = (struct - self.struct_mean) / self.struct_std
        label = 0.0 if self.labels is None else float(self.labels[idx])
        # Avoid torch.from_numpy here: the server environment can load numpy in
        # a way that makes PyTorch reject otherwise normal ndarray instances.
        return seqs, torch.tensor(struct.tolist(), dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def collate_batch(samples: Sequence[Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]]) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    num_lengths = len(samples[0][0])
    seqs = [torch.stack([sample[0][idx] for sample in samples], dim=0) for idx in range(num_lengths)]
    struct = torch.stack([sample[1] for sample in samples], dim=0)
    labels = torch.stack([sample[2] for sample in samples], dim=0)
    return seqs, struct, labels


class ConvSequenceBranch(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int, kernels: Sequence[int], dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(5, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, out_dim, kernel_size=k, padding=k // 2) for k in kernels])
        self.proj = nn.Sequential(
            nn.Linear(len(kernels) * out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = F.gelu(conv(emb))
            pooled.append(h.max(dim=-1).values)
            pooled.append(h.mean(dim=-1))
        return self.proj(torch.cat(pooled, dim=-1))


class ASiteNeuralSpecialist(nn.Module):
    def __init__(
        self,
        num_lengths: int,
        struct_dim: int,
        embed_dim: int,
        branch_dim: int,
        hidden_dim: int,
        kernels: Sequence[int],
        dropout: float,
        share_sequence_encoder: bool,
    ):
        super().__init__()
        if share_sequence_encoder:
            shared = ConvSequenceBranch(embed_dim, branch_dim, kernels, dropout)
            self.branches = nn.ModuleList([shared for _ in range(num_lengths)])
        else:
            self.branches = nn.ModuleList(
                [ConvSequenceBranch(embed_dim, branch_dim, kernels, dropout) for _ in range(num_lengths)]
            )
        self.struct_mlp = nn.Sequential(
            nn.Linear(struct_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fusion_dim = branch_dim * (num_lengths + 1)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, seqs: Sequence[torch.Tensor], struct: torch.Tensor) -> torch.Tensor:
        parts = [branch(seq) for branch, seq in zip(self.branches, seqs)]
        parts.append(self.struct_mlp(struct))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def compute_struct_norm(
    rows: Sequence[Mapping[str, str]],
    struct_by_id: Mapping[str, Mapping[str, str]],
    struct_fields: Sequence[str],
    struct_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    vectors = np.stack([structure_vector(row, struct_by_id, struct_fields, struct_window) for row in rows], axis=0)
    return vectors.mean(axis=0).astype(np.float32), vectors.std(axis=0).astype(np.float32)


def make_balanced_sampler(labels: np.ndarray, epoch_size_multiplier: float) -> WeightedRandomSampler:
    counts = Counter(int(x) for x in labels.tolist())
    weights = np.asarray([1.0 / max(1, counts[int(label)]) for label in labels], dtype=np.float64)
    num_samples = max(1, int(round(len(labels) * epoch_size_multiplier)))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)


def move_batch(
    seqs: Sequence[torch.Tensor],
    struct: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    return [seq.to(device, non_blocking=True) for seq in seqs], struct.to(device, non_blocking=True), labels.to(device, non_blocking=True)


@torch.no_grad()
def predict_prob(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    for seqs, struct, labels in loader:
        seqs, struct, labels = move_batch(seqs, struct, labels, device)
        chunks.append(torch.sigmoid(model(seqs, struct)).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    out = {"AUPR": float("nan"), "AUC": float("nan")}
    if len(np.unique(y_true)) > 1:
        out["AUPR"] = float(average_precision_score(y_true, prob))
        out["AUC"] = float(roc_auc_score(y_true, prob))
    return out


def calibrate_probability(prob: np.ndarray, temperature: float, bias: float) -> np.ndarray:
    prob = np.clip(prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(prob / (1.0 - prob))
    return 1.0 / (1.0 + np.exp(-(logits / max(temperature, 1e-6) + bias)))


def specialist_matrix(m1a_prob: np.ndarray) -> np.ndarray:
    p = np.clip(m1a_prob.astype(np.float64), 1e-8, 1.0 - 1e-8)
    return np.stack([p, 1.0 - p], axis=1)


def binary_f1_from_pred(labels: np.ndarray, pred: np.ndarray, positive_index: int) -> float:
    y = labels == positive_index
    q = pred == positive_index
    tp = float(np.logical_and(y, q).sum())
    fp = float(np.logical_and(~y, q).sum())
    fn = float(np.logical_and(y, ~q).sum())
    denom = 2.0 * tp + fp + fn
    return 0.0 if denom <= 0.0 else (2.0 * tp / denom)


def guarded_selection_score(
    metrics: Mapping[str, object],
    base_metrics: Mapping[str, object],
    target_class: str,
    guard_classes: Sequence[str],
    m6a_drop_penalty: float,
    macro_drop_penalty: float,
) -> float:
    per_class = metrics.get("per_class", {})
    base_pc = base_metrics.get("per_class", {})
    if not isinstance(per_class, Mapping) or not isinstance(base_pc, Mapping):
        return float("-inf")
    target_values = per_class.get(target_class, {})
    if not isinstance(target_values, Mapping):
        return float("-inf")
    target = safe_float(target_values.get("AUPR_ovr"), float("nan"))
    target_f1 = safe_float(target_values.get("F1"), 0.0)
    if not math.isfinite(target):
        return float("-inf")
    score = target + 0.02 * target_f1
    for name in guard_classes:
        values = per_class.get(name, {})
        base_values = base_pc.get(name, {})
        if isinstance(values, Mapping) and isinstance(base_values, Mapping):
            drop = safe_float(base_values.get("AUPR_ovr"), 0.0) - safe_float(values.get("AUPR_ovr"), 0.0)
            score -= m6a_drop_penalty * max(0.0, drop)
    macro_drop = safe_float(base_metrics.get("macro_AUPR_ovr"), 0.0) - safe_float(metrics.get("macro_AUPR_ovr"), 0.0)
    score -= macro_drop_penalty * max(0.0, macro_drop)
    return score


def search_fusion(
    base_probs: np.ndarray,
    labels: np.ndarray,
    raw_specialist_m1a: np.ndarray,
    class_names: Sequence[str],
    alphas: Sequence[float],
    temperatures: Sequence[float],
    biases: Sequence[float],
    target_class: str,
    guard_classes: Sequence[str],
    m6a_drop_penalty: float,
    macro_drop_penalty: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    m1a_idx = class_names.index("m1A")
    m6a_idx = class_names.index("m6A")
    focus_indices = [m1a_idx, m6a_idx]
    base_metrics = evaluate(base_probs, labels, class_names)
    base_pc = base_metrics["per_class"]  # type: ignore[index]
    base_m1a_aupr = float(base_pc["m1A"]["AUPR_ovr"])
    base_m6a_aupr = float(base_pc["m6A"]["AUPR_ovr"])
    base_macro_aupr = float(base_metrics["macro_AUPR_ovr"])
    y_m1a = (labels == m1a_idx).astype(np.int32)
    y_m6a = (labels == m6a_idx).astype(np.int32)
    other_probs = base_probs.copy()
    other_probs[:, [m1a_idx, m6a_idx]] = -1.0
    other_argmax = other_probs.argmax(axis=1)
    other_max = other_probs[np.arange(len(labels)), other_argmax]
    focus_mass = base_probs[:, m1a_idx] + base_probs[:, m6a_idx]
    base_cond_m1a = base_probs[:, m1a_idx] / np.maximum(focus_mass, 1e-12)
    candidates: List[Dict[str, object]] = []
    best: Optional[Dict[str, object]] = None
    for alpha in alphas:
        for temp in temperatures:
            for bias in biases:
                calibrated = calibrate_probability(raw_specialist_m1a, temp, bias)
                mixed_m1a = (1.0 - alpha) * base_cond_m1a + alpha * calibrated
                m1a_prob = focus_mass * mixed_m1a
                m6a_prob = focus_mass * (1.0 - mixed_m1a)
                m1a_aupr = float(average_precision_score(y_m1a, m1a_prob)) if y_m1a.max() > 0 else float("nan")
                m6a_aupr = float(average_precision_score(y_m6a, m6a_prob)) if y_m6a.max() > 0 else float("nan")
                focus_best = np.maximum(m1a_prob, m6a_prob)
                pred = np.where(
                    focus_best > other_max,
                    np.where(m1a_prob >= m6a_prob, m1a_idx, m6a_idx),
                    other_argmax,
                )
                m1a_f1 = binary_f1_from_pred(labels, pred, m1a_idx)
                m6a_f1 = binary_f1_from_pred(labels, pred, m6a_idx)
                macro_aupr = base_macro_aupr + (m1a_aupr - base_m1a_aupr + m6a_aupr - base_m6a_aupr) / max(1, len(class_names))
                score = m1a_aupr + 0.02 * m1a_f1
                score -= m6a_drop_penalty * max(0.0, base_m6a_aupr - m6a_aupr)
                score -= macro_drop_penalty * max(0.0, base_macro_aupr - macro_aupr)
                row: Dict[str, object] = {
                    "alpha": alpha,
                    "temperature": temp,
                    "bias": bias,
                    "selection_score": score,
                    "valid_macro_AUPR_ovr": macro_aupr,
                    "valid_m1A_AUPR_ovr": m1a_aupr,
                    "valid_m6A_AUPR_ovr": m6a_aupr,
                    "valid_m1A_F1": m1a_f1,
                    "valid_m6A_F1": m6a_f1,
                }
                candidates.append(row)
                if best is None or score > float(best["selection_score"]):
                    best = row
    if best is None:
        raise RuntimeError("No fusion candidate was evaluated.")
    return best, candidates


def apply_fusion(
    base_probs: np.ndarray,
    specialist_m1a: np.ndarray,
    class_names: Sequence[str],
    alpha: float,
    temperature: float,
    bias: float,
) -> np.ndarray:
    focus_indices = [class_names.index(name) for name in FOCUS_CLASSES]
    calibrated = calibrate_probability(specialist_m1a, temperature, bias)
    return adjust_probability_mass(base_probs, focus_indices, specialist_matrix(calibrated), alpha)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv")
    parser.add_argument("--rnafold", default="rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv")
    parser.add_argument("--base-model-dir", default="rmbase12_work/models/multirm12_multi_length_fusion_cdhit80/multi_length_independent")
    parser.add_argument("--out-dir", default="rmbase12_work/metrics/a_site_neural_two_stage/multi_length_independent")
    parser.add_argument("--split", default="split_cdhit80_cluster")
    parser.add_argument("--class-names", default="")
    parser.add_argument("--sequence-columns", default="sequence_41nt,sequence_101nt,sequence_201nt")
    parser.add_argument("--sequence-lengths", default="41,101,201")
    parser.add_argument("--struct-fields", default=",".join(DEFAULT_STRUCT_FIELDS))
    parser.add_argument("--struct-window", type=int, default=41)
    parser.add_argument("--max-ratio-to-minority", type=int, default=40)
    parser.add_argument("--max-train-per-class", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--branch-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--kernels", default="3,5,7,11")
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--share-sequence-encoder", action="store_true")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--sampler-epoch-size-multiplier", type=float, default=1.0)
    parser.add_argument("--pos-weight", default="none", choices=["none", "ratio", "sqrt_ratio"])
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--alphas", default="0,0.02,0.05,0.1,0.15,0.2,0.35,0.5,0.75,1")
    parser.add_argument("--temperatures", default="0.5,0.75,1,1.5,2")
    parser.add_argument("--biases", default="-2,-1,-0.5,0,0.5,1,2")
    parser.add_argument("--target-class", default="m1A")
    parser.add_argument("--guard-classes", default="m6A")
    parser.add_argument("--m6a-drop-penalty", type=float, default=5.0)
    parser.add_argument("--macro-drop-penalty", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-checkpoint", action="store_true", help="Load out-dir/checkpoint.pt and skip training.")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequence_columns = parse_csv(args.sequence_columns)
    sequence_lengths = [int(item) for item in parse_csv(args.sequence_lengths)]
    if len(sequence_columns) != len(sequence_lengths):
        raise ValueError("--sequence-columns and --sequence-lengths must have the same length.")
    struct_fields = parse_csv(args.struct_fields)
    kernels = [int(item) for item in parse_csv(args.kernels)]

    dataset_rows = read_tsv(Path(args.dataset))
    train_rows, _, _ = split_rows(dataset_rows, args.split)
    rows_by_id = {row["site_id"]: row for row in dataset_rows if row.get("site_id")}
    valid_pred_rows = read_tsv(Path(args.base_model_dir) / "valid_predictions.tsv")
    test_pred_rows = read_tsv(Path(args.base_model_dir) / "test_predictions.tsv")
    class_names = infer_class_names(test_pred_rows, args.class_names)
    for name in FOCUS_CLASSES:
        if name not in class_names:
            raise ValueError(f"Missing focus class in class names: {name}")
    struct_by_id = load_structure(args.rnafold)

    train_focus = cap_focus_train_rows(train_rows, args.max_ratio_to_minority, args.max_train_per_class, args.seed)
    y_train = np.asarray([1.0 if row.get("mod_type") == "m1A" else 0.0 for row in train_focus], dtype=np.float32)
    valid_focus_rows, y_valid_focus = rows_from_predictions(valid_pred_rows, rows_by_id, only_focus=True)
    valid_all_rows, _ = rows_from_predictions(valid_pred_rows, rows_by_id, only_focus=False)
    test_all_rows, _ = rows_from_predictions(test_pred_rows, rows_by_id, only_focus=False)
    if len(valid_all_rows) != len(valid_pred_rows) or len(test_all_rows) != len(test_pred_rows):
        raise ValueError("Prediction rows and dataset rows do not have matching site_id coverage.")

    struct_mean, struct_std = compute_struct_norm(train_focus, struct_by_id, struct_fields, args.struct_window)
    train_dataset = ASiteDataset(
        train_focus,
        y_train,
        sequence_columns,
        sequence_lengths,
        struct_by_id,
        struct_fields,
        args.struct_window,
        struct_mean,
        struct_std,
    )
    valid_focus_dataset = ASiteDataset(
        valid_focus_rows,
        y_valid_focus,
        sequence_columns,
        sequence_lengths,
        struct_by_id,
        struct_fields,
        args.struct_window,
        struct_mean,
        struct_std,
    )
    valid_all_dataset = ASiteDataset(
        valid_all_rows,
        None,
        sequence_columns,
        sequence_lengths,
        struct_by_id,
        struct_fields,
        args.struct_window,
        struct_mean,
        struct_std,
    )
    test_all_dataset = ASiteDataset(
        test_all_rows,
        None,
        sequence_columns,
        sequence_lengths,
        struct_by_id,
        struct_fields,
        args.struct_window,
        struct_mean,
        struct_std,
    )

    sampler = make_balanced_sampler(y_train, args.sampler_epoch_size_multiplier) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    eval_kwargs = {
        "batch_size": args.batch_size * 2,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_batch,
    }
    valid_focus_loader = DataLoader(valid_focus_dataset, **eval_kwargs)
    valid_all_loader = DataLoader(valid_all_dataset, **eval_kwargs)
    test_all_loader = DataLoader(test_all_dataset, **eval_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASiteNeuralSpecialist(
        num_lengths=len(sequence_columns),
        struct_dim=len(struct_mean),
        embed_dim=args.embed_dim,
        branch_dim=args.branch_dim,
        hidden_dim=args.hidden_dim,
        kernels=kernels,
        dropout=args.dropout,
        share_sequence_encoder=args.share_sequence_encoder,
    ).to(device)

    pos_count = float((y_train == 1).sum())
    neg_count = float((y_train == 0).sum())
    pos_weight_value = 1.0
    if args.pos_weight == "ratio":
        pos_weight_value = max(1.0, neg_count / max(1.0, pos_count))
    elif args.pos_weight == "sqrt_ratio":
        pos_weight_value = math.sqrt(max(1.0, neg_count / max(1.0, pos_count)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(
        "[load] "
        f"dataset_rows={len(dataset_rows)} train_focus_counts={dict(Counter(row.get('mod_type', '') for row in train_focus))} "
        f"valid_focus={len(valid_focus_rows)} valid_all={len(valid_all_rows)} test_all={len(test_all_rows)} "
        f"struct_dim={len(struct_mean)} device={device} pos_weight={pos_weight_value:.4f}",
        flush=True,
    )

    history: List[Dict[str, object]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    checkpoint_path = out_dir / "checkpoint.pt"
    if args.reuse_checkpoint and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        best_state = checkpoint.get("best_state_dict") or checkpoint.get("model_state_dict")
        if best_state is None:
            raise RuntimeError(f"No model state found in {checkpoint_path}")
        best_score = float(checkpoint.get("best_score", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
        history = list(checkpoint.get("history", []))
        print(f"[reuse_checkpoint] {checkpoint_path} best_epoch={best_epoch} best_score={best_score:.6f}", flush=True)
    else:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for seqs, struct, labels in train_loader:
                seqs, struct, labels = move_batch(seqs, struct, labels, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(seqs, struct)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()

            valid_focus_prob = predict_prob(model, valid_focus_loader, device)
            valid_binary = binary_metrics(y_valid_focus, valid_focus_prob)
            score = valid_binary["AUPR"]
            improved = score > best_score + args.early_stopping_min_delta
            if improved:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "valid_focus_AUPR": valid_binary["AUPR"],
                "valid_focus_AUC": valid_binary["AUC"],
                "improved": improved,
                "no_improve": no_improve,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"epoch={epoch} loss={row['loss']:.4f} valid_focus_AUPR={row['valid_focus_AUPR']:.6f} "
                f"valid_focus_AUC={row['valid_focus_AUC']:.6f} no_improve={no_improve}",
                flush=True,
            )
            torch.save(
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "best_state_dict": best_state,
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "args": vars(args),
                    "struct_mean": struct_mean,
                    "struct_std": struct_std,
                    "sequence_columns": sequence_columns,
                    "sequence_lengths": sequence_lengths,
                    "struct_fields": struct_fields,
                },
                checkpoint_path,
            )
            if no_improve >= args.early_stopping_patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_score={best_score:.6f}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("[predict] valid/test specialist probabilities", flush=True)
    valid_specialist = predict_prob(model, valid_all_loader, device)
    test_specialist = predict_prob(model, test_all_loader, device)
    valid_base_probs, valid_labels, _, _ = prediction_matrix(valid_pred_rows, class_names)
    test_base_probs, test_labels, _, _ = prediction_matrix(test_pred_rows, class_names)
    valid_base_metrics = evaluate(valid_base_probs, valid_labels, class_names)
    test_base_metrics = evaluate(test_base_probs, test_labels, class_names)

    print("[fusion] validation grid search", flush=True)
    best_fusion, candidates = search_fusion(
        valid_base_probs,
        valid_labels,
        valid_specialist,
        class_names,
        [float(item) for item in parse_csv(args.alphas)],
        [float(item) for item in parse_csv(args.temperatures)],
        [float(item) for item in parse_csv(args.biases)],
        args.target_class,
        parse_csv(args.guard_classes),
        args.m6a_drop_penalty,
        args.macro_drop_penalty,
    )
    print(f"[fusion] best={best_fusion}", flush=True)
    test_adjusted = apply_fusion(
        test_base_probs,
        test_specialist,
        class_names,
        float(best_fusion["alpha"]),
        float(best_fusion["temperature"]),
        float(best_fusion["bias"]),
    )
    test_metrics = evaluate(test_adjusted, test_labels, class_names)

    write_tsv(out_dir / "training_history.tsv", history)
    write_tsv(out_dir / "validation_fusion_grid.tsv", candidates)
    write_tsv(out_dir / "test_per_class_metrics.tsv", per_class_rows(test_metrics, class_names))
    write_tsv(out_dir / "test_base_vs_neural_two_stage_per_class.tsv", summarize_comparison(test_base_metrics, test_metrics, class_names, FOCUS_CLASSES))
    write_predictions(out_dir / "test_predictions.tsv", test_pred_rows, test_adjusted, class_names)

    summary = {
        "focus_classes": FOCUS_CLASSES,
        "class_names": list(class_names),
        "base_model_dir": args.base_model_dir,
        "dataset": args.dataset,
        "rnafold": args.rnafold,
        "train_focus_counts": dict(Counter(row.get("mod_type", "") for row in train_focus)),
        "valid_focus_counts": {"m1A": int((y_valid_focus == 1).sum()), "m6A": int((y_valid_focus == 0).sum())},
        "best_epoch": best_epoch,
        "best_focus_validation_AUPR": best_score,
        "best_validation_fusion": best_fusion,
        "validation_base": valid_base_metrics,
        "test_base": test_base_metrics,
        "test_neural_two_stage": test_metrics,
        "args": vars(args),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    base_m1a = test_base_metrics["per_class"]["m1A"]["AUPR_ovr"]  # type: ignore[index]
    new_m1a = test_metrics["per_class"]["m1A"]["AUPR_ovr"]  # type: ignore[index]
    base_m6a = test_base_metrics["per_class"]["m6A"]["AUPR_ovr"]  # type: ignore[index]
    new_m6a = test_metrics["per_class"]["m6A"]["AUPR_ovr"]  # type: ignore[index]
    print(
        f"[done] best_epoch={best_epoch} alpha={best_fusion['alpha']} temp={best_fusion['temperature']} "
        f"bias={best_fusion['bias']} test_m1A_AUPR={new_m1a} delta_m1A={float(new_m1a) - float(base_m1a)} "
        f"test_m6A_AUPR={new_m6a} delta_m6A={float(new_m6a) - float(base_m6a)}",
        flush=True,
    )
    print(out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
