#!/usr/bin/env python
"""Train an A-family neural specialist and calibrate two-stage fusion.

The fixed 12-class model stays unchanged. This script trains a local neural
classifier on A-site-related RNA modifications and uses the validation set to
decide how much A-family probability mass should be redistributed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_a_site_neural_two_stage import (
    ASiteDataset,
    ConvSequenceBranch,
    DEFAULT_STRUCT_FIELDS,
    collate_batch,
    compute_struct_norm,
    set_seed,
)
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


A_FAMILY_CLASSES = ["Am", "m1A", "m6A", "m6Am", "AtoI"]


def cap_family_train_rows(
    rows: Sequence[Dict[str, str]],
    focus_classes: Sequence[str],
    max_ratio_to_minority: int,
    max_per_class: int,
    cap_minority_floor: int,
    seed: int,
) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        mod = str(row.get("mod_type", ""))
        if mod in focus_classes:
            grouped[mod].append(row)
    counts = [len(grouped[name]) for name in focus_classes if grouped.get(name)]
    if not counts:
        raise ValueError("No A-family training rows were found.")
    reference = max(max(1, min(counts)), int(cap_minority_floor))
    limit = reference * max(1, int(max_ratio_to_minority))
    if max_per_class > 0:
        limit = min(limit, int(max_per_class))
    rng = random.Random(seed)
    capped: List[Dict[str, str]] = []
    for name in focus_classes:
        group = list(grouped.get(name, []))
        rng.shuffle(group)
        capped.extend(group[: min(len(group), limit)])
    rng.shuffle(capped)
    return capped


def rows_from_predictions(
    pred_rows: Sequence[Mapping[str, str]],
    rows_by_id: Mapping[str, Dict[str, str]],
    focus_classes: Sequence[str],
    only_focus: bool,
) -> Tuple[List[Dict[str, str]], np.ndarray, List[Mapping[str, str]]]:
    focus_to_idx = {name: idx for idx, name in enumerate(focus_classes)}
    rows: List[Dict[str, str]] = []
    labels: List[int] = []
    matched_predictions: List[Mapping[str, str]] = []
    for pred in pred_rows:
        site_id = str(pred.get("site_id", ""))
        if site_id not in rows_by_id:
            continue
        mod = str(pred.get("true_mod_type", ""))
        if only_focus and mod not in focus_to_idx:
            continue
        rows.append(rows_by_id[site_id])
        labels.append(focus_to_idx.get(mod, -1))
        matched_predictions.append(pred)
    return rows, np.asarray(labels, dtype=np.int64), matched_predictions


def prediction_prior_vector(pred: Mapping[str, str], class_names: Sequence[str], focus_classes: Sequence[str]) -> np.ndarray:
    probs = np.asarray([safe_float(pred.get(f"prob_{name}"), 0.0) for name in class_names], dtype=np.float64)
    focus_indices = [class_names.index(name) for name in focus_classes]
    focus_probs = probs[focus_indices]
    focus_mass = float(focus_probs.sum())
    cond = focus_probs / max(focus_mass, 1e-12)
    if focus_mass <= 1e-12:
        cond = np.full(len(focus_classes), 1.0 / max(1, len(focus_classes)), dtype=np.float64)
    sorted_cond = np.sort(cond)[::-1]
    top1 = float(sorted_cond[0]) if len(sorted_cond) else 0.0
    top2 = float(sorted_cond[1]) if len(sorted_cond) > 1 else 0.0
    entropy = float(-(cond * np.log(np.clip(cond, 1e-12, 1.0))).sum())
    other = probs.copy()
    other[focus_indices] = -1.0
    other_max = float(other.max()) if len(other) else 0.0
    focus_abs_max = float(focus_probs.max()) if len(focus_probs) else 0.0
    m1a_abs = float(probs[class_names.index("m1A")]) if "m1A" in class_names else 0.0
    m1a_cond = float(cond[focus_classes.index("m1A")]) if "m1A" in focus_classes else 0.0
    mass_logit = math.log(max(focus_mass, 1e-8) / max(1.0 - focus_mass, 1e-8))
    extras = np.asarray(
        [
            focus_mass,
            mass_logit,
            entropy,
            top1,
            top1 - top2,
            focus_abs_max,
            other_max,
            focus_abs_max - other_max,
            m1a_abs,
            m1a_cond,
        ],
        dtype=np.float64,
    )
    return np.concatenate([cond.astype(np.float64), extras], axis=0).astype(np.float32)


def prior_by_site_id(
    pred_rows: Sequence[Mapping[str, str]],
    class_names: Sequence[str],
    focus_classes: Sequence[str],
) -> Dict[str, np.ndarray]:
    return {
        str(row.get("site_id", "")): prediction_prior_vector(row, class_names, focus_classes)
        for row in pred_rows
        if row.get("site_id")
    }


def prior_matrix_from_rows(
    rows: Sequence[Mapping[str, str]],
    prior_by_id: Mapping[str, np.ndarray],
    prior_dim: int,
) -> np.ndarray:
    if prior_dim <= 0:
        return np.zeros((len(rows), 0), dtype=np.float32)
    values = []
    zero = np.zeros(prior_dim, dtype=np.float32)
    for row in rows:
        values.append(prior_by_id.get(str(row.get("site_id", "")), zero))
    return np.stack(values, axis=0).astype(np.float32) if values else np.zeros((0, prior_dim), dtype=np.float32)


def prior_matrix_from_predictions(pred_rows: Sequence[Mapping[str, str]], class_names: Sequence[str], focus_classes: Sequence[str]) -> np.ndarray:
    values = [prediction_prior_vector(row, class_names, focus_classes) for row in pred_rows]
    return np.stack(values, axis=0).astype(np.float32) if values else np.zeros((0, 0), dtype=np.float32)


def normalize_prior_features(
    values: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.shape[1] == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return values.astype(np.float32), empty, empty
    if mean is None:
        mean = values.mean(axis=0).astype(np.float32)
    if std is None:
        std = values.std(axis=0).astype(np.float32)
    std = np.maximum(std.astype(np.float32), 1e-6)
    return ((values.astype(np.float32) - mean) / std).astype(np.float32), mean, std


class AFamilyPriorDataset(Dataset):
    def __init__(self, base: ASiteDataset, prior_features: np.ndarray):
        self.base = base
        prior_features = np.asarray(prior_features, dtype=np.float32)
        if len(base) != prior_features.shape[0]:
            raise ValueError(f"Prior feature rows do not match dataset rows: {prior_features.shape[0]} vs {len(base)}")
        self.prior = torch.tensor(prior_features.tolist(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        seqs, struct, label = self.base[idx]
        return seqs, struct, self.prior[idx], label


def collate_prior_batch(
    samples: Sequence[Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    num_lengths = len(samples[0][0])
    seqs = [torch.stack([sample[0][idx] for sample in samples], dim=0) for idx in range(num_lengths)]
    struct = torch.stack([sample[1] for sample in samples], dim=0)
    prior = torch.stack([sample[2] for sample in samples], dim=0)
    labels = torch.stack([sample[3] for sample in samples], dim=0)
    return seqs, struct, prior, labels


class AFamilyNeuralSpecialist(nn.Module):
    def __init__(
        self,
        num_lengths: int,
        struct_dim: int,
        num_classes: int,
        embed_dim: int,
        branch_dim: int,
        hidden_dim: int,
        kernels: Sequence[int],
        dropout: float,
        share_sequence_encoder: bool,
        prior_dim: int = 0,
    ):
        super().__init__()
        self.prior_dim = int(prior_dim)
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
        if self.prior_dim > 0:
            self.prior_mlp = nn.Sequential(
                nn.Linear(self.prior_dim, branch_dim),
                nn.LayerNorm(branch_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.prior_mlp = None
        fusion_dim = branch_dim * (num_lengths + 1 + (1 if self.prior_mlp is not None else 0))
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, seqs: Sequence[torch.Tensor], struct: torch.Tensor, prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts = [branch(seq) for branch, seq in zip(self.branches, seqs)]
        parts.append(self.struct_mlp(struct))
        if self.prior_mlp is not None:
            if prior is None:
                raise ValueError("prior features are required when prior_dim > 0")
            parts.append(self.prior_mlp(prior))
        return self.head(torch.cat(parts, dim=-1))


def make_balanced_sampler(labels: np.ndarray, epoch_size_multiplier: float) -> WeightedRandomSampler:
    counts = Counter(int(x) for x in labels.tolist())
    weights = np.asarray([1.0 / max(1, counts[int(label)]) for label in labels], dtype=np.float64)
    num_samples = max(1, int(round(len(labels) * epoch_size_multiplier)))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)


def class_weight_tensor(labels: np.ndarray, num_classes: int, power: float, max_weight: float, device: torch.device) -> torch.Tensor:
    counts = Counter(int(x) for x in labels.tolist())
    total = max(1, len(labels))
    weights = []
    for idx in range(num_classes):
        count = max(1, counts.get(idx, 0))
        value = (total / max(1.0, num_classes * count)) ** max(0.0, power)
        weights.append(min(float(max_weight), float(value)))
    arr = np.asarray(weights, dtype=np.float32)
    positive = arr[arr > 0]
    if len(positive):
        arr = arr / max(1e-6, float(positive.mean()))
    return torch.tensor(arr, dtype=torch.float32, device=device)


def move_batch(
    seqs: Sequence[torch.Tensor],
    struct: torch.Tensor,
    prior: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        [seq.to(device, non_blocking=True) for seq in seqs],
        struct.to(device, non_blocking=True),
        prior.to(device, non_blocking=True),
        labels.to(device, non_blocking=True).long(),
    )


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    for seqs, struct, prior, labels in loader:
        seqs, struct, prior, labels = move_batch(seqs, struct, prior, labels, device)
        chunks.append(model(seqs, struct, prior).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.maximum(exp_z.sum(axis=1, keepdims=True), 1e-12)


def calibrated_specialist_probs(
    logits: np.ndarray,
    temperature: float,
    m1a_bias: float,
    focus_classes: Sequence[str],
) -> np.ndarray:
    z = logits.astype(np.float64) / max(temperature, 1e-6)
    if "m1A" in focus_classes:
        z[:, focus_classes.index("m1A")] += float(m1a_bias)
    return softmax_np(z)


def focus_aupr(
    labels: np.ndarray,
    probs: np.ndarray,
    class_idx: int,
) -> float:
    y = (labels == class_idx).astype(np.int32)
    if y.max() <= 0 or y.min() >= 1:
        return float("nan")
    return float(average_precision_score(y, probs))


def focus_f1(labels: np.ndarray, pred: np.ndarray, global_idx: int) -> float:
    y = (labels == global_idx).astype(np.int32)
    q = (pred == global_idx).astype(np.int32)
    return float(f1_score(y, q, zero_division=0))


def specialist_validation_metrics(
    labels_local: np.ndarray,
    probs: np.ndarray,
    focus_classes: Sequence[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for idx, name in enumerate(focus_classes):
        y = (labels_local == idx).astype(np.int32)
        if len(np.unique(y)) > 1:
            out[f"{name}_AUPR"] = float(average_precision_score(y, probs[:, idx]))
            out[f"{name}_AUC"] = float(roc_auc_score(y, probs[:, idx]))
        else:
            out[f"{name}_AUPR"] = float("nan")
            out[f"{name}_AUC"] = float("nan")
    values = [value for key, value in out.items() if key.endswith("_AUPR") and math.isfinite(value)]
    out["macro_AUPR"] = float(np.mean(values)) if values else float("nan")
    return out


def search_fusion(
    base_probs: np.ndarray,
    labels: np.ndarray,
    specialist_logits: np.ndarray,
    class_names: Sequence[str],
    focus_classes: Sequence[str],
    alphas: Sequence[float],
    temperatures: Sequence[float],
    m1a_biases: Sequence[float],
    target_class: str,
    guard_classes: Sequence[str],
    guard_drop_penalty: float,
    macro_drop_penalty: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    focus_indices = [class_names.index(name) for name in focus_classes]
    focus_global = np.asarray(focus_indices, dtype=np.int64)
    base_metrics = evaluate(base_probs, labels, class_names)
    base_pc = base_metrics["per_class"]  # type: ignore[index]
    base_macro = safe_float(base_metrics.get("macro_AUPR_ovr"), 0.0)
    base_focus_aupr = {
        name: safe_float(base_pc.get(name, {}).get("AUPR_ovr"), float("nan"))  # type: ignore[union-attr]
        for name in focus_classes
    }
    other_probs = base_probs.copy()
    other_probs[:, focus_indices] = -1.0
    other_argmax = other_probs.argmax(axis=1)
    other_max = other_probs[np.arange(len(labels)), other_argmax]
    focus_mass = base_probs[:, focus_indices].sum(axis=1)
    base_cond = base_probs[:, focus_indices] / np.maximum(focus_mass[:, None], 1e-12)
    uniform = np.full_like(base_cond, 1.0 / max(1, len(focus_indices)))
    base_cond = np.where(focus_mass[:, None] > 1e-12, base_cond, uniform)

    candidates: List[Dict[str, object]] = []
    best: Optional[Dict[str, object]] = None
    for alpha in alphas:
        for temp in temperatures:
            for m1a_bias in m1a_biases:
                specialist = calibrated_specialist_probs(specialist_logits, temp, m1a_bias, focus_classes)
                mixed = (1.0 - alpha) * base_cond + alpha * specialist
                new_focus_probs = focus_mass[:, None] * mixed
                pred_local = new_focus_probs.argmax(axis=1)
                focus_best = new_focus_probs[np.arange(len(labels)), pred_local]
                pred = np.where(focus_best > other_max, focus_global[pred_local], other_argmax)
                row: Dict[str, object] = {
                    "alpha": float(alpha),
                    "temperature": float(temp),
                    "m1a_bias": float(m1a_bias),
                }
                delta_sum = 0.0
                finite_delta_n = 0
                for local_idx, name in enumerate(focus_classes):
                    aupr = focus_aupr(labels, new_focus_probs[:, local_idx], class_names.index(name))
                    f1 = focus_f1(labels, pred, class_names.index(name))
                    base_aupr = base_focus_aupr.get(name, float("nan"))
                    row[f"valid_{name}_AUPR_ovr"] = aupr
                    row[f"valid_{name}_F1"] = f1
                    if math.isfinite(aupr) and math.isfinite(base_aupr):
                        delta_sum += aupr - base_aupr
                        finite_delta_n += 1
                macro_approx = base_macro + delta_sum / max(1, len(class_names))
                target_aupr = safe_float(row.get(f"valid_{target_class}_AUPR_ovr"), float("nan"))
                target_f1 = safe_float(row.get(f"valid_{target_class}_F1"), 0.0)
                if not math.isfinite(target_aupr):
                    score = float("-inf")
                else:
                    score = target_aupr + 0.02 * target_f1
                    for name in guard_classes:
                        current = safe_float(row.get(f"valid_{name}_AUPR_ovr"), float("nan"))
                        base_value = base_focus_aupr.get(name, float("nan"))
                        if math.isfinite(current) and math.isfinite(base_value):
                            score -= guard_drop_penalty * max(0.0, base_value - current)
                    score -= macro_drop_penalty * max(0.0, base_macro - macro_approx)
                row["valid_macro_AUPR_ovr_approx"] = macro_approx
                row["selection_score"] = score
                candidates.append(row)
                if best is None or score > safe_float(best.get("selection_score"), float("-inf")):
                    best = row
    if best is None:
        raise RuntimeError("No fusion candidate was evaluated.")
    return best, candidates


def apply_fusion(
    base_probs: np.ndarray,
    specialist_logits: np.ndarray,
    class_names: Sequence[str],
    focus_classes: Sequence[str],
    alpha: float,
    temperature: float,
    m1a_bias: float,
) -> np.ndarray:
    focus_indices = [class_names.index(name) for name in focus_classes]
    specialist = calibrated_specialist_probs(specialist_logits, temperature, m1a_bias, focus_classes)
    return adjust_probability_mass(base_probs, focus_indices, specialist, alpha)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv")
    parser.add_argument("--rnafold", default="rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv")
    parser.add_argument("--base-model-dir", default="rmbase12_work/models/multirm12_multi_length_fusion_cdhit80/multi_length_independent")
    parser.add_argument("--train-predictions", default="", help="Optional path to fixed main-model train_predictions.tsv.")
    parser.add_argument("--out-dir", default="rmbase12_work/metrics/a_family_neural_two_stage/multi_length_independent")
    parser.add_argument("--split", default="split_cdhit80_cluster")
    parser.add_argument("--class-names", default="")
    parser.add_argument("--focus-classes", default=",".join(A_FAMILY_CLASSES))
    parser.add_argument("--sequence-columns", default="sequence_41nt,sequence_101nt,sequence_201nt")
    parser.add_argument("--sequence-lengths", default="41,101,201")
    parser.add_argument("--struct-fields", default=",".join(DEFAULT_STRUCT_FIELDS))
    parser.add_argument("--struct-window", type=int, default=41)
    parser.add_argument("--max-ratio-to-minority", type=int, default=40)
    parser.add_argument("--cap-minority-floor", type=int, default=500)
    parser.add_argument("--max-train-per-class", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--branch-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=320)
    parser.add_argument("--kernels", default="3,5,7,11")
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--share-sequence-encoder", action="store_true")
    parser.add_argument("--use-base-prior", action="store_true", help="Use fixed main-model A-family probabilities as prior features.")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--sampler-epoch-size-multiplier", type=float, default=1.0)
    parser.add_argument("--class-weight-power", type=float, default=0.25)
    parser.add_argument("--max-class-weight", type=float, default=8.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--alphas", default="0,0.02,0.05,0.1,0.15,0.2,0.35,0.5,0.75,1")
    parser.add_argument("--temperatures", default="0.5,0.75,1,1.5,2")
    parser.add_argument("--m1a-biases", default="-3,-2,-1,-0.5,0,0.5,1,2,3")
    parser.add_argument("--target-class", default="m1A")
    parser.add_argument("--guard-classes", default="Am,m6A,AtoI")
    parser.add_argument("--guard-drop-penalty", type=float, default=4.0)
    parser.add_argument("--macro-drop-penalty", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-checkpoint", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_classes = parse_csv(args.focus_classes)
    sequence_columns = parse_csv(args.sequence_columns)
    sequence_lengths = [int(item) for item in parse_csv(args.sequence_lengths)]
    struct_fields = parse_csv(args.struct_fields)
    kernels = [int(item) for item in parse_csv(args.kernels)]
    if len(sequence_columns) != len(sequence_lengths):
        raise ValueError("--sequence-columns and --sequence-lengths must have the same length.")

    dataset_rows = read_tsv(Path(args.dataset))
    train_rows, _, _ = split_rows(dataset_rows, args.split)
    rows_by_id = {row["site_id"]: row for row in dataset_rows if row.get("site_id")}
    valid_pred_rows = read_tsv(Path(args.base_model_dir) / "valid_predictions.tsv")
    test_pred_rows = read_tsv(Path(args.base_model_dir) / "test_predictions.tsv")
    train_pred_rows: List[Mapping[str, str]] = []
    class_names = infer_class_names(test_pred_rows, args.class_names)
    missing = [name for name in focus_classes if name not in class_names]
    if missing:
        raise ValueError(f"Focus class(es) missing from class names: {', '.join(missing)}")
    if args.target_class not in focus_classes:
        raise ValueError("--target-class must be one of --focus-classes")
    struct_by_id = load_structure(args.rnafold)
    train_prior_by_id: Dict[str, np.ndarray] = {}
    prior_dim = 0
    if args.use_base_prior:
        train_predictions_path = Path(args.train_predictions) if args.train_predictions else Path(args.base_model_dir) / "train_predictions.tsv"
        if not train_predictions_path.exists():
            raise FileNotFoundError(
                f"Missing train predictions for --use-base-prior: {train_predictions_path}. "
                "Run export_motif_struct_fusion_v4_predictions.py first."
            )
        train_pred_rows = read_tsv(train_predictions_path)
        train_prior_by_id = prior_by_site_id(train_pred_rows, class_names, focus_classes)
        prior_dim = len(next(iter(train_prior_by_id.values()))) if train_prior_by_id else 0
        train_rows = [row for row in train_rows if str(row.get("site_id", "")) in train_prior_by_id]

    train_focus = cap_family_train_rows(
        train_rows,
        focus_classes,
        args.max_ratio_to_minority,
        args.max_train_per_class,
        args.cap_minority_floor,
        args.seed,
    )
    focus_to_idx = {name: idx for idx, name in enumerate(focus_classes)}
    y_train = np.asarray([focus_to_idx[str(row.get("mod_type", ""))] for row in train_focus], dtype=np.int64)
    valid_focus_rows, y_valid_focus, valid_focus_pred_rows = rows_from_predictions(valid_pred_rows, rows_by_id, focus_classes, only_focus=True)
    valid_all_rows, _, valid_all_pred_rows = rows_from_predictions(valid_pred_rows, rows_by_id, focus_classes, only_focus=False)
    test_all_rows, _, test_all_pred_rows = rows_from_predictions(test_pred_rows, rows_by_id, focus_classes, only_focus=False)
    if len(valid_all_rows) != len(valid_pred_rows) or len(test_all_rows) != len(test_pred_rows):
        raise ValueError("Prediction rows and dataset rows do not have matching site_id coverage.")
    if args.use_base_prior:
        train_prior_raw = prior_matrix_from_rows(train_focus, train_prior_by_id, prior_dim)
        valid_focus_prior_raw = prior_matrix_from_predictions(valid_focus_pred_rows, class_names, focus_classes)
        valid_all_prior_raw = prior_matrix_from_predictions(valid_all_pred_rows, class_names, focus_classes)
        test_all_prior_raw = prior_matrix_from_predictions(test_all_pred_rows, class_names, focus_classes)
        prior_dim = train_prior_raw.shape[1]
        train_prior, prior_mean, prior_std = normalize_prior_features(train_prior_raw)
        valid_focus_prior, _, _ = normalize_prior_features(valid_focus_prior_raw, prior_mean, prior_std)
        valid_all_prior, _, _ = normalize_prior_features(valid_all_prior_raw, prior_mean, prior_std)
        test_all_prior, _, _ = normalize_prior_features(test_all_prior_raw, prior_mean, prior_std)
    else:
        prior_mean = np.zeros((0,), dtype=np.float32)
        prior_std = np.zeros((0,), dtype=np.float32)
        train_prior = np.zeros((len(train_focus), 0), dtype=np.float32)
        valid_focus_prior = np.zeros((len(valid_focus_rows), 0), dtype=np.float32)
        valid_all_prior = np.zeros((len(valid_all_rows), 0), dtype=np.float32)
        test_all_prior = np.zeros((len(test_all_rows), 0), dtype=np.float32)

    struct_mean, struct_std = compute_struct_norm(train_focus, struct_by_id, struct_fields, args.struct_window)
    train_base_dataset = ASiteDataset(
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
    valid_focus_base_dataset = ASiteDataset(
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
    valid_all_base_dataset = ASiteDataset(
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
    test_all_base_dataset = ASiteDataset(
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
    train_dataset = AFamilyPriorDataset(train_base_dataset, train_prior)
    valid_focus_dataset = AFamilyPriorDataset(valid_focus_base_dataset, valid_focus_prior)
    valid_all_dataset = AFamilyPriorDataset(valid_all_base_dataset, valid_all_prior)
    test_all_dataset = AFamilyPriorDataset(test_all_base_dataset, test_all_prior)

    sampler = make_balanced_sampler(y_train, args.sampler_epoch_size_multiplier) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_prior_batch,
    )
    eval_kwargs = {
        "batch_size": args.batch_size * 2,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_prior_batch,
    }
    valid_focus_loader = DataLoader(valid_focus_dataset, **eval_kwargs)
    valid_all_loader = DataLoader(valid_all_dataset, **eval_kwargs)
    test_all_loader = DataLoader(test_all_dataset, **eval_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AFamilyNeuralSpecialist(
        num_lengths=len(sequence_columns),
        struct_dim=len(struct_mean),
        num_classes=len(focus_classes),
        embed_dim=args.embed_dim,
        branch_dim=args.branch_dim,
        hidden_dim=args.hidden_dim,
        kernels=kernels,
        dropout=args.dropout,
        share_sequence_encoder=args.share_sequence_encoder,
        prior_dim=prior_dim,
    ).to(device)
    weights = class_weight_tensor(y_train, len(focus_classes), args.class_weight_power, args.max_class_weight, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_counts = dict(Counter(row.get("mod_type", "") for row in train_focus))
    valid_counts = dict(Counter(row.get("mod_type", "") for row in valid_focus_rows))
    print(
        "[load] "
        f"dataset_rows={len(dataset_rows)} train_focus_counts={train_counts} valid_focus_counts={valid_counts} "
        f"valid_all={len(valid_all_rows)} test_all={len(test_all_rows)} struct_dim={len(struct_mean)} "
        f"prior_dim={prior_dim} use_base_prior={args.use_base_prior} "
        f"device={device} class_weights={[float(x) for x in weights.detach().cpu().tolist()]}",
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
        target_local = focus_classes.index(args.target_class)
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for seqs, struct, prior, labels in train_loader:
                seqs, struct, prior, labels = move_batch(seqs, struct, prior, labels, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(seqs, struct, prior)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()

            valid_focus_logits = predict_logits(model, valid_focus_loader, device)
            valid_focus_probs = softmax_np(valid_focus_logits)
            valid_binary = specialist_validation_metrics(y_valid_focus, valid_focus_probs, focus_classes)
            score = valid_binary.get(f"{args.target_class}_AUPR", float("nan"))
            if not math.isfinite(score):
                score = -1.0
            improved = score > best_score + args.early_stopping_min_delta
            if improved:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            row: Dict[str, object] = {
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "valid_target_AUPR": valid_binary.get(f"{args.target_class}_AUPR", float("nan")),
                "valid_macro_AUPR": valid_binary.get("macro_AUPR", float("nan")),
                "improved": improved,
                "no_improve": no_improve,
                "lr": optimizer.param_groups[0]["lr"],
            }
            for name in focus_classes:
                row[f"valid_{name}_AUPR"] = valid_binary.get(f"{name}_AUPR", float("nan"))
                row[f"valid_{name}_AUC"] = valid_binary.get(f"{name}_AUC", float("nan"))
            history.append(row)
            print(
                f"epoch={epoch} loss={row['loss']:.4f} valid_{args.target_class}_AUPR={row['valid_target_AUPR']:.6f} "
                f"valid_macro_AUPR={row['valid_macro_AUPR']:.6f} no_improve={no_improve}",
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
                    "focus_classes": focus_classes,
                    "prior_dim": prior_dim,
                    "prior_mean": prior_mean,
                    "prior_std": prior_std,
                },
                checkpoint_path,
            )
            if no_improve >= args.early_stopping_patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_score={best_score:.6f}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("[predict] valid/test A-family specialist logits", flush=True)
    valid_logits = predict_logits(model, valid_all_loader, device)
    test_logits = predict_logits(model, test_all_loader, device)
    valid_base_probs, valid_labels, _, _ = prediction_matrix(valid_pred_rows, class_names)
    test_base_probs, test_labels, _, _ = prediction_matrix(test_pred_rows, class_names)
    valid_base_metrics = evaluate(valid_base_probs, valid_labels, class_names)
    test_base_metrics = evaluate(test_base_probs, test_labels, class_names)

    print("[fusion] validation grid search", flush=True)
    best_fusion, candidates = search_fusion(
        valid_base_probs,
        valid_labels,
        valid_logits,
        class_names,
        focus_classes,
        [float(item) for item in parse_csv(args.alphas)],
        [float(item) for item in parse_csv(args.temperatures)],
        [float(item) for item in parse_csv(args.m1a_biases)],
        args.target_class,
        parse_csv(args.guard_classes),
        args.guard_drop_penalty,
        args.macro_drop_penalty,
    )
    print(f"[fusion] best={best_fusion}", flush=True)
    test_adjusted = apply_fusion(
        test_base_probs,
        test_logits,
        class_names,
        focus_classes,
        float(best_fusion["alpha"]),
        float(best_fusion["temperature"]),
        float(best_fusion["m1a_bias"]),
    )
    test_metrics = evaluate(test_adjusted, test_labels, class_names)

    write_tsv(out_dir / "training_history.tsv", history)
    write_tsv(out_dir / "validation_fusion_grid.tsv", candidates)
    write_tsv(out_dir / "test_per_class_metrics.tsv", per_class_rows(test_metrics, class_names))
    write_tsv(out_dir / "test_base_vs_a_family_neural_per_class.tsv", summarize_comparison(test_base_metrics, test_metrics, class_names, focus_classes))
    write_predictions(out_dir / "test_predictions.tsv", test_pred_rows, test_adjusted, class_names)

    summary = {
        "focus_classes": focus_classes,
        "class_names": list(class_names),
        "base_model_dir": args.base_model_dir,
        "dataset": args.dataset,
        "rnafold": args.rnafold,
        "train_focus_counts": train_counts,
        "valid_focus_counts": valid_counts,
        "best_epoch": best_epoch,
        "best_focus_validation_target_AUPR": best_score,
        "best_validation_fusion": best_fusion,
        "validation_base": valid_base_metrics,
        "test_base": test_base_metrics,
        "test_a_family_neural_two_stage": test_metrics,
        "args": vars(args),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    base_pc = test_base_metrics["per_class"]  # type: ignore[index]
    new_pc = test_metrics["per_class"]  # type: ignore[index]
    parts = []
    for name in focus_classes:
        base_value = safe_float(base_pc.get(name, {}).get("AUPR_ovr"), float("nan"))  # type: ignore[union-attr]
        new_value = safe_float(new_pc.get(name, {}).get("AUPR_ovr"), float("nan"))  # type: ignore[union-attr]
        delta = new_value - base_value if math.isfinite(base_value) and math.isfinite(new_value) else float("nan")
        parts.append(f"{name}_AUPR={new_value} delta={delta}")
    print(
        f"[done] best_epoch={best_epoch} alpha={best_fusion['alpha']} temp={best_fusion['temperature']} "
        f"m1a_bias={best_fusion['m1a_bias']} " + " ".join(parts),
        flush=True,
    )
    print(out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
