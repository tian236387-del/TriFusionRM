#!/usr/bin/env python
"""Train relational window-level multilabel StructDualRM on RMBase11."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from train_structdualrm_multiclass import (
    BASE_TO_ID,
    STRUCT_FIELDS,
    CenterAttentionPool,
    GraphLayer,
    encode_sequence,
    metric_or_nan,
    parse_float,
    parse_int,
    parse_paired_flags,
    parse_partner_indices,
    read_tsv,
)


DEFAULT_CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "Y", "AtoI"]
CLASS_NAMES = list(DEFAULT_CLASS_NAMES)
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
LABEL_SPLIT = re.compile(r"[;,|]+")


def configure_classes(value: str) -> None:
    global CLASS_NAMES, CLASS_TO_ID
    names = [name.strip() for name in str(value or "").split(",") if name.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("--class-names must contain a non-empty, unique comma-separated class order")
    CLASS_NAMES = names
    CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_labels(value: str) -> List[int]:
    labels = []
    for label in LABEL_SPLIT.split(str(value or "")):
        label = label.strip()
        if label in CLASS_TO_ID:
            labels.append(CLASS_TO_ID[label])
    return sorted(set(labels))


def labels_to_tensor(value: str) -> torch.Tensor:
    target = torch.zeros(len(CLASS_NAMES), dtype=torch.float32)
    indices = parse_labels(value)
    if indices:
        target[indices] = 1.0
    return target


@dataclass
class RelationalConfig:
    input_length: int = 41
    num_classes: int = len(CLASS_NAMES)
    d_model: int = 96
    num_layers: int = 2
    num_heads: int = 4
    label_heads: int = 4
    dropout: float = 0.2
    struct_dim: int = 48


class RelationalStructDualRM(nn.Module):
    def __init__(self, cfg: RelationalConfig, relation_message: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.base_emb = nn.Embedding(5, cfg.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.input_length, cfg.d_model)
        self.dist_emb = nn.Embedding(cfg.input_length, cfg.d_model)
        seq_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(seq_layer, num_layers=cfg.num_layers)
        self.graph_layers = nn.ModuleList([GraphLayer(cfg.d_model, cfg.dropout) for _ in range(2)])
        self.seq_attn = CenterAttentionPool(cfg.d_model)
        self.graph_attn = CenterAttentionPool(cfg.d_model)
        self.struct_mlp = nn.Sequential(
            nn.Linear(len(STRUCT_FIELDS), cfg.struct_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.struct_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.struct_dim, cfg.struct_dim),
        )

        global_dim = cfg.d_model * 7 + cfg.struct_dim
        self.global_project = nn.Sequential(
            nn.Linear(global_dim, cfg.d_model * 2),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model * 2),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
        )
        self.token_fuse = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.label_queries = nn.Parameter(torch.empty(cfg.num_classes, cfg.d_model))
        nn.init.xavier_uniform_(self.label_queries)
        self.relation_project = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout))
        self.label_self_attention = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.label_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.label_norm1 = nn.LayerNorm(cfg.d_model)
        self.label_ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
        )
        self.label_norm2 = nn.LayerNorm(cfg.d_model)
        self.window_output = nn.Parameter(torch.empty(cfg.num_classes, cfg.d_model))
        self.center_output = nn.Parameter(torch.empty(cfg.num_classes, cfg.d_model))
        nn.init.xavier_uniform_(self.window_output)
        nn.init.xavier_uniform_(self.center_output)
        self.window_bias = nn.Parameter(torch.zeros(cfg.num_classes))
        self.center_bias = nn.Parameter(torch.zeros(cfg.num_classes))
        self.register_buffer("relation_message", relation_message)

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, length)
        dist = torch.abs(pos - (length // 2))
        return self.transformer(self.base_emb(seq) + self.pos_emb(pos) + self.dist_emb(dist))

    def pool_seq(self, h: torch.Tensor) -> torch.Tensor:
        center_idx = h.shape[1] // 2
        center = h[:, center_idx, :]
        local = h[:, max(0, center_idx - 5) : min(h.shape[1], center_idx + 6), :].mean(dim=1)
        return torch.cat([center, local, self.seq_attn(h)], dim=-1)

    def pool_graph(self, g: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
        center_idx = g.shape[1] // 2
        center = g[:, center_idx, :]
        local = g[:, max(0, center_idx - 10) : min(g.shape[1], center_idx + 11), :].mean(dim=1)
        center_partner = partner[:, center_idx]
        mask = (center_partner >= 0).float().unsqueeze(-1)
        gather_idx = center_partner.clamp_min(0).view(-1, 1, 1).expand(-1, 1, g.shape[-1])
        pair = torch.gather(g, 1, gather_idx).squeeze(1) * mask
        return torch.cat([center, local, self.graph_attn(g), pair], dim=-1)

    def forward(
        self,
        seq: torch.Tensor,
        paired_flags: torch.Tensor,
        pair_partner: torch.Tensor,
        struct: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h = self.encode_sequence(seq)
        g = h + paired_flags.unsqueeze(-1)
        for layer in self.graph_layers:
            g = layer(g, pair_partner)

        z_global = self.global_project(torch.cat([self.pool_seq(h), self.pool_graph(g, pair_partner), self.struct_mlp(struct)], dim=-1))
        token = self.token_fuse(torch.cat([h, g], dim=-1))
        cross_scores = torch.einsum("bld,cd->bcl", token, self.label_queries) / math.sqrt(self.cfg.d_model)
        cross_attention = torch.softmax(cross_scores, dim=-1)
        label_states = torch.einsum("bcl,bld->bcd", cross_attention, token)
        relation_message = torch.einsum("ck,bkd->bcd", self.relation_message, label_states)
        label_states = self.label_norm1(label_states + z_global.unsqueeze(1) + self.relation_project(relation_message))
        attended, label_attention = self.label_self_attention(
            label_states, label_states, label_states, need_weights=True, average_attn_weights=False
        )
        label_states = self.label_norm2(label_states + attended + self.label_ffn(label_states + attended))
        scale = math.sqrt(self.cfg.d_model)
        window_logits = torch.einsum("bcd,cd->bc", label_states, self.window_output) / scale + self.window_bias
        center_logits = torch.einsum("bcd,cd->bc", label_states, self.center_output) / scale + self.center_bias
        return {
            "window_logits": window_logits,
            "center_logits": center_logits,
            "label_states": label_states,
            "cross_attention": cross_attention,
            "label_attention": label_attention,
        }


class MultilabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        fold_by_id: Dict[str, Dict[str, str]],
        input_length: int,
        sequence_column: str,
        stats: Optional[Dict[str, List[float]]] = None,
    ):
        self.rows = list(rows)
        self.fold_by_id = fold_by_id
        self.input_length = input_length
        self.sequence_column = sequence_column
        values = np.asarray(
            [[parse_float(fold_by_id[row["_fold_site_id"]].get(field, 0.0)) for field in STRUCT_FIELDS] for row in self.rows],
            dtype=np.float32,
        )
        if stats is None:
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            std[std < 1e-6] = 1.0
            self.stats = {"mean": mean.tolist(), "std": std.tolist()}
        else:
            self.stats = stats
        self.mean = torch.tensor(self.stats["mean"], dtype=torch.float32)
        self.std = torch.tensor(self.stats["std"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        fold = self.fold_by_id[row["_fold_site_id"]]
        struct = torch.tensor([parse_float(fold.get(field, 0.0)) for field in STRUCT_FIELDS], dtype=torch.float32)
        return {
            "seq": encode_sequence(row[self.sequence_column], self.input_length),
            "paired_flags": parse_paired_flags(fold.get("paired_flags", ""), self.input_length),
            "pair_partner": parse_partner_indices(fold.get("base_pair_list", ""), self.input_length),
            "struct": (struct - self.mean) / self.std,
            "window_target": labels_to_tensor(row["window_labels"]),
            "center_target": labels_to_tensor(row["center_labels"]),
            "center_label": torch.tensor(parse_int(row["class_index"]), dtype=torch.long),
            "site_id": row["site_id"],
            "center_labels_text": row["center_labels"],
            "window_labels_text": row["window_labels"],
        }


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    tensor_keys = ["seq", "paired_flags", "pair_partner", "struct", "window_target", "center_target", "center_label"]
    output = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}  # type: ignore[arg-type]
    for key in ["site_id", "center_labels_text", "window_labels_text"]:
        output[key] = [str(item[key]) for item in batch]
    return output


def load_rows(
    dataset_path: Path,
    rnafold_path: Path,
    split_name: str,
    target_field: str,
    fold_site_id_column: str,
    max_rows: int,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    fold_by_id = {row["site_id"]: row for row in read_tsv(rnafold_path)}
    rows = []
    unknown_labels = set()
    for source in read_tsv(dataset_path):
        fold_site_id = source.get(fold_site_id_column, "") or source.get("site_id", "")
        if fold_site_id not in fold_by_id:
            continue
        labels = [label.strip() for label in LABEL_SPLIT.split(source.get(target_field, "")) if label.strip()]
        unknown_labels.update(label for label in labels if label not in CLASS_TO_ID)
        labels = [label for label in labels if label in CLASS_TO_ID]
        if not labels:
            continue
        row = dict(source)
        normalized = ",".join(sorted(set(labels), key=CLASS_TO_ID.__getitem__))
        row["window_labels"] = normalized
        row["center_labels"] = normalized
        row["class_index"] = str(CLASS_TO_ID[labels[0]])
        row["_fold_site_id"] = fold_site_id
        rows.append(row)
    if unknown_labels:
        raise ValueError(f"Dataset contains labels absent from --class-names: {sorted(unknown_labels)}")
    random.Random(seed).shuffle(rows)
    if max_rows:
        rows = rows[:max_rows]
    partitions = tuple([row for row in rows if row.get(split_name) == split] for split in ("train", "valid", "test"))
    if any(not partition for partition in partitions):
        raise ValueError(f"Split {split_name!r} produced empty train/valid/test partitions: {[len(x) for x in partitions]}")
    return partitions[0], partitions[1], partitions[2], fold_by_id


def build_relation_prior(rows: Sequence[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    pair_counts = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.float64)
    for row in rows:
        labels = parse_labels(row["window_labels"])
        counts[labels] += 1
        for left in labels:
            for right in labels:
                pair_counts[left, right] += 1
    denominator = np.sqrt(np.maximum(counts[:, None] * counts[None, :], 1.0))
    relation_target = pair_counts / denominator
    np.fill_diagonal(relation_target, 1.0)
    relation_target = np.clip(relation_target, 0.0, 1.0)
    message = relation_target.copy()
    message[message < 0.005] = 0.0
    message += np.eye(len(CLASS_NAMES), dtype=np.float64)
    message /= np.maximum(message.sum(axis=1, keepdims=True), 1e-8)
    summary = {
        "train_window_positive_counts": {name: int(counts[idx]) for idx, name in enumerate(CLASS_NAMES)},
        "train_pair_counts": pair_counts.astype(int).tolist(),
        "relation_target": relation_target.tolist(),
        "relation_message": message.tolist(),
    }
    return relation_target.astype(np.float32), message.astype(np.float32), summary


def positive_weights(rows: Sequence[Dict[str, str]], field: str, cap: float, device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for row in rows:
        counts[parse_labels(row[field])] += 1
    weights = (max(len(rows), 1) - counts) / np.maximum(counts, 1.0)
    return torch.tensor(np.clip(weights, 1.0, cap), dtype=torch.float32, device=device)


def center_class_weights(rows: Sequence[Dict[str, str]], power: float, device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for row in rows:
        counts[parse_int(row["class_index"])] += 1
    counts[counts < 1] = 1.0
    weights = (counts.sum() / counts) ** power
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def asymmetric_focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    negative_weight: float,
    gamma_pos: float,
    gamma_neg: float,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    focal = targets * (1.0 - probs).pow(gamma_pos) + (1.0 - targets) * probs.pow(gamma_neg)
    observation_weight = targets + (1.0 - targets) * negative_weight
    return (loss * focal * observation_weight).mean()


def relation_embedding_loss(label_states: torch.Tensor, relation_target: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(label_states.mean(dim=0), dim=-1)
    similarity = (normalized @ normalized.transpose(0, 1) + 1.0) / 2.0
    mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
    return F.mse_loss(similarity[mask], relation_target[mask])


def cardinality_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(torch.sigmoid(logits).sum(dim=1), targets.sum(dim=1))


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    window_pos_weight: torch.Tensor,
    center_pos_weight: torch.Tensor,
    center_ce: nn.Module,
    relation_target: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    sums = {"loss": 0.0, "window": 0.0, "center": 0.0, "relation": 0.0, "cardinality": 0.0}
    rows = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["seq"], batch["paired_flags"], batch["pair_partner"], batch["struct"])
        window_loss = asymmetric_focal_bce(
            output["window_logits"],
            batch["window_target"],
            window_pos_weight,
            args.negative_weight,
            args.gamma_pos,
            args.gamma_neg,
        )
        if args.center_loss == "ce":
            center_loss = center_ce(output["center_logits"], batch["center_label"])
        else:
            center_loss = asymmetric_focal_bce(
                output["center_logits"],
                batch["center_target"],
                center_pos_weight,
                args.negative_weight,
                args.gamma_pos,
                args.gamma_neg,
            )
        relation_loss = relation_embedding_loss(output["label_states"], relation_target)
        count_loss = cardinality_loss(output["window_logits"], batch["window_target"])
        loss = (
            args.lambda_window * window_loss
            + args.lambda_center * center_loss
            + args.lambda_relation * relation_loss
            + args.lambda_cardinality * count_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = int(batch["window_target"].shape[0])
        rows += batch_size
        for key, value in [
            ("loss", loss),
            ("window", window_loss),
            ("center", center_loss),
            ("relation", relation_loss),
            ("cardinality", count_loss),
        ]:
            sums[key] += float(value.detach().cpu()) * batch_size
    return {key: value / max(rows, 1) for key, value in sums.items()}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, center_activation: str = "softmax") -> Dict[str, object]:
    model.eval()
    output: Dict[str, List[object]] = {
        "window_probs": [],
        "center_probs": [],
        "window_targets": [],
        "center_targets": [],
        "center_labels": [],
        "site_ids": [],
        "center_labels_text": [],
        "window_labels_text": [],
        "label_attention": [],
    }
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch["seq"], batch["paired_flags"], batch["pair_partner"], batch["struct"])
        output["window_probs"].append(torch.sigmoid(pred["window_logits"]).cpu().numpy())
        if center_activation == "sigmoid":
            center_probs = torch.sigmoid(pred["center_logits"])
        else:
            center_probs = torch.softmax(pred["center_logits"], dim=-1)
        output["center_probs"].append(center_probs.cpu().numpy())
        output["window_targets"].append(batch["window_target"].cpu().numpy())
        output["center_targets"].append(batch["center_target"].cpu().numpy())
        output["center_labels"].append(batch["center_label"].cpu().numpy())
        output["label_attention"].append(pred["label_attention"].mean(dim=(0, 1)).cpu().numpy())
        for key in ["site_ids", "center_labels_text", "window_labels_text"]:
            source_key = "site_id" if key == "site_ids" else key
            output[key].extend(batch[source_key])  # type: ignore[arg-type]
    for key in ["window_probs", "center_probs", "window_targets", "center_targets"]:
        output[key] = np.vstack(output[key])  # type: ignore[arg-type]
    output["center_labels"] = np.concatenate(output["center_labels"])  # type: ignore[arg-type]
    output["label_attention"] = np.mean(np.stack(output["label_attention"]), axis=0)  # type: ignore[arg-type]
    return output


def optimize_thresholds(targets: np.ndarray, probs: np.ndarray) -> np.ndarray:
    thresholds = np.full(probs.shape[1], 0.5, dtype=np.float32)
    candidates = np.linspace(0.05, 0.95, 19)
    for idx in range(probs.shape[1]):
        y = targets[:, idx].astype(int)
        best = (-1.0, 0.5)
        for threshold in candidates:
            score = f1_score(y, (probs[:, idx] >= threshold).astype(int), zero_division=0)
            if score > best[0]:
                best = (float(score), float(threshold))
        thresholds[idx] = best[1]
    return thresholds


def evaluate_multilabel(targets: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> Dict[str, object]:
    pred = (probs >= thresholds.reshape(1, -1)).astype(int)
    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        y = targets[:, idx].astype(int)
        p = probs[:, idx]
        q = pred[:, idx]
        per_class[name] = {
            "support": int(y.sum()),
            "AUC": metric_or_nan(roc_auc_score, y, p) if len(np.unique(y)) > 1 else float("nan"),
            "AUPR": metric_or_nan(average_precision_score, y, p) if len(np.unique(y)) > 1 else float("nan"),
            "F1": float(f1_score(y, q, zero_division=0)),
            "Precision": float(precision_score(y, q, zero_division=0)),
            "Recall": float(recall_score(y, q, zero_division=0)),
            "MCC": metric_or_nan(matthews_corrcoef, y, q),
            "threshold": float(thresholds[idx]),
        }
    return {
        "macro_AUPR": float(np.nanmean([per_class[name]["AUPR"] for name in CLASS_NAMES])),
        "macro_AUC": float(np.nanmean([per_class[name]["AUC"] for name in CLASS_NAMES])),
        "macro_F1": float(f1_score(targets, pred, average="macro", zero_division=0)),
        "micro_F1": float(f1_score(targets, pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(targets, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, pred, average="macro", zero_division=0)),
        "macro_jaccard": float(jaccard_score(targets, pred, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(targets, pred)),
        "hamming_loss": float(hamming_loss(targets, pred)),
        "mean_true_cardinality": float(targets.sum(axis=1).mean()),
        "mean_predicted_cardinality": float(pred.sum(axis=1).mean()),
        "per_class": per_class,
    }


def evaluate_center_multiclass(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(axis=1)
    per_class_aupr = []
    for idx in range(len(CLASS_NAMES)):
        y = (labels == idx).astype(int)
        per_class_aupr.append(average_precision_score(y, probs[:, idx]) if len(np.unique(y)) > 1 else float("nan"))
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_F1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, pred)),
        "macro_AUPR_ovr": float(np.nanmean(per_class_aupr)),
    }


def selection_score(
    selection_metric: str,
    window_metrics: Dict[str, object],
    center_multiclass_metrics: Dict[str, float],
    center_multilabel_metrics: Dict[str, object],
) -> float:
    if selection_metric in {"macro_AUPR", "window_macro_AUPR"}:
        return float(window_metrics["macro_AUPR"])
    if selection_metric in {"macro_F1", "window_macro_F1"}:
        return float(window_metrics["macro_F1"])
    if selection_metric in {"micro_F1", "window_micro_F1"}:
        return float(window_metrics["micro_F1"])
    if selection_metric == "center_macro_AUPR":
        return float(center_multiclass_metrics["macro_AUPR_ovr"])
    if selection_metric == "center_macro_F1":
        return float(center_multiclass_metrics["macro_F1"])
    if selection_metric == "center_balanced_accuracy":
        return float(center_multiclass_metrics["balanced_accuracy"])
    if selection_metric == "center_multilabel_macro_AUPR":
        return float(center_multilabel_metrics["macro_AUPR"])
    if selection_metric == "center_multilabel_macro_F1":
        return float(center_multilabel_metrics["macro_F1"])
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def write_matrix(path: Path, matrix: np.ndarray, row_name: str = "label") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([row_name] + CLASS_NAMES)
        for idx, name in enumerate(CLASS_NAMES):
            writer.writerow([name] + [float(value) for value in matrix[idx]])


def write_predictions(path: Path, predictions: Dict[str, object], thresholds: np.ndarray) -> None:
    probs = predictions["window_probs"]
    targets = predictions["window_targets"]
    assert isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray)
    fields = ["site_id", "center_labels", "window_labels", "predicted_window_labels"]
    fields += [f"target_{name}" for name in CLASS_NAMES] + [f"prob_{name}" for name in CLASS_NAMES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, site_id in enumerate(predictions["site_ids"]):  # type: ignore[assignment]
            predicted = [name for class_idx, name in enumerate(CLASS_NAMES) if probs[idx, class_idx] >= thresholds[class_idx]]
            row = {
                "site_id": site_id,
                "center_labels": predictions["center_labels_text"][idx],  # type: ignore[index]
                "window_labels": predictions["window_labels_text"][idx],  # type: ignore[index]
                "predicted_window_labels": ",".join(predicted),
            }
            row.update({f"target_{name}": int(targets[idx, class_idx]) for class_idx, name in enumerate(CLASS_NAMES)})
            row.update({f"prob_{name}": float(probs[idx, class_idx]) for class_idx, name in enumerate(CLASS_NAMES)})
            writer.writerow(row)


@torch.no_grad()
def measure_efficiency(model: nn.Module, loader: DataLoader, device: torch.device, steps: int) -> Dict[str, object]:
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    try:
        batch = next(iter(loader))
    except StopIteration:
        return {"parameters": parameters, "trainable_parameters": trainable}
    batch = move_batch(batch, device)

    def forward() -> None:
        model(batch["seq"], batch["paired_flags"], batch["pair_partner"], batch["struct"])

    model.eval()
    for _ in range(3):
        forward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    repeat = max(1, int(steps))
    for _ in range(repeat):
        forward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    batch_size = int(batch["seq"].shape[0])
    profiled_flops = None
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as profile:
            forward()
        profiled_flops = int(sum(int(event.flops or 0) for event in profile.key_averages()))
    except (RuntimeError, AttributeError, TypeError):
        pass
    return {
        "parameters": parameters,
        "trainable_parameters": trainable,
        "profiled_forward_flops": profiled_flops,
        "flops_per_sample": float(profiled_flops) / batch_size if profiled_flops is not None else None,
        "inference_batch_size": batch_size,
        "inference_repetitions": repeat,
        "inference_seconds": elapsed,
        "samples_per_second": batch_size * repeat / max(elapsed, 1e-12),
        "milliseconds_per_sample": 1000.0 * elapsed / max(batch_size * repeat, 1),
    }


def run_training(args: argparse.Namespace) -> Dict[str, object]:
    configure_classes(args.class_names)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_rows, valid_rows, test_rows, fold_by_id = load_rows(
        Path(args.dataset),
        Path(args.rnafold),
        args.split,
        args.target_field,
        args.fold_site_id_column,
        args.max_rows,
        args.seed,
    )
    relation_target_np, relation_message_np, relation_summary = build_relation_prior(train_rows)
    cfg = RelationalConfig(
        input_length=args.input_length,
        num_classes=len(CLASS_NAMES),
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        label_heads=args.label_heads,
        dropout=args.dropout,
    )
    train_ds = MultilabelDataset(train_rows, fold_by_id, cfg.input_length, args.sequence_column)
    valid_ds = MultilabelDataset(valid_rows, fold_by_id, cfg.input_length, args.sequence_column, stats=train_ds.stats)
    test_ds = MultilabelDataset(test_rows, fold_by_id, cfg.input_length, args.sequence_column, stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    relation_target = torch.tensor(relation_target_np, dtype=torch.float32, device=device)
    relation_message = torch.tensor(relation_message_np, dtype=torch.float32, device=device)
    model = RelationalStructDualRM(cfg, relation_message).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    window_pos_weight = positive_weights(train_rows, "window_labels", args.pos_weight_cap, device)
    center_pos_weight = positive_weights(train_rows, "center_labels", args.pos_weight_cap, device)
    center_ce = nn.CrossEntropyLoss(weight=center_class_weights(train_rows, args.class_weight_power, device))

    history = []
    best_state = None
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    training_seconds = 0.0
    best_thresholds = np.full(len(CLASS_NAMES), 0.5, dtype=np.float32)
    best_center_thresholds = np.full(len(CLASS_NAMES), 0.5, dtype=np.float32)
    start_epoch = 1
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint.get("history", [])
        best_state = checkpoint.get("best_state_dict")
        best_score = float(checkpoint.get("best_score", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        training_seconds = float(checkpoint.get("training_seconds", 0.0))
        best_thresholds = np.asarray(checkpoint.get("best_thresholds", best_thresholds), dtype=np.float32)
        best_center_thresholds = np.asarray(
            checkpoint.get("best_center_thresholds", best_center_thresholds), dtype=np.float32
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"resumed_from={checkpoint_path} start_epoch={start_epoch} best_score={best_score:.4f}", flush=True)

    if start_epoch > args.epochs:
        print(f"checkpoint already reached epoch {start_epoch - 1}; running final evaluation only", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        losses = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            window_pos_weight,
            center_pos_weight,
            center_ce,
            relation_target,
            args,
        )
        scheduler.step()
        center_activation = "sigmoid" if args.center_loss == "bce" else "softmax"
        valid_predictions = predict(model, valid_loader, device, center_activation=center_activation)
        thresholds = optimize_thresholds(valid_predictions["window_targets"], valid_predictions["window_probs"])  # type: ignore[arg-type]
        center_thresholds = optimize_thresholds(
            valid_predictions["center_targets"], valid_predictions["center_probs"]  # type: ignore[arg-type]
        )
        valid_metrics = evaluate_multilabel(
            valid_predictions["window_targets"], valid_predictions["window_probs"], thresholds  # type: ignore[arg-type]
        )
        center_metrics = evaluate_center_multiclass(
            valid_predictions["center_labels"], valid_predictions["center_probs"]  # type: ignore[arg-type]
        )
        center_multilabel_metrics = evaluate_multilabel(
            valid_predictions["center_targets"], valid_predictions["center_probs"], center_thresholds  # type: ignore[arg-type]
        )
        score = selection_score(args.selection_metric, valid_metrics, center_metrics, center_multilabel_metrics)
        training_seconds += time.perf_counter() - epoch_started
        improved = score > best_score + args.early_stopping_min_delta
        history.append(
            {
                "epoch": epoch,
                "losses": losses,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation": valid_metrics,
                "validation_center_multiclass": center_metrics,
                "validation_center_multilabel": center_multilabel_metrics,
                "selection_score": score,
                "improved": improved,
            }
        )
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_thresholds = thresholds.copy()
            best_center_thresholds = center_thresholds.copy()
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_state_dict": best_state,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "training_seconds": training_seconds,
                    "best_thresholds": best_thresholds,
                    "best_center_thresholds": best_center_thresholds,
                    "history": history,
                    "config": asdict(cfg),
                    "class_names": CLASS_NAMES,
                },
                checkpoint_path,
            )
        print(
            f"epoch={epoch} loss={losses['loss']:.4f} valid_{args.selection_metric}={score:.4f} "
            f"center_macro_AUPR={center_metrics['macro_AUPR_ovr']:.4f}",
            flush=True,
        )
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"early_stop epoch={epoch} best_epoch={best_epoch} best_score={best_score:.4f}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    center_activation = "sigmoid" if args.center_loss == "bce" else "softmax"
    valid_predictions = predict(model, valid_loader, device, center_activation=center_activation)
    test_predictions = predict(model, test_loader, device, center_activation=center_activation)
    validation = evaluate_multilabel(
        valid_predictions["window_targets"], valid_predictions["window_probs"], best_thresholds  # type: ignore[arg-type]
    )
    test = evaluate_multilabel(
        test_predictions["window_targets"], test_predictions["window_probs"], best_thresholds  # type: ignore[arg-type]
    )
    validation_center = evaluate_center_multiclass(
        valid_predictions["center_labels"], valid_predictions["center_probs"]  # type: ignore[arg-type]
    )
    test_center = evaluate_center_multiclass(
        test_predictions["center_labels"], test_predictions["center_probs"]  # type: ignore[arg-type]
    )
    validation_center_multilabel = evaluate_multilabel(
        valid_predictions["center_targets"], valid_predictions["center_probs"], best_center_thresholds  # type: ignore[arg-type]
    )
    test_center_multilabel = evaluate_multilabel(
        test_predictions["center_targets"], test_predictions["center_probs"], best_center_thresholds  # type: ignore[arg-type]
    )
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    efficiency = measure_efficiency(model, test_loader, device, args.efficiency_steps)
    efficiency["training_seconds"] = training_seconds
    efficiency["training_peak_memory_bytes"] = peak_memory
    metrics: Dict[str, object] = {
        "dataset": args.dataset,
        "rnafold": args.rnafold,
        "task": f"RMBase_hg38_{len(CLASS_NAMES)}_label_coordinate_multilabel",
        "classes": CLASS_NAMES,
        "device": str(device),
        "split": args.split,
        "row_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "selection_metric": args.selection_metric,
        "best_score": float(best_score),
        "best_epoch": best_epoch,
        "best_epoch_by_validation_window_macro_AUPR": int(max(history, key=lambda item: item["validation"]["macro_AUPR"])["epoch"])
        if history
        else None,
        "best_epoch_by_validation_center_multilabel_macro_AUPR": int(
            max(history, key=lambda item: item["validation_center_multilabel"]["macro_AUPR"])["epoch"]
        )
        if history
        else None,
        "thresholds_selected_on_validation": {name: float(best_thresholds[idx]) for idx, name in enumerate(CLASS_NAMES)},
        "center_thresholds_selected_on_validation": {
            name: float(best_center_thresholds[idx]) for idx, name in enumerate(CLASS_NAMES)
        },
        "loss": {
            "formula": "L_window_asymmetric_focal_BCE + lambda_center*L_center + lambda_relation*L_relation + lambda_cardinality*L_cardinality",
            "center_loss": args.center_loss,
            "negative_weight": args.negative_weight,
            "gamma_pos": args.gamma_pos,
            "gamma_neg": args.gamma_neg,
            "lambda_window": args.lambda_window,
            "lambda_center": args.lambda_center,
            "lambda_relation": args.lambda_relation,
            "lambda_cardinality": args.lambda_cardinality,
        },
        "relation_prior": relation_summary,
        "validation": validation,
        "test": test,
        "validation_center_multiclass": validation_center,
        "test_center_multiclass": test_center,
        "validation_center_multilabel": validation_center_multilabel,
        "test_center_multilabel": test_center_multilabel,
        "efficiency": efficiency,
        "history": history,
        "caveats": [
            "Labels are coordinate-collapsed RMBase annotations; co-occurring labels are retained rather than forced into mutually exclusive classes.",
            "Unobserved labels are positive-unlabeled rather than experimentally verified negatives; negative BCE terms are down-weighted.",
            "All thresholds and checkpoints are selected on validation data only.",
        ],
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "class_names": CLASS_NAMES,
            "struct_stats": train_ds.stats,
            "relation_target": relation_target_np,
            "relation_message": relation_message_np,
            "thresholds": best_thresholds,
            "center_thresholds": best_center_thresholds,
        },
        out_dir / "model.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    write_matrix(out_dir / "relation_target.tsv", relation_target_np)
    write_matrix(out_dir / "learned_label_attention.tsv", test_predictions["label_attention"])  # type: ignore[arg-type]
    write_predictions(out_dir / "valid_predictions.tsv", valid_predictions, best_thresholds)
    write_predictions(out_dir / "test_predictions.tsv", test_predictions, best_thresholds)
    (out_dir / "README.md").write_text(
        "# Relational multilabel StructDualRM\n\n"
        "Predicts coordinate-collapsed RMBase modifications with sigmoid multilabel heads and retains biological "
        "co-occurrence such as Am/m6Am.\n\n"
        "The model combines sequence Transformer encoding, RNAfold pair-aware graph encoding, label-token cross-attention, "
        "label self-attention, and a training-only co-annotation relation graph.\n\n"
        "The composite objective contains asymmetric focal multilabel BCE, center-label auxiliary loss, relation-embedding "
        "consistency loss, and label-cardinality loss.\n",
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train relational multilabel StructDualRM.")
    parser.add_argument(
        "--dataset",
        default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_window_multilabel_strict.tsv",
    )
    parser.add_argument("--rnafold", default="rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv")
    parser.add_argument("--out-dir", default="rmbase11_full_work/models/M7_relational_multilabel_strict_full")
    parser.add_argument("--split", default="split_multilabel_component")
    parser.add_argument("--class-names", default=",".join(DEFAULT_CLASS_NAMES))
    parser.add_argument("--target-field", default="label_set")
    parser.add_argument("--fold-site-id-column", default="source_site_id")
    parser.add_argument("--sequence-column", default="sequence_41nt")
    parser.add_argument("--input-length", type=int, default=41)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--label-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-cap", type=float, default=100.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--negative-weight", type=float, default=0.2)
    parser.add_argument("--gamma-pos", type=float, default=0.0)
    parser.add_argument("--gamma-neg", type=float, default=2.0)
    parser.add_argument("--center-loss", choices=["ce", "bce"], default="bce")
    parser.add_argument("--lambda-window", type=float, default=1.0)
    parser.add_argument("--lambda-center", type=float, default=0.0)
    parser.add_argument("--lambda-relation", type=float, default=0.1)
    parser.add_argument("--lambda-cardinality", type=float, default=0.05)
    parser.add_argument(
        "--selection-metric",
        choices=[
            "macro_AUPR",
            "macro_F1",
            "micro_F1",
            "window_macro_AUPR",
            "window_macro_F1",
            "window_micro_F1",
            "center_macro_AUPR",
            "center_macro_F1",
            "center_balanced_accuracy",
            "center_multilabel_macro_AUPR",
            "center_multilabel_macro_F1",
        ],
        default="macro_AUPR",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--efficiency-steps", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
