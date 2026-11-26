#!/usr/bin/env python
"""Train multiclass StructDualRM variants on the RMBase hg38 11-class dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


BASE_TO_ID = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4}
CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
REGION_TO_ID = {"5UTR": 0, "CDS": 1, "3UTR": 2, "intron": 3, "exon": 4, "intergenic": 5, "other": 6}
STRUCT_FIELDS = [
    "center_paired",
    "central_11nt_paired_fraction",
    "central_21nt_paired_fraction",
    "paired_fraction_window",
    "MFE_per_nt",
    "loop_span_around_center",
]


def encode_sequence(seq: str, length: int) -> torch.Tensor:
    seq = str(seq or "").upper().replace("U", "T")
    ids = [BASE_TO_ID.get(base, 0) for base in seq[:length]]
    ids.extend([0] * max(0, length - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: object, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_paired_flags(value: str, length: int) -> torch.Tensor:
    flags = [parse_float(x) for x in str(value or "").split(",") if x != ""]
    flags = flags[:length] + [0.0] * max(0, length - len(flags))
    return torch.tensor(flags, dtype=torch.float32)


def parse_partner_indices(base_pair_list: str, length: int) -> torch.Tensor:
    partners = [-1] * length
    for item in str(base_pair_list or "").split(";"):
        if not item or "-" not in item:
            continue
        left, right = [parse_int(x) for x in item.split("-", 1)]
        if 1 <= left <= length and 1 <= right <= length:
            partners[left - 1] = right - 1
            partners[right - 1] = left - 1
    return torch.tensor(partners, dtype=torch.long)


@dataclass
class MultiClassConfig:
    input_length: int = 41
    num_classes: int = len(CLASS_NAMES)
    d_model: int = 64
    num_layers: int = 1
    num_heads: int = 4
    dropout: float = 0.2
    struct_dim: int = 32
    context_dim: int = 24
    rbp_input_dim: int = 0
    rbp_dim: int = 32


class CenterAttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        center = h[:, h.shape[1] // 2 : h.shape[1] // 2 + 1, :].expand_as(h)
        weights = torch.softmax(self.score(torch.cat([h, center], dim=-1)).squeeze(-1), dim=-1)
        return torch.sum(h * weights.unsqueeze(-1), dim=1)


class GraphLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.self_lin = nn.Linear(d_model, d_model)
        self.backbone_lin = nn.Linear(d_model, d_model)
        self.pair_lin = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
        left = torch.zeros_like(h)
        right = torch.zeros_like(h)
        left[:, 1:, :] = h[:, :-1, :]
        right[:, :-1, :] = h[:, 1:, :]
        backbone = (left + right) / 2.0
        mask = (partner >= 0).float().unsqueeze(-1)
        gather_idx = partner.clamp_min(0).unsqueeze(-1).expand(-1, -1, h.shape[-1])
        pair_h = torch.gather(h, 1, gather_idx) * mask
        out = self.self_lin(h) + self.backbone_lin(backbone) + self.pair_lin(pair_h)
        return self.norm(h + self.dropout(torch.relu(out)))


class StructDualRMMultiClass(nn.Module):
    def __init__(self, cfg: MultiClassConfig, variant: str):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.base_emb = nn.Embedding(5, cfg.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.input_length, cfg.d_model)
        self.dist_emb = nn.Embedding(cfg.input_length, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.seq_attn = CenterAttentionPool(cfg.d_model)
        self.graph_layers = nn.ModuleList([GraphLayer(cfg.d_model, cfg.dropout) for _ in range(2)])
        self.graph_attn = CenterAttentionPool(cfg.d_model)
        self.struct_mlp = nn.Sequential(
            nn.Linear(len(STRUCT_FIELDS), cfg.struct_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.struct_dim, cfg.struct_dim),
        )
        self.region_emb = nn.Embedding(len(REGION_TO_ID), cfg.context_dim // 2)
        self.context_mlp = nn.Sequential(nn.Linear(2, cfg.context_dim // 2), nn.ReLU(), nn.Linear(cfg.context_dim // 2, cfg.context_dim // 2))
        self.rbp_mlp = (
            nn.Sequential(
                nn.Linear(cfg.rbp_input_dim, cfg.rbp_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.rbp_dim, cfg.rbp_dim),
            )
            if cfg.rbp_input_dim > 0
            else None
        )

        self.seq_dim = cfg.d_model * 3
        self.graph_dim = cfg.d_model * 4
        self.rbp_out_dim = cfg.rbp_dim if cfg.rbp_input_dim > 0 else 0
        self.full_dim = self.seq_dim + self.graph_dim + cfg.struct_dim + cfg.context_dim + self.rbp_out_dim
        self.label_token_fuse = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.label_queries = nn.Parameter(torch.empty(cfg.num_classes, cfg.d_model))
        nn.init.xavier_uniform_(self.label_queries)
        self.label_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.full_dim + cfg.d_model, cfg.d_model * 2),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.d_model * 2, 1),
                )
                for _ in range(cfg.num_classes)
            ]
        )
        self.class_gate = nn.Sequential(
            nn.Linear(self.full_dim, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.num_classes),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(nn.Linear(self.full_dim, cfg.d_model), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.num_classes))

    def encode_seq(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, length)
        dist = torch.abs(pos - (length // 2))
        x = self.base_emb(seq) + self.pos_emb(pos) + self.dist_emb(dist)
        return self.transformer(x)

    def pool_seq(self, h: torch.Tensor) -> torch.Tensor:
        center_idx = h.shape[1] // 2
        center = h[:, center_idx, :]
        local = h[:, max(0, center_idx - 5) : min(h.shape[1], center_idx + 6), :].mean(dim=1)
        attn = self.seq_attn(h)
        return torch.cat([center, local, attn], dim=-1)

    def pool_graph(self, g: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
        center_idx = g.shape[1] // 2
        center = g[:, center_idx, :]
        local = g[:, max(0, center_idx - 10) : min(g.shape[1], center_idx + 11), :].mean(dim=1)
        attn = self.graph_attn(g)
        center_partner = partner[:, center_idx]
        mask = (center_partner >= 0).float().unsqueeze(-1)
        gather_idx = center_partner.clamp_min(0).view(-1, 1, 1).expand(-1, 1, g.shape[-1])
        pair = torch.gather(g, 1, gather_idx).squeeze(1) * mask
        return torch.cat([center, local, attn, pair], dim=-1)

    def label_attention_pool(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        token = self.label_token_fuse(torch.cat([h, g], dim=-1))
        scores = torch.einsum("bld,cd->bcl", token, self.label_queries)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bcl,bld->bcd", weights, token)

    def forward(
        self,
        seq: torch.Tensor,
        paired_flags: torch.Tensor,
        pair_partner: torch.Tensor,
        struct: torch.Tensor,
        region: torch.Tensor,
        context: torch.Tensor,
        rbp: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.encode_seq(seq)
        z_seq = self.pool_seq(h)
        if self.variant in {"M2", "M3", "M4", "M5"}:
            g = h + paired_flags.unsqueeze(-1)
            for layer in self.graph_layers:
                g = layer(g, pair_partner)
            z_graph = self.pool_graph(g, pair_partner)
        else:
            g = h
            z_graph = torch.zeros(seq.shape[0], self.graph_dim, device=seq.device)

        if self.variant in {"M1", "M3", "M4", "M5"}:
            z_struct = self.struct_mlp(struct)
        else:
            z_struct = torch.zeros(seq.shape[0], self.cfg.struct_dim, device=seq.device)

        if self.variant in {"M4", "M5"}:
            z_context = torch.cat([self.region_emb(region), self.context_mlp(context)], dim=-1)
        else:
            z_context = torch.zeros(seq.shape[0], self.cfg.context_dim, device=seq.device)

        if self.rbp_mlp is not None and rbp is not None:
            z_rbp = self.rbp_mlp(rbp)
        elif self.rbp_out_dim > 0:
            z_rbp = torch.zeros(seq.shape[0], self.rbp_out_dim, device=seq.device)
        else:
            z_rbp = torch.zeros(seq.shape[0], 0, device=seq.device)

        z = torch.cat([z_seq, z_graph, z_struct, z_context, z_rbp], dim=-1)
        if self.variant == "M5":
            gate = self.class_gate(z)
            graph_start = self.seq_dim
            graph_end = graph_start + self.graph_dim
            struct_end = graph_end + self.cfg.struct_dim
            class_tokens = self.label_attention_pool(h, g)
            z_base = torch.cat([z[:, :graph_start], z[:, struct_end:]], dim=-1)
            logits = []
            for class_idx in range(self.cfg.num_classes):
                class_gate = gate[:, class_idx : class_idx + 1]
                z_class = torch.cat(
                    [
                        z_base[:, : self.seq_dim],
                        class_gate * z[:, graph_start:graph_end],
                        class_gate * z[:, graph_end:struct_end],
                        z_base[:, self.seq_dim :],
                    ],
                    dim=-1,
                )
                head_input = torch.cat([class_tokens[:, class_idx, :], z_class], dim=-1)
                logits.append(self.label_heads[class_idx](head_input))
            logits_tensor = torch.cat(logits, dim=-1)
        elif self.variant in {"M3", "M4"}:
            gate = self.class_gate(z)
            graph_start = self.seq_dim
            graph_end = graph_start + self.graph_dim
            struct_end = graph_end + self.cfg.struct_dim
            z_base = torch.cat([z[:, :graph_start], z[:, struct_end:]], dim=-1)
            logits = []
            for class_idx in range(self.cfg.num_classes):
                class_gate = gate[:, class_idx : class_idx + 1]
                z_class = torch.cat(
                    [
                        z_base[:, : self.seq_dim],
                        class_gate * z[:, graph_start:graph_end],
                        class_gate * z[:, graph_end:struct_end],
                        z_base[:, self.seq_dim :],
                    ],
                    dim=-1,
                )
                logits.append(self.head(z_class)[:, class_idx : class_idx + 1])
            logits_tensor = torch.cat(logits, dim=-1)
        else:
            gate = torch.ones(seq.shape[0], self.cfg.num_classes, device=seq.device)
            logits_tensor = self.head(z)
        return {"logits": logits_tensor, "class_gate": gate}


class RMBase11Dataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        fold_by_id: Dict[str, Dict[str, str]],
        input_length: int,
        stats: Optional[Dict[str, List[float]]] = None,
        use_metadata: bool = True,
        rbp_by_id: Optional[Dict[str, np.ndarray]] = None,
        rbp_fields: Optional[Sequence[str]] = None,
        rbp_stats: Optional[Dict[str, List[float]]] = None,
    ):
        self.rows = list(rows)
        self.fold_by_id = fold_by_id
        self.input_length = input_length
        self.use_metadata = use_metadata
        self.rbp_fields = list(rbp_fields or [])
        self.has_rbp = bool(rbp_by_id) and bool(self.rbp_fields)
        struct_values = []
        for row in self.rows:
            fold = fold_by_id[row["site_id"]]
            struct_values.append([parse_float(fold.get(field, 0.0)) for field in STRUCT_FIELDS])
        arr = np.asarray(struct_values, dtype=np.float32)
        if stats is None:
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            std[std < 1e-6] = 1.0
            self.stats = {"mean": mean.tolist(), "std": std.tolist()}
        else:
            self.stats = stats
        self.mean = torch.tensor(self.stats["mean"], dtype=torch.float32)
        self.std = torch.tensor(self.stats["std"], dtype=torch.float32)
        if self.has_rbp:
            rbp_dim = len(self.rbp_fields)
            rbp_values = np.zeros((len(self.rows), rbp_dim), dtype=np.float32)
            assert rbp_by_id is not None
            for idx, row in enumerate(self.rows):
                values = rbp_by_id.get(row["site_id"])
                if values is not None and values.shape[0] == rbp_dim:
                    rbp_values[idx] = values
            if rbp_stats is None:
                mean = rbp_values.mean(axis=0)
                std = rbp_values.std(axis=0)
                std[std < 1e-6] = 1.0
                self.rbp_stats = {"mean": mean.tolist(), "std": std.tolist()}
            else:
                self.rbp_stats = rbp_stats
            rbp_mean = np.asarray(self.rbp_stats["mean"], dtype=np.float32)
            rbp_std = np.asarray(self.rbp_stats["std"], dtype=np.float32)
            rbp_std[rbp_std < 1e-6] = 1.0
            self.rbp_values = (rbp_values - rbp_mean) / rbp_std
        else:
            self.rbp_stats = {"mean": [], "std": []}
            self.rbp_values = np.zeros((len(self.rows), 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        fold = self.fold_by_id[row["site_id"]]
        struct = torch.tensor([parse_float(fold.get(field, 0.0)) for field in STRUCT_FIELDS], dtype=torch.float32)
        struct = (struct - self.mean) / self.std
        if self.use_metadata:
            support_log = math.log10(max(parse_float(row.get("support_num", 1.0)), 1.0))
            gene_known = 0.0 if row.get("gene_id", "") in {"", "intergenic", "na"} else 1.0
            region_id = REGION_TO_ID.get(row.get("region_type", "other"), REGION_TO_ID["other"])
        else:
            support_log = 0.0
            gene_known = 0.0
            region_id = REGION_TO_ID["other"]
        return {
            "seq": encode_sequence(row["sequence_41nt"], self.input_length),
            "paired_flags": parse_paired_flags(fold.get("paired_flags", ""), self.input_length),
            "pair_partner": parse_partner_indices(fold.get("base_pair_list", ""), self.input_length),
            "struct": struct,
            "region": torch.tensor(region_id, dtype=torch.long),
            "context": torch.tensor([support_log, gene_known], dtype=torch.float32),
            "rbp": torch.tensor(self.rbp_values[idx], dtype=torch.float32),
            "label": torch.tensor(parse_int(row["class_index"]), dtype=torch.long),
            "site_id": row["site_id"],
            "mod_type": row["mod_type"],
        }


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    tensor_keys = ["seq", "paired_flags", "pair_partner", "struct", "region", "context", "rbp", "label"]
    out = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}  # type: ignore[arg-type]
    out["site_id"] = [str(item["site_id"]) for item in batch]
    out["mod_type"] = [str(item["mod_type"]) for item in batch]
    return out


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_rows(dataset_path: Path, rnafold_path: Path, max_rows: int, seed: int) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    fold_by_id = {row["site_id"]: row for row in read_tsv(rnafold_path)}
    rows = [row for row in read_tsv(dataset_path) if row.get("site_id") in fold_by_id]
    random.Random(seed).shuffle(rows)
    if max_rows:
        rows = rows[:max_rows]
    return rows, fold_by_id


def load_rbp_features(path: Optional[str]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    if not path:
        return {}, []
    rbp_path = Path(path)
    if not rbp_path.exists():
        raise FileNotFoundError(f"RBP feature file not found: {rbp_path}")
    rbp_by_id: Dict[str, np.ndarray] = {}
    with rbp_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "site_id" not in reader.fieldnames:
            raise ValueError(f"RBP feature file must contain a site_id column: {rbp_path}")
        rbp_fields = [field for field in reader.fieldnames if field != "site_id"]
        for row in reader:
            rbp_by_id[row["site_id"]] = np.asarray([parse_float(row.get(field, 0.0)) for field in rbp_fields], dtype=np.float32)
    return rbp_by_id, rbp_fields


def split_rows(rows: Sequence[Dict[str, str]], split_name: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    return (
        [row for row in rows if row.get(split_name) == "train"],
        [row for row in rows if row.get(split_name) == "valid"],
        [row for row in rows if row.get(split_name) == "test"],
    )


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def class_weights_from_rows(rows: Sequence[Dict[str, str]], device: torch.device, power: float = 0.5) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for row in rows:
        counts[parse_int(row["class_index"])] += 1
    counts[counts < 1] = 1
    weights = (counts.sum() / counts) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def bce_pos_weight_from_rows(rows: Sequence[Dict[str, str]], device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for row in rows:
        counts[parse_int(row["class_index"])] += 1
    counts[counts < 1] = 1
    neg = max(len(rows), 1) - counts
    weights = np.clip(neg / counts, 1.0, 500.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_sampler(rows: Sequence[Dict[str, str]]) -> WeightedRandomSampler:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    labels = []
    for row in rows:
        label = parse_int(row["class_index"])
        labels.append(label)
        counts[label] += 1
    weights = [1.0 / max(counts[label], 1.0) for label in labels]
    return WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def onehot(labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=len(CLASS_NAMES)).float()


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device, objective: str) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["seq"],
            batch["paired_flags"],
            batch["pair_partner"],
            batch["struct"],
            batch["region"],
            batch["context"],
            batch["rbp"],
        )["logits"]
        if objective == "bce":
            loss = criterion(logits, onehot(batch["label"]))
        else:
            loss = criterion(logits, batch["label"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu()) * int(batch["label"].shape[0])
        count += int(batch["label"].shape[0])
    return total / max(count, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, objective: str = "ce") -> Tuple[np.ndarray, np.ndarray, List[str], List[str], np.ndarray]:
    model.eval()
    probs, labels, gates = [], [], []
    site_ids: List[str] = []
    mod_types: List[str] = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(
            batch["seq"],
            batch["paired_flags"],
            batch["pair_partner"],
            batch["struct"],
            batch["region"],
            batch["context"],
            batch["rbp"],
        )
        if objective == "bce":
            probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
        else:
            probs.append(torch.softmax(out["logits"], dim=-1).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        gates.append(out["class_gate"].cpu().numpy())
        site_ids.extend(batch["site_id"])  # type: ignore[arg-type]
        mod_types.extend(batch["mod_type"])  # type: ignore[arg-type]
    return np.vstack(probs), np.concatenate(labels), site_ids, mod_types, np.vstack(gates)


def metric_or_nan(fn, *args) -> float:
    try:
        return float(fn(*args))
    except ValueError:
        return float("nan")


def evaluate(probs: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    pred = probs.argmax(axis=1)
    onehot = np.eye(len(CLASS_NAMES), dtype=np.int32)[labels]
    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        y = (labels == idx).astype(int)
        p = probs[:, idx]
        pred_bin = (pred == idx).astype(int)
        per_class[name] = {
            "support": int(y.sum()),
            "AUC_ovr": metric_or_nan(roc_auc_score, y, p) if len(np.unique(y)) > 1 else float("nan"),
            "AUPR_ovr": metric_or_nan(average_precision_score, y, p) if len(np.unique(y)) > 1 else float("nan"),
            "F1": metric_or_nan(lambda a, b: f1_score(a, b, zero_division=0), y, pred_bin),
            "Precision": metric_or_nan(lambda a, b: precision_score(a, b, zero_division=0), y, pred_bin),
            "Recall": metric_or_nan(lambda a, b: recall_score(a, b, zero_division=0), y, pred_bin),
            "MCC_ovr": metric_or_nan(matthews_corrcoef, y, pred_bin),
        }

    normalized_probs = probs.copy()
    row_sums = normalized_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    normalized_probs = normalized_probs / row_sums
    macro_auc = metric_or_nan(lambda a, b: roc_auc_score(a, b, multi_class="ovr", average="macro"), onehot, normalized_probs)
    macro_aupr = float(np.nanmean([per_class[name]["AUPR_ovr"] for name in CLASS_NAMES]))
    return {
        "accuracy": metric_or_nan(accuracy_score, labels, pred),
        "balanced_accuracy": metric_or_nan(balanced_accuracy_score, labels, pred),
        "macro_F1": metric_or_nan(lambda a, b: f1_score(a, b, average="macro", zero_division=0), labels, pred),
        "weighted_F1": metric_or_nan(lambda a, b: f1_score(a, b, average="weighted", zero_division=0), labels, pred),
        "MCC": metric_or_nan(matthews_corrcoef, labels, pred),
        "macro_AUC_ovr": macro_auc,
        "macro_AUPR_ovr": macro_aupr,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
    }


def write_predictions(path: Path, probs: np.ndarray, labels: np.ndarray, site_ids: Sequence[str], mod_types: Sequence[str], gates: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["site_id", "true_mod_type", "true_class_index", "pred_mod_type", "pred_class_index"] + [
        f"prob_{name}" for name in CLASS_NAMES
    ] + [f"gate_{name}" for name in CLASS_NAMES]
    pred = probs.argmax(axis=1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for i, site_id in enumerate(site_ids):
            row = {
                "site_id": site_id,
                "true_mod_type": mod_types[i],
                "true_class_index": int(labels[i]),
                "pred_mod_type": CLASS_NAMES[int(pred[i])],
                "pred_class_index": int(pred[i]),
            }
            row.update({f"prob_{name}": float(probs[i, idx]) for idx, name in enumerate(CLASS_NAMES)})
            row.update({f"gate_{name}": float(gates[i, idx]) for idx, name in enumerate(CLASS_NAMES)})
            writer.writerow(row)


def write_confusion_matrix(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["true_mod_type"] + [f"pred_{name}" for name in CLASS_NAMES]
    rows = []
    for idx, name in enumerate(CLASS_NAMES):
        row = {"true_mod_type": name}
        row.update({f"pred_{CLASS_NAMES[j]}": matrix[idx][j] for j in range(len(CLASS_NAMES))})
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_training(args: argparse.Namespace) -> Dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    rows, fold_by_id = load_rows(Path(args.dataset), Path(args.rnafold), args.max_rows, args.seed)
    rbp_by_id, rbp_fields = load_rbp_features(args.rbp_features)
    train_rows, valid_rows, test_rows = split_rows(rows, args.split)
    cfg = MultiClassConfig(
        input_length=args.input_length,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        rbp_input_dim=len(rbp_fields),
        rbp_dim=args.rbp_dim,
    )
    train_ds = RMBase11Dataset(
        train_rows,
        fold_by_id,
        cfg.input_length,
        use_metadata=not args.no_metadata,
        rbp_by_id=rbp_by_id,
        rbp_fields=rbp_fields,
    )
    valid_ds = RMBase11Dataset(
        valid_rows,
        fold_by_id,
        cfg.input_length,
        stats=train_ds.stats,
        use_metadata=not args.no_metadata,
        rbp_by_id=rbp_by_id,
        rbp_fields=rbp_fields,
        rbp_stats=train_ds.rbp_stats,
    )
    test_ds = RMBase11Dataset(
        test_rows,
        fold_by_id,
        cfg.input_length,
        stats=train_ds.stats,
        use_metadata=not args.no_metadata,
        rbp_by_id=rbp_by_id,
        rbp_fields=rbp_fields,
        rbp_stats=train_ds.rbp_stats,
    )
    sampler = make_sampler(train_rows) if args.balanced_sampler else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = StructDualRMMultiClass(cfg, variant=args.variant).to(device)
    if args.objective == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight_from_rows(train_rows, device))
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_from_rows(train_rows, device, args.class_weight_power))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_state = None
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, args.objective)
        valid_probs, valid_labels, _, _, _ = predict(model, valid_loader, device, args.objective)
        valid_metrics = evaluate(valid_probs, valid_labels)
        score = float(valid_metrics[args.selection_metric])
        history.append({"epoch": epoch, "train_loss": loss, "validation": valid_metrics})
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(f"epoch={epoch} loss={loss:.4f} valid_{args.selection_metric}={score:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    valid_probs, valid_labels, valid_ids, valid_mods, valid_gates = predict(model, valid_loader, device, args.objective)
    test_probs, test_labels, test_ids, test_mods, test_gates = predict(model, test_loader, device, args.objective)
    valid_metrics = evaluate(valid_probs, valid_labels)
    test_metrics = evaluate(test_probs, test_labels)
    metrics: Dict[str, object] = {
        "dataset": args.dataset,
        "rnafold": args.rnafold,
        "variant": args.variant,
        "task": "RMBase_hg38_11_class_multiclass",
        "classes": CLASS_NAMES,
        "device": str(device),
        "objective": args.objective,
        "row_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "split": args.split,
        "use_metadata": not args.no_metadata,
        "rbp_features": args.rbp_features or "",
        "use_rbp_features": bool(rbp_fields),
        "rbp_feature_count": len(rbp_fields),
        "rbp_feature_names": rbp_fields,
        "validation": valid_metrics,
        "test": test_metrics,
        "history": history,
        "caveats": [
            "This RMBase v3 robustness benchmark uses hg38 public single-site annotations, separate from the original local hg19 m1A/m6A workflow.",
            f"Split column used for this run: {args.split}.",
            f"Metadata context branch enabled: {not args.no_metadata}. When disabled, region/support/gene-known inputs are zeroed or set to 'other'.",
            f"RBP feature branch enabled: {bool(rbp_fields)}. RBP features are external ENCODE eCLIP overlap/distance annotations, not RMBase support metadata.",
            "m5U and D have very small sample sizes; their per-class metrics should be interpreted cautiously.",
        ],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "variant": args.variant,
            "struct_stats": train_ds.stats,
            "class_names": CLASS_NAMES,
            "class_to_id": CLASS_TO_ID,
            "region_to_id": REGION_TO_ID,
            "base_to_id": BASE_TO_ID,
            "rbp_feature_names": rbp_fields,
            "rbp_stats": train_ds.rbp_stats,
        },
        out_dir / "model.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions(out_dir / "valid_predictions.tsv", valid_probs, valid_labels, valid_ids, valid_mods, valid_gates)
    write_predictions(out_dir / "test_predictions.tsv", test_probs, test_labels, test_ids, test_mods, test_gates)
    write_confusion_matrix(out_dir / "test_confusion_matrix.tsv", test_metrics["confusion_matrix"])  # type: ignore[arg-type]
    (out_dir / "README.md").write_text(
        f"# RMBase 11-class {args.variant} model\n\n"
        f"Trained on `{args.dataset}` with RNAfold features from `{args.rnafold}`.\n\n"
        f"RBP features: `{args.rbp_features or 'disabled'}`.\n\n"
        f"Device: `{device}`. Row counts: {metrics['row_counts']}.\n\n"
        "Files:\n"
        "- `model.pt`: model weights and preprocessing metadata\n"
        "- `metrics.json`: validation/test multiclass and one-vs-rest metrics\n"
        "- `test_predictions.tsv`: class probabilities and class-specific gate values\n"
        "- `test_confusion_matrix.tsv`: test confusion matrix\n",
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train multiclass StructDualRM on RMBase 11-class dataset.")
    parser.add_argument("--dataset", default="rmbase11_work/datasets/rmbase11_hg38_41nt_cdhit80_like.tsv")
    parser.add_argument("--rnafold", default="rmbase11_work/structure/rnafold_41nt_rmbase11.tsv")
    parser.add_argument("--out-dir", default="rmbase11_work/models/M4_structdualrm_multiclass")
    parser.add_argument("--variant", choices=["M0", "M1", "M2", "M3", "M4", "M5"], default="M4")
    parser.add_argument("--split", default="split_random")
    parser.add_argument("--input-length", type=int, default=41)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--rbp-features", default="", help="Optional TSV with site_id plus numeric RBP overlap/distance features.")
    parser.add_argument("--rbp-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--objective", choices=["ce", "bce"], default="ce")
    parser.add_argument("--selection-metric", choices=["macro_AUPR_ovr", "macro_F1", "MCC", "balanced_accuracy"], default="macro_AUPR_ovr")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--no-metadata", action="store_true", help="Disable region_type/support_num/gene-known context inputs for reviewer-safe ablation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
