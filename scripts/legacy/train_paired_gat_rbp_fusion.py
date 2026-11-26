#!/usr/bin/env python
"""Train a position-aware sequence/structure/RBP fusion model on RMBase11.

This is a fresh model family, separate from the earlier M4-M10 experiments.
It follows the technical route:

* sequence Transformer branch with relative-position and center-distance encodings
* RNAfold pair-aware GAT branch over backbone and base-pair edges
* RBP feature branch using the currently available local RBP overlap features
* softmax-gated multimodal fusion

The current RMBase11 benchmark files are 41 nt windows.  The script is written
with a configurable input length, but the default keeps 41 nt so that the model
is directly comparable with the already trained MultiRM/TransRNAm/EvoRMD
baselines under the same strict split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
from torch.utils.data import DataLoader, Dataset


BASE_TO_ID = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4}
DEFAULT_CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
CLASS_NAMES = list(DEFAULT_CLASS_NAMES)
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
STRUCT_FIELDS = [
    "center_paired",
    "central_11nt_paired_fraction",
    "central_21nt_paired_fraction",
    "paired_fraction_window",
    "MFE_per_nt",
    "loop_span_around_center",
]
NODE_FEATURE_FIELDS = ["paired_flags"]
MAX_PAIR_CANDIDATES = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def configure_classes(class_names: Sequence[str]) -> None:
    global CLASS_TO_ID
    names = [str(name).strip() for name in class_names if str(name).strip()]
    if not names:
        raise ValueError("No class names were provided or inferred")
    CLASS_NAMES[:] = names
    CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def infer_class_names(rows: Sequence[Mapping[str, str]], explicit: str = "") -> List[str]:
    if explicit.strip():
        return [name.strip() for name in explicit.split(",") if name.strip()]
    by_index: Dict[int, str] = {}
    fallback: List[str] = []
    seen = set()
    for row in rows:
        mod = row.get("mod_type", "")
        if mod and mod not in seen:
            seen.add(mod)
            fallback.append(mod)
        try:
            idx = int(float(row.get("class_index", "")))
            if mod:
                by_index.setdefault(idx, mod)
        except ValueError:
            continue
    if by_index:
        return [by_index[idx] for idx in sorted(by_index)]
    return fallback or list(DEFAULT_CLASS_NAMES)


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def parse_int(value: object, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_label(row: Mapping[str, str]) -> int:
    if row.get("class_index", "") != "":
        return int(float(row["class_index"]))
    return CLASS_TO_ID[row["mod_type"]]


def encode_sequence(seq: str, length: int) -> torch.Tensor:
    seq = str(seq or "").upper().replace("U", "T")
    ids = [BASE_TO_ID.get(base, 0) for base in seq[:length]]
    ids.extend([0] * max(0, length - len(ids)))
    return torch.tensor(ids, dtype=torch.uint8)


def parse_paired_flags(value: str, length: int) -> torch.Tensor:
    flags = [parse_float(x) for x in str(value or "").split(",") if x != ""]
    flags = flags[:length] + [0.0] * max(0, length - len(flags))
    return torch.tensor(flags, dtype=torch.float32)


def parse_node_features(row: Mapping[str, str], length: int) -> torch.Tensor:
    features = [parse_paired_flags(str(row.get(field, "")), length) for field in NODE_FEATURE_FIELDS]
    if not features:
        features = [torch.zeros(length, dtype=torch.float32)]
    if len(features) == 1:
        return features[0]
    return torch.stack(features, dim=-1)


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


def configure_pair_candidates(max_pair_candidates: int) -> None:
    global MAX_PAIR_CANDIDATES
    MAX_PAIR_CANDIDATES = max(1, int(max_pair_candidates))


def parse_probability_pair_graph(
    row: Mapping[str, str],
    length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse top-k partition-function contacts with an MFE-compatible fallback."""

    candidates: List[List[Tuple[int, float]]] = [[] for _ in range(length)]
    for item in str(row.get("pair_probability_list", "") or "").split(";"):
        if not item or ":" not in item or "-" not in item:
            continue
        pair_text, probability_text = item.rsplit(":", 1)
        left_text, right_text = pair_text.split("-", 1)
        left = parse_int(left_text) - 1
        right = parse_int(right_text) - 1
        probability = max(0.0, min(1.0, parse_float(probability_text)))
        if 0 <= left < length and 0 <= right < length and left != right and probability > 0.0:
            candidates[left].append((right, probability))
            candidates[right].append((left, probability))

    if not any(candidates):
        mfe_partner = parse_partner_indices(str(row.get("base_pair_list", "")), length)
        partners = mfe_partner.unsqueeze(-1)
        weights = (partners >= 0).to(dtype=torch.float32)
        if MAX_PAIR_CANDIDATES > 1:
            padding = torch.full((length, MAX_PAIR_CANDIDATES - 1), -1, dtype=torch.long)
            partners = torch.cat([partners, padding], dim=-1)
            weights = torch.cat([weights, torch.zeros_like(padding, dtype=torch.float32)], dim=-1)
        return partners, weights

    partners = torch.full((length, MAX_PAIR_CANDIDATES), -1, dtype=torch.long)
    weights = torch.zeros((length, MAX_PAIR_CANDIDATES), dtype=torch.float32)
    for source, values in enumerate(candidates):
        for slot, (target, probability) in enumerate(
            sorted(values, key=lambda item: item[1], reverse=True)[:MAX_PAIR_CANDIDATES]
        ):
            partners[source, slot] = target
            weights[source, slot] = probability
    return partners, weights


def shuffle_node_feature_rows(features: torch.Tensor, seed: int) -> torch.Tensor:
    """Permute positions within every sample while preserving each feature marginal."""

    output = features.clone()
    generator = torch.Generator().manual_seed(int(seed))
    for row_index in range(output.shape[0]):
        permutation = torch.randperm(output.shape[1], generator=generator)
        output[row_index] = output[row_index, permutation]
    return output


def shuffle_positive_pair_weights(weights: torch.Tensor, seed: int) -> torch.Tensor:
    output = weights.clone()
    generator = torch.Generator().manual_seed(int(seed))
    for row_index in range(output.shape[0]):
        valid = output[row_index] > 0
        values = output[row_index][valid]
        if values.numel() > 1:
            output[row_index][valid] = values[torch.randperm(values.numel(), generator=generator)]
    return output


def rewire_probability_graph(partners: torch.Tensor, weights: torch.Tensor, seed: int) -> torch.Tensor:
    """Reassign directed targets while preserving source out-degree and target in-degree."""

    output = partners.clone()
    rng = random.Random(int(seed))
    for row_index in range(output.shape[0]):
        valid_slots = torch.nonzero(weights[row_index] > 0, as_tuple=False)
        if valid_slots.shape[0] < 2:
            continue
        sources = [int(slot[0]) for slot in valid_slots]
        targets = [int(output[row_index, slot[0], slot[1]]) for slot in valid_slots]
        proposal = list(targets)
        for _ in range(100):
            rng.shuffle(proposal)
            seen = set()
            valid = True
            for source, target in zip(sources, proposal):
                edge = (source, target)
                if source == target or edge in seen:
                    valid = False
                    break
                seen.add(edge)
            if valid:
                break
        else:
            continue
        for slot, target in zip(valid_slots, proposal):
            output[row_index, slot[0], slot[1]] = target
    return output


def split_rows(rows: Sequence[Dict[str, str]], split_name: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    return (
        [row for row in rows if row.get(split_name) == "train"],
        [row for row in rows if row.get(split_name) == "valid"],
        [row for row in rows if row.get(split_name) == "test"],
    )


def load_rnafold(path: Path) -> Dict[str, Dict[str, str]]:
    return {row["site_id"]: row for row in read_tsv(path)}


def load_rbp_features(path: Optional[str]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    if not path:
        return {}, []
    rbp_path = Path(path)
    if not rbp_path.exists():
        raise FileNotFoundError(f"RBP feature file not found: {rbp_path}")
    with rbp_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "site_id" not in reader.fieldnames:
            raise ValueError(f"RBP feature file must contain site_id: {rbp_path}")
        fields = [field for field in reader.fieldnames if field != "site_id"]
        values = {
            row["site_id"]: np.asarray([parse_float(row.get(field, 0.0)) for field in fields], dtype=np.float32)
            for row in reader
        }
    return values, fields


def normalize_matrix(values: np.ndarray, stats: Optional[Dict[str, List[float]]] = None) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    if values.shape[1] == 0:
        return values.astype(np.float32), {"mean": [], "std": []}
    if stats is None:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-6] = 1.0
        stats = {"mean": mean.tolist(), "std": std.tolist()}
    mean_arr = np.asarray(stats["mean"], dtype=np.float32)
    std_arr = np.asarray(stats["std"], dtype=np.float32)
    std_arr[std_arr < 1e-6] = 1.0
    return ((values - mean_arr) / std_arr).astype(np.float32), stats


class RMBaseFusionDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        fold_by_id: Mapping[str, Mapping[str, str]],
        input_length: int,
        rbp_by_id: Optional[Mapping[str, np.ndarray]] = None,
        rbp_dim: int = 0,
        struct_stats: Optional[Dict[str, List[float]]] = None,
        rbp_stats: Optional[Dict[str, List[float]]] = None,
        shuffle_structure: bool = False,
        shuffle_rbp: bool = False,
        disable_pair_edges: bool = False,
        shuffle_structure_nodes: bool = False,
        shuffle_pair_weights: bool = False,
        rewire_pair_edges: bool = False,
        seed: int = 42,
        sequence_column: str = "sequence_41nt",
    ):
        self.rows = list(rows)
        self.fold_by_id = fold_by_id
        self.input_length = input_length
        self.site_ids = [row["site_id"] for row in self.rows]
        self.mod_types = [row["mod_type"] for row in self.rows]
        self.labels = np.asarray([parse_label(row) for row in self.rows], dtype=np.int64)
        missing_sequence = [row["site_id"] for row in self.rows if not row.get(sequence_column)]
        if missing_sequence:
            raise ValueError(
                f"Missing {sequence_column} for {len(missing_sequence)} rows; first site_id={missing_sequence[0]}"
            )
        self.seq = torch.stack([encode_sequence(row[sequence_column], input_length) for row in self.rows])

        rng = random.Random(seed)
        fold_ids = self.site_ids.copy()
        if shuffle_structure:
            rng.shuffle(fold_ids)
        self.fold_ids = fold_ids

        rbp_ids = self.site_ids.copy()
        if shuffle_rbp:
            rng.shuffle(rbp_ids)
        self.rbp_ids = rbp_ids

        struct_values = np.asarray(
            [[parse_float(fold_by_id[fold_id].get(field, 0.0)) for field in STRUCT_FIELDS] for fold_id in self.fold_ids],
            dtype=np.float32,
        )
        self.struct, self.struct_stats = normalize_matrix(struct_values, struct_stats)

        rbp_values = np.zeros((len(self.rows), rbp_dim), dtype=np.float32)
        if rbp_by_id and rbp_dim:
            for idx, rbp_id in enumerate(self.rbp_ids):
                values = rbp_by_id.get(rbp_id)
                if values is not None and values.shape[0] == rbp_dim:
                    rbp_values[idx] = values
        self.rbp, self.rbp_stats = normalize_matrix(rbp_values, rbp_stats)

        self.struct_tensor = torch.as_tensor(self.struct, dtype=torch.float32)
        self.rbp_tensor = torch.as_tensor(self.rbp, dtype=torch.float32)
        self.label_tensor = torch.as_tensor(self.labels, dtype=torch.long)
        self.paired_flags_tensor = torch.stack(
            [parse_node_features(fold_by_id[fold_id], input_length) for fold_id in self.fold_ids]
        )
        pair_graphs = [parse_probability_pair_graph(fold_by_id[fold_id], input_length) for fold_id in self.fold_ids]
        self.pair_partner_tensor = torch.stack([item[0] for item in pair_graphs]).to(torch.int16)
        self.pair_weight_tensor = torch.stack([item[1] for item in pair_graphs]).to(torch.float32)
        if shuffle_structure_nodes:
            self.paired_flags_tensor = shuffle_node_feature_rows(self.paired_flags_tensor, seed + 101)
        if shuffle_pair_weights:
            self.pair_weight_tensor = shuffle_positive_pair_weights(self.pair_weight_tensor, seed + 211)
        if rewire_pair_edges:
            self.pair_partner_tensor = rewire_probability_graph(
                self.pair_partner_tensor,
                self.pair_weight_tensor,
                seed + 307,
            ).to(torch.int16)
        if disable_pair_edges:
            self.pair_partner_tensor.fill_(-1)
            self.pair_weight_tensor.zero_()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "seq": self.seq[idx],
            "paired_flags": self.paired_flags_tensor[idx],
            "pair_partner": self.pair_partner_tensor[idx],
            "pair_weights": self.pair_weight_tensor[idx],
            "struct": self.struct_tensor[idx],
            "rbp": self.rbp_tensor[idx],
            "label": self.label_tensor[idx],
            "site_id": self.site_ids[idx],
            "mod_type": self.mod_types[idx],
        }


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    tensor_keys = ["seq", "paired_flags", "pair_partner", "pair_weights", "struct", "rbp", "label"]
    out: Dict[str, object] = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}  # type: ignore[arg-type]
    out["site_id"] = [str(item["site_id"]) for item in batch]
    out["mod_type"] = [str(item["mod_type"]) for item in batch]
    return out


class CenterAttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        center = h[:, h.shape[1] // 2 : h.shape[1] // 2 + 1, :].expand_as(h)
        weights = torch.softmax(self.score(torch.cat([h, center], dim=-1)).squeeze(-1), dim=-1)
        return torch.sum(h * weights.unsqueeze(-1), dim=1)


class PairAwareGATLayer(nn.Module):
    """A light GAT over self/backbone/base-pair candidate edges."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.edge_type = nn.Embedding(4, d_model)
        self.msg = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        partner: torch.Tensor,
        pair_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length, dim = h.shape
        if partner.dim() == 2:
            partner = partner.unsqueeze(-1)
        pair_count = partner.shape[-1]
        if pair_weights is None:
            pair_weights = (partner >= 0).to(dtype=h.dtype)
        elif pair_weights.dim() == 2:
            pair_weights = pair_weights.unsqueeze(-1)
        pair_weights = pair_weights.to(device=h.device, dtype=h.dtype).clamp(0.0, 1.0)
        self_h = h
        left_h = torch.zeros_like(h)
        right_h = torch.zeros_like(h)
        left_h[:, 1:, :] = h[:, :-1, :]
        right_h[:, :-1, :] = h[:, 1:, :]
        gather_idx = partner.long().clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, dim)
        gather_source = h.unsqueeze(2).expand(-1, -1, pair_count, -1)
        pair_h = torch.gather(gather_source, 1, gather_idx)
        candidates = torch.cat([torch.stack([self_h, left_h, right_h], dim=2), pair_h], dim=2)
        type_ids = torch.cat(
            [
                torch.arange(3, device=h.device),
                torch.full((pair_count,), 3, device=h.device, dtype=torch.long),
            ]
        ).view(1, 1, 3 + pair_count)
        candidates = candidates + self.edge_type(type_ids)

        valid = torch.ones(batch, length, 3 + pair_count, dtype=torch.bool, device=h.device)
        valid[:, 0, 1] = False
        valid[:, -1, 2] = False
        valid[:, :, 3:] = partner >= 0

        messages = self.msg(candidates)
        q = self.query(h).unsqueeze(2)
        k = self.key(messages)
        scores = (q * k).sum(dim=-1) / math.sqrt(dim)
        pair_log_prior = torch.log(pair_weights.clamp_min(1e-6))
        scores[:, :, 3:] = scores[:, :, 3:] + pair_log_prior
        scores = scores.masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=-1)
        update = torch.sum(weights.unsqueeze(-1) * messages, dim=2)
        h = self.norm1(h + self.dropout(self.out(update)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h, weights


class SequenceBranch(nn.Module):
    def __init__(self, d_model: int, length: int, layers: int, heads: int, dropout: float, use_position: bool):
        super().__init__()
        self.length = length
        self.use_position = use_position
        self.base_emb = nn.Embedding(5, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(length, d_model)
        self.dist_emb = nn.Embedding(length, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pool = CenterAttentionPool(d_model)
        self.project = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        x = self.base_emb(seq.long())
        if self.use_position:
            pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, length)
            dist = torch.abs(pos - (length // 2))
            x = x + self.pos_emb(pos) + self.dist_emb(dist)
        h = self.encoder(x)
        center = h[:, length // 2, :]
        local = h[:, max(0, length // 2 - 5) : min(length, length // 2 + 6), :].mean(dim=1)
        pooled = self.pool(h)
        return self.project(torch.cat([center, local, pooled], dim=-1))


class StructureBranch(nn.Module):
    def __init__(self, d_model: int, length: int, layers: int, dropout: float, use_position: bool):
        super().__init__()
        self.length = length
        self.use_position = use_position
        self.base_emb = nn.Embedding(5, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(length, d_model)
        self.dist_emb = nn.Embedding(length, d_model)
        self.paired_proj = nn.Linear(max(1, len(NODE_FEATURE_FIELDS)), d_model)
        self.scalar_proj = nn.Sequential(nn.Linear(len(STRUCT_FIELDS), d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.layers = nn.ModuleList([PairAwareGATLayer(d_model, dropout) for _ in range(layers)])
        self.pool = CenterAttentionPool(d_model)
        self.project = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(
        self,
        seq: torch.Tensor,
        paired_flags: torch.Tensor,
        partner: torch.Tensor,
        struct: torch.Tensor,
        pair_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length = seq.shape
        node_features = paired_flags if paired_flags.dim() == 3 else paired_flags.unsqueeze(-1)
        h = self.base_emb(seq.long()) + self.paired_proj(node_features.float())
        if self.use_position:
            pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, length)
            dist = torch.abs(pos - (length // 2))
            h = h + self.pos_emb(pos) + self.dist_emb(dist)
        attentions = []
        for layer in self.layers:
            h, attn = layer(h, partner, pair_weights)
            attentions.append(attn)
        center = h[:, length // 2, :]
        local = h[:, max(0, length // 2 - 10) : min(length, length // 2 + 11), :].mean(dim=1)
        pooled = self.pool(h)
        scalar = self.scalar_proj(struct)
        return self.project(torch.cat([center, local, pooled, scalar], dim=-1)), torch.stack(attentions).mean(dim=0)


class RBPBranch(nn.Module):
    def __init__(self, rbp_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.rbp_dim = rbp_dim
        self.project = (
            nn.Sequential(
                nn.Linear(rbp_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            if rbp_dim > 0
            else None
        )

    def forward(self, rbp: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.project is None:
            return torch.zeros(batch_size, 0, device=device)
        return self.project(rbp.float())


@dataclass
class FusionConfig:
    input_length: int = 41
    num_classes: int = len(CLASS_NAMES)
    d_model: int = 128
    seq_layers: int = 2
    seq_heads: int = 4
    gat_layers: int = 2
    dropout: float = 0.3
    rbp_dim: int = 0
    use_structure: bool = True
    use_rbp: bool = True
    use_position: bool = True
    use_gate: bool = True


class PairedGATRBPFusion(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_branch = SequenceBranch(cfg.d_model, cfg.input_length, cfg.seq_layers, cfg.seq_heads, cfg.dropout, cfg.use_position)
        self.struct_branch = StructureBranch(cfg.d_model, cfg.input_length, cfg.gat_layers, cfg.dropout, cfg.use_position)
        self.rbp_branch = RBPBranch(cfg.rbp_dim, cfg.d_model, cfg.dropout)
        self.branch_norm = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(3)])
        self.gate = nn.Sequential(nn.Linear(cfg.d_model * 3, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, 3))
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.num_classes),
        )

    def forward(
        self,
        seq: torch.Tensor,
        paired_flags: torch.Tensor,
        partner: torch.Tensor,
        struct: torch.Tensor,
        rbp: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch = seq.shape[0]
        device = seq.device
        z_seq = self.branch_norm[0](self.seq_branch(seq))
        if self.cfg.use_structure:
            z_struct, edge_attention = self.struct_branch(seq, paired_flags, partner, struct)
            z_struct = self.branch_norm[1](z_struct)
        else:
            z_struct = torch.zeros_like(z_seq)
            edge_attention = torch.zeros(batch, self.cfg.input_length, 4, device=device)
        if self.cfg.use_rbp and self.cfg.rbp_dim > 0:
            z_rbp = self.branch_norm[2](self.rbp_branch(rbp, batch, device))
        else:
            z_rbp = torch.zeros_like(z_seq)
        branches = torch.stack([z_seq, z_struct, z_rbp], dim=1)
        active = torch.tensor(
            [True, self.cfg.use_structure, self.cfg.use_rbp and self.cfg.rbp_dim > 0],
            device=device,
            dtype=torch.bool,
        )
        if self.cfg.use_gate:
            logits = self.gate(torch.cat([z_seq, z_struct, z_rbp], dim=-1))
            logits = logits.masked_fill(~active.view(1, 3), -1e9)
            weights = torch.softmax(logits, dim=-1)
        else:
            weights = active.float().view(1, 3).expand(batch, 3)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        fused = torch.sum(weights.unsqueeze(-1) * branches, dim=1)
        return {"logits": self.classifier(fused), "modality_gate": weights, "edge_attention": edge_attention}


def class_weights_from_rows(rows: Sequence[Dict[str, str]], power: float, device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for row in rows:
        counts[parse_label(row)] += 1
    counts[counts < 1] = 1.0
    weights = (counts.sum() / counts) ** power
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalCrossEntropy(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor], gamma: float):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else torch.ones(len(CLASS_NAMES)))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        p = torch.exp(logp)
        idx = labels.view(-1, 1)
        logp_t = logp.gather(1, idx).squeeze(1)
        p_t = p.gather(1, idx).squeeze(1)
        w_t = self.weight.gather(0, labels)
        return (-(1.0 - p_t).pow(self.gamma) * logp_t * w_t).mean()


def move_batch(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device) -> Dict[str, float]:
    model.train()
    total = 0.0
    rows = 0
    gate_sum = np.zeros(3, dtype=np.float64)
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["seq"], batch["paired_flags"], batch["pair_partner"], batch["struct"], batch["rbp"])  # type: ignore[arg-type]
        labels = batch["label"]  # type: ignore[assignment]
        loss = criterion(out["logits"], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = int(labels.shape[0])
        total += float(loss.detach().cpu()) * batch_size
        rows += batch_size
        gate_sum += out["modality_gate"].detach().cpu().numpy().sum(axis=0)
    gate_mean = gate_sum / max(rows, 1)
    return {
        "loss": total / max(rows, 1),
        "gate_seq": float(gate_mean[0]),
        "gate_structure": float(gate_mean[1]),
        "gate_rbp": float(gate_mean[2]),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    probs, labels = [], []
    gate_sum = np.zeros(3, dtype=np.float64)
    rows = 0
    site_ids: List[str] = []
    mod_types: List[str] = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["seq"], batch["paired_flags"], batch["pair_partner"], batch["struct"], batch["rbp"])  # type: ignore[arg-type]
        batch_probs = torch.softmax(out["logits"], dim=-1)
        probs.append(batch_probs.detach().cpu().numpy())
        labels.append(batch["label"].detach().cpu().numpy())  # type: ignore[union-attr]
        batch_size = int(batch["label"].shape[0])  # type: ignore[index]
        gate_sum += out["modality_gate"].detach().cpu().numpy().sum(axis=0)
        rows += batch_size
        site_ids.extend(batch["site_id"])  # type: ignore[arg-type]
        mod_types.extend(batch["mod_type"])  # type: ignore[arg-type]
    return {
        "probs": np.vstack(probs),
        "labels": np.concatenate(labels),
        "site_ids": site_ids,
        "mod_types": mod_types,
        "gate_mean": (gate_sum / max(rows, 1)).tolist(),
    }


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
        q = (pred == idx).astype(int)
        per_class[name] = {
            "support": int(y.sum()),
            "AUC_ovr": metric_or_nan(roc_auc_score, y, p) if len(np.unique(y)) > 1 else float("nan"),
            "AUPR_ovr": metric_or_nan(average_precision_score, y, p) if len(np.unique(y)) > 1 else float("nan"),
            "F1": metric_or_nan(lambda a, b: f1_score(a, b, zero_division=0), y, q),
            "Precision": metric_or_nan(lambda a, b: precision_score(a, b, zero_division=0), y, q),
            "Recall": metric_or_nan(lambda a, b: recall_score(a, b, zero_division=0), y, q),
            "MCC_ovr": metric_or_nan(matthews_corrcoef, y, q),
        }
    return {
        "accuracy": metric_or_nan(accuracy_score, labels, pred),
        "balanced_accuracy": metric_or_nan(balanced_accuracy_score, labels, pred),
        "macro_F1": metric_or_nan(lambda a, b: f1_score(a, b, average="macro", zero_division=0), labels, pred),
        "weighted_F1": metric_or_nan(lambda a, b: f1_score(a, b, average="weighted", zero_division=0), labels, pred),
        "MCC": metric_or_nan(matthews_corrcoef, labels, pred),
        "macro_AUC_ovr": metric_or_nan(lambda a, b: roc_auc_score(a, b, multi_class="ovr", average="macro"), onehot, probs),
        "macro_AUPR_ovr": float(np.nanmean([per_class[name]["AUPR_ovr"] for name in CLASS_NAMES])),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
    }


def write_predictions(path: Path, predictions: Dict[str, object]) -> None:
    probs = predictions["probs"]
    labels = predictions["labels"]
    assert isinstance(probs, np.ndarray) and isinstance(labels, np.ndarray)
    pred = probs.argmax(axis=1)
    fields = ["site_id", "true_mod_type", "true_class_index", "pred_mod_type", "pred_class_index"]
    fields += [f"prob_{name}" for name in CLASS_NAMES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, site_id in enumerate(predictions["site_ids"]):  # type: ignore[index]
            row = {
                "site_id": site_id,
                "true_mod_type": predictions["mod_types"][idx],  # type: ignore[index]
                "true_class_index": int(labels[idx]),
                "pred_mod_type": CLASS_NAMES[int(pred[idx])],
                "pred_class_index": int(pred[idx]),
            }
            row.update({f"prob_{name}": float(probs[idx, j]) for j, name in enumerate(CLASS_NAMES)})
            writer.writerow(row)


def write_confusion_matrix(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    fields = ["true_mod_type"] + [f"pred_{name}" for name in CLASS_NAMES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, name in enumerate(CLASS_NAMES):
            row = {"true_mod_type": name}
            row.update({f"pred_{CLASS_NAMES[j]}": int(matrix[idx][j]) for j in range(len(CLASS_NAMES))})
            writer.writerow(row)


def write_per_class_metrics(path: Path, metrics: Mapping[str, object]) -> None:
    per_class = metrics["per_class"]
    assert isinstance(per_class, Mapping)
    fields = ["mod_type", "support", "AUPR_ovr", "AUC_ovr", "F1", "Precision", "Recall", "MCC_ovr"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for name in CLASS_NAMES:
            values = per_class[name]
            assert isinstance(values, Mapping)
            row = {"mod_type": name}
            row.update({field: values[field] for field in fields if field != "mod_type"})
            writer.writerow(row)


def write_gate_summary(path: Path, gate: Sequence[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["modality", "mean_gate"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for name, value in zip(["sequence", "structure", "rbp"], gate):
            writer.writerow({"modality": name, "mean_gate": float(value)})


def run(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"

    rows = read_tsv(Path(args.dataset))
    configure_classes(infer_class_names(rows, args.class_names))
    fold_by_id = load_rnafold(Path(args.rnafold))
    rows = [row for row in rows if row.get("site_id") in fold_by_id]
    if args.max_rows:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_rows]
    train_rows, valid_rows, test_rows = split_rows(rows, args.split)
    rbp_by_id, rbp_fields = load_rbp_features(args.rbp_features)
    rbp_dim = len(rbp_fields) if not args.disable_rbp else 0

    train_ds = RMBaseFusionDataset(
        train_rows,
        fold_by_id,
        args.input_length,
        rbp_by_id=rbp_by_id,
        rbp_dim=rbp_dim,
        shuffle_structure=args.shuffle_structure,
        shuffle_rbp=args.shuffle_rbp,
        seed=args.seed,
    )
    valid_ds = RMBaseFusionDataset(
        valid_rows,
        fold_by_id,
        args.input_length,
        rbp_by_id=rbp_by_id,
        rbp_dim=rbp_dim,
        struct_stats=train_ds.struct_stats,
        rbp_stats=train_ds.rbp_stats,
        shuffle_structure=False,
        shuffle_rbp=False,
        seed=args.seed,
    )
    test_ds = RMBaseFusionDataset(
        test_rows,
        fold_by_id,
        args.input_length,
        rbp_by_id=rbp_by_id,
        rbp_dim=rbp_dim,
        struct_stats=train_ds.struct_stats,
        rbp_stats=train_ds.rbp_stats,
        shuffle_structure=False,
        shuffle_rbp=False,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    cfg = FusionConfig(
        input_length=args.input_length,
        num_classes=len(CLASS_NAMES),
        d_model=args.d_model,
        seq_layers=args.seq_layers,
        seq_heads=args.seq_heads,
        gat_layers=args.gat_layers,
        dropout=args.dropout,
        rbp_dim=rbp_dim,
        use_structure=not args.disable_structure,
        use_rbp=not args.disable_rbp,
        use_position=not args.disable_position,
        use_gate=not args.disable_gate,
    )
    model = PairedGATRBPFusion(cfg).to(device)
    weights = class_weights_from_rows(train_rows, args.class_weight_power, device)
    criterion: nn.Module
    if args.loss == "focal":
        criterion = FocalCrossEntropy(weights, args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: List[Dict[str, object]] = []
    best_state = None
    best_score = -1.0
    best_epoch = 0
    start_epoch = 1
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint.get("history", [])
        best_state = checkpoint.get("best_state_dict")
        best_score = float(checkpoint.get("best_score", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"resumed_from={checkpoint_path} start_epoch={start_epoch} best_score={best_score:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        losses = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        valid_pred = predict(model, valid_loader, device)
        valid_metrics = evaluate(valid_pred["probs"], valid_pred["labels"])  # type: ignore[arg-type]
        score = float(valid_metrics[args.selection_metric])
        history.append({"epoch": epoch, "train": losses, "validation": valid_metrics, "learning_rate": optimizer.param_groups[0]["lr"]})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_state_dict": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": history,
                "config": asdict(cfg),
            },
            checkpoint_path,
        )
        print(
            f"epoch={epoch} loss={losses['loss']:.4f} valid_{args.selection_metric}={score:.4f} "
            f"gate=({losses['gate_seq']:.2f},{losses['gate_structure']:.2f},{losses['gate_rbp']:.2f})",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_pred = predict(model, valid_loader, device)
    test_pred = predict(model, test_loader, device)
    valid_metrics = evaluate(valid_pred["probs"], valid_pred["labels"])  # type: ignore[arg-type]
    test_metrics = evaluate(test_pred["probs"], test_pred["labels"])  # type: ignore[arg-type]

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "class_names": CLASS_NAMES,
            "struct_stats": train_ds.struct_stats,
            "rbp_fields": rbp_fields,
            "rbp_stats": train_ds.rbp_stats,
            "training_args": vars(args),
        },
        out_dir / "model.pt",
    )
    metrics: Dict[str, object] = {
        "dataset": args.dataset,
        "rnafold": args.rnafold,
        "rbp_features": args.rbp_features,
        "variant": "PairedGAT_RBP_GatedFusion",
        "task": f"RMBase_hg38_{len(CLASS_NAMES)}_class_center_site_prediction",
        "classes": CLASS_NAMES,
        "device": str(device),
        "split": args.split,
        "row_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "config": asdict(cfg),
        "loss": args.loss,
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "validation": valid_metrics,
        "test": test_metrics,
        "validation_gate_mean": valid_pred["gate_mean"],
        "test_gate_mean": test_pred["gate_mean"],
        "history": history,
        "caveats": [
            "The current benchmark uses 41 nt windows to match the existing RMBase11 baselines; the 101 nt route requires rebuilding data, RNAfold structures, RBP windows and all baselines.",
            "The current RBP branch uses sample-level local overlap/count/distance features because position-level 101 x K RBP matrices have not yet been generated.",
            "No support_num, publication count, cell-list hash or other evidence-strength metadata is used as a model input.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions(out_dir / "valid_predictions.tsv", valid_pred)
    write_predictions(out_dir / "test_predictions.tsv", test_pred)
    write_per_class_metrics(out_dir / "valid_per_class_metrics.tsv", valid_metrics)
    write_per_class_metrics(out_dir / "test_per_class_metrics.tsv", test_metrics)
    write_confusion_matrix(out_dir / "test_confusion_matrix.tsv", test_metrics["confusion_matrix"])  # type: ignore[arg-type]
    write_gate_summary(out_dir / "test_modality_gate_summary.tsv", test_pred["gate_mean"])  # type: ignore[arg-type]
    (out_dir / "README.md").write_text(
        "# PairedGAT-RBP gated fusion\n\n"
        "Fresh model following the sequence-structure-RBP technical route. It uses a sequence Transformer branch, "
        "a pair-aware GAT branch over RNAfold backbone/base-pair edges, a local RBP feature MLP branch and "
        "softmax-gated multimodal fusion.\n\n"
        f"Split: `{args.split}`. Selection metric: `{args.selection_metric}`. Best epoch: {best_epoch}.\n",
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PairedGAT-RBP gated fusion model on RMBase11.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_mmseqs80_strict_splits.tsv")
    parser.add_argument("--class-names", default="", help="Optional comma-separated class order; inferred from class_index if omitted.")
    parser.add_argument("--rnafold", default="rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv")
    parser.add_argument("--rbp-features", default="rmbase11_full_work/rbp/rmbase11_rbp_features.tsv")
    parser.add_argument("--split", default="split_mmseqs80_cluster")
    parser.add_argument("--out-dir", default="rmbase11_full_work/models/PairedGAT_RBP_fusion_strict_full")
    parser.add_argument("--input-length", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seq-layers", type=int, default=2)
    parser.add_argument("--seq-heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--selection-metric", choices=["macro_AUPR_ovr", "macro_F1", "MCC", "balanced_accuracy"], default="macro_AUPR_ovr")
    parser.add_argument("--disable-structure", action="store_true")
    parser.add_argument("--disable-rbp", action="store_true")
    parser.add_argument("--disable-position", action="store_true")
    parser.add_argument("--disable-gate", action="store_true")
    parser.add_argument("--shuffle-structure", action="store_true")
    parser.add_argument("--shuffle-rbp", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
