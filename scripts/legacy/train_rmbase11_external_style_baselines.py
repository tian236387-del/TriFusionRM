#!/usr/bin/env python
"""Train same-split external-model style baselines on the RMBase hg38 11-class data.

The public repositories for EvoRMD, MultiRM and TransRNAm use different input
formats and old dependency stacks.  This script keeps the main architectural
idea of each model while forcing a shared dataset, split and metric protocol:

* MultiRM-style: 3-mer Word2Vec embedding + BiLSTM + label-specific attention.
* TransRNAm-style: 3-mer Word2Vec embedding + Transformer pooling + CNN heads.
* EvoRMD-lite: sequence encoder + biological-context fusion before attention.

Rows are evaluated as an 11-class mutually exclusive problem, with one-vs-rest
AUC/AUPR also reported per modification class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
BASE_TO_ID = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4}
ALPHABET = "ACGT"
DEFAULT_CONTEXT_FIELDS = ["region_type", "gene_type", "cell_list", "seq_type_list"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def normalize_sequence(seq: str) -> str:
    return str(seq or "").upper().replace("U", "T")


def encode_bases(seq: str, length: int) -> np.ndarray:
    seq = normalize_sequence(seq)
    out = np.zeros(length, dtype=np.int64)
    for idx, base in enumerate(seq[:length]):
        out[idx] = BASE_TO_ID.get(base, 0)
    return out


def split_rows(rows: Sequence[Dict[str, str]], split_name: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    return (
        [row for row in rows if row.get(split_name) == "train"],
        [row for row in rows if row.get(split_name) == "valid"],
        [row for row in rows if row.get(split_name) == "test"],
    )


def parse_label(row: Mapping[str, str]) -> int:
    if row.get("class_index", "") != "":
        return int(float(row["class_index"]))
    return CLASS_TO_ID[row["mod_type"]]


def split_tokens(value: str) -> List[str]:
    text = str(value or "unknown").strip()
    if not text or text.lower() in {"na", "nan", "none"}:
        return ["unknown"]
    tokens = []
    for part in text.replace(";", ",").split(","):
        token = part.strip()
        if token:
            tokens.append(token)
    return tokens or ["unknown"]


def hash_tokens(tokens: Sequence[str], dim: int, salt: str) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        digest = hashlib.blake2b(f"{salt}:{token}".encode("utf-8"), digest_size=8).hexdigest()
        idx = int(digest, 16) % dim
        out[idx] = 1.0
    total = out.sum()
    if total > 1:
        out /= total
    return out


@dataclass
class ContextSpec:
    region_vocab: Dict[str, int]
    gene_vocab: Dict[str, int]
    context_mode: str = "full"
    cell_hash_dim: int = 64
    seqtype_hash_dim: int = 32

    @property
    def dim(self) -> int:
        base_dim = len(self.region_vocab) + len(self.gene_vocab)
        if self.context_mode == "full":
            return base_dim + self.cell_hash_dim + self.seqtype_hash_dim + 2
        if self.context_mode == "safe":
            return base_dim + 1
        if self.context_mode == "none":
            return 1
        raise ValueError(f"Unsupported context mode: {self.context_mode}")


def build_context_spec(
    rows: Sequence[Dict[str, str]],
    max_gene_types: int = 24,
    cell_hash_dim: int = 64,
    seqtype_hash_dim: int = 32,
    context_mode: str = "full",
) -> ContextSpec:
    regions = sorted({row.get("region_type", "unknown") or "unknown" for row in rows} | {"unknown"})
    gene_counts = Counter(row.get("gene_type", "unknown") or "unknown" for row in rows)
    gene_types = ["unknown"] + [name for name, _ in gene_counts.most_common(max_gene_types) if name != "unknown"]
    return ContextSpec(
        region_vocab={name: idx for idx, name in enumerate(regions)},
        gene_vocab={name: idx for idx, name in enumerate(gene_types)},
        context_mode=context_mode,
        cell_hash_dim=cell_hash_dim,
        seqtype_hash_dim=seqtype_hash_dim,
    )


def encode_context(row: Mapping[str, str], spec: ContextSpec) -> np.ndarray:
    out = np.zeros(spec.dim, dtype=np.float32)
    if spec.context_mode == "none":
        return out
    offset = 0
    region = row.get("region_type", "unknown") or "unknown"
    out[offset + spec.region_vocab.get(region, spec.region_vocab["unknown"])] = 1.0
    offset += len(spec.region_vocab)

    gene = row.get("gene_type", "unknown") or "unknown"
    out[offset + spec.gene_vocab.get(gene, spec.gene_vocab["unknown"])] = 1.0
    offset += len(spec.gene_vocab)

    gene_id = str(row.get("gene_id", "") or "")
    gene_known = 0.0 if gene_id in {"", "na", "intergenic"} else 1.0
    if spec.context_mode == "safe":
        out[offset] = gene_known
        return out

    out[offset : offset + spec.cell_hash_dim] = hash_tokens(split_tokens(row.get("cell_list", "")), spec.cell_hash_dim, "cell")
    offset += spec.cell_hash_dim

    out[offset : offset + spec.seqtype_hash_dim] = hash_tokens(split_tokens(row.get("seq_type_list", "")), spec.seqtype_hash_dim, "seqtype")
    offset += spec.seqtype_hash_dim

    try:
        support = max(float(row.get("support_num", "1") or 1.0), 1.0)
    except ValueError:
        support = 1.0
    out[offset] = math.log10(support)
    out[offset + 1] = gene_known
    return out


def load_embedding_dict(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    return {str(key).upper().replace("U", "T"): np.asarray(value, dtype=np.float32) for key, value in raw.items()}


def build_trimer_vocab(embedding_dict: Mapping[str, np.ndarray]) -> Dict[str, int]:
    all_trimers = {"".join(parts) for parts in __import__("itertools").product(ALPHABET, repeat=3)}
    keys = sorted(set(embedding_dict) | all_trimers)
    return {key: idx + 1 for idx, key in enumerate(keys)}


def build_embedding_matrix(vocab: Mapping[str, int], embedding_dict: Mapping[str, np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    first = next(iter(embedding_dict.values()))
    matrix = np.zeros((len(vocab) + 1, int(first.shape[0])), dtype=np.float32)
    for mer, idx in vocab.items():
        if mer in embedding_dict:
            matrix[idx] = embedding_dict[mer]
        else:
            matrix[idx] = rng.normal(0.0, 0.02, size=matrix.shape[1]).astype(np.float32)
    return matrix


def encode_trimers(seq: str, vocab: Mapping[str, int], length: int, k: int = 3) -> np.ndarray:
    seq = normalize_sequence(seq)
    width = max(0, length - k + 1)
    out = np.zeros(width, dtype=np.int64)
    for idx in range(width):
        out[idx] = vocab.get(seq[idx : idx + k], 0)
    return out


class RMBase11ExternalDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        model_kind: str,
        input_length: int,
        trimer_vocab: Optional[Mapping[str, int]] = None,
        context_spec: Optional[ContextSpec] = None,
    ):
        self.rows = list(rows)
        self.model_kind = model_kind
        self.labels = np.asarray([parse_label(row) for row in self.rows], dtype=np.int64)
        self.site_ids = [row["site_id"] for row in self.rows]
        self.mod_types = [row["mod_type"] for row in self.rows]
        if model_kind == "evormd_lite":
            if context_spec is None:
                raise ValueError("context_spec is required for evormd_lite")
            self.x = np.stack([encode_bases(row["sequence_41nt"], input_length) for row in self.rows])
            self.context = np.stack([encode_context(row, context_spec) for row in self.rows])
        else:
            if trimer_vocab is None:
                raise ValueError("trimer_vocab is required for k-mer models")
            self.x = np.stack([encode_trimers(row["sequence_41nt"], trimer_vocab, input_length) for row in self.rows])
            self.context = None
        self.x_tensor = torch.as_tensor(self.x, dtype=torch.long)
        self.label_tensor = torch.as_tensor(self.labels, dtype=torch.long)
        self.context_tensor = torch.as_tensor(self.context, dtype=torch.float32) if self.context is not None else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item: Dict[str, object] = {
            "x": self.x_tensor[idx],
            "label": self.label_tensor[idx],
            "site_id": self.site_ids[idx],
            "mod_type": self.mod_types[idx],
        }
        if self.context_tensor is not None:
            item["context"] = self.context_tensor[idx]
        return item


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "x": torch.stack([item["x"] for item in batch]),  # type: ignore[arg-type]
        "label": torch.stack([item["label"] for item in batch]),  # type: ignore[arg-type]
        "site_id": [str(item["site_id"]) for item in batch],
        "mod_type": [str(item["mod_type"]) for item in batch],
    }
    if "context" in batch[0]:
        out["context"] = torch.stack([item["context"] for item in batch])  # type: ignore[arg-type]
    return out


class MultiRMStyleModel(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        self.encoder = nn.LSTM(input_size=weights.shape[1], hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        encoded_dim = hidden_dim * 2
        self.attn_proj = nn.Linear(encoded_dim, encoded_dim)
        self.label_queries = nn.Parameter(torch.empty(num_classes, encoded_dim))
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Dropout(dropout), nn.Linear(encoded_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)) for _ in range(num_classes)]
        )
        nn.init.xavier_uniform_(self.label_queries)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h, _ = self.encoder(self.embedding(x.long()))
        scores = torch.einsum("bld,td->blt", torch.tanh(self.attn_proj(h)), self.label_queries)
        attention = torch.softmax(scores.transpose(1, 2), dim=-1)
        pooled = torch.einsum("btl,bld->btd", attention, h)
        logits = torch.cat([head(pooled[:, idx, :]) for idx, head in enumerate(self.heads)], dim=-1)
        return {"logits": logits, "attention": attention}


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float, max_len: int = 256):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.shape[1], :])


class ConvProjection(nn.Module):
    def __init__(self, input_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(32, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        l1 = (input_size - 7) // 2 + 1
        l2 = (l1 - 2) // 2 + 1
        l3 = (l2 - 2) // 2 + 1
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * l3, 256), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x.unsqueeze(1)))


class TransRNAmStyleModel(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, num_classes: int, dropout: float, heads: int, ff_dim: int, transformer_layers: int):
        super().__init__()
        weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        dim = weights.shape[1]
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        self.position = PositionalEncoding(dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.to_512 = nn.Linear(dim, 512)
        self.cnn = ConvProjection(512, dropout)
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)) for _ in range(num_classes)]
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h = self.position(self.embedding(x.long()))
        h = self.transformer(h)
        pooled = self.to_512(h.mean(dim=1))
        conv = self.cnn(pooled)
        z = torch.cat([pooled, conv], dim=-1)
        logits = torch.cat([head(z) for head in self.heads], dim=-1)
        return {"logits": logits}


class EvoRMDLiteModel(nn.Module):
    def __init__(self, context_dim: int, num_classes: int, seq_dim: int, meta_dim: int, dropout: float, mlp_depth: int):
        super().__init__()
        self.base_emb = nn.Embedding(5, seq_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(64, seq_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=seq_dim,
            nhead=4,
            dim_feedforward=seq_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.meta = nn.Sequential(nn.Linear(context_dim, meta_dim), nn.ReLU(), nn.LayerNorm(meta_dim), nn.Dropout(dropout))
        fused_dim = seq_dim + meta_dim
        self.attention = nn.Linear(fused_dim, 1)
        if mlp_depth <= 1:
            self.classifier = nn.Linear(fused_dim, num_classes)
        elif mlp_depth == 2:
            self.classifier = nn.Sequential(nn.Linear(fused_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if context is None:
            raise ValueError("EvoRMD-lite requires context")
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], x.shape[1])
        seq_h = self.encoder(self.base_emb(x.long()) + self.pos_emb(pos))
        meta = self.meta(context).unsqueeze(1).expand(-1, x.shape[1], -1)
        token = torch.cat([seq_h, meta], dim=-1)
        weights = torch.softmax(self.attention(token).squeeze(-1), dim=-1)
        pooled = torch.sum(token * weights.unsqueeze(-1), dim=1)
        return {"logits": self.classifier(pooled), "attention": weights}


def class_weights_from_rows(rows: Sequence[Dict[str, str]], power: float, device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for row in rows:
        counts[parse_label(row)] += 1
    counts[counts < 1] = 1
    weights = (counts.sum() / counts) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def multilabel_pos_weight(rows: Sequence[Dict[str, str]], device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for row in rows:
        counts[parse_label(row)] += 1
    counts[counts < 1] = 1
    neg = len(rows) - counts
    weights = neg / counts
    weights = np.clip(weights, 1.0, 500.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()


def move_batch(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, loss_fn, device: torch.device, objective: str) -> float:
    model.train()
    total = 0.0
    rows = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["x"], batch.get("context"))  # type: ignore[arg-type]
        labels = batch["label"]  # type: ignore[assignment]
        if objective == "ce":
            loss = loss_fn(out["logits"], labels)
        else:
            loss = loss_fn(out["logits"], onehot(labels, len(CLASS_NAMES)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = int(labels.shape[0])
        total += float(loss.detach().cpu()) * batch_size
        rows += batch_size
    return total / max(rows, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, objective: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    model.eval()
    probs, labels = [], []
    site_ids: List[str] = []
    mod_types: List[str] = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch["x"], batch.get("context"))["logits"]  # type: ignore[arg-type]
        if objective == "ce":
            batch_probs = torch.softmax(logits, dim=-1)
        else:
            batch_probs = torch.sigmoid(logits)
        probs.append(batch_probs.detach().cpu().numpy())
        labels.append(batch["label"].detach().cpu().numpy())  # type: ignore[union-attr]
        site_ids.extend(batch["site_id"])  # type: ignore[arg-type]
        mod_types.extend(batch["mod_type"])  # type: ignore[arg-type]
    return np.vstack(probs), np.concatenate(labels), site_ids, mod_types


def metric_or_nan(fn, *args) -> float:
    try:
        return float(fn(*args))
    except ValueError:
        return float("nan")


def evaluate(probs: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    pred = probs.argmax(axis=1)
    onehot_np = np.eye(len(CLASS_NAMES), dtype=np.int32)[labels]
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    normalized_probs = probs / row_sums
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
    macro_auc = metric_or_nan(lambda a, b: roc_auc_score(a, b, multi_class="ovr", average="macro"), onehot_np, normalized_probs)
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


def write_predictions(path: Path, probs: np.ndarray, labels: np.ndarray, site_ids: Sequence[str], mod_types: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["site_id", "true_mod_type", "true_class_index", "pred_mod_type", "pred_class_index"] + [f"prob_{name}" for name in CLASS_NAMES]
    pred = probs.argmax(axis=1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, site_id in enumerate(site_ids):
            row = {
                "site_id": site_id,
                "true_mod_type": mod_types[idx],
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
            row.update({f"pred_{CLASS_NAMES[j]}": matrix[idx][j] for j in range(len(CLASS_NAMES))})
            writer.writerow(row)


def build_model(args: argparse.Namespace, embedding_matrix: Optional[np.ndarray], context_spec: Optional[ContextSpec]) -> Tuple[nn.Module, str, Dict[str, object]]:
    if args.model == "multirm":
        if embedding_matrix is None:
            raise ValueError("embedding_matrix is required")
        return (
            MultiRMStyleModel(embedding_matrix, hidden_dim=args.hidden_dim, num_classes=len(CLASS_NAMES), dropout=args.dropout),
            "bce",
            {
                "official_repo": "https://github.com/Tsedao/MultiRM",
                "source_type": "style_reimplementation",
                "note": "Keeps MultiRM's 3-mer embedding, BiLSTM and label-specific attention, retrained on the RMBase hg38 11-class split.",
            },
        )
    if args.model == "transrnam":
        if embedding_matrix is None:
            raise ValueError("embedding_matrix is required")
        return (
            TransRNAmStyleModel(
                embedding_matrix,
                num_classes=len(CLASS_NAMES),
                dropout=args.dropout,
                heads=args.transformer_heads,
                ff_dim=args.transformer_ff_dim,
                transformer_layers=args.transformer_layers,
            ),
            "bce",
            {
                "official_repo": "https://github.com/lennylv/TransRNAm",
                "source_type": "style_reimplementation",
                "note": "Keeps TransRNAm's 3-mer embedding, Transformer pooling and CNN classifier idea, with hard-coded legacy paths removed.",
            },
        )
    if args.model == "evormd_lite":
        if context_spec is None:
            raise ValueError("context_spec is required")
        return (
            EvoRMDLiteModel(
                context_dim=context_spec.dim,
                num_classes=len(CLASS_NAMES),
                seq_dim=args.seq_dim,
                meta_dim=args.meta_dim,
                dropout=args.dropout,
                mlp_depth=args.mlp_depth,
            ),
            "ce",
            {
                "official_repo": "https://github.com/Gardeina/EvoRMD",
                "source_type": "adapted_lite_reimplementation",
                "note": f"Official EvoRMD requires RNA-FM/fm and its original metadata stack. This version keeps context fusion before attention, but replaces RNA-FM with a trainable sequence encoder. Context mode: {args.context_mode}.",
            },
        )
    raise ValueError(args.model)


def run(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rows = read_tsv(Path(args.dataset))
    if args.max_rows:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_rows]
    train_rows, valid_rows, test_rows = split_rows(rows, args.split)

    embedding_matrix = None
    trimer_vocab = None
    context_spec = None
    if args.model in {"multirm", "transrnam"}:
        embedding_dict = load_embedding_dict(Path(args.embedding_path))
        trimer_vocab = build_trimer_vocab(embedding_dict)
        embedding_matrix = build_embedding_matrix(trimer_vocab, embedding_dict, args.seed)
    else:
        context_spec = build_context_spec(
            train_rows,
            max_gene_types=args.max_gene_types,
            cell_hash_dim=args.cell_hash_dim,
            seqtype_hash_dim=args.seqtype_hash_dim,
            context_mode=args.context_mode,
        )

    train_ds = RMBase11ExternalDataset(train_rows, args.model, args.input_length, trimer_vocab=trimer_vocab, context_spec=context_spec)
    valid_ds = RMBase11ExternalDataset(valid_rows, args.model, args.input_length, trimer_vocab=trimer_vocab, context_spec=context_spec)
    test_ds = RMBase11ExternalDataset(test_rows, args.model, args.input_length, trimer_vocab=trimer_vocab, context_spec=context_spec)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    model, objective, source_info = build_model(args, embedding_matrix, context_spec)
    model = model.to(device)
    if objective == "ce":
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_from_rows(train_rows, args.class_weight_power, device))
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=multilabel_pos_weight(train_rows, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_state = None
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device, objective)
        valid_probs, valid_labels, _, _ = predict(model, valid_loader, device, objective)
        valid_metrics = evaluate(valid_probs, valid_labels)
        score = float(valid_metrics[args.selection_metric])
        history.append({"epoch": epoch, "train_loss": loss, "validation": valid_metrics})
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(f"model={args.model} epoch={epoch} loss={loss:.4f} valid_{args.selection_metric}={score:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    valid_probs, valid_labels, valid_ids, valid_mods = predict(model, valid_loader, device, objective)
    test_probs, test_labels, test_ids, test_mods = predict(model, test_loader, device, objective)
    valid_metrics = evaluate(valid_probs, valid_labels)
    test_metrics = evaluate(test_probs, test_labels)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "model": args.model,
            "objective": objective,
            "trimer_vocab": trimer_vocab,
            "context_spec": asdict(context_spec) if context_spec is not None else None,
            "training_args": vars(args),
        },
        out_dir / "model.pt",
    )
    metrics: Dict[str, object] = {
        "dataset": args.dataset,
        "variant": args.model,
        "display_name": {
            "multirm": "MultiRM-style retrained",
            "transrnam": "TransRNAm-style retrained",
            "evormd_lite": "EvoRMD-lite adapted",
        }[args.model],
        "task": "RMBase_hg38_11_class_multiclass",
        "classes": CLASS_NAMES,
        "device": str(device),
        "objective": objective,
        "row_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "source_info": source_info,
        "validation": valid_metrics,
        "test": test_metrics,
        "history": history,
        "caveats": [
            "All rows come from the same RMBase hg38 11-class split used by StructDualRM.",
            "These runs are same-data architecture-style baselines. They are not cross-paper metric comparisons.",
            "m5U and D have very small test supports; per-class metrics for those classes are unstable.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions(out_dir / "valid_predictions.tsv", valid_probs, valid_labels, valid_ids, valid_mods)
    write_predictions(out_dir / "test_predictions.tsv", test_probs, test_labels, test_ids, test_mods)
    write_confusion_matrix(out_dir / "test_confusion_matrix.tsv", test_metrics["confusion_matrix"])  # type: ignore[arg-type]
    (out_dir / "README.md").write_text(
        f"# {metrics['display_name']}\n\n"
        f"Official/source repository: {source_info['official_repo']}\n\n"
        f"Source type: `{source_info['source_type']}`.\n\n"
        f"{source_info['note']}\n\n"
        f"Dataset: `{args.dataset}`. Split column: `{args.split}`. Device: `{device}`.\n",
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RMBase 11-class external-model style baselines.")
    parser.add_argument("--model", choices=["multirm", "transrnam", "evormd_lite"], required=True)
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites.tsv")
    parser.add_argument("--split", default="split_random")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--embedding-path", default="external/MultiRM-master/Embeddings/embeddings_12RM.pkl")
    parser.add_argument("--input-length", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seq-dim", type=int, default=128)
    parser.add_argument("--meta-dim", type=int, default=128)
    parser.add_argument("--mlp-depth", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=6)
    parser.add_argument("--transformer-ff-dim", type=int, default=512)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--selection-metric", choices=["macro_AUPR_ovr", "macro_F1", "MCC", "balanced_accuracy"], default="macro_AUPR_ovr")
    parser.add_argument("--max-gene-types", type=int, default=24)
    parser.add_argument("--cell-hash-dim", type=int, default=64)
    parser.add_argument("--seqtype-hash-dim", type=int, default=32)
    parser.add_argument("--context-mode", choices=["full", "safe", "none"], default="full")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.out_dir:
        args.out_dir = f"rmbase11_full_work/baselines/{args.model}"
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
