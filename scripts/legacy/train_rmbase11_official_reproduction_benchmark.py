#!/usr/bin/env python
"""Run paper-like and strict-split official-architecture RNA modification baselines.

This script is intentionally separate from the StructDualRM model family.  It
implements the core architectures released with MultiRM and TransRNAm in a
portable form, an EvoRMD-style adapter, and an EvoRMD core reproduction using
the released RNA-FM T12 backbone.
All models are trained and evaluated on the same RMBase11 hg38 table so that
random-split and strict-split results are directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import time
from collections import Counter
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


DEFAULT_CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
CLASS_NAMES = list(DEFAULT_CLASS_NAMES)
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
BASE_TO_ID = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4}
ALPHABET = "ACGT"


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
    global CLASS_NAMES, CLASS_TO_ID
    names = [str(name).strip() for name in class_names if str(name).strip()]
    if not names:
        raise ValueError("No class names were provided or inferred")
    CLASS_NAMES = names
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


def split_rows(rows: Sequence[Dict[str, str]], split_name: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    return (
        [row for row in rows if row.get(split_name) == "train"],
        [row for row in rows if row.get(split_name) == "valid"],
        [row for row in rows if row.get(split_name) == "test"],
    )


def normalize_sequence(seq: str) -> str:
    return str(seq or "").upper().replace("U", "T")


def parse_label(row: Mapping[str, str]) -> int:
    if row.get("class_index", "") != "":
        return int(float(row["class_index"]))
    return CLASS_TO_ID[row["mod_type"]]


def load_embedding_dict(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    return {str(key).upper().replace("U", "T"): np.asarray(value, dtype=np.float32) for key, value in raw.items()}


def build_trimer_vocab(embedding_dict: Mapping[str, np.ndarray]) -> Dict[str, int]:
    trimers = {"".join(parts) for parts in __import__("itertools").product(ALPHABET, repeat=3)}
    keys = sorted(set(embedding_dict) | trimers)
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


def encode_bases(seq: str, length: int) -> np.ndarray:
    seq = normalize_sequence(seq)
    out = np.zeros(length, dtype=np.int64)
    for idx, base in enumerate(seq[:length]):
        out[idx] = BASE_TO_ID.get(base, 0)
    return out


def onehot(labels: torch.Tensor) -> torch.Tensor:
    return F.one_hot(labels, num_classes=len(CLASS_NAMES)).float()


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
        out[int(digest, 16) % dim] = 1.0
    total = out.sum()
    if total > 1:
        out /= total
    return out


@dataclass
class ContextSpec:
    region_vocab: Dict[str, int]
    gene_vocab: Dict[str, int]
    cell_hash_dim: int = 64
    seqtype_hash_dim: int = 32

    @property
    def dim(self) -> int:
        return len(self.region_vocab) + len(self.gene_vocab) + self.cell_hash_dim + self.seqtype_hash_dim + 3

    @property
    def safe_dim(self) -> int:
        return len(self.region_vocab) + len(self.gene_vocab) + 1


def build_context_spec(rows: Sequence[Dict[str, str]], max_gene_types: int, cell_hash_dim: int, seqtype_hash_dim: int) -> ContextSpec:
    regions = sorted({row.get("region_type", "unknown") or "unknown" for row in rows} | {"unknown"})
    gene_counts = Counter(row.get("gene_type", "unknown") or "unknown" for row in rows)
    genes = ["unknown"] + [name for name, _ in gene_counts.most_common(max_gene_types) if name != "unknown"]
    return ContextSpec(
        region_vocab={name: idx for idx, name in enumerate(regions)},
        gene_vocab={name: idx for idx, name in enumerate(genes)},
        cell_hash_dim=cell_hash_dim,
        seqtype_hash_dim=seqtype_hash_dim,
    )


def encode_context(row: Mapping[str, str], spec: ContextSpec, context_mode: str = "full") -> np.ndarray:
    if context_mode not in {"full", "safe"}:
        raise ValueError(f"Unsupported context mode for context encoding: {context_mode}")
    out = np.zeros(spec.dim if context_mode == "full" else spec.safe_dim, dtype=np.float32)
    offset = 0
    region = row.get("region_type", "unknown") or "unknown"
    out[offset + spec.region_vocab.get(region, spec.region_vocab["unknown"])] = 1.0
    offset += len(spec.region_vocab)
    gene = row.get("gene_type", "unknown") or "unknown"
    out[offset + spec.gene_vocab.get(gene, spec.gene_vocab["unknown"])] = 1.0
    offset += len(spec.gene_vocab)
    if context_mode == "safe":
        out[offset] = 1.0 if row.get("gene_id", "") not in {"", "na", "intergenic"} else 0.0
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
    out[offset + 1] = 1.0 if row.get("gene_id", "") not in {"", "na", "intergenic"} else 0.0
    out[offset + 2] = 1.0
    return out


class OfficialBenchmarkDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        model_name: str,
        input_length: int,
        sequence_column: str = "sequence_41nt",
        trimer_vocab: Optional[Mapping[str, int]] = None,
        context_spec: Optional[ContextSpec] = None,
        context_mode: str = "full",
        shuffle_context: bool = False,
        context_shuffle_seed: int = 0,
    ):
        self.rows = list(rows)
        self.model_name = model_name
        self.labels = np.asarray([parse_label(row) for row in self.rows], dtype=np.int64)
        self.site_ids = [row["site_id"] for row in self.rows]
        self.mod_types = [row["mod_type"] for row in self.rows]
        if model_name in {"multirm_official", "transrnam_official"}:
            if trimer_vocab is None:
                raise ValueError("trimer_vocab is required for k-mer models")
            self.x = np.stack([encode_trimers(row[sequence_column], trimer_vocab, input_length) for row in self.rows])
            self.context = None
        else:
            if context_mode != "sequence_only" and context_spec is None:
                raise ValueError("context_spec is required for evormd_adapted")
            self.x = np.stack([encode_bases(row[sequence_column], input_length) for row in self.rows])
            if context_mode == "sequence_only":
                self.context = None
            else:
                self.context = np.stack([encode_context(row, context_spec, context_mode) for row in self.rows])
                if shuffle_context:
                    permutation = np.random.default_rng(context_shuffle_seed).permutation(len(self.context))
                    self.context = self.context[permutation]
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


class BahdanauAttention(nn.Module):
    """Official MultiRM/TransRNAm attention block, made device-agnostic."""

    def __init__(self, in_features: int, hidden_units: int, num_task: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_units)
        self.w2 = nn.Linear(in_features, hidden_units)
        self.v = nn.Linear(hidden_units, num_task)

    def forward(self, hidden_states: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden_states.unsqueeze(1)
        scores = self.v(torch.tanh(self.w1(values) + self.w2(hidden)))
        weights = torch.softmax(scores, dim=1)
        context = torch.matmul(values.transpose(1, 2), weights).transpose(1, 2)
        return context, weights


class MultiRMOfficialModel(nn.Module):
    """Portable implementation of MultiRM Scripts/models.py model_v3."""

    def __init__(self, embedding_matrix: np.ndarray, num_classes: int, freeze_embedding: bool = True):
        super().__init__()
        weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze_embedding, padding_idx=0)
        self.bilstm = nn.LSTM(input_size=weights.shape[1], hidden_size=256, batch_first=True, bidirectional=True)
        self.attention = BahdanauAttention(in_features=512, hidden_units=100, num_task=num_classes)
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(), nn.Linear(128, 1)) for _ in range(num_classes)]
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        emb = self.embedding(x.long())
        output, _ = self.bilstm(emb)
        hidden = torch.cat([output[:, -1, :256], output[:, 0, 256:]], dim=-1)
        context_vector, attention = self.attention(hidden, output)
        logits = torch.cat([head(context_vector[:, idx, :]) for idx, head in enumerate(self.heads)], dim=-1)
        return {"logits": logits, "attention": attention.transpose(1, 2)}


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.shape[1], :])


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        if embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        self.d_k = embedding_dim // heads
        self.heads = heads
        self.linears = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch = query.shape[0]
        query, key, value = [
            layer(x).view(batch, -1, self.heads, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linears, (query, key, value))
        ]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, value).transpose(1, 2).contiguous().view(batch, -1, self.heads * self.d_k)
        return self.linears[-1](out)


class TransRNAmTransformerBlock(nn.Module):
    """Portable version of TransRNAm MyTransformerModel."""

    def __init__(self, embedding_matrix: np.ndarray, dropout: float, heads: int, output_size: int, freeze_embedding: bool = True):
        super().__init__()
        weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze_embedding, padding_idx=0)
        self.position = PositionalEncoding(weights.shape[1], dropout)
        self.attention = MultiHeadedAttention(heads, weights.shape[1], dropout)
        self.norm = nn.LayerNorm(weights.shape[1])
        self.linear = nn.Linear(weights.shape[1], output_size)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.position(self.embedding(x.long()))
        att = self.attention(emb, emb, emb)
        att = self.norm(att + emb)
        att = self.norm(att)
        pooled = att.sum(dim=1) / (emb.shape[1] + 1e-5)
        return self.linear(pooled)


class TransRNAmCNN(nn.Module):
    """Portable version of TransRNAm NaiveNet(input_size=512)."""

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
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * l3, 256), nn.ReLU(), nn.Dropout())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x.unsqueeze(1)))


class TransRNAmOfficialModel(nn.Module):
    """Portable implementation of TransRNAm Scripts/models.py model_v11_test."""

    def __init__(self, embedding_matrix: np.ndarray, num_classes: int, dropout: float, heads: int, freeze_embedding: bool = True):
        super().__init__()
        self.transformer = TransRNAmTransformerBlock(
            embedding_matrix=embedding_matrix,
            dropout=dropout,
            heads=heads,
            output_size=512,
            freeze_embedding=freeze_embedding,
        )
        self.cnn = TransRNAmCNN(input_size=512, dropout=dropout)
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(), nn.Linear(128, 1)) for _ in range(num_classes)]
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        transformed = self.transformer(x.long())
        conv = self.cnn(transformed)
        fused = torch.cat([transformed, conv], dim=-1)
        logits = torch.cat([head(fused) for head in self.heads], dim=-1)
        return {"logits": logits}


class EvoRMDAdaptedModel(nn.Module):
    """EvoRMD-style model when RNA-FM/fm is not available in the environment."""

    def __init__(
        self,
        context_dim: int,
        num_classes: int,
        seq_dim: int,
        meta_dim: int,
        dropout: float,
        mlp_depth: int,
        use_context: bool = True,
    ):
        super().__init__()
        self.use_context = use_context
        self.base_emb = nn.Embedding(5, seq_dim, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=seq_dim,
            nhead=4,
            dim_feedforward=seq_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.sequence_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.context_encoder = (
            nn.Sequential(nn.Linear(context_dim, meta_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(meta_dim, seq_dim))
            if use_context
            else None
        )
        self.attention = nn.Linear(seq_dim, 1)
        in_dim = seq_dim * 2 if use_context else seq_dim
        if mlp_depth <= 1:
            self.classifier = nn.Linear(in_dim, num_classes)
        elif mlp_depth == 2:
            self.classifier = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        token = self.sequence_encoder(self.base_emb(x.long()))
        meta = None
        fused_tokens = token
        if self.use_context:
            if context is None or self.context_encoder is None:
                raise ValueError("context is required for this EvoRMD adapted model")
            meta = self.context_encoder(context.float()).unsqueeze(1)
            fused_tokens = token + meta
        weights = torch.softmax(self.attention(fused_tokens).squeeze(-1), dim=-1)
        pooled = torch.sum(fused_tokens * weights.unsqueeze(-1), dim=1)
        classifier_input = torch.cat([pooled, meta.squeeze(1)], dim=-1) if meta is not None else pooled
        logits = self.classifier(classifier_input)
        return {"logits": logits, "attention": weights}


class EvoRMDRNAFMModel(nn.Module):
    """EvoRMD core architecture with the released RNA-FM T12 backbone."""

    def __init__(
        self,
        context_dim: int,
        num_classes: int,
        meta_dim: int,
        mlp_depth: int,
        use_context: bool = True,
        train_rnafm: bool = False,
    ):
        super().__init__()
        try:
            import fm
        except ImportError as exc:
            raise RuntimeError("rna-fm==0.2.2 is required for evormd_rnafm_official_core") from exc

        self.rnafm, alphabet = fm.pretrained.rna_fm_t12()
        self.use_context = use_context
        self.train_rnafm = train_rnafm
        for parameter in self.rnafm.parameters():
            parameter.requires_grad = train_rnafm

        token_lookup = [
            int(alphabet.unk_idx),
            int(alphabet.get_idx("A")),
            int(alphabet.get_idx("C")),
            int(alphabet.get_idx("G")),
            int(alphabet.get_idx("U")),
        ]
        self.register_buffer("token_lookup", torch.tensor(token_lookup, dtype=torch.long))
        self.bos_idx = int(alphabet.cls_idx)
        self.eos_idx = int(alphabet.eos_idx)
        self.padding_idx = int(alphabet.padding_idx)

        self.context_encoder = None
        if use_context:
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, meta_dim),
                nn.LayerNorm(meta_dim),
            )
        fused_dim = 640 + (meta_dim if use_context else 0)
        self.attention = nn.Linear(fused_dim, 1)
        if mlp_depth <= 1:
            self.classifier = nn.Linear(fused_dim, num_classes)
        elif mlp_depth == 2:
            self.classifier = nn.Sequential(nn.Linear(fused_dim, 128), nn.ReLU(), nn.Linear(128, num_classes))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )

    def train(self, mode: bool = True) -> "EvoRMDRNAFMModel":
        super().train(mode)
        if not self.train_rnafm:
            self.rnafm.eval()
        return self

    def task_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in super().state_dict().items()
            if not key.startswith("rnafm.")
        }

    def load_task_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        missing, unexpected = super().load_state_dict(state_dict, strict=False)
        invalid_missing = [key for key in missing if not key.startswith("rnafm.")]
        if invalid_missing or unexpected:
            raise RuntimeError(f"Invalid EvoRMD task checkpoint: missing={invalid_missing}, unexpected={unexpected}")

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        base_tokens = self.token_lookup[x.long().clamp(min=0, max=4)]
        batch_size = base_tokens.shape[0]
        bos = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=x.device)
        eos = torch.full((batch_size, 1), self.eos_idx, dtype=torch.long, device=x.device)
        tokens = torch.cat([bos, base_tokens, eos], dim=1)

        if self.train_rnafm:
            results = self.rnafm(tokens, repr_layers=[12])
        else:
            with torch.no_grad():
                results = self.rnafm(tokens, repr_layers=[12])
        fused_tokens = results["representations"][12]

        if self.use_context:
            if context is None or self.context_encoder is None:
                raise ValueError("context is required for this EvoRMD RNA-FM model")
            metadata = self.context_encoder(context.float())
            metadata_tokens = metadata.unsqueeze(1).expand(-1, fused_tokens.shape[1], -1)
            fused_tokens = torch.cat([fused_tokens, metadata_tokens], dim=-1)

        attention = torch.softmax(self.attention(fused_tokens).squeeze(-1), dim=-1)
        pooled = torch.sum(fused_tokens * attention.unsqueeze(-1), dim=1)
        return {"logits": self.classifier(pooled), "attention": attention}


def checkpoint_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    if isinstance(model, EvoRMDRNAFMModel) and not model.train_rnafm:
        return model.task_state_dict()
    return model.state_dict()


def load_checkpoint_model_state(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    if isinstance(model, EvoRMDRNAFMModel) and not model.train_rnafm:
        model.load_task_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def downsample_training_rows(
    rows: Sequence[Dict[str, str]],
    target_classes: Sequence[str],
    alpha: float,
    seed: int,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    targets = {name.strip().lower() for name in target_classes if name.strip()}
    counts = Counter(row["mod_type"] for row in rows)
    other_counts = [count for name, count in counts.items() if name.lower() not in targets and count > 0]
    if not targets or not other_counts:
        return list(rows), {"enabled": False, "before": dict(counts), "after": dict(counts)}

    geometric_mean = float(np.exp(np.mean(np.log(np.asarray(other_counts, dtype=np.float64)))))
    rng = random.Random(seed)
    selected: List[Dict[str, str]] = []
    targets_by_name: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        if row["mod_type"].lower() in targets:
            targets_by_name.setdefault(row["mod_type"], []).append(row)
        else:
            selected.append(row)

    requested: Dict[str, int] = {}
    for name, class_rows in targets_by_name.items():
        keep = int(round((geometric_mean ** (1.0 - alpha)) * (len(class_rows) ** alpha)))
        keep = min(max(keep, 1), len(class_rows))
        requested[name] = keep
        selected.extend(rng.sample(class_rows, keep) if keep < len(class_rows) else class_rows)
    rng.shuffle(selected)
    after = Counter(row["mod_type"] for row in selected)
    return selected, {
        "enabled": True,
        "alpha": alpha,
        "target_classes": sorted(targets),
        "geometric_mean_other_classes": geometric_mean,
        "requested": requested,
        "before": dict(counts),
        "after": dict(after),
    }


def effective_num_pos_weight(rows: Sequence[Dict[str, str]], beta: float, cap: float, device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for row in rows:
        counts[parse_label(row)] += 1
    counts[counts < 1] = 1
    effective = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective, 1e-12)
    weights = weights / weights.mean()
    # Convert official effective-number class weights into a BCE pos_weight scale.
    inverse = (len(rows) - counts) / counts
    combined = np.sqrt(np.maximum(weights, 1e-8) * np.maximum(inverse, 1.0))
    return torch.tensor(np.clip(combined, 1.0, cap), dtype=torch.float32, device=device)


def class_weights(rows: Sequence[Dict[str, str]], power: float, device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for row in rows:
        counts[parse_label(row)] += 1
    counts[counts < 1] = 1
    weights = np.power(counts.sum() / counts, power)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


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
            loss = loss_fn(out["logits"], onehot(labels))
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


@torch.no_grad()
def measure_inference_efficiency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_steps: int = 2,
    timed_steps: int = 5,
) -> Dict[str, object]:
    """Measure model size, one-batch FLOPs and steady-state throughput."""
    try:
        raw_batch = next(iter(loader))
    except StopIteration:
        return {"available": False, "reason": "empty_loader"}
    batch = move_batch(raw_batch, device)
    batch_size = int(batch["label"].shape[0])  # type: ignore[index,union-attr]

    def forward_once() -> None:
        model(batch["x"], batch.get("context"))

    model.eval()
    for _ in range(max(0, warmup_steps)):
        forward_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(max(1, timed_steps)):
        forward_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    profiled_flops = None
    profiler_error = ""
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as profile:
            forward_once()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        profiled_flops = int(sum(int(event.flops or 0) for event in profile.key_averages()))
    except Exception as exc:
        profiler_error = f"{type(exc).__name__}: {exc}"

    timed_steps = max(1, timed_steps)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    milliseconds_per_sample = 1000.0 * elapsed / max(batch_size * timed_steps, 1)
    return {
        "available": True,
        "parameter_count": int(total_parameters),
        "trainable_parameter_count": int(trainable_parameters),
        "profile_batch_size": batch_size,
        "profiled_forward_flops": profiled_flops,
        "flops_per_sample": (float(profiled_flops) / batch_size) if profiled_flops is not None and batch_size else None,
        "timed_steps": timed_steps,
        "timed_seconds": elapsed,
        "samples_per_second": batch_size * timed_steps / max(elapsed, 1e-12),
        "milliseconds_per_sample": milliseconds_per_sample,
        "latency_ms_per_sample": milliseconds_per_sample,
        "profiler_error": profiler_error,
    }


def metric_or_nan(fn, *args) -> float:
    try:
        return float(fn(*args))
    except ValueError:
        return float("nan")


def evaluate(probs: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    pred = probs.argmax(axis=1)
    onehot_np = np.eye(len(CLASS_NAMES), dtype=np.int32)[labels]
    norm = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
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
    return {
        "accuracy": metric_or_nan(accuracy_score, labels, pred),
        "balanced_accuracy": metric_or_nan(balanced_accuracy_score, labels, pred),
        "macro_F1": metric_or_nan(lambda a, b: f1_score(a, b, average="macro", zero_division=0), labels, pred),
        "weighted_F1": metric_or_nan(lambda a, b: f1_score(a, b, average="weighted", zero_division=0), labels, pred),
        "MCC": metric_or_nan(matthews_corrcoef, labels, pred),
        "macro_AUC_ovr": metric_or_nan(lambda a, b: roc_auc_score(a, b, multi_class="ovr", average="macro"), onehot_np, norm),
        "macro_AUPR_ovr": float(np.nanmean([per_class[name]["AUPR_ovr"] for name in CLASS_NAMES])),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
    }


def write_predictions(path: Path, probs: np.ndarray, labels: np.ndarray, site_ids: Sequence[str], mod_types: Sequence[str]) -> None:
    fields = ["site_id", "true_mod_type", "true_class_index", "pred_mod_type", "pred_class_index"]
    fields += [f"prob_{name}" for name in CLASS_NAMES]
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


def write_per_class_metrics(path: Path, metrics: Mapping[str, object]) -> None:
    fields = ["mod_type", "support", "AUPR_ovr", "AUC_ovr", "F1", "Precision", "Recall", "MCC_ovr"]
    per_class = metrics.get("per_class", {})
    if not isinstance(per_class, Mapping):
        raise ValueError("Metrics do not contain a per_class mapping")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for name in CLASS_NAMES:
            values = per_class.get(name, {})
            if not isinstance(values, Mapping):
                values = {}
            row = {"mod_type": name}
            row.update({field: values.get(field, "") for field in fields if field != "mod_type"})
            writer.writerow(row)


def build_model(
    args: argparse.Namespace,
    embedding_matrix: Optional[np.ndarray],
    context_spec: Optional[ContextSpec],
) -> Tuple[nn.Module, str, Dict[str, object]]:
    if args.model == "multirm_official":
        if embedding_matrix is None:
            raise ValueError("embedding_matrix is required")
        return (
            MultiRMOfficialModel(embedding_matrix, len(CLASS_NAMES), freeze_embedding=not args.train_embedding),
            "bce",
            {
                "official_repo": "https://github.com/Tsedao/MultiRM",
                "source_type": "official_architecture_reproduction",
                "official_files": ["Scripts/models.py:model_v3", "Scripts/util_layers.py:BahdanauAttention"],
                "limitations": "Training/evaluation loop is reimplemented to use the unified RMBase11 splits and metrics.",
            },
        )
    if args.model == "transrnam_official":
        if embedding_matrix is None:
            raise ValueError("embedding_matrix is required")
        return (
            TransRNAmOfficialModel(
                embedding_matrix,
                len(CLASS_NAMES),
                dropout=args.dropout,
                heads=args.transformer_heads,
                freeze_embedding=not args.train_embedding,
            ),
            "bce",
            {
                "official_repo": "https://github.com/lennylv/TransRNAm",
                "source_type": "official_architecture_reproduction",
                "official_files": ["Scripts/models.py:model_v11_test", "Scripts/util_layers.py:MyTransformerModel"],
                "limitations": "Hard-coded GPU ids and pre-generated JSON/HDF loaders are replaced by the unified RMBase11 loader.",
            },
        )
    if args.model == "evormd_adapted":
        use_context = args.context_mode != "sequence_only"
        if use_context and context_spec is None:
            raise ValueError("context_spec is required")
        context_dim = 0
        if context_spec is not None:
            context_dim = context_spec.dim if args.context_mode == "full" else context_spec.safe_dim
        return (
            EvoRMDAdaptedModel(
                context_dim=context_dim,
                num_classes=len(CLASS_NAMES),
                seq_dim=args.seq_dim,
                meta_dim=args.meta_dim,
                dropout=args.dropout,
                mlp_depth=args.mlp_depth,
                use_context=use_context,
            ),
            "ce",
            {
                "official_repo": "https://github.com/Gardeina/EvoRMD",
                "source_type": "adapted_reproduction_no_rna_fm",
                "official_files": ["Script/model.py:RNAFMDataset/TrainableAttention/MulticlassClassifier"],
                "context_mode": args.context_mode,
                "context_shuffled_within_split": args.shuffle_context,
                "excluded_context_fields": ["cell_list", "seq_type_list", "support_num"] if args.context_mode != "full" else [],
                "limitations": "The fm/RNA-FM package and RNA-FM weights are not available locally; a trainable nucleotide Transformer replaces RNA-FM. Context inputs are controlled by --context-mode.",
            },
        )
    if args.model == "evormd_rnafm_official_core":
        use_context = args.context_mode != "sequence_only"
        if use_context and context_spec is None:
            raise ValueError("context_spec is required")
        context_dim = 0
        if context_spec is not None:
            context_dim = context_spec.dim if args.context_mode == "full" else context_spec.safe_dim
        return (
            EvoRMDRNAFMModel(
                context_dim=context_dim,
                num_classes=len(CLASS_NAMES),
                meta_dim=args.meta_dim,
                mlp_depth=args.mlp_depth,
                use_context=use_context,
                train_rnafm=args.train_rnafm,
            ),
            "ce",
            {
                "official_repo": "https://github.com/Gardeina/EvoRMD",
                "source_type": "official_core_rnafm_strict_split_adaptation",
                "official_files": [
                    "Script/main.py:fm.pretrained.rna_fm_t12",
                    "Script/embedding.py:get_rnafm_with_hier_species_before_attention",
                    "Script/model.py:TrainableAttention/MulticlassClassifier",
                ],
                "rnafm": {
                    "package": "rna-fm==0.2.2",
                    "backbone": "rna_fm_t12",
                    "representation_layer": 12,
                    "representation_dim": 640,
                    "trainable": args.train_rnafm,
                },
                "context_mode": args.context_mode,
                "context_shuffled_within_split": args.shuffle_context,
                "excluded_context_fields": ["cell_list", "seq_type_list", "support_num"] if args.context_mode != "full" else [],
                "limitations": "The released RNA-FM T12 backbone and EvoRMD pre-attention fusion are used. The unified strict-split table does not provide EvoRMD's original organ/subcell/species schema, so available RMBase context fields are encoded through the same pre-attention metadata pathway.",
            },
        )
    raise ValueError(args.model)


def run(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    if args.shuffle_context and args.context_mode == "sequence_only":
        raise ValueError("--shuffle-context requires --context-mode full or safe")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows = read_tsv(Path(args.dataset))
    configure_classes(infer_class_names(rows, args.class_names))
    if args.max_rows:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_rows]
    train_rows, valid_rows, test_rows = split_rows(rows, args.split)
    if not train_rows or not valid_rows or not test_rows:
        raise ValueError(f"Split {args.split} does not contain train/valid/test rows")
    original_train_count = len(train_rows)
    downsampling_info: Dict[str, object] = {"enabled": False}
    if args.model == "evormd_rnafm_official_core" and args.downsample_train_classes.strip():
        train_rows, downsampling_info = downsample_training_rows(
            train_rows,
            args.downsample_train_classes.split(","),
            args.downsample_alpha,
            args.seed,
        )
        print(
            f"training_downsampling before={original_train_count} after={len(train_rows)} "
            f"classes={args.downsample_train_classes} alpha={args.downsample_alpha}",
            flush=True,
        )

    embedding_matrix = None
    trimer_vocab = None
    context_spec = None
    if args.model in {"multirm_official", "transrnam_official"}:
        embedding_dict = load_embedding_dict(Path(args.embedding_path))
        trimer_vocab = build_trimer_vocab(embedding_dict)
        embedding_matrix = build_embedding_matrix(trimer_vocab, embedding_dict, args.seed)
    elif args.context_mode != "sequence_only":
        context_spec = build_context_spec(
            train_rows,
            max_gene_types=args.max_gene_types,
            cell_hash_dim=args.cell_hash_dim,
            seqtype_hash_dim=args.seqtype_hash_dim,
        )

    train_ds = OfficialBenchmarkDataset(
        train_rows,
        args.model,
        args.input_length,
        sequence_column=args.sequence_column,
        trimer_vocab=trimer_vocab,
        context_spec=context_spec,
        context_mode=args.context_mode,
        shuffle_context=args.shuffle_context,
        context_shuffle_seed=args.seed + 1001,
    )
    valid_ds = OfficialBenchmarkDataset(
        valid_rows,
        args.model,
        args.input_length,
        sequence_column=args.sequence_column,
        trimer_vocab=trimer_vocab,
        context_spec=context_spec,
        context_mode=args.context_mode,
        shuffle_context=args.shuffle_context,
        context_shuffle_seed=args.seed + 2001,
    )
    test_ds = OfficialBenchmarkDataset(
        test_rows,
        args.model,
        args.input_length,
        sequence_column=args.sequence_column,
        trimer_vocab=trimer_vocab,
        context_spec=context_spec,
        context_mode=args.context_mode,
        shuffle_context=args.shuffle_context,
        context_shuffle_seed=args.seed + 3001,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    model, objective, source_info = build_model(args, embedding_matrix, context_spec)
    model = model.to(device)
    if objective == "ce":
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_rows, args.class_weight_power, device))
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=effective_num_pos_weight(train_rows, args.beta, args.pos_weight_cap, device))
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    history = []
    best_state = None
    best_score = -1.0
    best_epoch = 0
    start_epoch = 1
    accumulated_training_seconds = 0.0
    epochs_without_improvement = 0
    early_stopped = False
    stop_reason = ""
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_checkpoint_model_state(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint.get("history", [])
        best_state = checkpoint.get("best_state_dict")
        best_score = float(checkpoint.get("best_score", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        accumulated_training_seconds = float(checkpoint.get("accumulated_training_seconds", 0.0))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", max(0, start_epoch - 1 - best_epoch))
        )
        print(f"resumed_from={checkpoint_path} start_epoch={start_epoch} best_score={best_score:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device, objective)
        scheduler.step()
        valid_probs, valid_labels, _, _ = predict(model, valid_loader, device, objective)
        valid_metrics = evaluate(valid_probs, valid_labels)
        accumulated_training_seconds += time.perf_counter() - epoch_started
        score = float(valid_metrics[args.selection_metric])
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation": valid_metrics,
            }
        )
        improved = score > best_score + args.early_stopping_min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in checkpoint_model_state(model).items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": checkpoint_model_state(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_state_dict": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": history,
                "accumulated_training_seconds": accumulated_training_seconds,
                "epochs_without_improvement": epochs_without_improvement,
                "class_names": CLASS_NAMES,
                "source_info": source_info,
            },
            checkpoint_path,
        )
        print(
            f"model={args.model} split={args.split} epoch={epoch} loss={loss:.4f} "
            f"valid_{args.selection_metric}={score:.4f} no_improve={epochs_without_improvement}",
            flush=True,
        )
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            early_stopped = True
            stop_reason = (
                f"validation {args.selection_metric} did not improve by at least "
                f"{args.early_stopping_min_delta} for {args.early_stopping_patience} epochs"
            )
            print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_score={best_score:.4f}: {stop_reason}", flush=True)
            break

    if best_state is not None:
        load_checkpoint_model_state(model, best_state)
    valid_probs, valid_labels, valid_ids, valid_mods = predict(model, valid_loader, device, objective)
    test_probs, test_labels, test_ids, test_mods = predict(model, test_loader, device, objective)
    valid_metrics = evaluate(valid_probs, valid_labels)
    test_metrics = evaluate(test_probs, test_labels)
    training_peak_memory_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    efficiency = measure_inference_efficiency(model, test_loader, device)
    efficiency["training_seconds"] = accumulated_training_seconds
    efficiency["training_peak_memory_bytes"] = training_peak_memory_bytes

    torch.save(
        {
            "model_state_dict": checkpoint_model_state(model),
            "class_names": CLASS_NAMES,
            "model": args.model,
            "objective": objective,
            "trimer_vocab": trimer_vocab,
            "context_spec": asdict(context_spec) if context_spec is not None else None,
            "source_info": source_info,
            "training_args": vars(args),
        },
        out_dir / "model.pt",
    )
    metrics: Dict[str, object] = {
        "dataset": args.dataset,
        "model": args.model,
        "task": f"RMBase_hg38_{len(CLASS_NAMES)}_class_center_site_prediction",
        "classes": CLASS_NAMES,
        "device": str(device),
        "objective": objective,
        "split_column": args.split,
        "row_counts": {
            "train_before_downsampling": original_train_count,
            "train": len(train_rows),
            "valid": len(valid_rows),
            "test": len(test_rows),
        },
        "training_downsampling": downsampling_info,
        "context_shuffle": {
            "enabled": args.shuffle_context,
            "scope": "independent global permutation within each split",
            "preserves_metadata_bundle": True,
        },
        "source_info": source_info,
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "early_stopping": {
            "patience": args.early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "early_stopped": early_stopped,
            "stop_reason": stop_reason,
            "epochs_without_improvement_at_end": epochs_without_improvement,
        },
        "validation": valid_metrics,
        "test": test_metrics,
        "efficiency": efficiency,
        "history": history,
        "caveats": [
            "The train/validation/test loader and metric code are unified across models to isolate the effect of split strictness.",
            "Random-split results are supplementary because random splits can overestimate generalization when similar sequences or nearby sites cross partitions.",
            "Strict-split results use the same MMseqs2-80 cluster partition used by the StructDualRM strict benchmark.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions(out_dir / "valid_predictions.tsv", valid_probs, valid_labels, valid_ids, valid_mods)
    write_predictions(out_dir / "test_predictions.tsv", test_probs, test_labels, test_ids, test_mods)
    write_per_class_metrics(out_dir / "valid_per_class_metrics.tsv", valid_metrics)
    write_per_class_metrics(out_dir / "test_per_class_metrics.tsv", test_metrics)
    write_confusion_matrix(out_dir / "test_confusion_matrix.tsv", test_metrics["confusion_matrix"])  # type: ignore[arg-type]
    (out_dir / "README.md").write_text(
        f"# {args.model}\n\n"
        f"- Dataset: `{args.dataset}`\n"
        f"- Split column: `{args.split}`\n"
        f"- Source type: `{source_info['source_type']}`\n"
        f"- Official repository: {source_info['official_repo']}\n"
        f"- Limitations: {source_info['limitations']}\n"
        f"- Best epoch: {best_epoch}\n"
        f"- Test macro AUPR: {test_metrics['macro_AUPR_ovr']:.6f}\n",
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train official-architecture RMBase11 reproduction baselines.")
    parser.add_argument(
        "--model",
        choices=["multirm_official", "transrnam_official", "evormd_adapted", "evormd_rnafm_official_core"],
        required=True,
    )
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites.tsv")
    parser.add_argument("--class-names", default="", help="Optional comma-separated class order; inferred from class_index if omitted.")
    parser.add_argument("--split", default="split_random")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--embedding-path", default="external/MultiRM-master/Embeddings/embeddings_12RM.pkl")
    parser.add_argument("--input-length", type=int, default=41)
    parser.add_argument("--sequence-column", default="sequence_41nt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-decay", type=float, default=0.98)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.99999)
    parser.add_argument("--pos-weight-cap", type=float, default=500.0)
    parser.add_argument("--train-embedding", action="store_true")
    parser.add_argument("--transformer-heads", type=int, default=6)
    parser.add_argument("--seq-dim", type=int, default=128)
    parser.add_argument("--meta-dim", type=int, default=128)
    parser.add_argument("--mlp-depth", type=int, default=3)
    parser.add_argument("--train-rnafm", action="store_true", help="Fine-tune RNA-FM instead of freezing it.")
    parser.add_argument(
        "--downsample-train-classes",
        default="m6A,m5C",
        help="Comma-separated EvoRMD classes downsampled in the training partition only; empty disables it.",
    )
    parser.add_argument("--downsample-alpha", type=float, default=0.6)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--max-gene-types", type=int, default=24)
    parser.add_argument("--cell-hash-dim", type=int, default=64)
    parser.add_argument("--seqtype-hash-dim", type=int, default=32)
    parser.add_argument(
        "--context-mode",
        choices=["full", "safe", "sequence_only"],
        default="full",
        help="EvoRMD-adapted inputs: full metadata, safe region/gene context, or sequence only.",
    )
    parser.add_argument(
        "--shuffle-context",
        action="store_true",
        help="Globally permute complete context vectors within each split while preserving sequences and labels.",
    )
    parser.add_argument("--selection-metric", choices=["macro_AUPR_ovr", "macro_F1", "MCC", "balanced_accuracy"], default="macro_AUPR_ovr")
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
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
