#!/usr/bin/env python
"""Train a class-specific structure-gated motif model on RMBase11.

V4 keeps the strong motif/k-mer sequence backbone from V3, but changes RNAfold
integration into a per-class logit residual:

    logits_c = sequence_logits_c + gate_c(x) * structure_delta_c(x)

This lets structure help modification types that are structure-sensitive while
keeping the sequence-dominated classes close to the sequence-only model.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

from train_motif_struct_fusion_v3 import MotifSequenceBranch, make_kmer_ids
from train_paired_gat_rbp_fusion import (
    CLASS_NAMES,
    CenterAttentionPool,
    FocalCrossEntropy,
    NODE_FEATURE_FIELDS,
    PairAwareGATLayer,
    RBPBranch,
    RMBaseFusionDataset,
    STRUCT_FIELDS,
    StructureBranch,
    class_weights_from_rows,
    collate_batch,
    configure_pair_candidates,
    configure_classes,
    encode_sequence,
    evaluate,
    infer_class_names,
    load_rbp_features,
    load_rnafold,
    move_batch,
    read_tsv,
    set_seed,
    split_rows,
    write_confusion_matrix,
    write_gate_summary,
    write_per_class_metrics,
    write_predictions,
)


@dataclass
class MotifStructV4Config:
    input_length: int = 41
    num_classes: int = len(CLASS_NAMES)
    d_model: int = 128
    gat_layers: int = 2
    dropout: float = 0.3
    rbp_dim: int = 0
    rbp_pos_shape: Tuple[int, int, int] = (0, 0, 0)
    use_structure: bool = True
    use_rbp: bool = False
    use_rbp_positional: bool = False
    use_position: bool = True
    structure_uses_sequence: bool = False
    structure_encoder: str = "gat"
    rbp_pos_encoder: str = "attention"
    structure_scale: float = 0.5
    rbp_scale: float = 0.05
    rbp_pos_scale: float = 0.05
    gate_init_logit: float = -2.0
    rbp_pos_gate_init_logit: float = -2.0
    modality_dropout_struct: float = 0.0
    modality_dropout_rbp: float = 0.0
    seq_encoder: str = "motif"
    seq_residual_encoder: str = "none"
    seq_residual_focus_indices: Tuple[int, ...] = ()
    seq_residual_scale: float = 0.1
    seq_residual_gate_init_logit: float = -2.0
    multi_length_columns: Tuple[str, ...] = ("sequence_41nt", "sequence_101nt", "sequence_201nt")
    multi_length_lengths: Tuple[int, ...] = (41, 101, 201)
    multi_length_base_encoder: str = "motif"
    multi_length_share_encoder: bool = False
    m1a_aux_weight: float = 0.0
    m1a_aux_pos_weight: float = 1.0
    afamily_classes: Tuple[str, ...] = ()
    afamily_indices: Tuple[int, ...] = ()
    afamily_aux_weight: float = 0.0
    afamily_residual_scale: float = 0.0
    afamily_fusion_mode: str = "legacy"
    afamily_m1a_expert_scale: float = 0.0
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff_mult: int = 4
    max_pair_candidates: int = 1
    confidence_gated_structure: bool = False
    structure_confidence_floor: float = 0.1
    struct_fields: Tuple[str, ...] = tuple(STRUCT_FIELDS)
    node_feature_fields: Tuple[str, ...] = tuple(NODE_FEATURE_FIELDS)


class EmptyStructureFeatures(Mapping[str, Mapping[str, str]]):
    """Return zero-valued structure records for sequence-only controls."""

    def __getitem__(self, key: str) -> Mapping[str, str]:
        return {}

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def __contains__(self, key: object) -> bool:
        return True


class TransformerMotifSequenceBranch(nn.Module):
    """K-mer token encoder with an optional motif CNN stem and Transformer layers."""

    def __init__(
        self,
        d_model: int,
        length: int,
        dropout: float,
        use_position: bool,
        num_layers: int,
        num_heads: int,
        ff_mult: int,
        use_motif_convs: bool,
    ):
        super().__init__()
        if d_model % max(1, num_heads) != 0:
            raise ValueError(f"d_model={d_model} must be divisible by transformer_heads={num_heads}")
        self.length = length
        self.use_position = use_position
        self.use_motif_convs = use_motif_convs
        self.base_emb = nn.Embedding(5, d_model, padding_idx=0)
        self.k2_emb = nn.Embedding(25, d_model, padding_idx=0)
        self.k3_emb = nn.Embedding(125, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(length, d_model)
        self.dist_emb = nn.Embedding(length, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        if use_motif_convs:
            self.convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                    for k in (3, 5, 7, 9)
                ]
            )
            self.conv_project = nn.Sequential(
                nn.Linear(d_model * 4, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
        else:
            self.convs = nn.ModuleList()
            self.conv_project = nn.Identity()
        encoder_kwargs = {
            "d_model": d_model,
            "nhead": max(1, num_heads),
            "dim_feedforward": d_model * max(1, ff_mult),
            "dropout": dropout,
            "activation": "gelu",
        }
        try:
            encoder_layer = nn.TransformerEncoderLayer(**encoder_kwargs, batch_first=True)
            self.transformer_batch_first = True
        except TypeError:
            encoder_layer = nn.TransformerEncoderLayer(**encoder_kwargs)
            self.transformer_batch_first = False
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
        self.output_norm = nn.LayerNorm(d_model)
        self.pool = CenterAttentionPool(d_model)
        self.project = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        x = self.base_emb(seq.long()) + self.k2_emb(make_kmer_ids(seq, 2)) + self.k3_emb(make_kmer_ids(seq, 3))
        if self.use_position:
            pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, length)
            dist = torch.abs(pos - (length // 2))
            x = x + 0.5 * (self.pos_emb(pos) + self.dist_emb(dist))
        x = self.input_norm(x)
        if self.use_motif_convs:
            conv_input = x.transpose(1, 2)
            conv_h = torch.cat([conv(conv_input).transpose(1, 2) for conv in self.convs], dim=-1)
            x = x + self.conv_project(conv_h)
        if self.transformer_batch_first:
            h = self.transformer(x)
        else:
            h = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        h = self.output_norm(h)
        center = h[:, length // 2, :]
        local = h[:, max(0, length // 2 - 5) : min(length, length // 2 + 6), :].mean(dim=1)
        pooled = self.pool(h)
        maxed = h.max(dim=1).values
        return self.project(torch.cat([center, local, pooled, maxed], dim=-1))


class ExtendedKmerMotifSequenceBranch(nn.Module):
    """Motif CNN + BiGRU encoder with 2/3/4/5-mer token embeddings."""

    def __init__(self, d_model: int, length: int, dropout: float, use_position: bool):
        super().__init__()
        self.length = length
        self.use_position = use_position
        self.kmers = (2, 3, 4, 5)
        self.base_emb = nn.Embedding(5, d_model, padding_idx=0)
        self.kmer_embs = nn.ModuleList([nn.Embedding(5**k, d_model, padding_idx=0) for k in self.kmers])
        self.pos_emb = nn.Embedding(length, d_model)
        self.dist_emb = nn.Embedding(length, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for k in (3, 5, 7, 9)
            ]
        )
        self.conv_project = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.bigru = nn.GRU(d_model, d_model // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.pool = CenterAttentionPool(d_model)
        self.project = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def _token_embed(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.base_emb(seq.long())
        for k, emb in zip(self.kmers, self.kmer_embs):
            x = x + emb(make_kmer_ids(seq, k))
        return x

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        x = self._token_embed(seq)
        if self.use_position:
            pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, length)
            dist = torch.abs(pos - (length // 2))
            x = x + 0.5 * (self.pos_emb(pos) + self.dist_emb(dist))
        x = self.input_norm(x)
        conv_input = x.transpose(1, 2)
        conv_h = torch.cat([conv(conv_input).transpose(1, 2) for conv in self.convs], dim=-1)
        h = self.conv_project(conv_h)
        h, _ = self.bigru(h)
        center = h[:, length // 2, :]
        local = h[:, max(0, length // 2 - 5) : min(length, length // 2 + 6), :].mean(dim=1)
        pooled = self.pool(h)
        maxed = h.max(dim=1).values
        return self.project(torch.cat([center, local, pooled, maxed], dim=-1))


class DilatedConvBlock(nn.Module):
    """Residual dilated 1D convolution block for local motif context expansion."""

    def __init__(self, d_model: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.net(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(h + y)


class DilatedMotifSequenceBranch(ExtendedKmerMotifSequenceBranch):
    """Extended k-mer motif encoder with residual dilated CNN blocks before BiGRU."""

    def __init__(self, d_model: int, length: int, dropout: float, use_position: bool):
        super().__init__(d_model, length, dropout, use_position)
        self.dilated_blocks = nn.ModuleList(
            [DilatedConvBlock(d_model, dilation=dilation, dropout=dropout) for dilation in (1, 2, 4)]
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        x = self._token_embed(seq)
        if self.use_position:
            pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, length)
            dist = torch.abs(pos - (length // 2))
            x = x + 0.5 * (self.pos_emb(pos) + self.dist_emb(dist))
        x = self.input_norm(x)
        conv_input = x.transpose(1, 2)
        conv_h = torch.cat([conv(conv_input).transpose(1, 2) for conv in self.convs], dim=-1)
        h = self.conv_project(conv_h)
        for block in self.dilated_blocks:
            h = block(h)
        h, _ = self.bigru(h)
        center = h[:, length // 2, :]
        local = h[:, max(0, length // 2 - 5) : min(length, length // 2 + 6), :].mean(dim=1)
        pooled = self.pool(h)
        maxed = h.max(dim=1).values
        return self.project(torch.cat([center, local, pooled, maxed], dim=-1))


def build_sequence_branch_by_name(cfg: MotifStructV4Config, encoder_name: str, input_length: int) -> nn.Module:
    if encoder_name == "motif":
        return MotifSequenceBranch(cfg.d_model, input_length, cfg.dropout, cfg.use_position)
    if encoder_name == "kmer_extended":
        return ExtendedKmerMotifSequenceBranch(cfg.d_model, input_length, cfg.dropout, cfg.use_position)
    if encoder_name == "dilated":
        return DilatedMotifSequenceBranch(cfg.d_model, input_length, cfg.dropout, cfg.use_position)
    if encoder_name in {"transformer", "hybrid"}:
        return TransformerMotifSequenceBranch(
            cfg.d_model,
            input_length,
            cfg.dropout,
            cfg.use_position,
            cfg.transformer_layers,
            cfg.transformer_heads,
            cfg.transformer_ff_mult,
            use_motif_convs=encoder_name == "hybrid",
        )
    raise ValueError(f"Unsupported sequence encoder: {encoder_name}")


class MultiLengthSequenceBranch(nn.Module):
    """Class-specific evidence fusion over 41/101/201 nt sequence windows."""

    def __init__(self, cfg: MotifStructV4Config):
        super().__init__()
        if len(cfg.multi_length_lengths) != len(cfg.multi_length_columns):
            raise ValueError("multi_length_lengths and multi_length_columns must have the same length.")
        if not cfg.multi_length_lengths:
            raise ValueError("At least one multi-length sequence branch is required.")
        self.lengths = tuple(int(length) for length in cfg.multi_length_lengths)
        self.keys = tuple(f"seq_len_{length}" for length in self.lengths)
        self.num_lengths = len(self.lengths)
        self.num_classes = cfg.num_classes
        self.share_encoder = bool(cfg.multi_length_share_encoder)
        max_length = max(self.lengths)
        if self.share_encoder:
            shared = build_sequence_branch_by_name(cfg, cfg.multi_length_base_encoder, max_length)
            self.branches = nn.ModuleList([shared])
        else:
            self.branches = nn.ModuleList(
                [build_sequence_branch_by_name(cfg, cfg.multi_length_base_encoder, length) for length in self.lengths]
            )
        self.length_heads = nn.ModuleList([nn.Linear(cfg.d_model, cfg.num_classes) for _ in self.lengths])
        self.length_gate_net = nn.Sequential(
            nn.Linear(cfg.d_model * self.num_lengths, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes * self.num_lengths),
        )
        nn.init.zeros_(self.length_gate_net[-1].weight)
        nn.init.zeros_(self.length_gate_net[-1].bias)
        self.class_length_prior = nn.Parameter(torch.zeros(cfg.num_classes, self.num_lengths))
        self.fuse_project = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def _branch(self, index: int) -> nn.Module:
        return self.branches[0] if self.share_encoder else self.branches[index]

    def forward(
        self,
        seq: torch.Tensor,
        seq_multi: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        seq_multi = seq_multi or {}
        z_by_length: List[torch.Tensor] = []
        logits_by_length: List[torch.Tensor] = []
        for index, (length, key) in enumerate(zip(self.lengths, self.keys)):
            seq_tensor = seq_multi.get(key)
            if seq_tensor is None:
                if int(seq.shape[1]) == int(length):
                    seq_tensor = seq
                else:
                    raise KeyError(f"Missing multi-length sequence tensor '{key}' for length={length}.")
            z = self._branch(index)(seq_tensor)
            z_by_length.append(z)
            logits_by_length.append(self.length_heads[index](z))
        z_stack = torch.stack(z_by_length, dim=1)
        branch_logits = torch.stack(logits_by_length, dim=-1)
        gate_input = torch.cat(z_by_length, dim=-1)
        gate_logits = self.length_gate_net(gate_input).view(
            seq.shape[0],
            self.num_classes,
            self.num_lengths,
        )
        gate_logits = gate_logits + self.class_length_prior.view(1, self.num_classes, self.num_lengths)
        length_gate = torch.softmax(gate_logits, dim=-1)
        seq_logits = (branch_logits * length_gate).sum(dim=-1)
        sample_length_gate = length_gate.mean(dim=1)
        z_fused = torch.einsum("bl,bld->bd", sample_length_gate, z_stack)
        return {
            "z": self.fuse_project(z_fused),
            "logits": seq_logits,
            "length_gate": length_gate,
            "branch_logits": branch_logits,
        }


def build_sequence_branch(cfg: MotifStructV4Config) -> nn.Module:
    if cfg.seq_encoder == "multi_length":
        return MultiLengthSequenceBranch(cfg)
    return build_sequence_branch_by_name(cfg, cfg.seq_encoder, cfg.input_length)


class ClassSpecificStructureGate(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float, gate_init_logit: float):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.class_bias = nn.Parameter(torch.full((num_classes,), float(gate_init_logit)))

    def forward(self, interaction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.delta(interaction)
        gate = torch.sigmoid(self.gate(interaction) + self.class_bias.view(1, -1))
        return delta, gate


class ClassSpecificSequenceResidualGate(nn.Module):
    """Class-specific logit residual from a secondary sequence encoder."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float,
        gate_init_logit: float,
        focus_indices: Sequence[int],
    ):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.class_bias = nn.Parameter(torch.full((num_classes,), float(gate_init_logit)))
        mask = torch.ones(num_classes, dtype=torch.float32)
        if focus_indices:
            mask.zero_()
            for idx in focus_indices:
                if 0 <= int(idx) < num_classes:
                    mask[int(idx)] = 1.0
        self.register_buffer("class_mask", mask)

    def forward(self, interaction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = self.delta(interaction)
        gate = torch.sigmoid(self.gate(interaction) + self.class_bias.view(1, -1))
        mask = self.class_mask.view(1, -1).to(device=interaction.device, dtype=interaction.dtype)
        return delta * mask, gate * mask


class ClassSpecificModalityGate(nn.Module):
    """Class-level priors plus sample-specific gates for auxiliary modalities."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float,
        struct_init_logit: float,
        rbp_init_logit: float,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.struct_prior = nn.Parameter(torch.full((num_classes,), float(struct_init_logit)))
        self.rbp_prior = nn.Parameter(torch.full((num_classes,), float(rbp_init_logit)))
        self.struct_gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.rbp_gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.struct_gate_net[-1].weight)
        nn.init.zeros_(self.struct_gate_net[-1].bias)
        nn.init.zeros_(self.rbp_gate_net[-1].weight)
        nn.init.zeros_(self.rbp_gate_net[-1].bias)

    def structure_gate(self, h_seq: torch.Tensor, h_struct: torch.Tensor) -> torch.Tensor:
        dynamic = self.struct_gate_net(torch.cat([h_seq, h_struct], dim=-1))
        return torch.sigmoid(dynamic + self.struct_prior.view(1, -1))

    def rbp_gate(self, h_main: torch.Tensor, h_rbp_cls: torch.Tensor) -> torch.Tensor:
        batch, num_classes, _ = h_rbp_cls.shape
        h_main_cls = h_main.unsqueeze(1).expand(batch, num_classes, -1)
        dynamic = self.rbp_gate_net(torch.cat([h_main_cls, h_rbp_cls], dim=-1)).squeeze(-1)
        return torch.sigmoid(dynamic + self.rbp_prior.view(1, -1))


class StructureOnlyBranch(nn.Module):
    """RNAfold branch without nucleotide identity leakage.

    The previous GAT structure branch included base embeddings.  That made the
    shuffled-structure control partly sequence-aware.  This branch keeps only
    RNAfold-derived node/edge/scalar signals: paired flags, base-pair graph,
    center distance and window-level structural statistics.
    """

    def __init__(self, d_model: int, length: int, layers: int, dropout: float, use_position: bool):
        super().__init__()
        self.length = length
        self.use_position = use_position
        self.paired_proj = nn.Linear(max(1, len(NODE_FEATURE_FIELDS)), d_model)
        self.pos_emb = nn.Embedding(length, d_model)
        self.dist_emb = nn.Embedding(length, d_model)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = paired_flags.shape[:2]
        device = paired_flags.device
        node_features = paired_flags if paired_flags.dim() == 3 else paired_flags.unsqueeze(-1)
        h = self.paired_proj(node_features.float())
        if self.use_position:
            pos = torch.arange(length, device=device).unsqueeze(0).expand(batch, length)
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


class CapacityMatchedStructureCNNLayer(nn.Module):
    """Generic local CNN block with approximately the same capacity as one GAT layer."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.local = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.mix = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        update = self.mix(F.gelu(self.local(h.transpose(1, 2)))).transpose(1, 2)
        h = self.norm1(h + self.dropout(update))
        return self.norm2(h + self.dropout(self.ffn(h)))


class CapacityMatchedStructureCNN(nn.Module):
    """Capacity control that sees RNAfold features but has no base-pair graph attention."""

    def __init__(self, d_model: int, length: int, layers: int, dropout: float, use_position: bool):
        super().__init__()
        self.length = length
        self.use_position = use_position
        self.paired_proj = nn.Linear(max(1, len(NODE_FEATURE_FIELDS)), d_model)
        self.pos_emb = nn.Embedding(length, d_model)
        self.dist_emb = nn.Embedding(length, d_model)
        self.scalar_proj = nn.Sequential(nn.Linear(len(STRUCT_FIELDS), d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.layers = nn.ModuleList([CapacityMatchedStructureCNNLayer(d_model, dropout) for _ in range(layers)])
        self.pool = CenterAttentionPool(d_model)
        self.project = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(
        self,
        seq: torch.Tensor,
        paired_flags: torch.Tensor,
        partner: torch.Tensor,
        struct: torch.Tensor,
        pair_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del seq, partner, pair_weights
        batch, length = paired_flags.shape[:2]
        device = paired_flags.device
        node_features = paired_flags if paired_flags.dim() == 3 else paired_flags.unsqueeze(-1)
        h = self.paired_proj(node_features.float())
        if self.use_position:
            pos = torch.arange(length, device=device).unsqueeze(0).expand(batch, length)
            dist = torch.abs(pos - (length // 2))
            h = h + self.pos_emb(pos) + self.dist_emb(dist)
        for layer in self.layers:
            h = layer(h)
        center = h[:, length // 2, :]
        local = h[:, max(0, length // 2 - 10) : min(length, length // 2 + 11), :].mean(dim=1)
        pooled = self.pool(h)
        scalar = self.scalar_proj(struct)
        dummy_attention = torch.zeros(batch, length, 4, dtype=h.dtype, device=device)
        return self.project(torch.cat([center, local, pooled, scalar], dim=-1)), dummy_attention


def load_rnafold_filtered(path: Path, include_site_ids: Optional[set[str]] = None) -> Dict[str, Dict[str, str]]:
    if include_site_ids is None:
        return load_rnafold(path)
    values: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "site_id" not in reader.fieldnames:
            raise ValueError(f"RNAfold TSV must contain site_id: {path}")
        for row in reader:
            site_id = row.get("site_id", "")
            if site_id in include_site_ids:
                values[site_id] = row
    return values


class RBPPositionalMemmap(Mapping[str, np.ndarray]):
    """Read-only site-indexed view over a binary RBP positional feature cache."""

    def __init__(self, directory: Path, include_site_ids: Optional[set[str]] = None):
        self.directory = directory
        self.values = np.load(directory / "values.npy", mmap_mode="r")
        self.index: Dict[str, int] = {}
        with (directory / "site_ids.txt").open("r", encoding="utf-8") as handle:
            for idx, raw in enumerate(handle):
                site_id = raw.rstrip("\r\n")
                if include_site_ids is None or site_id in include_site_ids:
                    self.index[site_id] = idx
        if self.values.ndim != 4:
            raise ValueError(f"RBP memmap must have shape [site,RBP,bin,feature]: {self.values.shape}")

    def __getitem__(self, key: str) -> np.ndarray:
        return self.values[self.index[key]]

    def __iter__(self):
        return iter(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def close(self) -> None:
        mmap = getattr(self.values, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass


def load_rbp_positional_features(
    path: Optional[str],
    include_site_ids: Optional[set[str]] = None,
) -> Tuple[Mapping[str, np.ndarray], Tuple[int, int, int], Dict[str, object]]:
    if not path:
        return {}, (0, 0, 0), {}
    rbp_path = Path(path)
    if not rbp_path.exists():
        raise FileNotFoundError(f"RBP positional feature file not found: {rbp_path}")
    if rbp_path.is_dir():
        meta_path = rbp_path / "metadata.json"
        if not meta_path.is_file() or not (rbp_path / "values.npy").is_file() or not (rbp_path / "site_ids.txt").is_file():
            raise ValueError(f"Incomplete RBP positional memmap cache: {rbp_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        shape_values = metadata.get("shape", [])
        shape = tuple(int(value) for value in shape_values)
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError(f"Invalid RBP positional shape in {meta_path}: {shape_values}")
        values = RBPPositionalMemmap(rbp_path, include_site_ids)
        if tuple(values.values.shape[1:]) != shape:
            raise ValueError(
                f"RBP positional memmap shape mismatch: values={values.values.shape[1:]} metadata={shape}"
            )
        metadata["storage"] = "numpy_memmap"
        return values, shape, metadata
    meta_path = rbp_path.with_suffix(rbp_path.suffix + ".json")
    metadata: Dict[str, object] = {}
    rbps: List[str] = []
    bins: List[Mapping[str, object]] = []
    features: List[str] = []
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        rbps = [str(x) for x in metadata.get("rbps", [])]
        bins = [x for x in metadata.get("bins", []) if isinstance(x, Mapping)]
        features = [str(x) for x in metadata.get("features", [])]
    with rbp_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "site_id" not in reader.fieldnames:
            raise ValueError(f"RBP positional TSV must contain site_id: {rbp_path}")
        fields = [field for field in reader.fieldnames if field != "site_id"]
        if not rbps or not bins or not features:
            parsed = [field.split("__") for field in fields if field.startswith("rbp_pos__")]
            rbps = sorted({parts[1] for parts in parsed if len(parts) == 4})
            bin_labels = sorted({parts[2] for parts in parsed if len(parts) == 4})
            features = sorted({parts[3] for parts in parsed if len(parts) == 4})
            bins = [{"label": label} for label in bin_labels]
        columns = [
            f"rbp_pos__{rbp}__{bin_info['label']}__{feature}"
            for rbp in rbps
            for bin_info in bins
            for feature in features
        ]
        shape = (len(rbps), len(bins), len(features))
        values: Dict[str, np.ndarray] = {}
        for row in reader:
            site_id = row.get("site_id", "")
            if include_site_ids is not None and site_id not in include_site_ids:
                continue
            arr = np.asarray([float(row.get(column, 0.0) or 0.0) for column in columns], dtype=np.float32)
            values[site_id] = arr.reshape(shape)
    metadata.setdefault("rbps", rbps)
    metadata.setdefault("bins", bins)
    metadata.setdefault("features", features)
    metadata.setdefault("shape", list(shape))
    return values, shape, metadata


def load_teacher_m1a_scores(path: Optional[str], field: str) -> Dict[str, float]:
    if not path:
        return {}
    teacher_path = Path(path)
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher prediction file not found: {teacher_path}")
    with teacher_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "site_id" not in reader.fieldnames or field not in reader.fieldnames:
            raise ValueError(f"Teacher TSV must contain site_id and {field}: {teacher_path}")
        return {
            str(row["site_id"]): float(row[field])
            for row in reader
        }


def multi_length_key(length: int) -> str:
    return f"seq_len_{int(length)}"


def build_multi_length_tensors(
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
    lengths: Sequence[int],
) -> Dict[str, torch.Tensor]:
    if len(columns) != len(lengths):
        raise ValueError("multi-length columns and lengths must have the same number of items.")
    tensors: Dict[str, torch.Tensor] = {}
    for column, length in zip(columns, lengths):
        missing = [str(row.get("site_id", "")) for row in rows if not row.get(column)]
        if missing:
            raise ValueError(f"Missing {column} for {len(missing)} rows; first site_id={missing[0]}")
        tensors[multi_length_key(int(length))] = torch.stack(
            [encode_sequence(str(row.get(column, "")), int(length)) for row in rows]
        )
    return tensors


def build_rbp_bin_permutations(num_rows: int, num_bins: int, seed: int) -> torch.Tensor:
    if num_bins < 1:
        return torch.empty((num_rows, 0), dtype=torch.int8)
    rng = np.random.default_rng(seed + 104729)
    random_keys = rng.random((num_rows, num_bins), dtype=np.float32)
    return torch.as_tensor(np.argsort(random_keys, axis=1), dtype=torch.int8)


class RMBaseFusionPositionalDataset(RMBaseFusionDataset):
    def __init__(
        self,
        *args,
        rbp_pos_by_id: Optional[Mapping[str, np.ndarray]] = None,
        rbp_pos_shape: Tuple[int, int, int] = (0, 0, 0),
        shuffle_rbp_bins: bool = False,
        **kwargs,
    ):
        seed = int(kwargs.get("seed", 42))
        super().__init__(*args, **kwargs)
        values = np.zeros((len(self.rows),) + tuple(rbp_pos_shape), dtype=np.float32)
        if rbp_pos_by_id and all(dim > 0 for dim in rbp_pos_shape):
            for idx, rbp_id in enumerate(self.rbp_ids):
                arr = rbp_pos_by_id.get(rbp_id)
                if arr is not None and tuple(arr.shape) == tuple(rbp_pos_shape):
                    values[idx] = arr
        self.rbp_pos_tensor = torch.as_tensor(values, dtype=torch.float32)
        self.rbp_bin_permutations = (
            build_rbp_bin_permutations(len(self.rows), rbp_pos_shape[1], seed)
            if shuffle_rbp_bins
            else None
        )

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = super().__getitem__(idx)
        rbp_pos = self.rbp_pos_tensor[idx]
        if self.rbp_bin_permutations is not None:
            rbp_pos = rbp_pos[:, self.rbp_bin_permutations[idx].long(), :]
        item["rbp_pos"] = rbp_pos
        return item


class RMBaseFusionMultiLengthDataset(RMBaseFusionDataset):
    def __init__(
        self,
        *args,
        multi_length_columns: Sequence[str],
        multi_length_lengths: Sequence[int],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.multi_length_tensors = build_multi_length_tensors(
            self.rows,
            multi_length_columns,
            multi_length_lengths,
        )

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = super().__getitem__(idx)
        for key, tensor in self.multi_length_tensors.items():
            item[key] = tensor[idx]
        return item


class RMBaseFusionMultiLengthPositionalDataset(RMBaseFusionPositionalDataset):
    def __init__(
        self,
        *args,
        multi_length_columns: Sequence[str],
        multi_length_lengths: Sequence[int],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.multi_length_tensors = build_multi_length_tensors(
            self.rows,
            multi_length_columns,
            multi_length_lengths,
        )

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = super().__getitem__(idx)
        for key, tensor in self.multi_length_tensors.items():
            item[key] = tensor[idx]
        return item


class TeacherM1AScoreDataset(Dataset):
    def __init__(self, base_dataset: Dataset, rows: Sequence[Mapping[str, str]], scores_by_id: Mapping[str, float]):
        self.base_dataset = base_dataset
        self.teacher_scores = torch.as_tensor(
            [float(scores_by_id[str(row.get("site_id", ""))]) for row in rows],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = dict(self.base_dataset[idx])  # type: ignore[index]
        item["teacher_m1a"] = self.teacher_scores[idx]
        return item


def collate_batch_v4(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    tensor_keys = ["seq", "paired_flags", "pair_partner", "pair_weights", "struct", "rbp", "label"]
    if "rbp_pos" in batch[0]:
        tensor_keys.append("rbp_pos")
    if "teacher_m1a" in batch[0]:
        tensor_keys.append("teacher_m1a")
    tensor_keys.extend(sorted(key for key in batch[0] if str(key).startswith("seq_len_")))
    out: Dict[str, object] = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}  # type: ignore[arg-type]
    out["site_id"] = [str(item["site_id"]) for item in batch]
    out["mod_type"] = [str(item["mod_type"]) for item in batch]
    return out


class RBPPositionalBranch(nn.Module):
    def __init__(
        self,
        num_rbps: int,
        num_bins: int,
        rbp_feat_dim: int,
        d_model: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.num_rbps = num_rbps
        self.num_bins = num_bins
        self.rbp_feat_dim = rbp_feat_dim
        self.rbp_emb = nn.Embedding(max(num_rbps, 1), d_model)
        self.bin_emb = nn.Embedding(max(num_bins, 1), d_model)
        self.feat_proj = nn.Sequential(
            nn.Linear(max(rbp_feat_dim, 1), d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.class_queries = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        self.attn_proj = nn.Linear(d_model, d_model)

    def forward(self, x_rbp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, num_rbps, num_bins, feat_dim = x_rbp.shape
        device = x_rbp.device
        rbp_ids = torch.arange(num_rbps, device=device)
        bin_ids = torch.arange(num_bins, device=device)
        rbp_e = self.rbp_emb(rbp_ids).view(1, num_rbps, 1, -1)
        bin_e = self.bin_emb(bin_ids).view(1, 1, num_bins, -1)
        feat_e = self.feat_proj(x_rbp.float())
        tokens = rbp_e + bin_e + feat_e
        tokens = tokens.view(batch, num_rbps * num_bins, -1)
        scores = torch.einsum("cd,bnd->bcn", self.class_queries, self.attn_proj(tokens)) / (tokens.shape[-1] ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        h_rbp_cls = torch.einsum("bcn,bnd->bcd", attn, tokens)
        return h_rbp_cls, attn.view(batch, self.class_queries.shape[0], num_rbps, num_bins)


class CapacityMatchedRBPMLPBranch(nn.Module):
    """Parameter-matched RBP control without class-specific token attention.

    It retains the same RBP/bin embeddings, feature projection, class vectors
    and d_model-by-d_model projection as ``RBPPositionalBranch``.  The only
    substantive change is mean pooling followed by an MLP projection, so any
    difference cannot be attributed to a larger parameter budget.
    """

    def __init__(
        self,
        num_rbps: int,
        num_bins: int,
        rbp_feat_dim: int,
        d_model: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.num_rbps = num_rbps
        self.num_bins = num_bins
        self.rbp_emb = nn.Embedding(max(num_rbps, 1), d_model)
        self.bin_emb = nn.Embedding(max(num_bins, 1), d_model)
        self.feat_proj = nn.Sequential(
            nn.Linear(max(rbp_feat_dim, 1), d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.class_vectors = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        self.mlp_proj = nn.Linear(d_model, d_model)

    def forward(self, x_rbp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, num_rbps, num_bins, _ = x_rbp.shape
        device = x_rbp.device
        rbp_ids = torch.arange(num_rbps, device=device)
        bin_ids = torch.arange(num_bins, device=device)
        tokens = self.feat_proj(x_rbp.float())
        tokens = tokens + self.rbp_emb(rbp_ids).view(1, num_rbps, 1, -1)
        tokens = tokens + self.bin_emb(bin_ids).view(1, 1, num_bins, -1)
        pooled = self.mlp_proj(tokens.mean(dim=(1, 2)))
        h_rbp_cls = pooled.unsqueeze(1) + self.class_vectors.unsqueeze(0)
        uniform = torch.full(
            (batch, self.class_vectors.shape[0], num_rbps, num_bins),
            1.0 / max(num_rbps * num_bins, 1),
            dtype=x_rbp.dtype,
            device=device,
        )
        return h_rbp_cls, uniform


class AFamilyEarlyFusionHead(nn.Module):
    """EvoRMD-style class-token fusion for A-family local correction.

    The head fuses sequence, RNAfold structure and RBP positional evidence
    before the A-family decision.  Its residual is mean-centered across the
    A-family classes in the caller so it mainly redistributes probability
    within A-related modifications instead of globally boosting all A classes.
    """

    modality_labels = ("sequence", "structure", "rbp_positional")

    def __init__(
        self,
        d_model: int,
        num_afamily: int,
        dropout: float,
        m1a_local_index: int = -1,
        m1a_expert_scale: float = 0.0,
    ):
        super().__init__()
        self.num_afamily = num_afamily
        self.m1a_local_index = int(m1a_local_index)
        self.m1a_expert_scale = float(m1a_expert_scale)
        self.class_queries = nn.Parameter(torch.randn(num_afamily, d_model) * 0.02)
        self.class_emb = nn.Parameter(torch.randn(num_afamily, d_model) * 0.02)
        self.modality_emb = nn.Parameter(torch.randn(len(self.modality_labels), d_model) * 0.02)
        self.seq_proj = nn.Linear(d_model, d_model)
        self.struct_proj = nn.Linear(d_model, d_model)
        self.rbp_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.local_logit = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.m1a_expert = (
            nn.Sequential(
                nn.Linear(d_model * 6, d_model * 2),
                nn.GELU(),
                nn.LayerNorm(d_model * 2),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            if 0 <= self.m1a_local_index < num_afamily and self.m1a_expert_scale > 0
            else None
        )

    def forward(
        self,
        z_seq: torch.Tensor,
        z_struct: torch.Tensor,
        h_rbp_afamily: torch.Tensor,
        use_structure: bool,
        use_rbp_positional: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = z_seq.shape[0]
        seq_token = self.seq_proj(z_seq).unsqueeze(1).expand(batch, self.num_afamily, -1)
        struct_token = self.struct_proj(z_struct).unsqueeze(1).expand(batch, self.num_afamily, -1)
        rbp_token = self.rbp_proj(h_rbp_afamily)
        tokens = torch.stack([seq_token, struct_token, rbp_token], dim=2)
        tokens = tokens + self.class_emb.view(1, self.num_afamily, 1, -1)
        tokens = tokens + self.modality_emb.view(1, 1, len(self.modality_labels), -1)

        keys = self.key_proj(tokens)
        values = self.value_proj(tokens)
        query = self.class_queries.view(1, self.num_afamily, 1, -1)
        scores = (query * keys).sum(dim=-1) / (keys.shape[-1] ** 0.5)
        available = torch.tensor(
            [1.0, 1.0 if use_structure else 0.0, 1.0 if use_rbp_positional else 0.0],
            dtype=scores.dtype,
            device=scores.device,
        ).view(1, 1, -1)
        scores = scores.masked_fill(available < 0.5, -1e4)
        attention = torch.softmax(scores, dim=-1)
        fused = torch.sum(attention.unsqueeze(-1) * values, dim=2)
        hidden = self.fuse(fused + self.class_emb.view(1, self.num_afamily, -1))
        local_logits = self.local_logit(hidden).squeeze(-1)
        if self.m1a_expert is not None:
            idx = self.m1a_local_index
            seq_m1a = seq_token[:, idx]
            struct_m1a = struct_token[:, idx] if use_structure else torch.zeros_like(seq_m1a)
            rbp_m1a = rbp_token[:, idx] if use_rbp_positional else torch.zeros_like(seq_m1a)
            expert_features = torch.cat(
                [
                    hidden[:, idx],
                    seq_m1a,
                    struct_m1a,
                    rbp_m1a,
                    seq_m1a * struct_m1a,
                    seq_m1a * rbp_m1a,
                ],
                dim=-1,
            )
            m1a_delta = self.m1a_expert(expert_features).squeeze(-1)
            local_logits = local_logits.clone()
            local_logits[:, idx] = local_logits[:, idx] + self.m1a_expert_scale * m1a_delta
        return local_logits, attention


class MotifStructFusionV4(nn.Module):
    def __init__(self, cfg: MotifStructV4Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        c = cfg.num_classes
        self.seq_branch = build_sequence_branch(cfg)
        self.use_multi_length_sequence = cfg.seq_encoder == "multi_length"
        self.multi_length_keys = tuple(multi_length_key(length) for length in cfg.multi_length_lengths)
        self.multi_length_labels = tuple(str(length) for length in cfg.multi_length_lengths)
        if cfg.seq_residual_encoder != "none":
            residual_cfg = replace(cfg, seq_encoder=cfg.seq_residual_encoder, seq_residual_encoder="none")
            self.seq_residual_branch: Optional[nn.Module] = build_sequence_branch(residual_cfg)
            self.seq_residual_norm: Optional[nn.Module] = nn.LayerNorm(d)
            self.seq_residual_gate: Optional[nn.Module] = ClassSpecificSequenceResidualGate(
                d,
                c,
                cfg.dropout,
                cfg.seq_residual_gate_init_logit,
                cfg.seq_residual_focus_indices,
            )
        else:
            self.seq_residual_branch = None
            self.seq_residual_norm = None
            self.seq_residual_gate = None
        if cfg.structure_encoder == "cnn_control":
            if cfg.structure_uses_sequence:
                raise ValueError("cnn_control is sequence-free; do not combine it with structure_uses_sequence")
            self.struct_branch = CapacityMatchedStructureCNN(
                d, cfg.input_length, cfg.gat_layers, cfg.dropout, cfg.use_position
            )
        elif cfg.structure_encoder == "gat":
            self.struct_branch = (
                StructureBranch(d, cfg.input_length, cfg.gat_layers, cfg.dropout, cfg.use_position)
                if cfg.structure_uses_sequence
                else StructureOnlyBranch(d, cfg.input_length, cfg.gat_layers, cfg.dropout, cfg.use_position)
            )
        else:
            raise ValueError(f"Unsupported structure_encoder: {cfg.structure_encoder}")
        self.rbp_branch = RBPBranch(cfg.rbp_dim, d, cfg.dropout)
        self.seq_norm = nn.LayerNorm(d)
        self.struct_norm = nn.LayerNorm(d)
        entropy_field = next(
            (
                field
                for field in ("multiscale_mean_entropy_flags", "pair_entropy_flags")
                if field in cfg.node_feature_fields
            ),
            "",
        )
        self.structure_entropy_index = (
            list(cfg.node_feature_fields).index(entropy_field) if entropy_field else -1
        )
        self.structure_variance_index = (
            list(cfg.node_feature_fields).index("multiscale_pair_variance_flags")
            if "multiscale_pair_variance_flags" in cfg.node_feature_fields
            else -1
        )
        self.rbp_norm = nn.LayerNorm(d)
        self.main_context = nn.Sequential(
            nn.Linear(d * 4, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(cfg.dropout),
        )
        self.seq_head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d, c))
        self.struct_head = nn.Linear(d, c)
        self.struct_gate = ClassSpecificStructureGate(d, c, cfg.dropout, cfg.gate_init_logit)
        self.class_modality_gate = ClassSpecificModalityGate(
            d,
            c,
            cfg.dropout,
            cfg.gate_init_logit,
            cfg.rbp_pos_gate_init_logit,
        )
        self.rbp_delta = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, c),
        )
        rbp_r, rbp_k, rbp_f = cfg.rbp_pos_shape
        rbp_pos_branch_cls = RBPPositionalBranch if cfg.rbp_pos_encoder == "attention" else CapacityMatchedRBPMLPBranch
        self.rbp_pos_branch = (
            rbp_pos_branch_cls(rbp_r, rbp_k, rbp_f, d, c, cfg.dropout)
            if cfg.use_rbp_positional and rbp_r > 0 and rbp_k > 0 and rbp_f > 0
            else None
        )
        self.rbp_pos_evidence_weight = nn.Parameter(torch.randn(c, d) * 0.02)
        self.rbp_pos_evidence_bias = nn.Parameter(torch.zeros(c))
        self.rbp_pos_interaction = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(cfg.dropout),
        )
        self.rbp_pos_interaction_weight = nn.Parameter(torch.randn(c, d) * 0.02)
        self.rbp_pos_interaction_bias = nn.Parameter(torch.zeros(c))
        self.m1a_aux_head: Optional[nn.Module]
        if cfg.m1a_aux_weight > 0:
            self.m1a_aux_head = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.LayerNorm(d),
                nn.Dropout(cfg.dropout),
                nn.Linear(d, 1),
            )
        else:
            self.m1a_aux_head = None
        self.afamily_indices = tuple(int(index) for index in cfg.afamily_indices)
        self.afamily_m1a_local_index = (
            tuple(cfg.afamily_classes).index("m1A") if "m1A" in tuple(cfg.afamily_classes) else -1
        )
        self.register_buffer(
            "afamily_index_tensor",
            torch.as_tensor(self.afamily_indices, dtype=torch.long),
            persistent=False,
        )
        if (cfg.afamily_aux_weight > 0 or cfg.afamily_residual_scale > 0) and len(self.afamily_indices) >= 2:
            if cfg.afamily_fusion_mode == "early_class":
                self.afamily_head: Optional[nn.Module] = AFamilyEarlyFusionHead(
                    d,
                    len(self.afamily_indices),
                    cfg.dropout,
                    self.afamily_m1a_local_index,
                    cfg.afamily_m1a_expert_scale,
                )
            elif cfg.afamily_fusion_mode == "legacy":
                self.afamily_head = nn.Sequential(
                    nn.Linear(d, d),
                    nn.GELU(),
                    nn.LayerNorm(d),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(d, len(self.afamily_indices)),
                )
            else:
                raise ValueError(f"Unsupported afamily_fusion_mode: {cfg.afamily_fusion_mode}")
        else:
            self.afamily_head = None

    def _interaction(self, z_seq: torch.Tensor, z_struct: torch.Tensor) -> torch.Tensor:
        return torch.cat([z_seq, z_struct, z_seq * z_struct, torch.abs(z_seq - z_struct)], dim=-1)

    def _drop_modality_gate(self, gate: torch.Tensor, drop_prob: float) -> torch.Tensor:
        if not self.training or drop_prob <= 0:
            return gate
        keep = (torch.rand(gate.shape[0], 1, device=gate.device) >= drop_prob).float()
        return gate * keep

    def _structure_confidence(self, node_features: torch.Tensor) -> torch.Tensor:
        """Estimate confidence from normalized ensemble entropy and scale variance."""

        batch = node_features.shape[0]
        if not self.cfg.confidence_gated_structure:
            return torch.ones(batch, 1, device=node_features.device, dtype=node_features.dtype)
        features = node_features if node_features.dim() == 3 else node_features.unsqueeze(-1)
        center = features.shape[1] // 2
        local = features[:, max(0, center - 10) : min(features.shape[1], center + 11), :]
        confidence = torch.ones(batch, device=features.device, dtype=features.dtype)
        if self.structure_entropy_index >= 0:
            entropy = local[:, :, self.structure_entropy_index].mean(dim=1).clamp(0.0, 1.0)
            confidence = confidence * (1.0 - entropy)
        if self.structure_variance_index >= 0:
            variance = local[:, :, self.structure_variance_index].mean(dim=1).clamp_min(0.0)
            confidence = confidence * torch.exp(-4.0 * variance)
        floor = float(max(0.0, min(1.0, self.cfg.structure_confidence_floor)))
        return (floor + (1.0 - floor) * confidence).unsqueeze(-1)

    def forward(
        self,
        seq: torch.Tensor,
        paired_flags: torch.Tensor,
        partner: torch.Tensor,
        struct: torch.Tensor,
        rbp: torch.Tensor,
        rbp_pos: Optional[torch.Tensor] = None,
        seq_multi: Optional[Mapping[str, torch.Tensor]] = None,
        pair_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch = seq.shape[0]
        device = seq.device
        if self.use_multi_length_sequence:
            seq_out = self.seq_branch(seq, seq_multi)  # type: ignore[operator]
            z_seq = self.seq_norm(seq_out["z"])
            seq_logits = seq_out["logits"]
            length_gate = seq_out["length_gate"]
            length_branch_logits = seq_out["branch_logits"]
        else:
            z_seq = self.seq_norm(self.seq_branch(seq))
            seq_logits = self.seq_head(z_seq)
            length_gate = torch.zeros(batch, self.cfg.num_classes, 0, device=device)
            length_branch_logits = torch.zeros(batch, self.cfg.num_classes, 0, device=device)

        if self.cfg.use_structure:
            z_struct, edge_attention = self.struct_branch(seq, paired_flags, partner, struct, pair_weights)
            z_struct = self.struct_norm(z_struct)
            interaction = self._interaction(z_seq, z_struct)
            struct_delta, _legacy_class_gate = self.struct_gate(interaction)
            class_gate = self.class_modality_gate.structure_gate(z_seq, z_struct)
            structure_confidence = self._structure_confidence(paired_flags)
            class_gate = class_gate * structure_confidence
            class_gate = self._drop_modality_gate(class_gate, self.cfg.modality_dropout_struct)
            logits = seq_logits + self.cfg.structure_scale * class_gate * struct_delta
            struct_logits = self.struct_head(z_struct)
            h_main = self.main_context(interaction)
        else:
            z_struct = torch.zeros_like(z_seq)
            edge_attention = torch.zeros(batch, self.cfg.input_length, 4, device=device)
            class_gate = torch.zeros(batch, self.cfg.num_classes, device=device)
            struct_delta = torch.zeros(batch, self.cfg.num_classes, device=device)
            logits = seq_logits
            struct_logits = torch.zeros(batch, self.cfg.num_classes, device=device)
            h_main = z_seq
            structure_confidence = torch.zeros(batch, 1, device=device)

        if self.cfg.use_rbp and self.cfg.rbp_dim > 0:
            z_rbp = self.rbp_norm(self.rbp_branch(rbp, batch, device))
            logits = logits + self.cfg.rbp_scale * self.rbp_delta(z_rbp)
            rbp_diag = torch.full((batch, 1), self.cfg.rbp_scale, device=device)
        else:
            rbp_diag = torch.zeros(batch, 1, device=device)

        if self.rbp_pos_branch is not None and rbp_pos is not None:
            h_rbp_cls, rbp_pos_attention = self.rbp_pos_branch(rbp_pos)
            h_main_cls = h_main.unsqueeze(1).expand(-1, self.cfg.num_classes, -1)
            rbp_pos_gate = self.class_modality_gate.rbp_gate(h_main, h_rbp_cls)
            rbp_pos_gate = self._drop_modality_gate(rbp_pos_gate, self.cfg.modality_dropout_rbp)
            rbp_pos_delta = (
                h_rbp_cls * self.rbp_pos_evidence_weight.view(1, self.cfg.num_classes, -1)
            ).sum(dim=-1) + self.rbp_pos_evidence_bias.view(1, -1)
            rbp_interaction = torch.cat([h_main_cls, h_rbp_cls, h_main_cls * h_rbp_cls], dim=-1)
            rbp_interaction_hidden = self.rbp_pos_interaction(rbp_interaction)
            rbp_interaction_delta = (
                rbp_interaction_hidden * self.rbp_pos_interaction_weight.view(1, self.cfg.num_classes, -1)
            ).sum(dim=-1) + self.rbp_pos_interaction_bias.view(1, -1)
            logits = logits + self.cfg.rbp_pos_scale * rbp_pos_gate * (rbp_pos_delta + rbp_interaction_delta)
            rbp_pos_diag = rbp_pos_gate.mean(dim=1, keepdim=True)
            rbp_pos_available = True
        else:
            h_rbp_cls = torch.zeros(batch, self.cfg.num_classes, self.cfg.d_model, device=device)
            rbp_pos_gate = torch.zeros(batch, self.cfg.num_classes, device=device)
            rbp_pos_attention = torch.zeros(
                batch,
                self.cfg.num_classes,
                max(1, self.cfg.rbp_pos_shape[0]),
                max(1, self.cfg.rbp_pos_shape[1]),
                device=device,
            )
            rbp_pos_diag = torch.zeros(batch, 1, device=device)
            rbp_pos_available = False

        if self.m1a_aux_head is not None:
            m1a_aux_logit = self.m1a_aux_head(h_main).squeeze(-1)
        else:
            m1a_aux_logit = torch.zeros(batch, device=device)

        if self.seq_residual_branch is not None:
            z_seq_residual = self.seq_residual_branch(seq)
            if self.seq_residual_norm is not None:
                z_seq_residual = self.seq_residual_norm(z_seq_residual)
            seq_residual_interaction = self._interaction(z_seq, z_seq_residual)
            if self.seq_residual_gate is None:
                raise RuntimeError("seq_residual_gate is missing while seq_residual_branch is enabled.")
            seq_residual_delta, seq_residual_gate = self.seq_residual_gate(seq_residual_interaction)
            logits = logits + self.cfg.seq_residual_scale * seq_residual_gate * seq_residual_delta
        else:
            seq_residual_delta = torch.zeros(batch, self.cfg.num_classes, device=device)
            seq_residual_gate = torch.zeros(batch, self.cfg.num_classes, device=device)

        if self.afamily_head is not None and self.afamily_index_tensor.numel() >= 2:
            if self.cfg.afamily_fusion_mode == "early_class":
                h_rbp_afamily = h_rbp_cls.index_select(1, self.afamily_index_tensor)
                afamily_logits, afamily_modality_attention = self.afamily_head(
                    z_seq,
                    z_struct,
                    h_rbp_afamily,
                    bool(self.cfg.use_structure),
                    rbp_pos_available,
                )
                afamily_delta = afamily_logits - afamily_logits.mean(dim=-1, keepdim=True)
            else:
                afamily_delta = self.afamily_head(h_main)
                afamily_logits = logits[:, self.afamily_index_tensor]
                afamily_modality_attention = torch.zeros(
                    batch,
                    len(self.afamily_indices),
                    len(AFamilyEarlyFusionHead.modality_labels),
                    device=device,
                )
            if self.cfg.afamily_residual_scale > 0:
                logits = logits.clone()
                logits[:, self.afamily_index_tensor] = (
                    logits[:, self.afamily_index_tensor] + self.cfg.afamily_residual_scale * afamily_delta
                )
            if self.cfg.afamily_fusion_mode != "early_class":
                afamily_logits = logits[:, self.afamily_index_tensor]
        else:
            afamily_delta = torch.zeros(batch, 0, device=device)
            afamily_logits = torch.zeros(batch, 0, device=device)
            afamily_modality_attention = torch.zeros(
                batch,
                0,
                len(AFamilyEarlyFusionHead.modality_labels),
                device=device,
            )

        structure_diag = class_gate.mean(dim=1, keepdim=True)
        sequence_diag = 1.0 - structure_diag
        diagnostic_mix = torch.cat([sequence_diag, structure_diag, rbp_diag + rbp_pos_diag], dim=-1)
        diagnostic_mix = diagnostic_mix / diagnostic_mix.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return {
            "logits": logits,
            "seq_logits": seq_logits,
            "struct_logits": struct_logits,
            "structure_delta": struct_delta,
            "structure_gate": class_gate,
            "structure_confidence": structure_confidence,
            "rbp_pos_gate": rbp_pos_gate,
            "seq_residual_delta": seq_residual_delta,
            "seq_residual_gate": seq_residual_gate,
            "m1a_aux_logit": m1a_aux_logit,
            "afamily_delta": afamily_delta,
            "afamily_logits": afamily_logits,
            "afamily_modality_attention": afamily_modality_attention,
            "length_gate": length_gate,
            "length_branch_logits": length_branch_logits,
            "rbp_pos_attention": rbp_pos_attention,
            "modality_gate": diagnostic_mix,
            "edge_attention": edge_attention,
        }


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor],
    gamma: float,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_probs = target_log_probs.exp()
    losses = (1.0 - target_probs).pow(max(0.0, float(gamma))) * (-target_log_probs)
    if weight is None:
        return losses.mean()
    sample_weights = weight.gather(0, targets)
    return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(logits)
    target_probs = torch.where(targets > 0.5, probs, 1.0 - probs)
    return ((1.0 - target_probs).pow(max(0.0, float(gamma))) * bce).mean()


def one_vs_rest_logit_margin(logits: torch.Tensor, class_index: int) -> torch.Tensor:
    target_logit = logits[:, class_index]
    mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
    mask[class_index] = False
    return target_logit - torch.logsumexp(logits[:, mask], dim=-1)


def teacher_binary_distillation_loss(
    student_margin: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float,
    teacher_positive_weight: float,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-3)
    clipped = teacher_probs.clamp(1e-6, 1.0 - 1e-6)
    teacher_logits = torch.logit(clipped)
    soft_targets = torch.sigmoid(teacher_logits / temperature)
    losses = F.binary_cross_entropy_with_logits(student_margin / temperature, soft_targets, reduction="none")
    sample_weights = 1.0 + max(0.0, float(teacher_positive_weight)) * soft_targets.detach()
    return temperature * temperature * (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def teacher_pairwise_ranking_loss(
    student_margin: torch.Tensor,
    teacher_probs: torch.Tensor,
    max_pairs: int,
    min_teacher_gap: float,
) -> torch.Tensor:
    pair_count = min(max(0, int(max_pairs)), student_margin.shape[0] // 2)
    if pair_count < 1:
        return torch.zeros((), dtype=student_margin.dtype, device=student_margin.device)
    order = torch.argsort(teacher_probs)
    low_indices = order[:pair_count]
    high_indices = order[-pair_count:]
    teacher_gap = teacher_probs[high_indices] - teacher_probs[low_indices]
    valid = teacher_gap >= max(0.0, float(min_teacher_gap))
    if not bool(valid.any()):
        return torch.zeros((), dtype=student_margin.dtype, device=student_margin.device)
    student_gap = student_margin[high_indices] - student_margin[low_indices]
    weights = teacher_gap[valid].detach()
    losses = F.softplus(-student_gap[valid])
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    aux_seq_weight: float,
    aux_struct_weight: float,
    gate_l1_weight: float,
    rbp_pos_gate_l1_weight: float,
    seq_residual_gate_l1_weight: float,
    m1a_aux_weight: float,
    m1a_aux_pos_weight: float,
    m1a_aux_loss_name: str,
    m1a_aux_focal_gamma: float,
    m1a_index: int,
    afamily_aux_weight: float,
    afamily_global_to_local: Optional[torch.Tensor],
    afamily_class_weights: Optional[torch.Tensor],
    afamily_loss_name: str,
    afamily_focal_gamma: float,
    distill_m1a_weight: float,
    distill_m1a_rank_weight: float,
    distill_temperature: float,
    distill_teacher_positive_weight: float,
    distill_rank_pairs: int,
    distill_min_teacher_gap: float,
    length_aux_weight: float,
    length_gate_entropy_weight: float,
    grad_accum_steps: int,
) -> Dict[str, float]:
    model.train()
    total = main_total = seq_aux_total = struct_aux_total = gate_total = rbp_pos_gate_total = seq_residual_gate_total = 0.0
    m1a_aux_total = afamily_aux_total = distill_m1a_total = distill_rank_total = 0.0
    length_aux_total = length_gate_entropy_total = 0.0
    rows = 0
    gate_sum = torch.zeros(3, dtype=torch.float64)
    use_structure = bool(getattr(model, "cfg").use_structure)
    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        batch = move_batch(batch, device)
        seq_multi = {key: batch[key] for key in getattr(model, "multi_length_keys", ()) if key in batch}
        out = model(
            batch["seq"],
            batch["paired_flags"],
            batch["pair_partner"],
            batch["struct"],
            batch["rbp"],
            batch.get("rbp_pos"),
            seq_multi=seq_multi,
            pair_weights=batch.get("pair_weights"),
        )  # type: ignore[arg-type]
        labels = batch["label"]  # type: ignore[assignment]
        main_loss = criterion(out["logits"], labels)
        seq_aux = criterion(out["seq_logits"], labels) if aux_seq_weight > 0 else torch.zeros((), device=device)
        struct_aux = criterion(out["struct_logits"], labels) if use_structure and aux_struct_weight > 0 else torch.zeros((), device=device)
        length_branch_logits = out.get("length_branch_logits")
        if (
            length_aux_weight > 0
            and torch.is_tensor(length_branch_logits)
            and length_branch_logits.dim() == 3
            and length_branch_logits.shape[-1] > 0
        ):
            length_aux = torch.stack(
                [criterion(length_branch_logits[:, :, idx], labels) for idx in range(length_branch_logits.shape[-1])]
            ).mean()
        else:
            length_aux = torch.zeros((), device=device)
        length_gate = out.get("length_gate")
        if torch.is_tensor(length_gate) and length_gate.dim() == 3 and length_gate.shape[-1] > 0:
            length_gate_entropy = -(length_gate * length_gate.clamp_min(1e-8).log()).sum(dim=-1).mean()
        else:
            length_gate_entropy = torch.zeros((), device=device)
        gate_l1 = out["structure_gate"].mean() if use_structure and gate_l1_weight > 0 else torch.zeros((), device=device)
        rbp_pos_gate_l1 = (
            out["rbp_pos_gate"].mean() if rbp_pos_gate_l1_weight > 0 and out["rbp_pos_gate"].numel() > 0 else torch.zeros((), device=device)
        )
        seq_residual_gate_l1 = (
            out["seq_residual_gate"].mean()
            if seq_residual_gate_l1_weight > 0 and out["seq_residual_gate"].numel() > 0
            else torch.zeros((), device=device)
        )
        m1a_aux_logit = out["m1a_aux_logit"]
        if m1a_aux_weight > 0 and torch.is_tensor(m1a_aux_logit) and m1a_aux_logit.numel() > 0:
            m1a_targets = (labels == m1a_index).float()
            pos_weight = torch.tensor(float(m1a_aux_pos_weight), device=device, dtype=m1a_aux_logit.dtype).clamp_min(1.0)
            if m1a_aux_loss_name == "focal":
                m1a_aux_loss = binary_focal_loss_with_logits(
                    m1a_aux_logit,
                    m1a_targets,
                    pos_weight,
                    m1a_aux_focal_gamma,
                )
            else:
                m1a_aux_loss = F.binary_cross_entropy_with_logits(
                    m1a_aux_logit,
                    m1a_targets,
                    pos_weight=pos_weight,
                )
        else:
            m1a_aux_loss = torch.zeros((), device=device)
        afamily_logits = out.get("afamily_logits")
        if (
            afamily_aux_weight > 0
            and afamily_global_to_local is not None
            and torch.is_tensor(afamily_logits)
            and afamily_logits.dim() == 2
            and afamily_logits.shape[-1] >= 2
        ):
            local_labels = afamily_global_to_local[labels]
            afamily_mask = local_labels >= 0
            if bool(afamily_mask.any()):
                if afamily_loss_name == "focal":
                    afamily_loss = focal_cross_entropy(
                        afamily_logits[afamily_mask],
                        local_labels[afamily_mask],
                        afamily_class_weights,
                        afamily_focal_gamma,
                    )
                else:
                    afamily_loss = F.cross_entropy(
                        afamily_logits[afamily_mask],
                        local_labels[afamily_mask],
                        weight=afamily_class_weights,
                    )
            else:
                afamily_loss = torch.zeros((), device=device)
        else:
            afamily_loss = torch.zeros((), device=device)
        teacher_m1a = batch.get("teacher_m1a")
        if (
            m1a_index >= 0
            and torch.is_tensor(teacher_m1a)
            and teacher_m1a.numel() == labels.numel()
            and (distill_m1a_weight > 0 or distill_m1a_rank_weight > 0)
        ):
            student_margin = one_vs_rest_logit_margin(out["logits"], m1a_index)
            teacher_m1a = teacher_m1a.to(dtype=student_margin.dtype)
            distill_m1a_loss = teacher_binary_distillation_loss(
                student_margin,
                teacher_m1a,
                distill_temperature,
                distill_teacher_positive_weight,
            )
            distill_rank_loss = teacher_pairwise_ranking_loss(
                student_margin,
                teacher_m1a,
                distill_rank_pairs,
                distill_min_teacher_gap,
            )
        else:
            distill_m1a_loss = torch.zeros((), device=device)
            distill_rank_loss = torch.zeros((), device=device)
        loss = (
            main_loss
            + aux_seq_weight * seq_aux
            + aux_struct_weight * struct_aux
            + length_aux_weight * length_aux
            + length_gate_entropy_weight * length_gate_entropy
            + gate_l1_weight * gate_l1
            + rbp_pos_gate_l1_weight * rbp_pos_gate_l1
            + seq_residual_gate_l1_weight * seq_residual_gate_l1
            + m1a_aux_weight * m1a_aux_loss
            + afamily_aux_weight * afamily_loss
            + distill_m1a_weight * distill_m1a_loss
            + distill_m1a_rank_weight * distill_rank_loss
        )
        (loss / grad_accum_steps).backward()
        should_step = (batch_index + 1) % grad_accum_steps == 0 or (batch_index + 1) == len(loader)
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = int(labels.shape[0])
        total += float(loss.detach().cpu()) * batch_size
        main_total += float(main_loss.detach().cpu()) * batch_size
        seq_aux_total += float(seq_aux.detach().cpu()) * batch_size
        struct_aux_total += float(struct_aux.detach().cpu()) * batch_size
        length_aux_total += float(length_aux.detach().cpu()) * batch_size
        length_gate_entropy_total += float(length_gate_entropy.detach().cpu()) * batch_size
        gate_total += float(gate_l1.detach().cpu()) * batch_size
        rbp_pos_gate_total += float(rbp_pos_gate_l1.detach().cpu()) * batch_size
        seq_residual_gate_total += float(seq_residual_gate_l1.detach().cpu()) * batch_size
        m1a_aux_total += float(m1a_aux_loss.detach().cpu()) * batch_size
        afamily_aux_total += float(afamily_loss.detach().cpu()) * batch_size
        distill_m1a_total += float(distill_m1a_loss.detach().cpu()) * batch_size
        distill_rank_total += float(distill_rank_loss.detach().cpu()) * batch_size
        rows += batch_size
        gate_sum += out["modality_gate"].detach().to(dtype=torch.float64, device="cpu").sum(dim=0)
    gate_mean = gate_sum / max(rows, 1)
    return {
        "loss": total / max(rows, 1),
        "main_loss": main_total / max(rows, 1),
        "seq_aux_loss": seq_aux_total / max(rows, 1),
        "struct_aux_loss": struct_aux_total / max(rows, 1),
        "length_aux_loss": length_aux_total / max(rows, 1),
        "length_gate_entropy": length_gate_entropy_total / max(rows, 1),
        "structure_gate_l1": gate_total / max(rows, 1),
        "rbp_pos_gate_l1": rbp_pos_gate_total / max(rows, 1),
        "seq_residual_gate_l1": seq_residual_gate_total / max(rows, 1),
        "m1a_aux_loss": m1a_aux_total / max(rows, 1),
        "afamily_aux_loss": afamily_aux_total / max(rows, 1),
        "distill_m1a_loss": distill_m1a_total / max(rows, 1),
        "distill_m1a_rank_loss": distill_rank_total / max(rows, 1),
        "gate_seq": float(gate_mean[0]),
        "gate_structure": float(gate_mean[1]),
        "gate_rbp": float(gate_mean[2]),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    probs, labels = [], []
    gate_sum = torch.zeros(3, dtype=torch.float64)
    class_gate_sum = torch.zeros(len(CLASS_NAMES), dtype=torch.float64)
    true_class_gate_sum = torch.zeros(len(CLASS_NAMES), dtype=torch.float64)
    rbp_pos_gate_sum = torch.zeros(len(CLASS_NAMES), dtype=torch.float64)
    true_rbp_pos_gate_sum = torch.zeros(len(CLASS_NAMES), dtype=torch.float64)
    seq_residual_gate_sum = torch.zeros(len(CLASS_NAMES), dtype=torch.float64)
    true_seq_residual_gate_sum = torch.zeros(len(CLASS_NAMES), dtype=torch.float64)
    true_class_count = torch.zeros(len(CLASS_NAMES), dtype=torch.float64)
    length_labels = [str(length) for length in getattr(model, "multi_length_labels", ())]
    length_gate_sum = torch.zeros((len(CLASS_NAMES), len(length_labels)), dtype=torch.float64)
    true_length_gate_sum = torch.zeros((len(CLASS_NAMES), len(length_labels)), dtype=torch.float64)
    rbp_pos_attention_sum: Optional[torch.Tensor] = None
    afamily_attention_sum: Optional[torch.Tensor] = None
    rows = 0
    site_ids: List[str] = []
    mod_types: List[str] = []
    for batch in loader:
        batch = move_batch(batch, device)
        seq_multi = {key: batch[key] for key in getattr(model, "multi_length_keys", ()) if key in batch}
        out = model(
            batch["seq"],
            batch["paired_flags"],
            batch["pair_partner"],
            batch["struct"],
            batch["rbp"],
            batch.get("rbp_pos"),
            seq_multi=seq_multi,
            pair_weights=batch.get("pair_weights"),
        )  # type: ignore[arg-type]
        batch_probs = torch.softmax(out["logits"], dim=-1)
        batch_labels = batch["label"].detach().cpu().tolist()  # type: ignore[union-attr]
        gates = out["structure_gate"].detach().to(dtype=torch.float64, device="cpu")
        rbp_pos_gates = out["rbp_pos_gate"].detach().to(dtype=torch.float64, device="cpu")
        seq_residual_gates = out["seq_residual_gate"].detach().to(dtype=torch.float64, device="cpu")
        length_gates = out["length_gate"].detach().to(dtype=torch.float64, device="cpu")
        rbp_pos_attention = out["rbp_pos_attention"].detach().to(dtype=torch.float64, device="cpu")
        afamily_attention = out["afamily_modality_attention"].detach().to(dtype=torch.float64, device="cpu")
        probs.extend(batch_probs.detach().cpu().tolist())
        labels.extend(batch_labels)
        batch_size = len(batch_labels)
        gate_sum += out["modality_gate"].detach().to(dtype=torch.float64, device="cpu").sum(dim=0)
        class_gate_sum += gates.sum(dim=0)
        rbp_pos_gate_sum += rbp_pos_gates.sum(dim=0)
        seq_residual_gate_sum += seq_residual_gates.sum(dim=0)
        if length_gates.ndim == 3 and length_gates.shape[-1] == len(length_labels):
            length_gate_sum += length_gates.sum(dim=0)
        if rbp_pos_attention_sum is None:
            rbp_pos_attention_sum = torch.zeros_like(rbp_pos_attention.sum(dim=0), dtype=torch.float64)
        rbp_pos_attention_sum += rbp_pos_attention.sum(dim=0)
        if afamily_attention.ndim == 3 and afamily_attention.shape[1] > 0:
            if afamily_attention_sum is None:
                afamily_attention_sum = torch.zeros_like(afamily_attention.sum(dim=0), dtype=torch.float64)
            afamily_attention_sum += afamily_attention.sum(dim=0)
        for idx, label in enumerate(batch_labels):
            label_idx = int(label)
            true_class_gate_sum[label_idx] += gates[idx, label_idx]
            true_rbp_pos_gate_sum[label_idx] += rbp_pos_gates[idx, label_idx]
            true_seq_residual_gate_sum[label_idx] += seq_residual_gates[idx, label_idx]
            if length_gates.ndim == 3 and length_gates.shape[-1] == len(length_labels):
                true_length_gate_sum[label_idx] += length_gates[idx, label_idx]
            true_class_count[label_idx] += 1
        rows += batch_size
        site_ids.extend(batch["site_id"])  # type: ignore[arg-type]
        mod_types.extend(batch["mod_type"])  # type: ignore[arg-type]
    return {
        "probs": np.asarray(probs, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "site_ids": site_ids,
        "mod_types": mod_types,
        "gate_mean": (gate_sum / max(rows, 1)).tolist(),
        "class_structure_gate_mean": (class_gate_sum / max(rows, 1)).tolist(),
        "true_class_structure_gate_mean": (true_class_gate_sum / torch.clamp_min(true_class_count, 1.0)).tolist(),
        "class_rbp_pos_gate_mean": (rbp_pos_gate_sum / max(rows, 1)).tolist(),
        "true_class_rbp_pos_gate_mean": (true_rbp_pos_gate_sum / torch.clamp_min(true_class_count, 1.0)).tolist(),
        "class_seq_residual_gate_mean": (seq_residual_gate_sum / max(rows, 1)).tolist(),
        "true_class_seq_residual_gate_mean": (true_seq_residual_gate_sum / torch.clamp_min(true_class_count, 1.0)).tolist(),
        "length_gate_labels": length_labels,
        "class_length_gate_mean": (length_gate_sum / max(rows, 1)).tolist(),
        "true_class_length_gate_mean": (
            true_length_gate_sum / torch.clamp_min(true_class_count, 1.0).unsqueeze(1)
            if len(length_labels)
            else true_length_gate_sum
        ).tolist(),
        "rbp_pos_attention_mean": (
            rbp_pos_attention_sum / max(rows, 1)
            if rbp_pos_attention_sum is not None
            else torch.zeros((len(CLASS_NAMES), 1, 1), dtype=torch.float64)
        ).tolist(),
        "afamily_classes": list(getattr(getattr(model, "cfg", None), "afamily_classes", ())),
        "afamily_modality_labels": list(AFamilyEarlyFusionHead.modality_labels),
        "afamily_modality_attention_mean": (
            afamily_attention_sum / max(rows, 1)
            if afamily_attention_sum is not None
            else torch.zeros((0, len(AFamilyEarlyFusionHead.modality_labels)), dtype=torch.float64)
        ).tolist(),
        "true_class_gate_support": true_class_count.to(dtype=torch.int64).tolist(),
    }


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
        seq_multi = {key: batch[key] for key in getattr(model, "multi_length_keys", ()) if key in batch}
        model(
            batch["seq"],
            batch["paired_flags"],
            batch["pair_partner"],
            batch["struct"],
            batch["rbp"],
            batch.get("rbp_pos"),
            seq_multi=seq_multi,
            pair_weights=batch.get("pair_weights"),
        )

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


def write_class_structure_gate(path: Path, predictions: Mapping[str, object]) -> None:
    class_mean = predictions.get("class_structure_gate_mean", [])
    true_mean = predictions.get("true_class_structure_gate_mean", [])
    support = predictions.get("true_class_gate_support", [])
    fields = ["mod_type", "support", "mean_gate_for_class_logit", "mean_gate_on_true_samples"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, name in enumerate(CLASS_NAMES):
            writer.writerow(
                {
                    "mod_type": name,
                    "support": int(support[idx]) if isinstance(support, list) and idx < len(support) else "",
                    "mean_gate_for_class_logit": float(class_mean[idx]) if isinstance(class_mean, list) and idx < len(class_mean) else "",
                    "mean_gate_on_true_samples": float(true_mean[idx]) if isinstance(true_mean, list) and idx < len(true_mean) else "",
                }
            )


def write_class_rbp_positional_gate(path: Path, predictions: Mapping[str, object]) -> None:
    class_mean = predictions.get("class_rbp_pos_gate_mean", [])
    true_mean = predictions.get("true_class_rbp_pos_gate_mean", [])
    support = predictions.get("true_class_gate_support", [])
    fields = ["mod_type", "support", "mean_rbp_pos_gate_for_class_logit", "mean_rbp_pos_gate_on_true_samples"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, name in enumerate(CLASS_NAMES):
            writer.writerow(
                {
                    "mod_type": name,
                    "support": int(support[idx]) if isinstance(support, list) and idx < len(support) else "",
                    "mean_rbp_pos_gate_for_class_logit": float(class_mean[idx]) if isinstance(class_mean, list) and idx < len(class_mean) else "",
                    "mean_rbp_pos_gate_on_true_samples": float(true_mean[idx]) if isinstance(true_mean, list) and idx < len(true_mean) else "",
                }
            )


def write_class_sequence_residual_gate(path: Path, predictions: Mapping[str, object]) -> None:
    class_mean = predictions.get("class_seq_residual_gate_mean", [])
    true_mean = predictions.get("true_class_seq_residual_gate_mean", [])
    support = predictions.get("true_class_gate_support", [])
    fields = ["mod_type", "support", "mean_seq_residual_gate_for_class_logit", "mean_seq_residual_gate_on_true_samples"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, name in enumerate(CLASS_NAMES):
            writer.writerow(
                {
                    "mod_type": name,
                    "support": int(support[idx]) if isinstance(support, list) and idx < len(support) else "",
                    "mean_seq_residual_gate_for_class_logit": float(class_mean[idx]) if isinstance(class_mean, list) and idx < len(class_mean) else "",
                    "mean_seq_residual_gate_on_true_samples": float(true_mean[idx]) if isinstance(true_mean, list) and idx < len(true_mean) else "",
                }
            )


def write_class_length_gate(path: Path, predictions: Mapping[str, object]) -> None:
    labels = [str(label) for label in predictions.get("length_gate_labels", [])]
    if not labels:
        return
    class_mean = predictions.get("class_length_gate_mean", [])
    true_mean = predictions.get("true_class_length_gate_mean", [])
    support = predictions.get("true_class_gate_support", [])
    fields = ["mod_type", "support"]
    fields.extend([f"mean_length_gate_{label}nt_for_class_logit" for label in labels])
    fields.extend([f"mean_length_gate_{label}nt_on_true_samples" for label in labels])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for class_idx, name in enumerate(CLASS_NAMES):
            row: Dict[str, object] = {
                "mod_type": name,
                "support": int(support[class_idx]) if isinstance(support, list) and class_idx < len(support) else "",
            }
            for length_idx, label in enumerate(labels):
                value = ""
                if (
                    isinstance(class_mean, list)
                    and class_idx < len(class_mean)
                    and isinstance(class_mean[class_idx], list)
                    and length_idx < len(class_mean[class_idx])
                ):
                    value = float(class_mean[class_idx][length_idx])
                row[f"mean_length_gate_{label}nt_for_class_logit"] = value
            for length_idx, label in enumerate(labels):
                value = ""
                if (
                    isinstance(true_mean, list)
                    and class_idx < len(true_mean)
                    and isinstance(true_mean[class_idx], list)
                    and length_idx < len(true_mean[class_idx])
                ):
                    value = float(true_mean[class_idx][length_idx])
                row[f"mean_length_gate_{label}nt_on_true_samples"] = value
            writer.writerow(row)


def write_rbp_positional_attention(path: Path, predictions: Mapping[str, object], metadata: Mapping[str, object]) -> None:
    attention = np.asarray(predictions.get("rbp_pos_attention_mean", []), dtype=np.float64)
    rbps = [str(x) for x in metadata.get("rbps", [])]
    bins = [x for x in metadata.get("bins", []) if isinstance(x, Mapping)]
    if attention.ndim != 3 or not rbps or not bins:
        return
    fields = ["mod_type", "rbp", "distance_bin", "bin_left", "bin_right", "mean_attention"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for class_idx, mod_type in enumerate(CLASS_NAMES):
            for rbp_idx, rbp in enumerate(rbps):
                for bin_idx, bin_info in enumerate(bins):
                    if class_idx >= attention.shape[0] or rbp_idx >= attention.shape[1] or bin_idx >= attention.shape[2]:
                        continue
                    writer.writerow(
                        {
                            "mod_type": mod_type,
                            "rbp": rbp,
                            "distance_bin": bin_info.get("label", str(bin_idx)),
                            "bin_left": bin_info.get("left", ""),
                            "bin_right": bin_info.get("right", ""),
                            "mean_attention": float(attention[class_idx, rbp_idx, bin_idx]),
                        }
                    )


def write_afamily_modality_attention(path: Path, predictions: Mapping[str, object]) -> None:
    attention = np.asarray(predictions.get("afamily_modality_attention_mean", []), dtype=np.float64)
    classes = [str(name) for name in predictions.get("afamily_classes", [])]
    modality_labels = [str(name) for name in predictions.get("afamily_modality_labels", [])]
    if attention.ndim != 2 or attention.shape[0] == 0 or not classes or not modality_labels:
        return
    fields = ["mod_type"] + modality_labels
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for class_idx, mod_type in enumerate(classes):
            if class_idx >= attention.shape[0]:
                continue
            row = {"mod_type": mod_type}
            for modality_idx, label in enumerate(modality_labels):
                row[label] = float(attention[class_idx, modality_idx]) if modality_idx < attention.shape[1] else ""
            writer.writerow(row)


def parse_csv_fields(value: str) -> List[str]:
    return [field.strip() for field in str(value or "").split(",") if field.strip()]


def parse_int_fields(value: str) -> List[int]:
    return [int(field) for field in parse_csv_fields(value)]


def parse_class_indices(value: str) -> Tuple[int, ...]:
    names = parse_csv_fields(value)
    if not names:
        return ()
    name_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    missing = [name for name in names if name not in name_to_idx]
    if missing:
        raise ValueError(f"Unknown class name(s) in --seq-residual-focus-classes: {', '.join(missing)}")
    return tuple(name_to_idx[name] for name in names)


def parse_named_class_indices(value: str, option_name: str) -> Tuple[int, ...]:
    names = parse_csv_fields(value)
    if not names:
        return ()
    name_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    missing = [name for name in names if name not in name_to_idx]
    if missing:
        raise ValueError(f"Unknown class name(s) in {option_name}: {', '.join(missing)}")
    return tuple(name_to_idx[name] for name in names)


def existing_class_names(value: str, option_name: str) -> Tuple[str, ...]:
    names = tuple(parse_csv_fields(value))
    if not names:
        return ()
    existing = []
    missing = []
    name_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    for name in names:
        if name in name_to_idx:
            existing.append(name)
        else:
            missing.append(name)
    if missing:
        print(
            f"warning: skipped class name(s) not present in current dataset for {option_name}: {', '.join(missing)}",
            flush=True,
        )
    return tuple(existing)


def afamily_class_weights_from_rows(
    rows: Sequence[Dict[str, str]],
    afamily_classes: Sequence[str],
    power: float,
    max_weight: float,
    m1a_weight_multiplier: float,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not afamily_classes:
        return None
    counts = Counter(str(row.get("mod_type", "")) for row in rows)
    raw = np.asarray([max(1, counts.get(name, 0)) for name in afamily_classes], dtype=np.float64)
    weights = (raw.sum() / raw) ** max(0.0, float(power))
    weights = weights / max(float(weights.mean()), 1e-8)
    if "m1A" in afamily_classes and float(m1a_weight_multiplier) != 1.0:
        weights[list(afamily_classes).index("m1A")] *= max(0.0, float(m1a_weight_multiplier))
        weights = weights / max(float(weights.mean()), 1e-8)
    weights = np.clip(weights, 1.0 / max(1.0, float(max_weight)), max(1.0, float(max_weight)))
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def metric_selection_score(metrics: Mapping[str, object], args: argparse.Namespace) -> float:
    score = float(metrics[args.selection_metric])
    focus_class = str(getattr(args, "selection_focus_class", "") or "")
    focus_weight = float(getattr(args, "selection_focus_weight", 0.0) or 0.0)
    focus_metric = str(getattr(args, "selection_focus_metric", "AUPR_ovr") or "AUPR_ovr")
    if focus_class and focus_weight:
        per_class = metrics.get("per_class", {})
        if isinstance(per_class, Mapping):
            focus_metrics = per_class.get(focus_class, {})
            if isinstance(focus_metrics, Mapping):
                focus_value = focus_metrics.get(focus_metric)
                try:
                    focus_float = float(focus_value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    focus_float = float("nan")
                if np.isfinite(focus_float):
                    score += focus_weight * focus_float
    return score


def add_grouped_detection_metrics(
    metrics: Dict[str, object],
    predictions: Mapping[str, object],
    rows: Sequence[Mapping[str, str]],
    group_column: str,
    positive_class: str,
    minimum_positive_rows: int,
) -> None:
    if not group_column:
        return
    if positive_class not in CLASS_NAMES:
        raise ValueError(f"Grouped detection positive class is absent: {positive_class}")
    group_by_id = {str(row.get("site_id", "")): str(row.get(group_column, "")) for row in rows}
    site_ids = [str(value) for value in predictions.get("site_ids", [])]
    labels = np.asarray(predictions.get("labels", []), dtype=np.int64)
    probabilities = np.asarray(predictions.get("probs", []), dtype=np.float64)
    if probabilities.ndim != 2 or len(site_ids) != len(labels) or len(labels) != len(probabilities):
        raise ValueError("Grouped detection metrics received inconsistent prediction arrays")
    positive_index = CLASS_NAMES.index(positive_class)
    groups = np.asarray([group_by_id.get(site_id, "") for site_id in site_ids], dtype=object)
    pred_labels = probabilities.argmax(axis=1)
    output: Dict[str, Dict[str, float]] = {}
    aupr_values = []
    for group in sorted({str(value) for value in groups if str(value)}):
        mask = groups == group
        y_true = (labels[mask] == positive_index).astype(np.int8)
        y_prob = probabilities[mask, positive_index]
        y_pred = (pred_labels[mask] == positive_index).astype(np.int8)
        if len(np.unique(y_true)) < 2:
            continue
        values = {
            "rows": int(mask.sum()),
            "positive_rows": int(y_true.sum()),
            "AUPR": float(average_precision_score(y_true, y_prob)),
            "AUC": float(roc_auc_score(y_true, y_prob)),
            "F1": float(f1_score(y_true, y_pred, zero_division=0)),
            "MCC": float(matthews_corrcoef(y_true, y_pred)),
        }
        output[group] = values
        if values["positive_rows"] >= max(1, int(minimum_positive_rows)):
            aupr_values.append(values["AUPR"])
    metrics["group_metric_column"] = group_column
    metrics["group_positive_class"] = positive_class
    metrics["group_metric_minimum_positive_rows"] = max(1, int(minimum_positive_rows))
    metrics["per_group_detection"] = output
    metrics["macro_group_AUPR"] = float(np.mean(aupr_values)) if aupr_values else float("nan")


def configure_structure_inputs(extra_struct_fields: str, node_feature_fields: str) -> None:
    for field in parse_csv_fields(extra_struct_fields):
        if field not in STRUCT_FIELDS:
            STRUCT_FIELDS.append(field)
    node_fields = parse_csv_fields(node_feature_fields) or ["paired_flags"]
    NODE_FEATURE_FIELDS[:] = node_fields


def make_focus_oversampler(
    rows: Sequence[Dict[str, str]],
    focus_classes: Sequence[str],
    factor: float,
    epoch_size_multiplier: float,
) -> Optional[WeightedRandomSampler]:
    focus = {str(name).strip() for name in focus_classes if str(name).strip()}
    if not focus or factor <= 1.0:
        return None
    weights = [float(factor) if str(row.get("mod_type", "")) in focus else 1.0 for row in rows]
    num_samples = max(1, int(round(len(weights) * max(0.1, float(epoch_size_multiplier)))))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def make_group_balanced_sampler(
    rows: Sequence[Dict[str, str]],
    column: str,
    power: float,
    max_weight: float,
    epoch_size_multiplier: float,
) -> Optional[WeightedRandomSampler]:
    if not column or power <= 0.0:
        return None
    missing = sum(not str(row.get(column, "")).strip() for row in rows)
    if missing:
        raise ValueError(f"Sampling group column {column!r} is missing for {missing} training rows")
    counts = Counter(str(row[column]) for row in rows)
    largest = max(counts.values(), default=1)
    cap = max(1.0, float(max_weight))
    weights = [min(cap, (largest / max(counts[str(row[column])], 1)) ** power) for row in rows]
    num_samples = max(1, int(round(len(weights) * max(0.1, float(epoch_size_multiplier)))))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def modality_gate_priors(model: nn.Module) -> Dict[str, List[float]]:
    gate = getattr(model, "class_modality_gate", None)
    if gate is None:
        return {"structure": [], "rbp_positional": []}
    with torch.no_grad():
        return {
            "structure": torch.sigmoid(gate.struct_prior).detach().cpu().tolist(),
            "rbp_positional": torch.sigmoid(gate.rbp_prior).detach().cpu().tolist(),
        }


def write_class_modality_gate(path: Path, predictions: Mapping[str, object], priors: Mapping[str, List[float]]) -> None:
    struct_mean = predictions.get("class_structure_gate_mean", [])
    struct_true = predictions.get("true_class_structure_gate_mean", [])
    rbp_mean = predictions.get("class_rbp_pos_gate_mean", [])
    rbp_true = predictions.get("true_class_rbp_pos_gate_mean", [])
    support = predictions.get("true_class_gate_support", [])
    struct_prior = priors.get("structure", [])
    rbp_prior = priors.get("rbp_positional", [])
    fields = [
        "mod_type",
        "support",
        "structure_gate_prior",
        "rbp_positional_gate_prior",
        "mean_sequence_gate_for_class_logit",
        "mean_structure_gate_for_class_logit",
        "mean_rbp_pos_gate_for_class_logit",
        "mean_structure_gate_on_true_samples",
        "mean_rbp_pos_gate_on_true_samples",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, name in enumerate(CLASS_NAMES):
            s_mean = float(struct_mean[idx]) if isinstance(struct_mean, list) and idx < len(struct_mean) else 0.0
            r_mean = float(rbp_mean[idx]) if isinstance(rbp_mean, list) and idx < len(rbp_mean) else 0.0
            writer.writerow(
                {
                    "mod_type": name,
                    "support": int(support[idx]) if isinstance(support, list) and idx < len(support) else "",
                    "structure_gate_prior": float(struct_prior[idx]) if idx < len(struct_prior) else "",
                    "rbp_positional_gate_prior": float(rbp_prior[idx]) if idx < len(rbp_prior) else "",
                    "mean_sequence_gate_for_class_logit": max(0.0, 1.0 - s_mean - r_mean),
                    "mean_structure_gate_for_class_logit": s_mean,
                    "mean_rbp_pos_gate_for_class_logit": r_mean,
                    "mean_structure_gate_on_true_samples": float(struct_true[idx]) if isinstance(struct_true, list) and idx < len(struct_true) else "",
                    "mean_rbp_pos_gate_on_true_samples": float(rbp_true[idx]) if isinstance(rbp_true, list) and idx < len(rbp_true) else "",
                }
            )


def run(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    configure_structure_inputs(args.extra_struct_fields, args.node_feature_fields)
    configure_pair_candidates(args.max_pair_candidates)

    rows = read_tsv(Path(args.dataset))
    configure_classes(infer_class_names(rows, args.class_names))
    seq_residual_focus_indices = parse_class_indices(args.seq_residual_focus_classes)
    afamily_classes = existing_class_names(args.afamily_classes, "--afamily-classes")
    afamily_indices = tuple({name: idx for idx, name in enumerate(CLASS_NAMES)}[name] for name in afamily_classes)
    m1a_index = CLASS_NAMES.index("m1A") if "m1A" in CLASS_NAMES else -1
    multi_length_columns = tuple(parse_csv_fields(args.multi_length_columns))
    multi_length_lengths = tuple(parse_int_fields(args.multi_length_lengths))
    if args.seq_encoder == "multi_length" and len(multi_length_columns) != len(multi_length_lengths):
        raise ValueError("--multi-length-columns and --multi-length-lengths must contain the same number of values.")
    if args.max_rows:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_rows]
    feature_site_filter = {str(row.get("site_id", "")) for row in rows} if args.max_rows else None
    if args.disable_structure and not args.rnafold.strip():
        fold_by_id: Mapping[str, Mapping[str, str]] = EmptyStructureFeatures()
    else:
        fold_by_id = load_rnafold_filtered(Path(args.rnafold), feature_site_filter)
        rows = [row for row in rows if row.get("site_id") in fold_by_id]
    train_rows, valid_rows, test_rows = split_rows(rows, args.split)
    teacher_m1a_scores = load_teacher_m1a_scores(args.distill_teacher_predictions, args.distill_teacher_field)
    if teacher_m1a_scores:
        missing_teacher = [row["site_id"] for row in train_rows if row["site_id"] not in teacher_m1a_scores]
        if missing_teacher:
            raise ValueError(
                f"Missing teacher m1A scores for {len(missing_teacher)} training rows; first site_id={missing_teacher[0]}"
            )
    if args.m1a_aux_weight > 0 and m1a_index < 0:
        raise ValueError("m1A auxiliary loss was requested, but m1A is not present in CLASS_NAMES.")
    if args.m1a_aux_pos_weight > 0:
        m1a_aux_pos_weight = float(args.m1a_aux_pos_weight)
    else:
        m1a_train_pos = sum(1 for row in train_rows if str(row.get("mod_type", "")) == "m1A")
        m1a_train_neg = max(1, len(train_rows) - m1a_train_pos)
        m1a_aux_pos_weight = float(max(1.0, m1a_train_neg / max(1, m1a_train_pos))) if m1a_train_pos > 0 else 1.0

    rbp_by_id, rbp_fields = load_rbp_features(args.rbp_features) if args.use_rbp else ({}, [])
    rbp_dim = len(rbp_fields) if args.use_rbp else 0
    rbp_pos_by_id, rbp_pos_shape, rbp_pos_metadata = (
        load_rbp_positional_features(args.rbp_positional_features, feature_site_filter)
        if args.use_rbp_positional
        else ({}, (0, 0, 0), {})
    )
    if args.seq_encoder == "multi_length":
        dataset_cls = RMBaseFusionMultiLengthPositionalDataset if args.use_rbp_positional else RMBaseFusionMultiLengthDataset
        dataset_extra = {
            "multi_length_columns": multi_length_columns,
            "multi_length_lengths": multi_length_lengths,
        }
        if args.use_rbp_positional:
            dataset_extra.update(
                {
                    "rbp_pos_by_id": rbp_pos_by_id,
                    "rbp_pos_shape": rbp_pos_shape,
                    "shuffle_rbp_bins": args.shuffle_rbp_bins,
                }
            )
    else:
        dataset_cls = RMBaseFusionPositionalDataset if args.use_rbp_positional else RMBaseFusionDataset
        dataset_extra = (
            {
                "rbp_pos_by_id": rbp_pos_by_id,
                "rbp_pos_shape": rbp_pos_shape,
                "shuffle_rbp_bins": args.shuffle_rbp_bins,
            }
            if args.use_rbp_positional
            else {}
        )
    train_ds = dataset_cls(
        train_rows,
        fold_by_id,
        args.input_length,
        rbp_by_id=rbp_by_id,
        rbp_dim=rbp_dim,
        shuffle_structure=args.shuffle_structure,
        shuffle_rbp=args.shuffle_rbp,
        disable_pair_edges=args.disable_pair_edges,
        shuffle_structure_nodes=args.shuffle_structure_nodes,
        shuffle_pair_weights=args.shuffle_pair_weights,
        rewire_pair_edges=args.rewire_pair_edges,
        seed=args.seed,
        sequence_column=args.sequence_column,
        **dataset_extra,
    )
    valid_ds = dataset_cls(
        valid_rows,
        fold_by_id,
        args.input_length,
        rbp_by_id=rbp_by_id,
        rbp_dim=rbp_dim,
        struct_stats=train_ds.struct_stats,
        rbp_stats=train_ds.rbp_stats,
        shuffle_structure=args.shuffle_structure,
        shuffle_rbp=args.shuffle_rbp,
        disable_pair_edges=args.disable_pair_edges,
        shuffle_structure_nodes=args.shuffle_structure_nodes,
        shuffle_pair_weights=args.shuffle_pair_weights,
        rewire_pair_edges=args.rewire_pair_edges,
        seed=args.seed,
        sequence_column=args.sequence_column,
        **dataset_extra,
    )
    test_ds = dataset_cls(
        test_rows,
        fold_by_id,
        args.input_length,
        rbp_by_id=rbp_by_id,
        rbp_dim=rbp_dim,
        struct_stats=train_ds.struct_stats,
        rbp_stats=train_ds.rbp_stats,
        shuffle_structure=args.shuffle_structure,
        shuffle_rbp=args.shuffle_rbp,
        disable_pair_edges=args.disable_pair_edges,
        shuffle_structure_nodes=args.shuffle_structure_nodes,
        shuffle_pair_weights=args.shuffle_pair_weights,
        rewire_pair_edges=args.rewire_pair_edges,
        seed=args.seed,
        sequence_column=args.sequence_column,
        **dataset_extra,
    )
    if args.use_rbp_positional:
        rbp_pos_by_id = {}
    oversample_classes = tuple(parse_csv_fields(args.oversample_classes))
    train_sampler = make_group_balanced_sampler(
        train_rows,
        args.sampling_group_column,
        args.sampling_group_power,
        args.sampling_group_max_weight,
        args.sampler_epoch_size_multiplier,
    )
    if train_sampler is None:
        train_sampler = make_focus_oversampler(
            train_rows,
            oversample_classes,
            args.oversample_factor,
            args.sampler_epoch_size_multiplier,
        )
    train_loader_dataset: Dataset = (
        TeacherM1AScoreDataset(train_ds, train_rows, teacher_m1a_scores)
        if teacher_m1a_scores
        else train_ds
    )
    train_loader = DataLoader(
        train_loader_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_batch_v4,
        num_workers=0,
    )
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch_v4, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch_v4, num_workers=0)

    cfg = MotifStructV4Config(
        input_length=args.input_length,
        num_classes=len(CLASS_NAMES),
        d_model=args.d_model,
        gat_layers=args.gat_layers,
        dropout=args.dropout,
        rbp_dim=rbp_dim,
        rbp_pos_shape=rbp_pos_shape,
        use_structure=not args.disable_structure,
        use_rbp=args.use_rbp,
        use_rbp_positional=args.use_rbp_positional,
        use_position=not args.disable_position,
        structure_uses_sequence=args.structure_uses_sequence,
        structure_encoder=args.structure_encoder,
        rbp_pos_encoder=args.rbp_pos_encoder,
        structure_scale=args.structure_scale,
        rbp_scale=args.rbp_scale,
        rbp_pos_scale=args.rbp_pos_scale,
        gate_init_logit=args.gate_init_logit,
        rbp_pos_gate_init_logit=args.rbp_pos_gate_init_logit,
        modality_dropout_struct=args.modality_dropout_struct,
        modality_dropout_rbp=args.modality_dropout_rbp,
        struct_fields=tuple(STRUCT_FIELDS),
        node_feature_fields=tuple(NODE_FEATURE_FIELDS),
        seq_encoder=args.seq_encoder,
        seq_residual_encoder=args.seq_residual_encoder,
        seq_residual_focus_indices=seq_residual_focus_indices,
        seq_residual_scale=args.seq_residual_scale,
        seq_residual_gate_init_logit=args.seq_residual_gate_init_logit,
        multi_length_columns=multi_length_columns,
        multi_length_lengths=multi_length_lengths,
        multi_length_base_encoder=args.multi_length_base_encoder,
        multi_length_share_encoder=args.multi_length_share_encoder,
        m1a_aux_weight=args.m1a_aux_weight,
        m1a_aux_pos_weight=m1a_aux_pos_weight,
        afamily_classes=afamily_classes,
        afamily_indices=afamily_indices,
        afamily_aux_weight=args.afamily_aux_weight,
        afamily_residual_scale=args.afamily_residual_scale,
        afamily_fusion_mode=args.afamily_fusion_mode,
        afamily_m1a_expert_scale=args.afamily_m1a_expert_scale,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ff_mult=args.transformer_ff_mult,
        max_pair_candidates=args.max_pair_candidates,
        confidence_gated_structure=args.confidence_gated_structure,
        structure_confidence_floor=args.structure_confidence_floor,
    )
    model = MotifStructFusionV4(cfg).to(device)
    initialized_from = ""
    if args.init_model_path:
        init_checkpoint = torch.load(args.init_model_path, map_location=device)
        init_state = init_checkpoint.get("model_state_dict", init_checkpoint)
        model.load_state_dict(init_state)
        initialized_from = str(args.init_model_path)
        print(f"initialized_from={initialized_from}", flush=True)
    weights = class_weights_from_rows(train_rows, args.class_weight_power, device)
    afamily_global_to_local: Optional[torch.Tensor] = None
    afamily_weights: Optional[torch.Tensor] = None
    if args.afamily_aux_weight > 0 and afamily_indices:
        afamily_global_to_local = torch.full((len(CLASS_NAMES),), -1, dtype=torch.long, device=device)
        for local_idx, global_idx in enumerate(afamily_indices):
            afamily_global_to_local[int(global_idx)] = int(local_idx)
        afamily_weights = afamily_class_weights_from_rows(
            train_rows,
            afamily_classes,
            args.afamily_class_weight_power,
            args.afamily_max_class_weight,
            args.afamily_m1a_weight_multiplier,
            device,
        )
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
    accumulated_training_seconds = 0.0
    epochs_without_improvement = 0
    early_stopped = False
    stop_reason = ""
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("variant") != "MotifStructFusionV4":
            raise RuntimeError(f"Checkpoint at {checkpoint_path} is not a MotifStructFusionV4 checkpoint.")
        model.load_state_dict(checkpoint["model_state_dict"])
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
        print(
            f"resumed_from={checkpoint_path} start_epoch={start_epoch} best_score={best_score:.4f} "
            f"no_improve={epochs_without_improvement}",
            flush=True,
        )
    elif initialized_from:
        initial_valid_pred = predict(model, valid_loader, device)
        initial_valid_metrics = evaluate(initial_valid_pred["probs"], initial_valid_pred["labels"])  # type: ignore[arg-type]
        add_grouped_detection_metrics(
            initial_valid_metrics,
            initial_valid_pred,
            valid_rows,
            args.evaluation_group_column,
            args.group_positive_class,
            args.group_min_positive,
        )
        best_score = metric_selection_score(initial_valid_metrics, args)
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"initial_validation selection_score={best_score:.4f} "
            f"valid_{args.selection_metric}={float(initial_valid_metrics[args.selection_metric]):.4f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        losses = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.aux_seq_weight,
            args.aux_struct_weight,
            args.gate_l1_weight,
            args.rbp_pos_gate_l1_weight,
            args.seq_residual_gate_l1_weight,
            args.m1a_aux_weight,
            m1a_aux_pos_weight,
            args.m1a_aux_loss,
            args.m1a_aux_focal_gamma,
            m1a_index,
            args.afamily_aux_weight,
            afamily_global_to_local,
            afamily_weights,
            args.afamily_loss,
            args.afamily_focal_gamma,
            args.distill_m1a_weight,
            args.distill_m1a_rank_weight,
            args.distill_temperature,
            args.distill_teacher_positive_weight,
            args.distill_rank_pairs,
            args.distill_min_teacher_gap,
            args.length_aux_weight,
            args.length_gate_entropy_weight,
            args.grad_accum_steps,
        )
        scheduler.step()
        valid_pred = predict(model, valid_loader, device)
        valid_metrics = evaluate(valid_pred["probs"], valid_pred["labels"])  # type: ignore[arg-type]
        add_grouped_detection_metrics(
            valid_metrics,
            valid_pred,
            valid_rows,
            args.evaluation_group_column,
            args.group_positive_class,
            args.group_min_positive,
        )
        accumulated_training_seconds += time.perf_counter() - epoch_started
        metric_value = float(valid_metrics[args.selection_metric])
        score = metric_selection_score(valid_metrics, args)
        improved = score > best_score + args.early_stopping_min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "train": losses,
                "validation": valid_metrics,
                "validation_metric_value": metric_value,
                "selection_score": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "improved": improved,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )
        torch.save(
            {
                "variant": "MotifStructFusionV4",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_state_dict": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "accumulated_training_seconds": accumulated_training_seconds,
                "history": history,
                "config": asdict(cfg),
            },
            checkpoint_path,
        )
        print(
            f"epoch={epoch} loss={losses['loss']:.4f} valid_{args.selection_metric}={metric_value:.4f} "
            f"selection_score={score:.4f} "
            f"gate=({losses['gate_seq']:.2f},{losses['gate_structure']:.2f},{losses['gate_rbp']:.2f}) "
            f"main={losses['main_loss']:.4f} struct_aux={losses['struct_aux_loss']:.4f} "
            f"length_aux={losses['length_aux_loss']:.4f} length_entropy={losses['length_gate_entropy']:.4f} "
            f"gate_l1={losses['structure_gate_l1']:.4f} rbp_pos_gate_l1={losses['rbp_pos_gate_l1']:.4f} "
            f"seq_res_gate_l1={losses['seq_residual_gate_l1']:.4f} m1a_aux={losses['m1a_aux_loss']:.4f} "
            f"afamily_aux={losses['afamily_aux_loss']:.4f} distill={losses['distill_m1a_loss']:.4f} "
            f"distill_rank={losses['distill_m1a_rank_loss']:.4f} "
            f"no_improve={epochs_without_improvement}",
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
        model.load_state_dict(best_state)
    valid_pred = predict(model, valid_loader, device)
    test_pred = predict(model, test_loader, device)
    valid_metrics = evaluate(valid_pred["probs"], valid_pred["labels"])  # type: ignore[arg-type]
    test_metrics = evaluate(test_pred["probs"], test_pred["labels"])  # type: ignore[arg-type]
    add_grouped_detection_metrics(
        valid_metrics,
        valid_pred,
        valid_rows,
        args.evaluation_group_column,
        args.group_positive_class,
        args.group_min_positive,
    )
    add_grouped_detection_metrics(
        test_metrics,
        test_pred,
        test_rows,
        args.evaluation_group_column,
        args.group_positive_class,
        args.group_min_positive,
    )
    gate_priors = modality_gate_priors(model)
    training_peak_memory_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    efficiency = measure_inference_efficiency(model, test_loader, device)
    efficiency["training_seconds"] = accumulated_training_seconds
    efficiency["training_peak_memory_bytes"] = training_peak_memory_bytes

    torch.save(
        {
            "variant": "MotifStructFusionV4",
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "class_names": CLASS_NAMES,
            "struct_stats": train_ds.struct_stats,
            "rbp_fields": rbp_fields,
            "rbp_stats": train_ds.rbp_stats,
            "rbp_positional_metadata": rbp_pos_metadata,
            "training_args": vars(args),
        },
        out_dir / "model.pt",
    )
    metrics: Dict[str, object] = {
        "dataset": args.dataset,
        "rnafold": args.rnafold,
        "rbp_features": args.rbp_features if args.use_rbp else "",
        "rbp_positional_features": args.rbp_positional_features if args.use_rbp_positional else "",
        "variant": "MotifStructFusionV4",
        "task": f"RMBase_hg38_{len(CLASS_NAMES)}_class_center_site_prediction",
        "classes": CLASS_NAMES,
        "device": str(device),
        "split": args.split,
        "row_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "config": asdict(cfg),
        "structure_fields": list(STRUCT_FIELDS),
        "node_feature_fields": list(NODE_FEATURE_FIELDS),
        "loss": args.loss,
        "aux_loss": {
            "seq": args.aux_seq_weight,
            "structure": args.aux_struct_weight,
            "length_branch": args.length_aux_weight,
            "m1A": args.m1a_aux_weight,
            "A_family": args.afamily_aux_weight,
        },
        "gate_l1_weight": args.gate_l1_weight,
        "rbp_pos_gate_l1_weight": args.rbp_pos_gate_l1_weight,
        "seq_residual_gate_l1_weight": args.seq_residual_gate_l1_weight,
        "m1a_aux_weight": args.m1a_aux_weight,
        "m1a_aux_pos_weight": m1a_aux_pos_weight,
        "m1a_aux_loss": args.m1a_aux_loss,
        "m1a_aux_focal_gamma": args.m1a_aux_focal_gamma,
        "afamily_classes": list(afamily_classes),
        "afamily_aux_weight": args.afamily_aux_weight,
        "afamily_residual_scale": args.afamily_residual_scale,
        "afamily_fusion_mode": args.afamily_fusion_mode,
        "afamily_m1a_expert_scale": args.afamily_m1a_expert_scale,
        "afamily_loss": args.afamily_loss,
        "afamily_focal_gamma": args.afamily_focal_gamma,
        "afamily_class_weight_power": args.afamily_class_weight_power,
        "afamily_max_class_weight": args.afamily_max_class_weight,
        "afamily_m1a_weight_multiplier": args.afamily_m1a_weight_multiplier,
        "afamily_effective_class_weights": (
            {
                name: float(afamily_weights[idx].detach().cpu())
                for idx, name in enumerate(afamily_classes)
            }
            if afamily_weights is not None
            else {}
        ),
        "distillation": {
            "teacher_predictions": args.distill_teacher_predictions,
            "teacher_field": args.distill_teacher_field,
            "m1a_weight": args.distill_m1a_weight,
            "m1a_rank_weight": args.distill_m1a_rank_weight,
            "temperature": args.distill_temperature,
            "teacher_positive_weight": args.distill_teacher_positive_weight,
            "rank_pairs": args.distill_rank_pairs,
            "min_teacher_gap": args.distill_min_teacher_gap,
            "initialized_from": initialized_from,
            "caveat": (
                "The full EvoRMD-adapted teacher is metadata-dominated and is used only as a leakage-sensitive diagnostic."
                if args.distill_teacher_field == "teacher_full_m1A"
                else ""
            ),
        },
        "length_gate_entropy_weight": args.length_gate_entropy_weight,
        "modality_gate_priors": gate_priors,
        "rbp_positional_metadata": rbp_pos_metadata,
        "seq_residual_focus_classes": parse_csv_fields(args.seq_residual_focus_classes),
        "oversampling": {
            "classes": list(oversample_classes),
            "factor": args.oversample_factor,
            "sampler_epoch_size_multiplier": args.sampler_epoch_size_multiplier,
            "enabled": train_sampler is not None,
        },
        "selection_metric": args.selection_metric,
        "selection_focus_class": args.selection_focus_class,
        "selection_focus_metric": args.selection_focus_metric,
        "selection_focus_weight": args.selection_focus_weight,
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
        "validation_gate_mean": valid_pred["gate_mean"],
        "test_gate_mean": test_pred["gate_mean"],
        "validation_class_structure_gate_mean": valid_pred["class_structure_gate_mean"],
        "test_class_structure_gate_mean": test_pred["class_structure_gate_mean"],
        "test_true_class_structure_gate_mean": test_pred["true_class_structure_gate_mean"],
        "validation_class_rbp_pos_gate_mean": valid_pred["class_rbp_pos_gate_mean"],
        "test_class_rbp_pos_gate_mean": test_pred["class_rbp_pos_gate_mean"],
        "test_true_class_rbp_pos_gate_mean": test_pred["true_class_rbp_pos_gate_mean"],
        "validation_class_seq_residual_gate_mean": valid_pred["class_seq_residual_gate_mean"],
        "test_class_seq_residual_gate_mean": test_pred["class_seq_residual_gate_mean"],
        "test_true_class_seq_residual_gate_mean": test_pred["true_class_seq_residual_gate_mean"],
        "m1a_aux_pos_weight_effective": m1a_aux_pos_weight,
        "length_gate_labels": test_pred["length_gate_labels"],
        "validation_class_length_gate_mean": valid_pred["class_length_gate_mean"],
        "test_class_length_gate_mean": test_pred["class_length_gate_mean"],
        "test_true_class_length_gate_mean": test_pred["true_class_length_gate_mean"],
        "history": history,
        "caveats": [
            "V4 uses class-specific structure gates; gate values should be interpreted per class.",
            "By default the structure branch is sequence-free, so shuffled-structure controls are stricter.",
            "The optional RBP positional branch represents RBP peaks as RBP-by-distance-bin tokens with class-specific attention and gates.",
            "The sequence encoder is configurable; motif is the historical V3 CNN+BiGRU backbone, kmer_extended adds 4/5-mer embeddings, dilated adds residual dilated CNN blocks, and hybrid adds a Transformer after the motif CNN stem.",
            "The optional sequence residual branch adds a class-gated secondary sequence encoder logit residual, optionally restricted to focus classes.",
            "The A-family head can run in legacy residual mode or early_class mode; early_class fuses sequence, structure and RBP tokens before the local A-family correction.",
            "The multi_length sequence encoder fuses 41/101/201 nt evidence with class-specific softmax gates when --seq-encoder multi_length is used.",
            "Structure benefit should be judged per class against sequence_only and shuffled_structure.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions(out_dir / "valid_predictions.tsv", valid_pred)
    write_predictions(out_dir / "test_predictions.tsv", test_pred)
    write_per_class_metrics(out_dir / "valid_per_class_metrics.tsv", valid_metrics)
    write_per_class_metrics(out_dir / "test_per_class_metrics.tsv", test_metrics)
    write_class_structure_gate(out_dir / "valid_per_class_structure_gate.tsv", valid_pred)
    write_class_structure_gate(out_dir / "test_per_class_structure_gate.tsv", test_pred)
    write_class_rbp_positional_gate(out_dir / "valid_per_class_rbp_positional_gate.tsv", valid_pred)
    write_class_rbp_positional_gate(out_dir / "test_per_class_rbp_positional_gate.tsv", test_pred)
    write_class_sequence_residual_gate(out_dir / "valid_per_class_sequence_residual_gate.tsv", valid_pred)
    write_class_sequence_residual_gate(out_dir / "test_per_class_sequence_residual_gate.tsv", test_pred)
    write_class_length_gate(out_dir / "valid_per_class_length_gate.tsv", valid_pred)
    write_class_length_gate(out_dir / "test_per_class_length_gate.tsv", test_pred)
    write_class_modality_gate(out_dir / "valid_per_class_modality_gate.tsv", valid_pred, gate_priors)
    write_class_modality_gate(out_dir / "test_per_class_modality_gate.tsv", test_pred, gate_priors)
    write_rbp_positional_attention(out_dir / "valid_rbp_positional_attention.tsv", valid_pred, rbp_pos_metadata)
    write_rbp_positional_attention(out_dir / "test_rbp_positional_attention.tsv", test_pred, rbp_pos_metadata)
    write_afamily_modality_attention(out_dir / "valid_afamily_modality_attention.tsv", valid_pred)
    write_afamily_modality_attention(out_dir / "test_afamily_modality_attention.tsv", test_pred)
    write_confusion_matrix(out_dir / "test_confusion_matrix.tsv", test_metrics["confusion_matrix"])  # type: ignore[arg-type]
    write_gate_summary(out_dir / "test_modality_gate_summary.tsv", test_pred["gate_mean"])  # type: ignore[arg-type]
    (out_dir / "README.md").write_text(
        "# MotifStructFusionV4\n\n"
        "Strong motif/k-mer sequence backbone plus sequence-free class-specific RNAfold structure gates and optional RBP positional branch.\n",
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MotifStructFusionV4 on RMBase11.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_mmseqs80_strict_splits.tsv")
    parser.add_argument("--class-names", default="", help="Optional comma-separated class order; inferred from class_index if omitted.")
    parser.add_argument(
        "--rnafold",
        default="rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv",
        help="RNAfold feature TSV. May be empty only together with --disable-structure.",
    )
    parser.add_argument("--rbp-features", default="rmbase11_full_work/rbp/rmbase11_rbp_features.tsv")
    parser.add_argument("--rbp-positional-features", default="rmbase11_full_work/rbp/rmbase11_rbp_positional_features.tsv")
    parser.add_argument("--split", default="split_mmseqs80_cluster")
    parser.add_argument("--out-dir", default="rmbase11_full_work/models/motif_struct_fusion_v4/full")
    parser.add_argument("--input-length", type=int, default=41)
    parser.add_argument("--sequence-column", default="sequence_41nt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument(
        "--max-pair-candidates",
        type=int,
        default=1,
        help="Maximum partition-function partner edges consumed per nucleotide.",
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--seq-encoder",
        choices=["motif", "kmer_extended", "dilated", "transformer", "hybrid", "multi_length"],
        default="motif",
    )
    parser.add_argument(
        "--multi-length-columns",
        default="sequence_41nt,sequence_101nt,sequence_201nt",
        help="Comma-separated sequence columns used when --seq-encoder multi_length.",
    )
    parser.add_argument(
        "--multi-length-lengths",
        default="41,101,201",
        help="Comma-separated sequence lengths matching --multi-length-columns.",
    )
    parser.add_argument(
        "--multi-length-base-encoder",
        choices=["motif", "kmer_extended", "dilated", "transformer", "hybrid"],
        default="motif",
        help="Base sequence encoder used inside each multi-length branch.",
    )
    parser.add_argument(
        "--multi-length-share-encoder",
        action="store_true",
        help="Share one length-agnostic sequence encoder across all multi-length windows; heads and gates stay class-specific.",
    )
    parser.add_argument(
        "--m1a-aux-weight",
        type=float,
        default=0.0,
        help="Optional auxiliary BCE loss weight for a dedicated m1A vs rest constraint.",
    )
    parser.add_argument(
        "--m1a-aux-pos-weight",
        type=float,
        default=0.0,
        help="Positive-class weight for the m1A auxiliary BCE loss. <=0 uses the training-set neg/pos ratio.",
    )
    parser.add_argument(
        "--m1a-aux-loss",
        choices=["bce", "focal"],
        default="bce",
        help="Loss used by the dedicated m1A-vs-rest auxiliary constraint.",
    )
    parser.add_argument("--m1a-aux-focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--distill-teacher-predictions",
        default="",
        help="Train-split TSV containing a teacher m1A probability keyed by site_id.",
    )
    parser.add_argument("--distill-teacher-field", default="teacher_full_m1A")
    parser.add_argument("--distill-m1a-weight", type=float, default=0.0)
    parser.add_argument("--distill-m1a-rank-weight", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--distill-teacher-positive-weight", type=float, default=0.0)
    parser.add_argument("--distill-rank-pairs", type=int, default=32)
    parser.add_argument("--distill-min-teacher-gap", type=float, default=0.05)
    parser.add_argument(
        "--afamily-classes",
        default="Am,m1A,m6A,m6Am,AtoI",
        help="Comma-separated A-family classes for the embedded local residual head.",
    )
    parser.add_argument(
        "--afamily-aux-weight",
        type=float,
        default=0.0,
        help="Auxiliary local CE loss weight for samples whose true label is in --afamily-classes.",
    )
    parser.add_argument(
        "--afamily-residual-scale",
        type=float,
        default=0.0,
        help="Scale for writing the embedded A-family residual head back into global logits.",
    )
    parser.add_argument(
        "--afamily-fusion-mode",
        choices=["legacy", "early_class"],
        default="legacy",
        help="legacy keeps the old h_main residual head; early_class fuses sequence/structure/RBP tokens before the A-family head.",
    )
    parser.add_argument(
        "--afamily-m1a-expert-scale",
        type=float,
        default=0.0,
        help="Scale of the dedicated multimodal m1A expert inside early_class A-family fusion; 0 disables it.",
    )
    parser.add_argument(
        "--afamily-loss",
        choices=["ce", "focal"],
        default="ce",
        help="Loss used by the embedded A-family local classification head.",
    )
    parser.add_argument("--afamily-focal-gamma", type=float, default=2.0)
    parser.add_argument("--afamily-class-weight-power", type=float, default=0.25)
    parser.add_argument("--afamily-max-class-weight", type=float, default=8.0)
    parser.add_argument(
        "--afamily-m1a-weight-multiplier",
        type=float,
        default=1.0,
        help="Additional multiplier applied to the m1A weight inside the A-family local loss.",
    )
    parser.add_argument(
        "--seq-residual-encoder",
        choices=["none", "kmer_extended", "dilated", "transformer", "hybrid"],
        default="none",
        help="Optional secondary sequence encoder added as a class-gated logit residual.",
    )
    parser.add_argument(
        "--seq-residual-focus-classes",
        default="",
        help="Comma-separated class names allowed to receive the sequence residual, e.g. Gm,m1A,m7G. Empty means all classes.",
    )
    parser.add_argument("--seq-residual-scale", type=float, default=0.1)
    parser.add_argument("--seq-residual-gate-init-logit", type=float, default=-2.0)
    parser.add_argument("--seq-residual-gate-l1-weight", type=float, default=0.0)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ff-mult", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--oversample-classes",
        default="",
        help="Comma-separated training classes to oversample with WeightedRandomSampler, e.g. m1A.",
    )
    parser.add_argument(
        "--oversample-factor",
        type=float,
        default=1.0,
        help="Sampling weight multiplier for --oversample-classes. Validation/test splits are unchanged.",
    )
    parser.add_argument(
        "--sampler-epoch-size-multiplier",
        type=float,
        default=1.0,
        help="Number of sampled training examples per epoch as a multiple of the original train rows.",
    )
    parser.add_argument(
        "--sampling-group-column",
        default="",
        help="Optional training column used for inverse-frequency group sampling, e.g. original_mod_type.",
    )
    parser.add_argument("--sampling-group-power", type=float, default=0.0)
    parser.add_argument("--sampling-group-max-weight", type=float, default=100.0)
    parser.add_argument("--aux-seq-weight", type=float, default=0.1)
    parser.add_argument("--aux-struct-weight", type=float, default=0.05)
    parser.add_argument("--length-aux-weight", type=float, default=0.0)
    parser.add_argument("--length-gate-entropy-weight", type=float, default=0.0)
    parser.add_argument("--gate-l1-weight", type=float, default=0.001)
    parser.add_argument("--structure-scale", type=float, default=0.5)
    parser.add_argument("--rbp-scale", type=float, default=0.05)
    parser.add_argument("--rbp-pos-scale", type=float, default=0.05)
    parser.add_argument("--gate-init-logit", type=float, default=-2.0)
    parser.add_argument("--rbp-pos-gate-init-logit", type=float, default=-2.0)
    parser.add_argument("--rbp-pos-gate-l1-weight", type=float, default=0.001)
    parser.add_argument("--modality-dropout-struct", type=float, default=0.0)
    parser.add_argument("--modality-dropout-rbp", type=float, default=0.0)
    parser.add_argument(
        "--selection-metric",
        choices=["macro_AUPR_ovr", "macro_F1", "MCC", "balanced_accuracy", "macro_group_AUPR"],
        default="macro_AUPR_ovr",
    )
    parser.add_argument(
        "--evaluation-group-column",
        default="",
        help="Optional column for validation/test subgroup metrics, e.g. original_mod_type.",
    )
    parser.add_argument("--group-positive-class", default="modified")
    parser.add_argument(
        "--group-min-positive",
        type=int,
        default=1,
        help="Minimum positive rows required for a group to enter macro_group_AUPR; all groups remain reported.",
    )
    parser.add_argument(
        "--selection-focus-class",
        default="",
        help="Optional class whose validation metric is added to the checkpoint selection score.",
    )
    parser.add_argument(
        "--selection-focus-metric",
        choices=["AUPR_ovr", "F1", "MCC_ovr"],
        default="AUPR_ovr",
    )
    parser.add_argument("--selection-focus-weight", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--init-model-path",
        default="",
        help="Optional model.pt used to initialize a fresh optimizer run before fine-tuning.",
    )
    parser.add_argument("--disable-structure", action="store_true")
    parser.add_argument("--structure-uses-sequence", action="store_true")
    parser.add_argument(
        "--structure-encoder",
        choices=["gat", "cnn_control"],
        default="gat",
        help="Use the proposed pair-aware GAT or a capacity-matched generic CNN control.",
    )
    parser.add_argument("--use-rbp", action="store_true")
    parser.add_argument("--use-rbp-positional", action="store_true")
    parser.add_argument(
        "--rbp-pos-encoder",
        choices=["attention", "mlp_control"],
        default="attention",
        help="Use class-token attention or a parameter-matched mean-pooling MLP control.",
    )
    parser.add_argument("--disable-position", action="store_true")
    parser.add_argument(
        "--extra-struct-fields",
        default="",
        help="Comma-separated scalar structure columns to append to the default STRUCT_FIELDS.",
    )
    parser.add_argument(
        "--node-feature-fields",
        default="paired_flags",
        help="Comma-separated per-position structure columns. Use paired_flags,loop_flags for RNAfold loop experiments.",
    )
    parser.add_argument("--shuffle-structure", action="store_true")
    parser.add_argument(
        "--shuffle-structure-nodes",
        action="store_true",
        help="Permute probability-derived node features within each sequence while preserving their marginals.",
    )
    parser.add_argument(
        "--shuffle-pair-weights",
        action="store_true",
        help="Permute positive pair probabilities within each sequence while preserving topology.",
    )
    parser.add_argument(
        "--rewire-pair-edges",
        action="store_true",
        help="Reassign pair targets while preserving directed source and target degree counts.",
    )
    parser.add_argument(
        "--confidence-gated-structure",
        action="store_true",
        help="Attenuate the structure residual using local ensemble entropy and optional cross-scale variance.",
    )
    parser.add_argument("--structure-confidence-floor", type=float, default=0.1)
    parser.add_argument("--shuffle-rbp", action="store_true")
    parser.add_argument(
        "--disable-pair-edges",
        action="store_true",
        help="Remove RNAfold base-pair edges while retaining paired flags, scalar structure features, and parameters.",
    )
    parser.add_argument(
        "--shuffle-rbp-bins",
        action="store_true",
        help="Randomly permute RBP distance bins per sample while preserving RBP feature values and model capacity.",
    )
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
