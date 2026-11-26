#!/usr/bin/env python
"""Train StructDualRM model variants on the prepared joint dataset."""

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
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


BASE_TO_ID = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4}
REGION_TO_ID = {"5UTR": 0, "CDS": 1, "3UTR": 2, "UTR": 3}
STRUCT_FIELDS = [
    "center_paired",
    "central_11nt_paired_fraction",
    "central_21nt_paired_fraction",
    "paired_fraction_101nt",
    "MFE_per_nt",
    "loop_span_around_center",
]


def encode_sequence(seq: str, length: int = 101) -> torch.Tensor:
    ids = [BASE_TO_ID.get(base.upper(), 0) for base in seq[:length]]
    ids.extend([0] * (length - len(ids)))
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


def parse_partner_indices(base_pair_list: str, length: int = 101) -> torch.Tensor:
    partners = [-1] * length
    for item in str(base_pair_list or "").split(";"):
        if not item or "-" not in item:
            continue
        left, right = [parse_int(x) for x in item.split("-", 1)]
        if 1 <= left <= length and 1 <= right <= length:
            partners[left - 1] = right - 1
            partners[right - 1] = left - 1
    return torch.tensor(partners, dtype=torch.long)


def parse_paired_flags(value: str, length: int = 101) -> torch.Tensor:
    flags = [parse_float(x) for x in str(value or "").split(",") if x != ""]
    flags = flags[:length] + [0.0] * max(0, length - len(flags))
    return torch.tensor(flags, dtype=torch.float32)


@dataclass
class StructDualRMConfig:
    input_length: int = 101
    d_model: int = 64
    num_layers: int = 1
    num_heads: int = 4
    dropout: float = 0.2
    context_dim: int = 32
    struct_dim: int = 32


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


class StructDualRMModel(nn.Module):
    def __init__(self, cfg: StructDualRMConfig, variant: str = "M4"):
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
        self.struct_mlp = nn.Sequential(nn.Linear(6, cfg.struct_dim), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.struct_dim, cfg.struct_dim))
        self.region_emb = nn.Embedding(len(REGION_TO_ID), cfg.context_dim // 2)
        self.context_mlp = nn.Sequential(nn.Linear(2, cfg.context_dim // 2), nn.ReLU(), nn.Linear(cfg.context_dim // 2, cfg.context_dim // 2))

        z_seq_dim = cfg.d_model * 3
        z_graph_dim = cfg.d_model * 4
        z_struct_dim = cfg.struct_dim
        z_context_dim = cfg.context_dim
        gate_dim = z_seq_dim + z_graph_dim + z_struct_dim + z_context_dim
        self.gate_m1a = nn.Sequential(nn.Linear(gate_dim, cfg.d_model), nn.ReLU(), nn.Linear(cfg.d_model, 1), nn.Sigmoid())
        self.gate_m6a = nn.Sequential(nn.Linear(gate_dim, cfg.d_model), nn.ReLU(), nn.Linear(cfg.d_model, 1), nn.Sigmoid())
        head_dim = z_seq_dim + z_graph_dim + z_struct_dim + z_context_dim
        self.head_m1a = nn.Sequential(nn.Linear(head_dim, cfg.d_model), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, 1))
        self.head_m6a = nn.Sequential(nn.Linear(head_dim, cfg.d_model), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, 1))

    def encode_seq(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, length)
        dist = torch.abs(pos - (length // 2))
        x = self.base_emb(seq) + self.pos_emb(pos) + self.dist_emb(dist)
        return self.transformer(x)

    def pool_seq(self, h: torch.Tensor) -> torch.Tensor:
        center = h[:, h.shape[1] // 2, :]
        local = h[:, h.shape[1] // 2 - 5 : h.shape[1] // 2 + 6, :].mean(dim=1)
        attn = self.seq_attn(h)
        return torch.cat([center, local, attn], dim=-1)

    def pool_graph(self, g: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
        center_idx = g.shape[1] // 2
        center = g[:, center_idx, :]
        local = g[:, center_idx - 10 : center_idx + 11, :].mean(dim=1)
        attn = self.graph_attn(g)
        center_partner = partner[:, center_idx]
        mask = (center_partner >= 0).float().unsqueeze(-1)
        gather_idx = center_partner.clamp_min(0).view(-1, 1, 1).expand(-1, 1, g.shape[-1])
        pair = torch.gather(g, 1, gather_idx).squeeze(1) * mask
        return torch.cat([center, local, attn, pair], dim=-1)

    def forward(
        self,
        seq: torch.Tensor,
        paired_flags: torch.Tensor,
        pair_partner: torch.Tensor,
        struct: torch.Tensor,
        region: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h = self.encode_seq(seq)
        z_seq = self.pool_seq(h)
        if self.variant in {"M2", "M3", "M4"}:
            g = h + paired_flags.unsqueeze(-1)
            for layer in self.graph_layers:
                g = layer(g, pair_partner)
            z_graph = self.pool_graph(g, pair_partner)
        else:
            z_graph = torch.zeros(seq.shape[0], self.cfg.d_model * 4, device=seq.device)

        if self.variant in {"M1", "M3", "M4"}:
            z_struct = self.struct_mlp(struct)
        else:
            z_struct = torch.zeros(seq.shape[0], self.cfg.struct_dim, device=seq.device)

        if self.variant == "M4":
            z_context = torch.cat([self.region_emb(region), self.context_mlp(context)], dim=-1)
        else:
            z_context = torch.zeros(seq.shape[0], self.cfg.context_dim, device=seq.device)

        gate_input = torch.cat([z_seq, z_graph, z_struct, z_context], dim=-1)
        if self.variant in {"M3", "M4"}:
            gate_m1a = self.gate_m1a(gate_input)
            gate_m6a = self.gate_m6a(gate_input)
        else:
            gate_m1a = torch.ones(seq.shape[0], 1, device=seq.device)
            gate_m6a = torch.ones(seq.shape[0], 1, device=seq.device)
        z_m1a = torch.cat([z_seq, gate_m1a * z_graph, gate_m1a * z_struct, z_context], dim=-1)
        z_m6a = torch.cat([z_seq, gate_m6a * z_graph, gate_m6a * z_struct, z_context], dim=-1)
        logits = torch.cat([self.head_m1a(z_m1a), self.head_m6a(z_m6a)], dim=-1)
        return {"logits": logits, "gate_m1A": gate_m1a, "gate_m6A": gate_m6a}


class JointDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, str]], fold_by_id: Dict[str, Dict[str, str]], stats: Optional[Dict[str, List[float]]] = None):
        self.rows = list(rows)
        self.fold_by_id = fold_by_id
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

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        fold = self.fold_by_id[row["site_id"]]
        struct = torch.tensor([parse_float(fold.get(field, 0.0)) for field in STRUCT_FIELDS], dtype=torch.float32)
        struct = (struct - self.mean) / self.std
        tx_len = max(parse_float(row.get("tx_len", 0.0)), 1.0)
        context = torch.tensor([parse_float(row.get("relative_pos", 0.0)), math.log10(tx_len)], dtype=torch.float32)
        return {
            "seq": encode_sequence(row["sequence_101nt"]),
            "paired_flags": parse_paired_flags(fold.get("paired_flags", "")),
            "pair_partner": parse_partner_indices(fold.get("base_pair_list", "")),
            "struct": struct,
            "region": torch.tensor(REGION_TO_ID.get(row.get("region_type", "UTR"), 3), dtype=torch.long),
            "context": context,
            "labels": torch.tensor([parse_float(row["label_m1A"]), parse_float(row["label_m6A"])], dtype=torch.float32),
            "site_id": row["site_id"],
        }


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    tensor_keys = ["seq", "paired_flags", "pair_partner", "struct", "region", "context", "labels"]
    out = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    out["site_id"] = [item["site_id"] for item in batch]
    return out


def load_rows(dataset_path: Path, rnafold_path: Path, max_rows: int, seed: int) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    with open(rnafold_path, "r", encoding="utf-8", newline="") as handle:
        fold_by_id = {row["site_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    rows = []
    with open(dataset_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["site_id"] in fold_by_id:
                rows.append(row)
    random.Random(seed).shuffle(rows)
    if max_rows:
        rows = rows[:max_rows]
    return rows, fold_by_id


def split_rows(rows: Sequence[Dict[str, str]], split_name: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    train = [row for row in rows if row.get(split_name) == "train"]
    valid = [row for row in rows if row.get(split_name) == "valid"]
    test = [row for row in rows if row.get(split_name) == "test"]
    return train, valid, test


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total = 0.0
    count = 0
    class_weights = torch.tensor([1.5, 1.0], dtype=torch.float32, device=device)
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["seq"], batch["paired_flags"], batch["pair_partner"], batch["struct"], batch["region"], batch["context"])["logits"]
        loss_by_class = criterion(logits, batch["labels"])
        loss = (loss_by_class * class_weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu()) * batch["labels"].shape[0]
        count += batch["labels"].shape[0]
    return total / max(count, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    model.eval()
    probs, labels, site_ids, gates1, gates6 = [], [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["seq"], batch["paired_flags"], batch["pair_partner"], batch["struct"], batch["region"], batch["context"])
        probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
        gates1.append(out["gate_m1A"].cpu().numpy())
        gates6.append(out["gate_m6A"].cpu().numpy())
        site_ids.extend(batch["site_id"])
    return np.vstack(probs), np.vstack(labels), site_ids, np.vstack(gates1), np.vstack(gates6)


def metric_or_nan(fn, y_true, y_pred) -> float:
    try:
        return float(fn(y_true, y_pred))
    except ValueError:
        return float("nan")


def calibrate_thresholds(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    thresholds = {}
    for idx, name in enumerate(["m1A", "m6A"]):
        best_t, best_mcc = 0.5, -2.0
        for t in np.linspace(0.05, 0.95, 91):
            mcc = matthews_corrcoef(labels[:, idx], probs[:, idx] >= t)
            if mcc > best_mcc:
                best_t, best_mcc = float(t), float(mcc)
        thresholds[name] = best_t
    return thresholds


def evaluate(probs: np.ndarray, labels: np.ndarray, thresholds: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    out = {}
    for idx, name in enumerate(["m1A", "m6A"]):
        pred = probs[:, idx] >= thresholds[name]
        y = labels[:, idx]
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        specificity = tn / max(tn + fp, 1)
        out[name] = {
            "AUC": metric_or_nan(roc_auc_score, y, probs[:, idx]),
            "AUPR": metric_or_nan(average_precision_score, y, probs[:, idx]),
            "F1": metric_or_nan(f1_score, y, pred),
            "MCC": metric_or_nan(matthews_corrcoef, y, pred),
            "Precision": metric_or_nan(lambda a, b: precision_score(a, b, zero_division=0), y, pred),
            "Recall": metric_or_nan(lambda a, b: recall_score(a, b, zero_division=0), y, pred),
            "Specificity": specificity,
            "Balanced accuracy": metric_or_nan(balanced_accuracy_score, y, pred),
            "prevalence": float(y.mean()),
        }
    out["macro"] = {
        "macro_AUPR": float(np.nanmean([out["m1A"]["AUPR"], out["m6A"]["AUPR"]])),
        "macro_MCC": float(np.nanmean([out["m1A"]["MCC"], out["m6A"]["MCC"]])),
        "average_F1": float(np.nanmean([out["m1A"]["F1"], out["m6A"]["F1"]])),
    }
    return out


def write_predictions(path: Path, probs: np.ndarray, labels: np.ndarray, site_ids: Sequence[str], gates1: np.ndarray, gates6: np.ndarray, thresholds: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        fields = ["site_id", "label_m1A", "label_m6A", "pred_m1A_prob", "pred_m6A_prob", "calibrated_m1A_label", "calibrated_m6A_label", "gate_m1A", "gate_m6A"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for i, site_id in enumerate(site_ids):
            writer.writerow(
                {
                    "site_id": site_id,
                    "label_m1A": int(labels[i, 0]),
                    "label_m6A": int(labels[i, 1]),
                    "pred_m1A_prob": probs[i, 0],
                    "pred_m6A_prob": probs[i, 1],
                    "calibrated_m1A_label": int(probs[i, 0] >= thresholds["m1A"]),
                    "calibrated_m6A_label": int(probs[i, 1] >= thresholds["m6A"]),
                    "gate_m1A": gates1[i, 0],
                    "gate_m6A": gates6[i, 0],
                }
            )


def run_training(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    rows, fold_by_id = load_rows(Path(args.dataset), Path(args.rnafold), args.max_rows, args.seed)
    train_rows, valid_rows, test_rows = split_rows(rows, args.split)
    train_ds = JointDataset(train_rows, fold_by_id)
    valid_ds = JointDataset(valid_rows, fold_by_id, stats=train_ds.stats)
    test_ds = JointDataset(test_rows, fold_by_id, stats=train_ds.stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    labels = np.asarray([[parse_float(row["label_m1A"]), parse_float(row["label_m6A"])] for row in train_rows], dtype=np.float32)
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    pos_weight = torch.tensor(neg / np.maximum(pos, 1.0), dtype=torch.float32, device=device)

    cfg = StructDualRMConfig(d_model=args.d_model, num_layers=args.layers, num_heads=args.heads, dropout=args.dropout)
    model = StructDualRMModel(cfg, variant=args.variant).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_state = None
    best_macro = -1.0
    best_thresholds = {"m1A": 0.5, "m6A": 0.5}
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_probs, valid_labels, _, _, _ = predict(model, valid_loader, device)
        thresholds = calibrate_thresholds(valid_probs, valid_labels)
        valid_metrics = evaluate(valid_probs, valid_labels, thresholds)
        score = valid_metrics["macro"]["macro_AUPR"]
        history.append({"epoch": epoch, "train_loss": loss, "valid_metrics": valid_metrics, "thresholds": thresholds})
        if score > best_macro:
            best_macro = score
            best_thresholds = thresholds
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(f"epoch={epoch} loss={loss:.4f} valid_macro_AUPR={score:.4f} thresholds={thresholds}")

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_probs, valid_labels, valid_ids, valid_g1, valid_g6 = predict(model, valid_loader, device)
    test_probs, test_labels, test_ids, test_g1, test_g6 = predict(model, test_loader, device)
    metrics = {
        "variant": args.variant,
        "device": str(device),
        "row_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "validation": evaluate(valid_probs, valid_labels, best_thresholds),
        "test": evaluate(test_probs, test_labels, best_thresholds),
        "history": history,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "variant": args.variant,
            "thresholds": best_thresholds,
            "struct_stats": train_ds.stats,
            "region_to_id": REGION_TO_ID,
            "base_to_id": BASE_TO_ID,
        },
        out_dir / "model.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out_dir / "thresholds_validation.json").write_text(json.dumps(best_thresholds, indent=2), encoding="utf-8")
    write_predictions(out_dir / "valid_predictions.tsv", valid_probs, valid_labels, valid_ids, valid_g1, valid_g6, best_thresholds)
    write_predictions(out_dir / "test_predictions.tsv", test_probs, test_labels, test_ids, test_g1, test_g6, best_thresholds)

    cal_dir = Path(args.calibration_out)
    if cal_dir:
        cal_dir.mkdir(parents=True, exist_ok=True)
        (cal_dir / "thresholds_validation.json").write_text(json.dumps(best_thresholds, indent=2), encoding="utf-8")
        write_predictions(cal_dir / "calibrated_test_predictions.tsv", test_probs, test_labels, test_ids, test_g1, test_g6, best_thresholds)

    readme = (
        f"# {args.variant} StructDualRM Model\n\n"
        f"Trained with PyTorch on {metrics['row_counts']} rows using split `{args.split}`.\n\n"
        "Files:\n"
        "- `model.pt`: model weights, config, thresholds, feature stats\n"
        "- `metrics.json`: validation/test metrics and training history\n"
        "- `thresholds_validation.json`: thresholds selected on validation MCC\n"
        "- `test_predictions.tsv`: probabilities and calibrated labels\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train StructDualRM")
    parser.add_argument("--dataset", default="05_datasets/dataset_joint_m1A_m6A_101nt.tsv")
    parser.add_argument("--rnafold", default="06_structure/rnafold_101nt.tsv")
    parser.add_argument("--out-dir", default="08_models/M4_structdualrm_full")
    parser.add_argument("--calibration-out", default="09_results/calibration")
    parser.add_argument("--variant", choices=["M0", "M1", "M2", "M3", "M4"], default="M4")
    parser.add_argument("--split", default="split_random")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
