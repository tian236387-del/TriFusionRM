#!/usr/bin/env python
"""Train a MultiRM-style label-attention model fused with RNAfold/RBP features.

M6 keeps the strongest strict baseline backbone observed so far:
3-mer embedding + BiLSTM + label-specific attention + BCE objective.
It then adds StructDualRM's reviewer-safe biological inputs:
RNAfold scalar structure features and optional local RBP peak features.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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

from train_rmbase11_external_style_baselines import (
    CLASS_NAMES,
    build_embedding_matrix,
    build_trimer_vocab,
    encode_trimers,
    load_embedding_dict,
    parse_label,
    read_tsv,
    split_rows,
)


STRUCT_FIELDS = [
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rnafold(path: Path) -> Dict[str, Dict[str, str]]:
    return {row["site_id"]: row for row in read_tsv(path)}


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
            raise ValueError(f"RBP feature file must contain site_id: {rbp_path}")
        fields = [field for field in reader.fieldnames if field != "site_id"]
        for row in reader:
            rbp_by_id[row["site_id"]] = np.asarray([parse_float(row.get(field, 0.0)) for field in fields], dtype=np.float32)
    return rbp_by_id, fields


@dataclass
class FeatureStats:
    mean: List[float]
    std: List[float]


def normalize_matrix(values: np.ndarray, stats: Optional[FeatureStats] = None) -> Tuple[np.ndarray, FeatureStats]:
    if values.shape[1] == 0:
        return values.astype(np.float32), FeatureStats([], [])
    if stats is None:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-6] = 1.0
        stats = FeatureStats(mean.tolist(), std.tolist())
    mean_arr = np.asarray(stats.mean, dtype=np.float32)
    std_arr = np.asarray(stats.std, dtype=np.float32)
    std_arr[std_arr < 1e-6] = 1.0
    return ((values - mean_arr) / std_arr).astype(np.float32), stats


class HybridDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        rnafold_by_id: Mapping[str, Mapping[str, str]],
        trimer_vocab: Mapping[str, int],
        input_length: int,
        rbp_by_id: Optional[Mapping[str, np.ndarray]] = None,
        rbp_dim: int = 0,
        struct_stats: Optional[FeatureStats] = None,
        rbp_stats: Optional[FeatureStats] = None,
    ):
        self.rows = list(rows)
        self.labels = np.asarray([parse_label(row) for row in self.rows], dtype=np.int64)
        self.site_ids = [row["site_id"] for row in self.rows]
        self.mod_types = [row["mod_type"] for row in self.rows]
        self.x = np.stack([encode_trimers(row["sequence_41nt"], trimer_vocab, input_length) for row in self.rows])

        struct_values = []
        for row in self.rows:
            fold = rnafold_by_id[row["site_id"]]
            struct_values.append([parse_float(fold.get(field, 0.0)) for field in STRUCT_FIELDS])
        struct_arr = np.asarray(struct_values, dtype=np.float32)
        self.struct, self.struct_stats = normalize_matrix(struct_arr, struct_stats)

        rbp_values = np.zeros((len(self.rows), rbp_dim), dtype=np.float32)
        if rbp_by_id and rbp_dim:
            for idx, row in enumerate(self.rows):
                values = rbp_by_id.get(row["site_id"])
                if values is not None and values.shape[0] == rbp_dim:
                    rbp_values[idx] = values
        self.rbp, self.rbp_stats = normalize_matrix(rbp_values, rbp_stats)

        self.x_tensor = torch.as_tensor(self.x, dtype=torch.long)
        self.struct_tensor = torch.as_tensor(self.struct, dtype=torch.float32)
        self.rbp_tensor = torch.as_tensor(self.rbp, dtype=torch.float32)
        self.label_tensor = torch.as_tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "x": self.x_tensor[idx],
            "struct": self.struct_tensor[idx],
            "rbp": self.rbp_tensor[idx],
            "label": self.label_tensor[idx],
            "site_id": self.site_ids[idx],
            "mod_type": self.mod_types[idx],
        }


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "x": torch.stack([item["x"] for item in batch]),  # type: ignore[arg-type]
        "struct": torch.stack([item["struct"] for item in batch]),  # type: ignore[arg-type]
        "rbp": torch.stack([item["rbp"] for item in batch]),  # type: ignore[arg-type]
        "label": torch.stack([item["label"] for item in batch]),  # type: ignore[arg-type]
        "site_id": [str(item["site_id"]) for item in batch],
        "mod_type": [str(item["mod_type"]) for item in batch],
    }


class MultiRMStructRBP(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden_dim: int,
        num_classes: int,
        struct_dim: int,
        rbp_input_dim: int,
        aux_dim: int,
        dropout: float,
    ):
        super().__init__()
        weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        self.encoder = nn.LSTM(input_size=weights.shape[1], hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        seq_dim = hidden_dim * 2
        self.attn_proj = nn.Linear(seq_dim, seq_dim)
        self.label_queries = nn.Parameter(torch.empty(num_classes, seq_dim))
        nn.init.xavier_uniform_(self.label_queries)

        self.struct_mlp = nn.Sequential(
            nn.Linear(struct_dim, aux_dim),
            nn.ReLU(),
            nn.LayerNorm(aux_dim),
            nn.Dropout(dropout),
            nn.Linear(aux_dim, aux_dim),
        )
        self.rbp_mlp = (
            nn.Sequential(
                nn.Linear(rbp_input_dim, aux_dim),
                nn.ReLU(),
                nn.LayerNorm(aux_dim),
                nn.Dropout(dropout),
                nn.Linear(aux_dim, aux_dim),
            )
            if rbp_input_dim > 0
            else None
        )
        rbp_out = aux_dim if rbp_input_dim > 0 else 0
        head_in = seq_dim + aux_dim + rbp_out
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(head_in, 1), nn.Sigmoid()) for _ in range(num_classes)])
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_in, 192),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(192, 1),
                )
                for _ in range(num_classes)
            ]
        )

    def forward(self, x: torch.Tensor, struct: torch.Tensor, rbp: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, _ = self.encoder(self.embedding(x.long()))
        scores = torch.einsum("bld,cd->bcl", torch.tanh(self.attn_proj(h)), self.label_queries)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bcl,bld->bcd", weights, h)
        z_struct = self.struct_mlp(struct)
        if self.rbp_mlp is not None:
            z_rbp = self.rbp_mlp(rbp)
            aux = torch.cat([z_struct, z_rbp], dim=-1)
        else:
            aux = z_struct
        aux_expanded = aux.unsqueeze(1).expand(-1, pooled.shape[1], -1)
        z = torch.cat([pooled, aux_expanded], dim=-1)
        gate_values = torch.cat([gate(z[:, idx, :]) for idx, gate in enumerate(self.gates)], dim=-1)
        logits = torch.cat([head(z[:, idx, :] * gate_values[:, idx : idx + 1]) for idx, head in enumerate(self.heads)], dim=-1)
        return {"logits": logits, "attention": weights, "class_gate": gate_values}


def onehot(labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=len(CLASS_NAMES)).float()


def multilabel_pos_weight(rows: Sequence[Dict[str, str]], device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for row in rows:
        counts[parse_label(row)] += 1
    counts[counts < 1] = 1
    neg = len(rows) - counts
    return torch.tensor(np.clip(neg / counts, 1.0, 500.0), dtype=torch.float32, device=device)


def move_batch(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, loss_fn, device: torch.device) -> float:
    model.train()
    total = 0.0
    rows = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["x"], batch["struct"], batch["rbp"])  # type: ignore[arg-type]
        labels = batch["label"]  # type: ignore[assignment]
        loss = loss_fn(out["logits"], onehot(labels))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = int(labels.shape[0])
        total += float(loss.detach().cpu()) * batch_size
        rows += batch_size
    return total / max(rows, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], np.ndarray]:
    model.eval()
    probs, labels, gates = [], [], []
    site_ids: List[str] = []
    mod_types: List[str] = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["x"], batch["struct"], batch["rbp"])  # type: ignore[arg-type]
        probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
        labels.append(batch["label"].detach().cpu().numpy())  # type: ignore[union-attr]
        gates.append(out["class_gate"].detach().cpu().numpy())
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
    onehot_np = np.eye(len(CLASS_NAMES), dtype=np.int32)[labels]
    norm = probs.copy()
    row_sums = norm.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    norm = norm / row_sums
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


def write_predictions(path: Path, probs: np.ndarray, labels: np.ndarray, site_ids: Sequence[str], mod_types: Sequence[str], gates: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["site_id", "true_mod_type", "true_class_index", "pred_mod_type", "pred_class_index"] + [
        f"prob_{name}" for name in CLASS_NAMES
    ] + [f"gate_{name}" for name in CLASS_NAMES]
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
            row.update({f"gate_{name}": float(gates[idx, j]) for j, name in enumerate(CLASS_NAMES)})
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


def run(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rows = read_tsv(Path(args.dataset))
    rnafold_by_id = load_rnafold(Path(args.rnafold))
    rows = [row for row in rows if row.get("site_id") in rnafold_by_id]
    if args.max_rows:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_rows]
    train_rows, valid_rows, test_rows = split_rows(rows, args.split)

    embedding_dict = load_embedding_dict(Path(args.embedding_path))
    trimer_vocab = build_trimer_vocab(embedding_dict)
    embedding_matrix = build_embedding_matrix(trimer_vocab, embedding_dict, args.seed)
    rbp_by_id, rbp_fields = load_rbp_features(args.rbp_features)

    train_ds = HybridDataset(train_rows, rnafold_by_id, trimer_vocab, args.input_length, rbp_by_id, len(rbp_fields))
    valid_ds = HybridDataset(
        valid_rows,
        rnafold_by_id,
        trimer_vocab,
        args.input_length,
        rbp_by_id,
        len(rbp_fields),
        struct_stats=train_ds.struct_stats,
        rbp_stats=train_ds.rbp_stats,
    )
    test_ds = HybridDataset(
        test_rows,
        rnafold_by_id,
        trimer_vocab,
        args.input_length,
        rbp_by_id,
        len(rbp_fields),
        struct_stats=train_ds.struct_stats,
        rbp_stats=train_ds.rbp_stats,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    model = MultiRMStructRBP(
        embedding_matrix=embedding_matrix,
        hidden_dim=args.hidden_dim,
        num_classes=len(CLASS_NAMES),
        struct_dim=len(STRUCT_FIELDS),
        rbp_input_dim=len(rbp_fields),
        aux_dim=args.aux_dim,
        dropout=args.dropout,
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=multilabel_pos_weight(train_rows, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_score = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_probs, valid_labels, _, _, _ = predict(model, valid_loader, device)
        valid_metrics = evaluate(valid_probs, valid_labels)
        score = float(valid_metrics[args.selection_metric])
        history.append({"epoch": epoch, "train_loss": loss, "validation": valid_metrics})
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(f"epoch={epoch} loss={loss:.4f} valid_{args.selection_metric}={score:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    valid_probs, valid_labels, valid_ids, valid_mods, valid_gates = predict(model, valid_loader, device)
    test_probs, test_labels, test_ids, test_mods, test_gates = predict(model, test_loader, device)
    valid_metrics = evaluate(valid_probs, valid_labels)
    test_metrics = evaluate(test_probs, test_labels)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "trimer_vocab": trimer_vocab,
            "struct_fields": STRUCT_FIELDS,
            "struct_stats": asdict(train_ds.struct_stats),
            "rbp_feature_names": rbp_fields,
            "rbp_stats": asdict(train_ds.rbp_stats),
            "training_args": vars(args),
        },
        out_dir / "model.pt",
    )
    metrics: Dict[str, object] = {
        "dataset": args.dataset,
        "rnafold": args.rnafold,
        "variant": "M6_MultiRMStructRBP",
        "display_name": "M6 MultiRM-style + Struct/RBP",
        "task": "RMBase_hg38_11_class_multiclass",
        "classes": CLASS_NAMES,
        "device": str(device),
        "objective": "bce_one_vs_rest",
        "row_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "split": args.split,
        "rbp_features": args.rbp_features,
        "rbp_feature_count": len(rbp_fields),
        "validation": valid_metrics,
        "test": test_metrics,
        "history": history,
        "caveats": [
            "This is a hybrid architecture-style model that keeps MultiRM-style label attention and adds RNAfold/RBP features.",
            "The RBP panel is limited to locally available complete ENCODE eCLIP peak files in the current environment.",
            "No RMBase support/cell/assay metadata is used.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions(out_dir / "valid_predictions.tsv", valid_probs, valid_labels, valid_ids, valid_mods, valid_gates)
    write_predictions(out_dir / "test_predictions.tsv", test_probs, test_labels, test_ids, test_mods, test_gates)
    write_confusion_matrix(out_dir / "test_confusion_matrix.tsv", test_metrics["confusion_matrix"])  # type: ignore[arg-type]
    (out_dir / "README.md").write_text(
        "# M6 MultiRM-style + Struct/RBP\n\n"
        "Hybrid architecture: 3-mer BiLSTM label-specific attention plus RNAfold scalar structure and optional RBP peak features.\n\n"
        f"Dataset: `{args.dataset}`. Split: `{args.split}`. Device: `{device}`.\n",
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train M6 MultiRM-style structure/RBP hybrid.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_mmseqs80_strict_splits.tsv")
    parser.add_argument("--rnafold", default="rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv")
    parser.add_argument("--rbp-features", default="rmbase11_full_work/rbp/rmbase11_rbp_features.tsv")
    parser.add_argument("--split", default="split_mmseqs80_cluster")
    parser.add_argument("--out-dir", default="rmbase11_full_work/models/M6_multirm_struct_rbp_mmseqs80_full")
    parser.add_argument("--embedding-path", default="external/MultiRM-master/Embeddings/embeddings_12RM.pkl")
    parser.add_argument("--input-length", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--aux-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selection-metric", choices=["macro_AUPR_ovr", "macro_F1", "MCC", "balanced_accuracy"], default="macro_AUPR_ovr")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
