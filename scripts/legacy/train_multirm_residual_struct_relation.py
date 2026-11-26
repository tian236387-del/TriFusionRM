#!/usr/bin/env python
"""Train a MultiRM backbone with conservative structure/relation residual logits."""

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

from train_multirm_struct_rbp_hybrid import (
    FeatureStats,
    STRUCT_FIELDS,
    load_rnafold,
    normalize_matrix,
    parse_float,
)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResidualDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        rnafold_by_id: Mapping[str, Mapping[str, str]],
        trimer_vocab: Mapping[str, int],
        input_length: int,
        struct_stats: Optional[FeatureStats] = None,
    ):
        self.rows = list(rows)
        self.labels = np.asarray([parse_label(row) for row in self.rows], dtype=np.int64)
        self.site_ids = [row["site_id"] for row in self.rows]
        self.mod_types = [row["mod_type"] for row in self.rows]
        self.x = np.stack([encode_trimers(row["sequence_41nt"], trimer_vocab, input_length) for row in self.rows])
        struct = np.asarray(
            [[parse_float(rnafold_by_id[row["site_id"]].get(field, 0.0)) for field in STRUCT_FIELDS] for row in self.rows],
            dtype=np.float32,
        )
        self.struct, self.struct_stats = normalize_matrix(struct, struct_stats)
        self.x_tensor = torch.as_tensor(self.x, dtype=torch.long)
        self.struct_tensor = torch.as_tensor(self.struct, dtype=torch.float32)
        self.label_tensor = torch.as_tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "x": self.x_tensor[idx],
            "struct": self.struct_tensor[idx],
            "label": self.label_tensor[idx],
            "site_id": self.site_ids[idx],
            "mod_type": self.mod_types[idx],
        }


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "x": torch.stack([item["x"] for item in batch]),  # type: ignore[arg-type]
        "struct": torch.stack([item["struct"] for item in batch]),  # type: ignore[arg-type]
        "label": torch.stack([item["label"] for item in batch]),  # type: ignore[arg-type]
        "site_id": [str(item["site_id"]) for item in batch],
        "mod_type": [str(item["mod_type"]) for item in batch],
    }


class MultiRMResidualStructRelation(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden_dim: int,
        num_classes: int,
        struct_dim: int,
        aux_dim: int,
        dropout: float,
        relation_message: torch.Tensor,
        max_delta: float,
    ):
        super().__init__()
        weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        self.encoder = nn.LSTM(input_size=weights.shape[1], hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        seq_dim = hidden_dim * 2
        self.attn_proj = nn.Linear(seq_dim, seq_dim)
        self.label_queries = nn.Parameter(torch.empty(num_classes, seq_dim))
        nn.init.xavier_uniform_(self.label_queries)
        self.base_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Dropout(dropout), nn.Linear(seq_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))
                for _ in range(num_classes)
            ]
        )
        self.struct_mlp = nn.Sequential(
            nn.Linear(struct_dim, aux_dim),
            nn.ReLU(),
            nn.LayerNorm(aux_dim),
            nn.Dropout(dropout),
            nn.Linear(aux_dim, aux_dim),
        )
        self.relation_project = nn.Sequential(nn.Linear(seq_dim, aux_dim), nn.ReLU(), nn.Dropout(dropout))
        self.delta_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(seq_dim + aux_dim * 2, 96),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(96, 1),
                )
                for _ in range(num_classes)
            ]
        )
        self.max_delta = max_delta
        self.register_buffer("relation_message", relation_message)

    def forward(self, x: torch.Tensor, struct: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, _ = self.encoder(self.embedding(x.long()))
        scores = torch.einsum("bld,cd->bcl", torch.tanh(self.attn_proj(h)), self.label_queries)
        attention = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bcl,bld->bcd", attention, h)
        base_logits = torch.cat([head(pooled[:, idx, :]) for idx, head in enumerate(self.base_heads)], dim=-1)
        relation_context = torch.einsum("ck,bkd->bcd", self.relation_message, pooled)
        relation_context = self.relation_project(relation_context)
        struct_context = self.struct_mlp(struct).unsqueeze(1).expand(-1, pooled.shape[1], -1)
        delta_input = torch.cat([pooled, struct_context, relation_context], dim=-1)
        raw_delta = torch.cat([head(delta_input[:, idx, :]) for idx, head in enumerate(self.delta_heads)], dim=-1)
        delta = self.max_delta * torch.tanh(raw_delta)
        return {"logits": base_logits + delta, "base_logits": base_logits, "delta": delta, "attention": attention}


def onehot(labels: torch.Tensor) -> torch.Tensor:
    return F.one_hot(labels, num_classes=len(CLASS_NAMES)).float()


def pos_weight(rows: Sequence[Dict[str, str]], device: torch.device, cap: float) -> torch.Tensor:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for row in rows:
        counts[parse_label(row)] += 1
    counts[counts < 1] = 1
    neg = len(rows) - counts
    return torch.tensor(np.clip(neg / counts, 1.0, cap), dtype=torch.float32, device=device)


def relation_prior(rows: Sequence[Dict[str, str]]) -> Tuple[np.ndarray, Dict[str, object]]:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for row in rows:
        counts[parse_label(row)] += 1
    # Conservative prior: same-center-base families get weak links, plus observed exact-coordinate coannotations.
    relation = np.eye(len(CLASS_NAMES), dtype=np.float64)
    center_groups = [
        ["Am", "m1A", "m6A"],
        ["Cm", "m5C"],
        ["Gm", "m7G"],
        ["Um", "D", "Y", "m5U"],
    ]
    name_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    for group in center_groups:
        for left in group:
            for right in group:
                if left != right:
                    relation[name_to_idx[left], name_to_idx[right]] = 0.15
    relation /= relation.sum(axis=1, keepdims=True)
    return relation.astype(np.float32), {"class_counts": {name: int(counts[idx]) for idx, name in enumerate(CLASS_NAMES)}, "message": relation.tolist()}


def move_batch(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device, lambda_delta: float) -> float:
    model.train()
    total = 0.0
    rows = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["x"], batch["struct"])  # type: ignore[arg-type]
        labels = batch["label"]  # type: ignore[assignment]
        loss = criterion(out["logits"], onehot(labels)) + lambda_delta * out["delta"].pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = int(labels.shape[0])
        total += float(loss.detach().cpu()) * batch_size
        rows += batch_size
    return total / max(rows, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    model.eval()
    probs, deltas, labels = [], [], []
    site_ids: List[str] = []
    mod_types: List[str] = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["x"], batch["struct"])  # type: ignore[arg-type]
        probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
        deltas.append(out["delta"].cpu().numpy())
        labels.append(batch["label"].cpu().numpy())  # type: ignore[union-attr]
        site_ids.extend(batch["site_id"])  # type: ignore[arg-type]
        mod_types.extend(batch["mod_type"])  # type: ignore[arg-type]
    return np.vstack(probs), np.concatenate(labels), np.vstack(deltas), site_ids, mod_types


def metric_or_nan(fn, *args) -> float:
    try:
        return float(fn(*args))
    except ValueError:
        return float("nan")


def evaluate(probs: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    pred = probs.argmax(axis=1)
    onehot_np = np.eye(len(CLASS_NAMES), dtype=np.int32)[labels]
    norm = probs.copy()
    norm /= np.maximum(norm.sum(axis=1, keepdims=True), 1e-8)
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


def write_predictions(path: Path, probs: np.ndarray, labels: np.ndarray, deltas: np.ndarray, site_ids: Sequence[str], mod_types: Sequence[str]) -> None:
    fields = ["site_id", "true_mod_type", "true_class_index", "pred_mod_type", "pred_class_index"]
    fields += [f"prob_{name}" for name in CLASS_NAMES] + [f"delta_{name}" for name in CLASS_NAMES]
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
            row.update({f"delta_{name}": float(deltas[idx, j]) for j, name in enumerate(CLASS_NAMES)})
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
    relation_message, relation_summary = relation_prior(train_rows)

    train_ds = ResidualDataset(train_rows, rnafold_by_id, trimer_vocab, args.input_length)
    valid_ds = ResidualDataset(valid_rows, rnafold_by_id, trimer_vocab, args.input_length, train_ds.struct_stats)
    test_ds = ResidualDataset(test_rows, rnafold_by_id, trimer_vocab, args.input_length, train_ds.struct_stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = MultiRMResidualStructRelation(
        embedding_matrix=embedding_matrix,
        hidden_dim=args.hidden_dim,
        num_classes=len(CLASS_NAMES),
        struct_dim=len(STRUCT_FIELDS),
        aux_dim=args.aux_dim,
        dropout=args.dropout,
        relation_message=torch.tensor(relation_message, dtype=torch.float32, device=device),
        max_delta=args.max_delta,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_rows, device, args.pos_weight_cap))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_state = None
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, args.lambda_delta)
        scheduler.step()
        valid_probs, valid_labels, _, _, _ = predict(model, valid_loader, device)
        valid_metrics = evaluate(valid_probs, valid_labels)
        score = float(valid_metrics[args.selection_metric])
        history.append({"epoch": epoch, "train_loss": loss, "validation": valid_metrics, "learning_rate": optimizer.param_groups[0]["lr"]})
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(f"epoch={epoch} loss={loss:.4f} valid_{args.selection_metric}={score:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_probs, valid_labels, valid_deltas, valid_ids, valid_mods = predict(model, valid_loader, device)
    test_probs, test_labels, test_deltas, test_ids, test_mods = predict(model, test_loader, device)
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
            "relation_message": relation_message,
            "training_args": vars(args),
        },
        out_dir / "model.pt",
    )
    metrics: Dict[str, object] = {
        "dataset": args.dataset,
        "rnafold": args.rnafold,
        "variant": "M9_MultiRMResidualStructRelation",
        "display_name": "M9 MultiRM residual + RNAfold/relation",
        "task": "RMBase_hg38_11_class_multiclass",
        "classes": CLASS_NAMES,
        "device": str(device),
        "objective": "bce_one_vs_rest_with_delta_regularization",
        "row_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "split": args.split,
        "max_delta": args.max_delta,
        "lambda_delta": args.lambda_delta,
        "relation_prior": relation_summary,
        "validation": valid_metrics,
        "test": test_metrics,
        "history": history,
        "caveats": [
            "This model keeps the MultiRM-style sequence backbone and constrains RNAfold/relation features to residual logit corrections.",
            "No RMBase support/cell/assay metadata or RBP features are used.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions(out_dir / "valid_predictions.tsv", valid_probs, valid_labels, valid_deltas, valid_ids, valid_mods)
    write_predictions(out_dir / "test_predictions.tsv", test_probs, test_labels, test_deltas, test_ids, test_mods)
    (out_dir / "README.md").write_text(
        "# M9 MultiRM residual + RNAfold/relation\n\n"
        "A conservative architecture revision: MultiRM-style 3-mer BiLSTM label attention is the primary branch; "
        "RNAfold scalar structure and a weak label relation prior only produce bounded residual logits.\n",
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MultiRM residual structure/relation model.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_mmseqs80_strict_splits.tsv")
    parser.add_argument("--rnafold", default="rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv")
    parser.add_argument("--split", default="split_mmseqs80_cluster")
    parser.add_argument("--out-dir", default="rmbase11_full_work/models/M9_multirm_residual_struct_relation_mmseqs80_full")
    parser.add_argument("--embedding-path", default="external/MultiRM-master/Embeddings/embeddings_12RM.pkl")
    parser.add_argument("--input-length", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--aux-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-delta", type=float, default=0.5)
    parser.add_argument("--lambda-delta", type=float, default=0.05)
    parser.add_argument("--pos-weight-cap", type=float, default=500.0)
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
