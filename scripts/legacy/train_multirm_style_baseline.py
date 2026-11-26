#!/usr/bin/env python
"""Train a MultiRM-style BiLSTM label-attention baseline on the frozen split."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from scripts.train_structdualrm import calibrate_thresholds, evaluate
except ModuleNotFoundError:  # pragma: no cover
    from train_structdualrm import calibrate_thresholds, evaluate


ALPHABET = "ACGT"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_trimer_vocab() -> Dict[str, int]:
    return {"".join(parts): idx + 1 for idx, parts in enumerate(itertools.product(ALPHABET, repeat=3))}


def encode_trimers(sequences: Sequence[str], vocab: Dict[str, int], k: int = 3) -> np.ndarray:
    if not sequences:
        return np.zeros((0, 0), dtype=np.int64)
    seqs = [seq.upper().replace("U", "T") for seq in sequences]
    width = max(len(seq) for seq in seqs) - k + 1
    encoded = np.zeros((len(seqs), max(width, 0)), dtype=np.int64)
    for row_idx, seq in enumerate(seqs):
        for pos in range(max(0, len(seq) - k + 1)):
            encoded[row_idx, pos] = vocab.get(seq[pos : pos + k], 0)
    return encoded


class MultiRMStyleModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 300, hidden_dim: int = 256, num_labels: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        encoded_dim = hidden_dim * 2
        self.attn_proj = nn.Linear(encoded_dim, encoded_dim)
        self.label_queries = nn.Parameter(torch.empty(num_labels, encoded_dim))
        self.heads = nn.ModuleList([nn.Sequential(nn.Dropout(dropout), nn.Linear(encoded_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)) for _ in range(num_labels)])
        nn.init.xavier_uniform_(self.label_queries)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, _ = self.encoder(self.embedding(x.long()))
        scores = torch.einsum("bld,td->blt", torch.tanh(self.attn_proj(h)), self.label_queries)
        attention = torch.softmax(scores.transpose(1, 2), dim=-1)
        context = torch.einsum("btl,bld->btd", attention, h)
        logits = torch.cat([head(context[:, idx, :]) for idx, head in enumerate(self.heads)], dim=-1)
        return {"logits": logits, "attention": attention}


class SequenceDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, str]], vocab: Dict[str, int]):
        self.rows = list(rows)
        self.x = encode_trimers([row["sequence"] for row in self.rows], vocab)
        self.y = np.asarray([[float(row["label_m1A"]), float(row["label_m6A"])] for row in self.rows], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.as_tensor(self.x[idx], dtype=torch.long),
            "y": torch.as_tensor(self.y[idx], dtype=torch.float32),
        }


def read_split_rows(path: Path, split_column: str) -> Dict[str, List[Dict[str, str]]]:
    groups = {"train": [], "valid": [], "test": []}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            split = row.get(split_column, "")
            if split in groups:
                groups[split].append(row)
    return groups


def labels_from_rows(rows: Sequence[Dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row["label_m1A"]), float(row["label_m6A"])] for row in rows], dtype=np.float32)


def make_loader(rows: Sequence[Dict[str, str]], vocab: Dict[str, int], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(SequenceDataset(rows, vocab), batch_size=batch_size, shuffle=shuffle)


def predict(model: MultiRMStyleModel, loader: DataLoader, device: torch.device, keep_attention: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    probs, labels, attentions = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"].to(device))
            probs.append(torch.sigmoid(out["logits"]).detach().cpu().numpy())
            labels.append(batch["y"].numpy())
            if keep_attention:
                attentions.append(out["attention"].detach().cpu().numpy())
    attention_arr = np.vstack(attentions) if keep_attention and attentions else None
    return np.vstack(probs), np.vstack(labels), attention_arr


def train_one_epoch(model: MultiRMStyleModel, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    total_loss = 0.0
    total_rows = 0
    model.train()
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)["logits"]
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * x.shape[0]
        total_rows += x.shape[0]
    return total_loss / max(total_rows, 1)


def write_predictions(path: Path, rows: Sequence[Dict[str, str]], labels: np.ndarray, probs: np.ndarray, thresholds: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["site_id", "label_m1A", "label_m6A", "pred_m1A_prob", "pred_m6A_prob", "calibrated_m1A_label", "calibrated_m6A_label"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    "site_id": row["site_id"],
                    "label_m1A": int(labels[idx, 0]),
                    "label_m6A": int(labels[idx, 1]),
                    "pred_m1A_prob": float(probs[idx, 0]),
                    "pred_m6A_prob": float(probs[idx, 1]),
                    "calibrated_m1A_label": int(probs[idx, 0] >= thresholds["m1A"]),
                    "calibrated_m6A_label": int(probs[idx, 1] >= thresholds["m6A"]),
                }
            )


def write_attention_outputs(out_dir: Path, rows: Sequence[Dict[str, str]], attention: np.ndarray) -> None:
    labels = ["m1A", "m6A"]
    positions = list(range(-(attention.shape[-1] // 2), attention.shape[-1] // 2 + 1))
    if len(positions) > attention.shape[-1]:
        positions = positions[: attention.shape[-1]]
    with (out_dir / "attention_weights.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["site_id", "label", *[f"pos_{pos}" for pos in positions]])
        for row_idx, row in enumerate(rows):
            for label_idx, label in enumerate(labels):
                writer.writerow([row["site_id"], label, *[float(x) for x in attention[row_idx, label_idx, :]]])

    with (out_dir / "attention_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "relative_position", "mean_attention"], delimiter="\t")
        writer.writeheader()
        mean_attention = attention.mean(axis=0)
        for label_idx, label in enumerate(labels):
            for pos_idx, pos in enumerate(positions):
                writer.writerow({"label": label, "relative_position": pos, "mean_attention": float(mean_attention[label_idx, pos_idx])})

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path("10_figures/model_interpretability")
        fig_dir.mkdir(parents=True, exist_ok=True)
        mean_attention = attention.mean(axis=0)
        for label_idx, label in enumerate(labels):
            plt.figure(figsize=(7, 3))
            plt.plot(positions, mean_attention[label_idx, : len(positions)], color="#2f6f8f", linewidth=2)
            plt.xlabel("Relative 3-mer position")
            plt.ylabel("Mean attention")
            plt.tight_layout()
            plt.savefig(fig_dir / f"multirm_style_attention_{label}.png", dpi=180)
            plt.close()
    except Exception as exc:  # pragma: no cover - plotting is optional in minimal test env
        (out_dir / "attention_plot_warning.txt").write_text(str(exc), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = build_trimer_vocab()
    rows_by_split = read_split_rows(Path(args.split_file), args.split_column)
    train_rows, valid_rows, test_rows = rows_by_split["train"], rows_by_split["valid"], rows_by_split["test"]
    train_loader = make_loader(train_rows, vocab, args.batch_size, shuffle=True)
    valid_loader = make_loader(valid_rows, vocab, args.batch_size, shuffle=False)
    test_loader = make_loader(test_rows, vocab, args.batch_size, shuffle=False)

    model = MultiRMStyleModel(vocab_size=len(vocab) + 1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    y_train = labels_from_rows(train_rows)
    positives = y_train.sum(axis=0)
    negatives = y_train.shape[0] - positives
    pos_weight = torch.as_tensor(negatives / np.maximum(positives, 1.0), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_thresholds = {"m1A": 0.5, "m6A": 0.5}
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_probs, valid_labels, _ = predict(model, valid_loader, device)
        thresholds = calibrate_thresholds(valid_probs, valid_labels)
        valid_metrics = evaluate(valid_probs, valid_labels, thresholds)
        score = valid_metrics["macro"]["macro_AUPR"]
        history.append({"epoch": epoch, "train_loss": loss, "valid_metrics": valid_metrics, "thresholds": thresholds})
        if score > best_score:
            best_score = score
            best_thresholds = thresholds
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(f"epoch={epoch} loss={loss:.4f} valid_macro_AUPR={score:.4f} thresholds={thresholds}")

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_probs, valid_labels, _ = predict(model, valid_loader, device)
    test_probs, test_labels, test_attention = predict(model, test_loader, device, keep_attention=True)

    metrics = {
        "model_name": "MultiRM_style_BiLSTM_attention",
        "variant": "multirm_style_bilstm_attention",
        "device": str(device),
        "split_file": args.split_file,
        "split_column": args.split_column,
        "row_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim},
        "validation": evaluate(valid_probs, valid_labels, best_thresholds),
        "test": evaluate(test_probs, test_labels, best_thresholds),
        "history": history,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "thresholds_validation.json").write_text(json.dumps(best_thresholds, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "vocab": vocab, "thresholds": best_thresholds, "args": vars(args)}, out_dir / "model.pt")
    write_predictions(out_dir / "valid_predictions.tsv", valid_rows, valid_labels, valid_probs, best_thresholds)
    write_predictions(out_dir / "test_predictions.tsv", test_rows, test_labels, test_probs, best_thresholds)
    if test_attention is not None:
        write_attention_outputs(out_dir, test_rows, test_attention)
    (out_dir / "README.md").write_text(
        "MultiRM-style BiLSTM label-attention baseline trained on the frozen m1A/m6A split.\n"
        "This is a same-dataset sequence-only comparator, not a claim of exact official MultiRM retraining.\n",
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'metrics.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MultiRM-style BiLSTM attention baseline.")
    parser.add_argument("--split-file", default="05_datasets/benchmark_splits/m1A_m6A_split_random_seed42.tsv")
    parser.add_argument("--split-column", default="split_random_seed42")
    parser.add_argument("--out-dir", default="12_benchmarks/multirm_style")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
