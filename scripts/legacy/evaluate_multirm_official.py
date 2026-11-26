#!/usr/bin/env python
"""Evaluate official pretrained MultiRM weights on the frozen m1A/m6A split."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
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


OFFICIAL_TASKS = ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "Psi", "AtoI"]
M1A_INDEX = OFFICIAL_TASKS.index("m1A")
M6A_INDEX = OFFICIAL_TASKS.index("m6A")


def load_embedding_vocab(path: Path) -> Dict[str, int]:
    with path.open("rb") as handle:
        embedding_dict = pickle.load(handle)
    return {word: idx for idx, word in enumerate(embedding_dict.keys())}


def encode_kmers(sequences: Sequence[str], vocab: Dict[str, int], k: int = 3) -> np.ndarray:
    if not sequences:
        return np.zeros((0, 0), dtype=np.int64)
    seqs = [seq.upper().replace("U", "T") for seq in sequences]
    width = max(len(seq) for seq in seqs) - k + 1
    encoded = np.zeros((len(seqs), max(width, 0)), dtype=np.int64)
    for row_idx, seq in enumerate(seqs):
        for pos in range(max(0, len(seq) - k + 1)):
            encoded[row_idx, pos] = vocab.get(seq[pos : pos + k], 0)
    return encoded


def select_m1a_m6a_probs(official_probs: np.ndarray) -> np.ndarray:
    return official_probs[:, [M1A_INDEX, M6A_INDEX]].astype(np.float32)


class BahdanauAttention(nn.Module):
    def __init__(self, in_features: int, hidden_units: int, num_task: int):
        super().__init__()
        self.W1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_with_time_axis = torch.unsqueeze(hidden_states, dim=1)
        score = self.V(torch.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = torch.softmax(score, dim=1)
        values = torch.transpose(values, 1, 2)
        context_vector = torch.matmul(values, attention_weights)
        context_vector = torch.transpose(context_vector, 1, 2)
        return context_vector, attention_weights


class EmbeddingSeq(nn.Module):
    def __init__(self, embedding_path: Path):
        super().__init__()
        with embedding_path.open("rb") as handle:
            weight_dict = pickle.load(handle)
        weights = torch.as_tensor(np.asarray(list(weight_dict.values())), dtype=torch.float32)
        self.embedding = nn.Embedding(num_embeddings=weights.shape[0], embedding_dim=weights.shape[1])
        self.embedding.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x.long())


class OfficialMultiRM(nn.Module):
    def __init__(self, embedding_path: Path, num_task: int = 12):
        super().__init__()
        self.num_task = num_task
        self.embed = EmbeddingSeq(embedding_path)
        self.NaiveBiLSTM = nn.LSTM(input_size=300, hidden_size=256, batch_first=True, bidirectional=True)
        self.Attention = BahdanauAttention(in_features=512, hidden_units=100, num_task=num_task)
        for idx in range(num_task):
            setattr(
                self,
                f"NaiveFC{idx}",
                nn.Sequential(
                    nn.Linear(in_features=512, out_features=128),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(in_features=128, out_features=1),
                    nn.Sigmoid(),
                ),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(x)
        batch_size = x.size(0)
        output, (h_n, _) = self.NaiveBiLSTM(x)
        hidden = h_n.view(batch_size, output.size()[-1])
        hidden = hidden.view(batch_size, output.size()[-1])
        context_vector, attention_weights = self.Attention(hidden, output)
        outs = []
        for idx in range(self.num_task):
            fc_layer = getattr(self, f"NaiveFC{idx}")
            y = torch.squeeze(fc_layer(context_vector[:, idx, :]), dim=-1)
            outs.append(y)
        return torch.stack(outs, dim=1), attention_weights


class EncodedSequenceDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, str]], vocab: Dict[str, int]):
        self.rows = list(rows)
        self.encoded = encode_kmers([row["sequence"] for row in self.rows], vocab)
        self.labels = np.asarray([[float(row["label_m1A"]), float(row["label_m6A"])] for row in self.rows], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.as_tensor(self.encoded[idx], dtype=torch.long), torch.as_tensor(self.labels[idx], dtype=torch.float32)


def read_split_rows(path: Path, split_column: str) -> Dict[str, List[Dict[str, str]]]:
    groups = {"train": [], "valid": [], "test": []}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            split = row.get(split_column, "")
            if split in groups:
                groups[split].append(row)
    return groups


def predict(model: OfficialMultiRM, rows: Sequence[Dict[str, str]], vocab: Dict[str, int], batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    dataset = EncodedSequenceDataset(rows, vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            official_probs, _ = model(x.to(device))
            probs.append(select_m1a_m6a_probs(official_probs.detach().cpu().numpy()))
            labels.append(y.numpy())
    return np.vstack(probs), np.vstack(labels)


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


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = load_embedding_vocab(Path(args.embedding_path))
    model = OfficialMultiRM(Path(args.embedding_path), num_task=len(OFFICIAL_TASKS)).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(state)

    rows_by_split = read_split_rows(Path(args.split_file), args.split_column)
    valid_probs, valid_labels = predict(model, rows_by_split["valid"], vocab, args.batch_size, device)
    test_probs, test_labels = predict(model, rows_by_split["test"], vocab, args.batch_size, device)
    thresholds = calibrate_thresholds(valid_probs, valid_labels)

    metrics = {
        "model_name": "MultiRM_official_101nt_zero_shot",
        "variant": "multirm_official_101nt_zero_shot",
        "device": str(device),
        "split_file": args.split_file,
        "split_column": args.split_column,
        "weights": args.weights,
        "note": "Official MultiRM 101nt pretrained weights evaluated directly on the frozen m1A/m6A split; no retraining on this dataset.",
        "row_counts": {key: len(value) for key, value in rows_by_split.items()},
        "validation": evaluate(valid_probs, valid_labels, thresholds),
        "test": evaluate(test_probs, test_labels, thresholds),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "thresholds_validation.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    write_predictions(out_dir / "valid_predictions.tsv", rows_by_split["valid"], valid_labels, valid_probs, thresholds)
    write_predictions(out_dir / "test_predictions.tsv", rows_by_split["test"], test_labels, test_probs, thresholds)
    (out_dir / "README.md").write_text(
        "Official MultiRM 101nt pretrained zero-shot benchmark on the frozen m1A/m6A split.\n"
        "This run loads the public 12-task MultiRM weights and reports only m1A and m6A heads.\n",
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'metrics.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate official pretrained MultiRM on the frozen benchmark split.")
    parser.add_argument("--split-file", default="05_datasets/benchmark_splits/m1A_m6A_split_random_seed42.tsv")
    parser.add_argument("--split-column", default="split_random_seed42")
    parser.add_argument("--embedding-path", default="external/MultiRM-master/Embeddings/embeddings_12RM.pkl")
    parser.add_argument("--weights", default="external/MultiRM-master/Weights/MultiRM/trained_model_101seqs.pkl")
    parser.add_argument("--out-dir", default="12_benchmarks/multirm_official_101nt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
