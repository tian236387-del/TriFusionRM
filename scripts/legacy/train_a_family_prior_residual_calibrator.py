#!/usr/bin/env python
"""Train an A-family residual calibrator on fixed main-model predictions."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_a_family_neural_two_stage import A_FAMILY_CLASSES, prediction_prior_vector
from train_a_site_two_stage_reranker import (
    evaluate,
    infer_class_names,
    parse_csv,
    per_class_rows,
    prediction_matrix,
    read_tsv,
    safe_float,
    summarize_comparison,
    write_predictions,
    write_tsv,
)


def cap_prediction_rows(
    pred_rows: Sequence[Mapping[str, str]],
    focus_classes: Sequence[str],
    max_ratio_to_minority: int,
    max_per_class: int,
    cap_minority_floor: int,
    seed: int,
) -> List[Mapping[str, str]]:
    grouped: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in pred_rows:
        mod = str(row.get("true_mod_type", ""))
        if mod in focus_classes:
            grouped[mod].append(row)
    counts = [len(grouped[name]) for name in focus_classes if grouped.get(name)]
    if not counts:
        raise ValueError("No A-family train prediction rows were found.")
    reference = max(max(1, min(counts)), int(cap_minority_floor))
    limit = reference * max(1, int(max_ratio_to_minority))
    if max_per_class > 0:
        limit = min(limit, int(max_per_class))
    rng = random.Random(seed)
    capped: List[Mapping[str, str]] = []
    for name in focus_classes:
        group = list(grouped.get(name, []))
        rng.shuffle(group)
        capped.extend(group[: min(len(group), limit)])
    rng.shuffle(capped)
    return capped


def focus_base_logits(pred_rows: Sequence[Mapping[str, str]], class_names: Sequence[str], focus_classes: Sequence[str]) -> np.ndarray:
    focus_indices = [class_names.index(name) for name in focus_classes]
    rows = []
    for row in pred_rows:
        probs = np.asarray([safe_float(row.get(f"prob_{name}"), 0.0) for name in class_names], dtype=np.float64)
        focus_probs = probs[focus_indices]
        mass = float(focus_probs.sum())
        if mass <= 1e-12:
            cond = np.full(len(focus_classes), 1.0 / max(1, len(focus_classes)), dtype=np.float64)
        else:
            cond = focus_probs / mass
        rows.append(np.log(np.clip(cond, 1e-8, 1.0)))
    return np.asarray(rows, dtype=np.float32)


def feature_matrix(pred_rows: Sequence[Mapping[str, str]], class_names: Sequence[str], focus_classes: Sequence[str]) -> np.ndarray:
    values = [prediction_prior_vector(row, class_names, focus_classes) for row in pred_rows]
    return np.stack(values, axis=0).astype(np.float32) if values else np.zeros((0, 0), dtype=np.float32)


def normalize_features(values: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.shape[1] == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return values.astype(np.float32), empty, empty
    if mean is None:
        mean = values.mean(axis=0).astype(np.float32)
    if std is None:
        std = values.std(axis=0).astype(np.float32)
    std = np.maximum(std.astype(np.float32), 1e-6)
    return ((values.astype(np.float32) - mean) / std).astype(np.float32), mean, std


def labels_local(pred_rows: Sequence[Mapping[str, str]], focus_classes: Sequence[str]) -> np.ndarray:
    to_idx = {name: idx for idx, name in enumerate(focus_classes)}
    return np.asarray([to_idx[str(row.get("true_mod_type", ""))] for row in pred_rows], dtype=np.int64)


class PriorResidualDataset(Dataset):
    def __init__(self, x: np.ndarray, base_logits: np.ndarray, labels: np.ndarray):
        self.x = torch.tensor(np.asarray(x, dtype=np.float32).tolist(), dtype=torch.float32)
        self.base_logits = torch.tensor(np.asarray(base_logits, dtype=np.float32).tolist(), dtype=torch.float32)
        self.labels = torch.tensor(np.asarray(labels, dtype=np.int64).tolist(), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.base_logits[idx], self.labels[idx]


class ResidualCalibrator(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_sampler(labels: np.ndarray, epoch_size_multiplier: float) -> WeightedRandomSampler:
    counts = Counter(int(label) for label in labels.tolist())
    weights = np.asarray([1.0 / max(1, counts[int(label)]) for label in labels], dtype=np.float64)
    num_samples = max(1, int(round(len(labels) * epoch_size_multiplier)))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)


def class_weights(labels: np.ndarray, num_classes: int, power: float, max_weight: float, device: torch.device) -> torch.Tensor:
    counts = Counter(int(label) for label in labels.tolist())
    total = max(1, len(labels))
    values = []
    for idx in range(num_classes):
        count = max(1, counts.get(idx, 0))
        values.append(min(float(max_weight), (total / max(1.0, num_classes * count)) ** max(0.0, power)))
    arr = np.asarray(values, dtype=np.float32)
    arr = arr / max(1e-6, float(arr[arr > 0].mean()))
    return torch.tensor(arr, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_delta(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(x), batch_size):
        batch = torch.tensor(x[start : start + batch_size].tolist(), dtype=torch.float32, device=device)
        chunks.append(model(batch).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def local_probs(base_logits: np.ndarray, delta: np.ndarray, residual_scale: float, temperature: float, m1a_bias: float, focus_classes: Sequence[str]) -> np.ndarray:
    z = base_logits.astype(np.float64) + float(residual_scale) * delta.astype(np.float64)
    if "m1A" in focus_classes:
        z[:, focus_classes.index("m1A")] += float(m1a_bias)
    z = z / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.maximum(exp_z.sum(axis=1, keepdims=True), 1e-12)


def apply_residual(
    base_probs: np.ndarray,
    delta: np.ndarray,
    class_names: Sequence[str],
    focus_classes: Sequence[str],
    residual_scale: float,
    temperature: float,
    m1a_bias: float,
) -> np.ndarray:
    focus_indices = [class_names.index(name) for name in focus_classes]
    mass = base_probs[:, focus_indices].sum(axis=1)
    base_cond = base_probs[:, focus_indices] / np.maximum(mass[:, None], 1e-12)
    uniform = np.full_like(base_cond, 1.0 / max(1, len(focus_indices)))
    base_cond = np.where(mass[:, None] > 1e-12, base_cond, uniform)
    base_logits = np.log(np.clip(base_cond, 1e-8, 1.0))
    cond = local_probs(base_logits, delta, residual_scale, temperature, m1a_bias, focus_classes)
    adjusted = base_probs.copy()
    adjusted[:, focus_indices] = mass[:, None] * cond
    return adjusted / np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)


def binary_f1(labels: np.ndarray, pred: np.ndarray, class_idx: int) -> float:
    return float(f1_score((labels == class_idx).astype(np.int32), (pred == class_idx).astype(np.int32), zero_division=0))


def search_residual(
    base_probs: np.ndarray,
    labels: np.ndarray,
    delta: np.ndarray,
    class_names: Sequence[str],
    focus_classes: Sequence[str],
    residual_scales: Sequence[float],
    temperatures: Sequence[float],
    m1a_biases: Sequence[float],
    target_class: str,
    guard_classes: Sequence[str],
    guard_drop_penalty: float,
    macro_drop_penalty: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    focus_indices = [class_names.index(name) for name in focus_classes]
    focus_global = np.asarray(focus_indices, dtype=np.int64)
    base_metrics = evaluate(base_probs, labels, class_names)
    base_pc = base_metrics["per_class"]  # type: ignore[index]
    base_macro = safe_float(base_metrics.get("macro_AUPR_ovr"), 0.0)
    base_focus_aupr = {
        name: safe_float(base_pc.get(name, {}).get("AUPR_ovr"), float("nan"))  # type: ignore[union-attr]
        for name in focus_classes
    }
    mass = base_probs[:, focus_indices].sum(axis=1)
    base_cond = base_probs[:, focus_indices] / np.maximum(mass[:, None], 1e-12)
    uniform = np.full_like(base_cond, 1.0 / max(1, len(focus_indices)))
    base_cond = np.where(mass[:, None] > 1e-12, base_cond, uniform)
    base_logits = np.log(np.clip(base_cond, 1e-8, 1.0))
    other_probs = base_probs.copy()
    other_probs[:, focus_indices] = -1.0
    other_argmax = other_probs.argmax(axis=1)
    other_max = other_probs[np.arange(len(labels)), other_argmax]
    candidates: List[Dict[str, object]] = []
    best: Optional[Dict[str, object]] = None
    for scale in residual_scales:
        for temp in temperatures:
            for bias in m1a_biases:
                cond = local_probs(base_logits, delta, scale, temp, bias, focus_classes)
                new_focus_probs = mass[:, None] * cond
                pred_local = new_focus_probs.argmax(axis=1)
                focus_best = new_focus_probs[np.arange(len(labels)), pred_local]
                pred = np.where(focus_best > other_max, focus_global[pred_local], other_argmax)
                row: Dict[str, object] = {
                    "residual_scale": float(scale),
                    "temperature": float(temp),
                    "m1a_bias": float(bias),
                }
                delta_sum = 0.0
                for local_idx, name in enumerate(focus_classes):
                    global_idx = class_names.index(name)
                    y = (labels == global_idx).astype(np.int32)
                    aupr = float(average_precision_score(y, new_focus_probs[:, local_idx])) if y.max() > 0 else float("nan")
                    row[f"valid_{name}_AUPR_ovr"] = aupr
                    row[f"valid_{name}_F1"] = binary_f1(labels, pred, global_idx)
                    base_value = base_focus_aupr.get(name, float("nan"))
                    if math.isfinite(aupr) and math.isfinite(base_value):
                        delta_sum += aupr - base_value
                macro_approx = base_macro + delta_sum / max(1, len(class_names))
                target = safe_float(row.get(f"valid_{target_class}_AUPR_ovr"), float("nan"))
                target_f1 = safe_float(row.get(f"valid_{target_class}_F1"), 0.0)
                if not math.isfinite(target):
                    score = float("-inf")
                else:
                    score = target + 0.02 * target_f1
                    for name in guard_classes:
                        current = safe_float(row.get(f"valid_{name}_AUPR_ovr"), float("nan"))
                        base_value = base_focus_aupr.get(name, float("nan"))
                        if math.isfinite(current) and math.isfinite(base_value):
                            score -= guard_drop_penalty * max(0.0, base_value - current)
                    score -= macro_drop_penalty * max(0.0, base_macro - macro_approx)
                row["valid_macro_AUPR_ovr_approx"] = macro_approx
                row["selection_score"] = score
                candidates.append(row)
                if best is None or score > safe_float(best.get("selection_score"), float("-inf")):
                    best = row
    if best is None:
        raise RuntimeError("No residual candidates were evaluated.")
    return best, candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-dir", default="rmbase12_work/models/multirm12_multi_length_fusion_cdhit80/multi_length_independent")
    parser.add_argument("--train-predictions", default="rmbase12_work/models/multirm12_multi_length_fusion_cdhit80/multi_length_independent/train_predictions_a_family.tsv")
    parser.add_argument("--out-dir", default="rmbase12_work/metrics/a_family_prior_residual_calibrator/multi_length_independent")
    parser.add_argument("--class-names", default="")
    parser.add_argument("--focus-classes", default=",".join(A_FAMILY_CLASSES))
    parser.add_argument("--target-class", default="m1A")
    parser.add_argument("--guard-classes", default="Am,m6A,AtoI")
    parser.add_argument("--max-ratio-to-minority", type=int, default=40)
    parser.add_argument("--cap-minority-floor", type=int, default=500)
    parser.add_argument("--max-train-per-class", type=int, default=40000)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--sampler-epoch-size-multiplier", type=float, default=1.0)
    parser.add_argument("--class-weight-power", type=float, default=0.25)
    parser.add_argument("--max-class-weight", type=float, default=8.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-residual-scale", type=float, default=0.2)
    parser.add_argument("--residual-l2", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--residual-scales", default="0,0.01,0.02,0.05,0.1,0.15,0.2,0.35,0.5,0.75,1")
    parser.add_argument("--temperatures", default="0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--m1a-biases", default="-3,-2,-1,-0.5,0,0.5,1,2,3")
    parser.add_argument("--guard-drop-penalty", type=float, default=4.0)
    parser.add_argument("--macro-drop-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_classes = parse_csv(args.focus_classes)
    train_rows_all = read_tsv(Path(args.train_predictions))
    valid_rows = read_tsv(Path(args.base_model_dir) / "valid_predictions.tsv")
    test_rows = read_tsv(Path(args.base_model_dir) / "test_predictions.tsv")
    class_names = infer_class_names(test_rows, args.class_names)
    for name in focus_classes:
        if name not in class_names:
            raise ValueError(f"Missing focus class in class_names: {name}")
    train_rows = cap_prediction_rows(
        train_rows_all,
        focus_classes,
        args.max_ratio_to_minority,
        args.max_train_per_class,
        args.cap_minority_floor,
        args.seed,
    )
    train_x_raw = feature_matrix(train_rows, class_names, focus_classes)
    train_x, feat_mean, feat_std = normalize_features(train_x_raw)
    train_logits = focus_base_logits(train_rows, class_names, focus_classes)
    train_y = labels_local(train_rows, focus_classes)
    valid_focus_rows = [row for row in valid_rows if str(row.get("true_mod_type", "")) in set(focus_classes)]
    valid_focus_x, _, _ = normalize_features(feature_matrix(valid_focus_rows, class_names, focus_classes), feat_mean, feat_std)
    valid_focus_logits = focus_base_logits(valid_focus_rows, class_names, focus_classes)
    valid_focus_y = labels_local(valid_focus_rows, focus_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualCalibrator(train_x.shape[1], len(focus_classes), args.hidden_dim, args.dropout).to(device)
    weights = class_weights(train_y, len(focus_classes), args.class_weight_power, args.max_class_weight, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    dataset = PriorResidualDataset(train_x, train_logits, train_y)
    sampler = make_sampler(train_y, args.sampler_epoch_size_multiplier) if args.balanced_sampler else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(
        f"[load] train_counts={dict(Counter(str(row.get('true_mod_type', '')) for row in train_rows))} "
        f"valid_focus={len(valid_focus_rows)} input_dim={train_x.shape[1]} device={device} "
        f"class_weights={[float(x) for x in weights.detach().cpu().tolist()]}",
        flush=True,
    )

    history: List[Dict[str, object]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    target_local = focus_classes.index(args.target_class)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, base_logits, labels in loader:
            x = x.to(device)
            base_logits = base_logits.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            delta = model(x)
            logits = base_logits + float(args.train_residual_scale) * delta
            loss = criterion(logits, labels) + float(args.residual_l2) * delta.pow(2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        valid_delta = predict_delta(model, valid_focus_x, args.batch_size * 2, device)
        probs = local_probs(valid_focus_logits, valid_delta, args.train_residual_scale, 1.0, 0.0, focus_classes)
        y = (valid_focus_y == target_local).astype(np.int32)
        score = float(average_precision_score(y, probs[:, target_local])) if y.max() > 0 else -1.0
        improved = score > best_score + args.early_stopping_min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            f"valid_{args.target_class}_AUPR": score,
            "improved": improved,
            "no_improve": no_improve,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch={epoch} loss={row['loss']:.4f} valid_{args.target_class}_AUPR={score:.6f} no_improve={no_improve}",
            flush=True,
        )
        if no_improve >= args.early_stopping_patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_score={best_score:.6f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_x, _, _ = normalize_features(feature_matrix(valid_rows, class_names, focus_classes), feat_mean, feat_std)
    test_x, _, _ = normalize_features(feature_matrix(test_rows, class_names, focus_classes), feat_mean, feat_std)
    valid_delta = predict_delta(model, valid_x, args.batch_size * 2, device)
    test_delta = predict_delta(model, test_x, args.batch_size * 2, device)
    valid_base_probs, valid_labels, _, _ = prediction_matrix(valid_rows, class_names)
    test_base_probs, test_labels, _, _ = prediction_matrix(test_rows, class_names)
    valid_base_metrics = evaluate(valid_base_probs, valid_labels, class_names)
    test_base_metrics = evaluate(test_base_probs, test_labels, class_names)
    best, candidates = search_residual(
        valid_base_probs,
        valid_labels,
        valid_delta,
        class_names,
        focus_classes,
        [float(x) for x in parse_csv(args.residual_scales)],
        [float(x) for x in parse_csv(args.temperatures)],
        [float(x) for x in parse_csv(args.m1a_biases)],
        args.target_class,
        parse_csv(args.guard_classes),
        args.guard_drop_penalty,
        args.macro_drop_penalty,
    )
    print(f"[fusion] best={best}", flush=True)
    test_adjusted = apply_residual(
        test_base_probs,
        test_delta,
        class_names,
        focus_classes,
        float(best["residual_scale"]),
        float(best["temperature"]),
        float(best["m1a_bias"]),
    )
    test_metrics = evaluate(test_adjusted, test_labels, class_names)
    write_tsv(out_dir / "training_history.tsv", history)
    write_tsv(out_dir / "validation_residual_grid.tsv", candidates)
    write_tsv(out_dir / "test_per_class_metrics.tsv", per_class_rows(test_metrics, class_names))
    write_tsv(out_dir / "test_base_vs_prior_residual_per_class.tsv", summarize_comparison(test_base_metrics, test_metrics, class_names, focus_classes))
    write_predictions(out_dir / "test_predictions.tsv", test_rows, test_adjusted, class_names)
    summary = {
        "focus_classes": focus_classes,
        "class_names": class_names,
        "train_predictions": args.train_predictions,
        "base_model_dir": args.base_model_dir,
        "train_counts": dict(Counter(str(row.get("true_mod_type", "")) for row in train_rows)),
        "best_epoch": best_epoch,
        "best_focus_validation_target_AUPR": best_score,
        "best_validation_residual": best,
        "validation_base": valid_base_metrics,
        "test_base": test_base_metrics,
        "test_prior_residual": test_metrics,
        "args": vars(args),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    base_pc = test_base_metrics["per_class"]  # type: ignore[index]
    new_pc = test_metrics["per_class"]  # type: ignore[index]
    parts = []
    for name in focus_classes:
        base_value = safe_float(base_pc.get(name, {}).get("AUPR_ovr"), float("nan"))  # type: ignore[union-attr]
        new_value = safe_float(new_pc.get(name, {}).get("AUPR_ovr"), float("nan"))  # type: ignore[union-attr]
        delta_value = new_value - base_value if math.isfinite(base_value) and math.isfinite(new_value) else float("nan")
        parts.append(f"{name}_AUPR={new_value} delta={delta_value}")
    print(
        f"[done] best_epoch={best_epoch} residual_scale={best['residual_scale']} temp={best['temperature']} "
        f"m1a_bias={best['m1a_bias']} " + " ".join(parts),
        flush=True,
    )
    print(out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
