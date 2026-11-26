#!/usr/bin/env python
"""Train an A-site second-stage reranker from an existing 12-class model.

The script does not retrain the main model.  It learns a specialist classifier
on training rows for either m1A/m6A or the broader A-family classes, selects the
blend weight on the validation predictions, and then rewrites only the
specialist class probability mass in the test predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline


DEFAULT_CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "Y", "AtoI"]
STAGE_FOCUS = {
    "binary_m1A_m6A": ["m1A", "m6A"],
    "a_family": ["Am", "m1A", "m6A", "m6Am", "AtoI"],
}
BASES = set("ACGTU")


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing TSV header: {path}")
        return [dict(row) for row in reader]


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def split_rows(rows: Sequence[Dict[str, str]], split_col: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    by_split: Dict[str, List[Dict[str, str]]] = {"train": [], "valid": [], "test": []}
    for row in rows:
        split = str(row.get(split_col, "")).strip().lower()
        if split in by_split:
            by_split[split].append(row)
    return by_split["train"], by_split["valid"], by_split["test"]


def infer_class_names(pred_rows: Sequence[Mapping[str, str]], explicit: str) -> List[str]:
    names = parse_csv(explicit)
    if names:
        return names
    if not pred_rows:
        return list(DEFAULT_CLASS_NAMES)
    prob_names = [field[5:] for field in pred_rows[0].keys() if field.startswith("prob_")]
    return prob_names or list(DEFAULT_CLASS_NAMES)


def normalize_sequence(seq: str) -> str:
    return "".join(base if base in BASES else "N" for base in str(seq or "").upper().replace("T", "U"))


def add_kmer_tokens(tokens: List[str], prefix: str, seq: str, k_values: Iterable[int]) -> None:
    seq = normalize_sequence(seq)
    for k in k_values:
        for i in range(0, max(0, len(seq) - k + 1)):
            kmer = seq[i : i + k]
            if "N" not in kmer:
                tokens.append(f"{prefix}_k{k}_{kmer}")


def add_center_position_tokens(tokens: List[str], prefix: str, seq: str, radius: int) -> None:
    seq = normalize_sequence(seq)
    if not seq:
        return
    center = len(seq) // 2
    left = max(0, center - radius)
    right = min(len(seq), center + radius + 1)
    for pos in range(left, right):
        base = seq[pos]
        if base in BASES:
            tokens.append(f"{prefix}_rel{pos - center}_{base}")


def load_structure(path: str) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    struct_path = Path(path)
    if not struct_path.exists():
        return {}
    rows = read_tsv(struct_path)
    return {row["site_id"]: row for row in rows if row.get("site_id")}


def numeric_bin(value: float, scale: float = 10.0, limit: int = 200) -> int:
    return max(-limit, min(limit, int(round(value * scale))))


def add_structure_tokens(tokens: List[str], struct_row: Mapping[str, str], center_radius: int) -> None:
    if not struct_row:
        return
    for key, value in struct_row.items():
        if key == "site_id":
            continue
        parsed = safe_float(value, float("nan"))
        if math.isfinite(parsed):
            tokens.append(f"struct_{key}_bin{numeric_bin(parsed)}")
    dot = str(struct_row.get("dot_bracket") or struct_row.get("structure") or "")
    if dot:
        center = len(dot) // 2
        for pos in range(max(0, center - center_radius), min(len(dot), center + center_radius + 1)):
            tokens.append(f"struct_dot_rel{pos - center}_{dot[pos]}")
        paired = sum(1 for ch in dot if ch in "()[]{}<>")
        if dot:
            tokens.append(f"struct_dot_paired_frac_bin{numeric_bin(paired / len(dot), scale=20.0, limit=20)}")


def build_feature_text(
    row: Mapping[str, str],
    sequence_columns: Sequence[str],
    k_values: Sequence[int],
    center_radius: int,
    struct_by_id: Mapping[str, Mapping[str, str]],
) -> str:
    tokens: List[str] = []
    for column in sequence_columns:
        seq = str(row.get(column, ""))
        if not seq:
            continue
        add_kmer_tokens(tokens, column, seq, k_values)
        add_center_position_tokens(tokens, column, seq, center_radius)
        gc = sum(1 for b in normalize_sequence(seq) if b in {"G", "C"}) / max(1, len(seq))
        tokens.append(f"{column}_gc_bin{numeric_bin(gc, scale=20.0, limit=20)}")
    site_id = str(row.get("site_id", ""))
    add_structure_tokens(tokens, struct_by_id.get(site_id, {}), center_radius)
    return " ".join(tokens)


def cap_focus_train_rows(
    rows: Sequence[Dict[str, str]],
    focus_classes: Sequence[str],
    max_ratio_to_minority: int,
    max_per_class: int,
    seed: int,
) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        mod = str(row.get("mod_type", ""))
        if mod in focus_classes:
            grouped[mod].append(row)
    non_empty_counts = [len(grouped[name]) for name in focus_classes if grouped.get(name)]
    if not non_empty_counts:
        return []
    minority = max(1, min(non_empty_counts))
    limit = max(1, minority * max_ratio_to_minority)
    if max_per_class > 0:
        limit = min(limit, max_per_class)
    rng = random.Random(seed)
    capped: List[Dict[str, str]] = []
    for name in focus_classes:
        group = list(grouped.get(name, []))
        rng.shuffle(group)
        capped.extend(group[: min(len(group), limit)])
    rng.shuffle(capped)
    return capped


def rows_to_xy(
    rows: Sequence[Mapping[str, str]],
    focus_classes: Sequence[str],
    sequence_columns: Sequence[str],
    k_values: Sequence[int],
    center_radius: int,
    struct_by_id: Mapping[str, Mapping[str, str]],
) -> Tuple[List[str], np.ndarray]:
    focus_to_idx = {name: idx for idx, name in enumerate(focus_classes)}
    x = [build_feature_text(row, sequence_columns, k_values, center_radius, struct_by_id) for row in rows]
    y = np.asarray([focus_to_idx[str(row.get("mod_type", ""))] for row in rows], dtype=np.int64)
    return x, y


def prediction_matrix(pred_rows: Sequence[Mapping[str, str]], class_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    probs = np.asarray(
        [[safe_float(row.get(f"prob_{name}", 0.0)) for name in class_names] for row in pred_rows],
        dtype=np.float64,
    )
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.maximum(row_sums, 1e-12)
    labels = np.asarray([int(float(row["true_class_index"])) for row in pred_rows], dtype=np.int64)
    site_ids = [str(row.get("site_id", "")) for row in pred_rows]
    mods = [str(row.get("true_mod_type", "")) for row in pred_rows]
    return probs, labels, site_ids, mods


def metric_or_nan(fn, *args) -> float:
    try:
        return float(fn(*args))
    except ValueError:
        return float("nan")


def evaluate(probs: np.ndarray, labels: np.ndarray, class_names: Sequence[str]) -> Dict[str, object]:
    pred = probs.argmax(axis=1)
    onehot = np.eye(len(class_names), dtype=np.int32)[labels]
    per_class: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(class_names):
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
        "macro_AUPR_ovr": float(np.nanmean([per_class[name]["AUPR_ovr"] for name in class_names])),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, pred, labels=list(range(len(class_names)))).tolist(),
    }


def predict_specialist_probs(
    model: Pipeline,
    x: Sequence[str],
    focus_classes: Sequence[str],
) -> np.ndarray:
    raw = model.predict_proba(x)
    class_to_col = {int(cls): idx for idx, cls in enumerate(model.named_steps["clf"].classes_)}
    out = np.zeros((len(x), len(focus_classes)), dtype=np.float64)
    for idx in range(len(focus_classes)):
        if idx in class_to_col:
            out[:, idx] = raw[:, class_to_col[idx]]
    row_sums = out.sum(axis=1, keepdims=True)
    return out / np.maximum(row_sums, 1e-12)


def adjust_probability_mass(
    base_probs: np.ndarray,
    focus_indices: Sequence[int],
    specialist_probs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    adjusted = base_probs.copy()
    focus = np.asarray(list(focus_indices), dtype=np.int64)
    mass = adjusted[:, focus].sum(axis=1)
    base_cond = adjusted[:, focus] / np.maximum(mass[:, None], 1e-12)
    uniform = np.full_like(base_cond, 1.0 / max(1, len(focus_indices)))
    base_cond = np.where(mass[:, None] > 1e-12, base_cond, uniform)
    mixed = (1.0 - alpha) * base_cond + alpha * specialist_probs
    adjusted[:, focus] = mass[:, None] * mixed
    return adjusted / np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)


def selection_score(metrics: Mapping[str, object], focus_classes: Sequence[str], target_class: str) -> float:
    per_class = metrics.get("per_class", {})
    if not isinstance(per_class, Mapping):
        return float("-inf")
    target_values = per_class.get(target_class, {})
    target = safe_float(target_values.get("AUPR_ovr") if isinstance(target_values, Mapping) else None, float("nan"))
    if not math.isfinite(target):
        return float("-inf")
    focus_values = []
    for name in focus_classes:
        values = per_class.get(name, {})
        if isinstance(values, Mapping):
            value = safe_float(values.get("AUPR_ovr"), float("nan"))
            if math.isfinite(value):
                focus_values.append(value)
    focus_mean = float(np.mean(focus_values)) if focus_values else target
    return 0.8 * target + 0.2 * focus_mean


def per_class_rows(metrics: Mapping[str, object], class_names: Sequence[str], prefix: str = "") -> List[Dict[str, object]]:
    per_class = metrics.get("per_class", {})
    rows = []
    for name in class_names:
        values = per_class.get(name, {}) if isinstance(per_class, Mapping) else {}
        row: Dict[str, object] = {"mod_type": name}
        for key in ["support", "AUPR_ovr", "AUC_ovr", "F1", "Precision", "Recall", "MCC_ovr"]:
            out_key = f"{prefix}{key}" if prefix else key
            row[out_key] = values.get(key, "") if isinstance(values, Mapping) else ""
        rows.append(row)
    return rows


def write_predictions(path: Path, pred_rows: Sequence[Mapping[str, str]], probs: np.ndarray, class_names: Sequence[str]) -> None:
    fields = ["site_id", "true_mod_type", "true_class_index", "pred_mod_type", "pred_class_index"]
    fields += [f"prob_{name}" for name in class_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = probs.argmax(axis=1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, row in enumerate(pred_rows):
            out = {
                "site_id": row.get("site_id", ""),
                "true_mod_type": row.get("true_mod_type", ""),
                "true_class_index": int(float(row.get("true_class_index", 0))),
                "pred_mod_type": class_names[int(pred[idx])],
                "pred_class_index": int(pred[idx]),
            }
            out.update({f"prob_{name}": float(probs[idx, j]) for j, name in enumerate(class_names)})
            writer.writerow(out)


def summarize_comparison(
    base_metrics: Mapping[str, object],
    adjusted_metrics: Mapping[str, object],
    class_names: Sequence[str],
    focus_classes: Sequence[str],
) -> List[Dict[str, object]]:
    base_pc = base_metrics.get("per_class", {})
    adj_pc = adjusted_metrics.get("per_class", {})
    rows = []
    for name in class_names:
        b = base_pc.get(name, {}) if isinstance(base_pc, Mapping) else {}
        a = adj_pc.get(name, {}) if isinstance(adj_pc, Mapping) else {}
        row = {
            "mod_type": name,
            "focus_class": int(name in focus_classes),
            "support": a.get("support", "") if isinstance(a, Mapping) else "",
            "base_AUPR_ovr": b.get("AUPR_ovr", "") if isinstance(b, Mapping) else "",
            "two_stage_AUPR_ovr": a.get("AUPR_ovr", "") if isinstance(a, Mapping) else "",
            "delta_AUPR_ovr": safe_float(a.get("AUPR_ovr") if isinstance(a, Mapping) else None, float("nan"))
            - safe_float(b.get("AUPR_ovr") if isinstance(b, Mapping) else None, float("nan")),
            "base_F1": b.get("F1", "") if isinstance(b, Mapping) else "",
            "two_stage_F1": a.get("F1", "") if isinstance(a, Mapping) else "",
            "base_MCC_ovr": b.get("MCC_ovr", "") if isinstance(b, Mapping) else "",
            "two_stage_MCC_ovr": a.get("MCC_ovr", "") if isinstance(a, Mapping) else "",
        }
        rows.append(row)
    return rows


def run_stage(
    stage: str,
    args: argparse.Namespace,
    rows_by_id: Mapping[str, Dict[str, str]],
    train_rows: Sequence[Dict[str, str]],
    valid_pred_rows: Sequence[Dict[str, str]],
    test_pred_rows: Sequence[Dict[str, str]],
    class_names: Sequence[str],
    struct_by_id: Mapping[str, Mapping[str, str]],
) -> Dict[str, object]:
    focus_classes = STAGE_FOCUS[stage]
    print(f"[stage:{stage}] focus_classes={','.join(focus_classes)}", flush=True)
    missing = [name for name in focus_classes if name not in class_names]
    if missing:
        raise ValueError(f"{stage} focus class(es) missing from class names: {', '.join(missing)}")
    focus_indices = [class_names.index(name) for name in focus_classes]
    train_focus = cap_focus_train_rows(
        train_rows,
        focus_classes,
        args.max_ratio_to_minority,
        args.max_train_per_class,
        args.seed,
    )
    print(
        f"[stage:{stage}] train_focus_counts={dict(Counter(row.get('mod_type', '') for row in train_focus))}",
        flush=True,
    )
    sequence_columns = parse_csv(args.sequence_columns)
    k_values = [int(item) for item in parse_csv(args.k_values)]
    print(f"[stage:{stage}] build train/valid/test hashed-token features", flush=True)
    x_train, y_train = rows_to_xy(train_focus, focus_classes, sequence_columns, k_values, args.center_radius, struct_by_id)

    valid_rows = [rows_by_id[row["site_id"]] for row in valid_pred_rows if row.get("site_id") in rows_by_id]
    test_rows = [rows_by_id[row["site_id"]] for row in test_pred_rows if row.get("site_id") in rows_by_id]
    if len(valid_rows) != len(valid_pred_rows) or len(test_rows) != len(test_pred_rows):
        raise ValueError("Prediction rows and dataset rows do not have matching site_id coverage.")
    x_valid_all = [build_feature_text(row, sequence_columns, k_values, args.center_radius, struct_by_id) for row in valid_rows]
    x_test_all = [build_feature_text(row, sequence_columns, k_values, args.center_radius, struct_by_id) for row in test_rows]
    print(f"[stage:{stage}] feature_rows train={len(x_train)} valid={len(x_valid_all)} test={len(x_test_all)}", flush=True)

    valid_base_probs, valid_labels, _, _ = prediction_matrix(valid_pred_rows, class_names)
    test_base_probs, test_labels, _, _ = prediction_matrix(test_pred_rows, class_names)
    valid_base_metrics = evaluate(valid_base_probs, valid_labels, class_names)
    test_base_metrics = evaluate(test_base_probs, test_labels, class_names)

    candidates: List[Dict[str, object]] = []
    best: Dict[str, object] | None = None
    best_model: Pipeline | None = None
    for c_value in [float(item) for item in parse_csv(args.c_values)]:
        print(f"[stage:{stage}] fit LogisticRegression C={c_value}", flush=True)
        model = Pipeline(
            [
                (
                    "vec",
                    HashingVectorizer(
                        analyzer="word",
                        token_pattern=r"(?u)\S+",
                        lowercase=False,
                        alternate_sign=False,
                        norm="l2",
                        n_features=args.hash_features,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=args.max_iter,
                        solver="saga",
                        penalty="l2",
                        n_jobs=args.n_jobs,
                        random_state=args.seed,
                        multi_class="auto",
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        valid_specialist = predict_specialist_probs(model, x_valid_all, focus_classes)
        for alpha in [float(item) for item in parse_csv(args.alphas)]:
            adjusted = adjust_probability_mass(valid_base_probs, focus_indices, valid_specialist, alpha)
            metrics = evaluate(adjusted, valid_labels, class_names)
            score = selection_score(metrics, focus_classes, args.target_class)
            row = {
                "stage": stage,
                "C": c_value,
                "alpha": alpha,
                "selection_score": score,
                "valid_macro_AUPR_ovr": metrics["macro_AUPR_ovr"],
            }
            for name in focus_classes:
                values = metrics["per_class"][name]  # type: ignore[index]
                row[f"valid_{name}_AUPR_ovr"] = values["AUPR_ovr"]
                row[f"valid_{name}_F1"] = values["F1"]
            candidates.append(row)
            if best is None or score > float(best["selection_score"]):
                best = row
                best_model = model
        print(f"[stage:{stage}] done C={c_value}", flush=True)
    if best is None or best_model is None:
        raise RuntimeError(f"No model candidate produced for stage {stage}")

    test_specialist = predict_specialist_probs(best_model, x_test_all, focus_classes)
    test_adjusted = adjust_probability_mass(test_base_probs, focus_indices, test_specialist, float(best["alpha"]))
    test_metrics = evaluate(test_adjusted, test_labels, class_names)

    out_dir = Path(args.out_dir) / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "validation_candidate_grid.tsv", candidates)
    write_tsv(out_dir / "test_per_class_metrics.tsv", per_class_rows(test_metrics, class_names))
    write_tsv(out_dir / "test_base_vs_two_stage_per_class.tsv", summarize_comparison(test_base_metrics, test_metrics, class_names, focus_classes))
    write_predictions(out_dir / "test_predictions.tsv", test_pred_rows, test_adjusted, class_names)
    summary = {
        "stage": stage,
        "focus_classes": focus_classes,
        "class_names": list(class_names),
        "base_model_dir": args.base_model_dir,
        "dataset": args.dataset,
        "rnafold": args.rnafold,
        "sequence_columns": sequence_columns,
        "k_values": k_values,
        "center_radius": args.center_radius,
        "train_focus_counts_before_cap": dict(Counter(row.get("mod_type", "") for row in train_rows if row.get("mod_type", "") in focus_classes)),
        "train_focus_counts_after_cap": dict(Counter(row.get("mod_type", "") for row in train_focus)),
        "best_validation_candidate": best,
        "validation_base": valid_base_metrics,
        "test_base": test_base_metrics,
        "test_two_stage": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[stage:{stage}] best_C={best['C']} best_alpha={best['alpha']} "
        f"test_m1A_AUPR={test_metrics['per_class'].get('m1A', {}).get('AUPR_ovr', '')}",
        flush=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv")
    parser.add_argument("--rnafold", default="rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv")
    parser.add_argument("--base-model-dir", default="rmbase12_work/models/multirm12_multi_length_fusion_cdhit80/multi_length_independent")
    parser.add_argument("--out-dir", default="rmbase12_work/metrics/a_site_two_stage_reranker/multi_length_independent")
    parser.add_argument("--split", default="split_cdhit80_cluster")
    parser.add_argument("--class-names", default="")
    parser.add_argument("--stages", default="binary_m1A_m6A,a_family")
    parser.add_argument("--target-class", default="m1A")
    parser.add_argument("--sequence-columns", default="sequence_41nt,sequence_101nt,sequence_201nt")
    parser.add_argument("--k-values", default="3,4,5")
    parser.add_argument("--center-radius", type=int, default=12)
    parser.add_argument("--hash-features", type=int, default=262144)
    parser.add_argument("--max-ratio-to-minority", type=int, default=20)
    parser.add_argument("--max-train-per-class", type=int, default=20000)
    parser.add_argument("--c-values", default="0.03,0.1,0.3,1.0,3.0")
    parser.add_argument("--alphas", default="0.0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_rows = read_tsv(Path(args.dataset))
    train_rows, _, _ = split_rows(dataset_rows, args.split)
    rows_by_id = {row["site_id"]: row for row in dataset_rows if row.get("site_id")}
    valid_pred_rows = read_tsv(Path(args.base_model_dir) / "valid_predictions.tsv")
    test_pred_rows = read_tsv(Path(args.base_model_dir) / "test_predictions.tsv")
    class_names = infer_class_names(test_pred_rows, args.class_names)
    struct_by_id = load_structure(args.rnafold)
    print(
        f"[load] dataset_rows={len(dataset_rows)} train_rows={len(train_rows)} "
        f"valid_predictions={len(valid_pred_rows)} test_predictions={len(test_pred_rows)} "
        f"struct_rows={len(struct_by_id)}",
        flush=True,
    )

    summaries = []
    for stage in parse_csv(args.stages):
        if stage not in STAGE_FOCUS:
            raise ValueError(f"Unknown stage: {stage}. Choices: {', '.join(STAGE_FOCUS)}")
        summaries.append(
            run_stage(
                stage,
                args,
                rows_by_id,
                train_rows,
                valid_pred_rows,
                test_pred_rows,
                class_names,
                struct_by_id,
            )
        )

    out_dir = Path(args.out_dir)
    compact_rows = []
    for summary in summaries:
        stage = str(summary["stage"])
        best = summary["best_validation_candidate"]
        test_metrics = summary["test_two_stage"]
        base_metrics = summary["test_base"]
        assert isinstance(best, Mapping)
        assert isinstance(test_metrics, Mapping)
        assert isinstance(base_metrics, Mapping)
        for name in STAGE_FOCUS[stage]:
            adj_pc = test_metrics["per_class"][name]  # type: ignore[index]
            base_pc = base_metrics["per_class"][name]  # type: ignore[index]
            compact_rows.append(
                {
                    "stage": stage,
                    "mod_type": name,
                    "best_C": best["C"],
                    "best_alpha": best["alpha"],
                    "support": adj_pc["support"],
                    "base_AUPR_ovr": base_pc["AUPR_ovr"],
                    "two_stage_AUPR_ovr": adj_pc["AUPR_ovr"],
                    "delta_AUPR_ovr": float(adj_pc["AUPR_ovr"]) - float(base_pc["AUPR_ovr"]),
                    "base_F1": base_pc["F1"],
                    "two_stage_F1": adj_pc["F1"],
                    "base_MCC_ovr": base_pc["MCC_ovr"],
                    "two_stage_MCC_ovr": adj_pc["MCC_ovr"],
                }
            )
    write_tsv(out_dir / "a_site_two_stage_summary.tsv", compact_rows)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
