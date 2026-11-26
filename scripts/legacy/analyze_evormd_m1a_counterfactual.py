#!/usr/bin/env python
"""Audit EvoRMD-adapted m1A ranking and export train-only teacher signals."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import train_rmbase11_official_reproduction_benchmark as evo


def read_prediction_scores(path: Optional[Path], score_field: str) -> Dict[str, float]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(row["site_id"]): float(row[score_field])
            for row in csv.DictReader(handle, delimiter="\t")
        }


def descending_percentiles(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    if len(scores) <= 1:
        return np.ones(len(scores), dtype=np.float64)
    return 1.0 - ranks.astype(np.float64) / float(len(scores) - 1)


def binary_ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return {"support": positives, "AUPR": float("nan"), "AUC": float("nan")}
    return {
        "support": positives,
        "AUPR": float(average_precision_score(labels, scores)),
        "AUC": float(roc_auc_score(labels, scores)),
    }


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, evo.ContextSpec, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = [str(name) for name in checkpoint["class_names"]]
    evo.configure_classes(class_names)
    context_spec = evo.ContextSpec(**checkpoint["context_spec"])
    training_args = dict(checkpoint.get("training_args", {}))
    model = evo.EvoRMDAdaptedModel(
        context_dim=context_spec.dim,
        num_classes=len(class_names),
        seq_dim=int(training_args.get("seq_dim", 128)),
        meta_dim=int(training_args.get("meta_dim", 128)),
        dropout=float(training_args.get("dropout", 0.2)),
        mlp_depth=int(training_args.get("mlp_depth", 3)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, context_spec, training_args


def build_context_masks(context_spec: evo.ContextSpec) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    region_end = len(context_spec.region_vocab)
    gene_end = region_end + len(context_spec.gene_vocab)
    cell_end = gene_end + context_spec.cell_hash_dim
    seqtype_end = cell_end + context_spec.seqtype_hash_dim
    support_idx = seqtype_end
    gene_known_idx = seqtype_end + 1
    constant_idx = seqtype_end + 2

    def mask_for(*spans: Tuple[int, int]) -> torch.Tensor:
        mask = torch.zeros(context_spec.dim, dtype=torch.float32)
        for start, end in spans:
            mask[start:end] = 1.0
        mask[constant_idx] = 1.0
        return mask

    masks["safe_context"] = mask_for((0, gene_end), (gene_known_idx, gene_known_idx + 1))
    masks["region_context_only"] = mask_for((0, region_end))
    masks["gene_context_only"] = mask_for((region_end, gene_end), (gene_known_idx, gene_known_idx + 1))
    masks["cell_context_only"] = mask_for((gene_end, cell_end))
    masks["seqtype_context_only"] = mask_for((cell_end, seqtype_end))
    masks["support_context_only"] = mask_for((support_idx, support_idx + 1))
    masks["source_context_only"] = mask_for(
        (gene_end, cell_end),
        (cell_end, seqtype_end),
        (support_idx, support_idx + 1),
    )
    return masks


@torch.no_grad()
def predict_modes(
    model: torch.nn.Module,
    dataset: evo.OfficialBenchmarkDataset,
    batch_size: int,
    device: torch.device,
    modes: Sequence[str],
    seed: int,
    context_masks: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, np.ndarray]:
    outputs: Dict[str, List[np.ndarray]] = {mode: [] for mode in modes}
    context_tensor = dataset.context_tensor
    if context_tensor is None:
        raise ValueError("EvoRMD counterfactual analysis requires context tensors")
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(len(dataset), generator=generator)
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        x = dataset.x_tensor[start:end].to(device)
        context = context_tensor[start:end].to(device)
        shuffled_context = context_tensor[shuffled_indices[start:end]].to(device)
        for mode in modes:
            if mode == "full":
                mode_x, mode_context = x, context
            elif mode == "zero_context":
                mode_x, mode_context = x, torch.zeros_like(context)
            elif mode == "shuffled_context":
                mode_x, mode_context = x, shuffled_context
            elif mode == "context_only":
                mode_x, mode_context = torch.zeros_like(x), context
            elif context_masks is not None and mode in context_masks:
                mask = context_masks[mode].to(device).view(1, -1)
                mode_context = context * mask
                mode_x = x if mode == "safe_context" else torch.zeros_like(x)
            else:
                raise ValueError(f"Unsupported counterfactual mode: {mode}")
            logits = model(mode_x, mode_context)["logits"]
            outputs[mode].append(torch.softmax(logits, dim=-1).cpu().numpy())
    return {mode: np.concatenate(parts, axis=0) for mode, parts in outputs.items()}


def write_teacher_scores(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    mode_probs: Mapping[str, np.ndarray],
    m1a_index: int,
) -> None:
    fields = ["site_id", "true_mod_type", "true_class_index"] + [
        f"teacher_{mode}_m1A" for mode in mode_probs
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, row in enumerate(rows):
            out = {
                "site_id": row["site_id"],
                "true_mod_type": row["mod_type"],
                "true_class_index": int(labels[idx]),
            }
            out.update(
                {
                    f"teacher_{mode}_m1A": float(probs[idx, m1a_index])
                    for mode, probs in mode_probs.items()
                }
            )
            writer.writerow(out)


def write_rows(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze_test_ranking(
    rows: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    mode_probs: Mapping[str, np.ndarray],
    class_names: Sequence[str],
    current_scores: Mapping[str, float],
    teacher_reference_scores: Mapping[str, float],
    out_dir: Path,
) -> Dict[str, object]:
    m1a_index = list(class_names).index("m1A")
    binary_labels = (labels == m1a_index).astype(np.int32)
    scores: Dict[str, np.ndarray] = {
        mode: probs[:, m1a_index].astype(np.float64)
        for mode, probs in mode_probs.items()
    }
    site_ids = [str(row["site_id"]) for row in rows]
    if current_scores:
        scores["current_main"] = np.asarray([current_scores[site_id] for site_id in site_ids], dtype=np.float64)
    if teacher_reference_scores:
        reference = np.asarray([teacher_reference_scores[site_id] for site_id in site_ids], dtype=np.float64)
        scores["saved_teacher_predictions"] = reference

    summary_rows: List[Dict[str, object]] = []
    support = int(binary_labels.sum())
    for name, values in scores.items():
        metric = binary_ranking_metrics(binary_labels, values)
        for multiplier in (1, 2, 5, 10):
            top_k = min(len(values), max(1, support * multiplier))
            top_indices = np.argsort(-values, kind="mergesort")[:top_k]
            metric[f"precision_at_{multiplier}x_support"] = float(binary_labels[top_indices].mean())
            metric[f"recall_at_{multiplier}x_support"] = float(binary_labels[top_indices].sum() / max(support, 1))
        summary_rows.append({"model_or_mode": name, **metric})
    summary_fields = list(summary_rows[0])
    write_rows(out_dir / "m1a_ranking_summary.tsv", summary_fields, summary_rows)

    quantile_rows: List[Dict[str, object]] = []
    for name, values in scores.items():
        for group_name, mask in (("true_m1A", binary_labels == 1), ("non_m1A", binary_labels == 0)):
            selected = values[mask]
            row: Dict[str, object] = {"model_or_mode": name, "group": group_name, "count": len(selected)}
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
                row[f"q{int(quantile * 100):03d}"] = float(np.quantile(selected, quantile))
            quantile_rows.append(row)
    write_rows(out_dir / "m1a_score_quantiles.tsv", list(quantile_rows[0]), quantile_rows)

    percentile_scores = {name: descending_percentiles(values) for name, values in scores.items()}
    positive_rows: List[Dict[str, object]] = []
    for idx in np.flatnonzero(binary_labels == 1):
        row = rows[int(idx)]
        out: Dict[str, object] = {
            "site_id": row["site_id"],
            "sequence_41nt": row.get("sequence_41nt", ""),
            "region_type": row.get("region_type", ""),
            "gene_type": row.get("gene_type", ""),
            "cell_list": row.get("cell_list", ""),
            "seq_type_list": row.get("seq_type_list", ""),
            "support_num": row.get("support_num", ""),
            "source_file": row.get("source_file", ""),
        }
        for name, values in scores.items():
            out[f"{name}_score"] = float(values[idx])
            out[f"{name}_rank_percentile"] = float(percentile_scores[name][idx])
        positive_rows.append(out)
    positive_rows.sort(key=lambda row: float(row.get("full_rank_percentile", 0.0)), reverse=True)
    write_rows(out_dir / "m1a_positive_rank_comparison.tsv", list(positive_rows[0]), positive_rows)

    full_scores = scores["full"]
    negative_indices = np.flatnonzero(binary_labels == 0)
    top_false = negative_indices[np.argsort(-full_scores[negative_indices], kind="mergesort")[:500]]
    false_rows: List[Dict[str, object]] = []
    false_counts: Dict[str, int] = {}
    for idx in top_false:
        source_class = str(rows[int(idx)]["mod_type"])
        false_counts[source_class] = false_counts.get(source_class, 0) + 1
        row = rows[int(idx)]
        out = {
            "site_id": row["site_id"],
            "true_mod_type": source_class,
            "sequence_41nt": row.get("sequence_41nt", ""),
            "region_type": row.get("region_type", ""),
            "gene_type": row.get("gene_type", ""),
            "cell_list": row.get("cell_list", ""),
            "seq_type_list": row.get("seq_type_list", ""),
            "support_num": row.get("support_num", ""),
            "source_file": row.get("source_file", ""),
        }
        out.update({f"{name}_score": float(values[idx]) for name, values in scores.items()})
        false_rows.append(out)
    write_rows(out_dir / "m1a_top_false_positives.tsv", list(false_rows[0]), false_rows)
    false_class_rows = [
        {"true_mod_type": name, "count_in_top500_teacher_false_positives": count}
        for name, count in sorted(false_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    write_rows(out_dir / "m1a_top_false_positive_classes.tsv", list(false_class_rows[0]), false_class_rows)

    agreement: Dict[str, object] = {}
    if "saved_teacher_predictions" in scores:
        agreement["max_abs_difference_full_vs_saved"] = float(
            np.max(np.abs(scores["full"] - scores["saved_teacher_predictions"]))
        )
    return {
        "test_rows": len(rows),
        "m1A_support": support,
        "ranking": {row["model_or_mode"]: row for row in summary_rows},
        "teacher_prediction_agreement": agreement,
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, context_spec, training_args = load_model(Path(args.model_path), device)
    context_masks = build_context_masks(context_spec)
    class_names = list(evo.CLASS_NAMES)
    m1a_index = class_names.index("m1A")

    rows = evo.read_tsv(Path(args.dataset))
    split_name = args.split or str(training_args.get("split", "split_cdhit80_cluster"))
    train_rows, valid_rows, test_rows = evo.split_rows(rows, split_name)
    if args.site_filter_dataset:
        allowed = {row["site_id"] for row in evo.read_tsv(Path(args.site_filter_dataset))}
        train_rows = [row for row in train_rows if row["site_id"] in allowed]
        valid_rows = [row for row in valid_rows if row["site_id"] in allowed]
        test_rows = [row for row in test_rows if row["site_id"] in allowed]

    split_rows = {"train": train_rows, "valid": valid_rows, "test": test_rows}
    exported: Dict[str, int] = {}
    test_mode_probs: Optional[Dict[str, np.ndarray]] = None
    test_labels: Optional[np.ndarray] = None
    for split in args.export_splits.split(","):
        split = split.strip()
        if not split:
            continue
        selected_rows = split_rows[split]
        dataset = evo.OfficialBenchmarkDataset(
            selected_rows,
            "evormd_adapted",
            int(training_args.get("input_length", 41)),
            context_spec=context_spec,
        )
        modes = ["full", "zero_context"]
        if split == "test":
            modes.extend(["shuffled_context", "context_only", *context_masks.keys()])
        mode_probs = predict_modes(
            model,
            dataset,
            args.batch_size,
            device,
            modes,
            args.seed,
            context_masks=context_masks,
        )
        labels = dataset.labels.astype(np.int64)
        write_teacher_scores(out_dir / f"{split}_teacher_m1a.tsv", selected_rows, labels, mode_probs, m1a_index)
        exported[split] = len(selected_rows)
        if split == "test":
            test_mode_probs = mode_probs
            test_labels = labels
        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result: Dict[str, object] = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "split": split_name,
        "site_filter_dataset": args.site_filter_dataset,
        "device": str(device),
        "classes": class_names,
        "exported_rows": exported,
        "counterfactual_definition": {
            "full": "sequence plus original RMBase-derived context",
            "zero_context": "sequence with all context values set to zero",
            "shuffled_context": "sequence paired with a deterministic random test context",
            "context_only": "context with every sequence token set to padding",
            "safe_context": "sequence plus region, gene type and gene-known indicator only",
            "region_context_only": "region block only with every sequence token set to padding",
            "gene_context_only": "gene-type block only with every sequence token set to padding",
            "cell_context_only": "cell-list hash block only with every sequence token set to padding",
            "seqtype_context_only": "sequence-type hash block only with every sequence token set to padding",
            "support_context_only": "support-count value only with every sequence token set to padding",
            "source_context_only": "cell-list, sequence-type and support-count blocks with every sequence token set to padding",
        },
    }
    if test_mode_probs is not None and test_labels is not None:
        result["counterfactual_metrics"] = {
            mode: evo.evaluate(probs, test_labels)
            for mode, probs in test_mode_probs.items()
        }
        result["m1a_ranking_analysis"] = analyze_test_ranking(
            test_rows,
            test_labels,
            test_mode_probs,
            class_names,
            read_prediction_scores(Path(args.current_test_predictions), "prob_m1A") if args.current_test_predictions else {},
            read_prediction_scores(Path(args.teacher_test_predictions), "prob_m1A") if args.teacher_test_predictions else {},
            out_dir,
        )
    (out_dir / "counterfactual_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="")
    parser.add_argument("--site-filter-dataset", default="")
    parser.add_argument("--current-test-predictions", default="")
    parser.add_argument("--teacher-test-predictions", default="")
    parser.add_argument("--export-splits", default="train,valid,test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
