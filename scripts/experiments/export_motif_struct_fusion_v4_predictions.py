#!/usr/bin/env python
"""Export train/valid/test predictions from a saved MotifStructFusionV4 model."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_motif_struct_fusion_v4 as v4


def parse_csv(value: object) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_int_csv(value: object) -> List[int]:
    return [int(item) for item in parse_csv(value)]


def namespace_get(mapping: Mapping[str, object], key: str, default: object) -> object:
    value = mapping.get(key, default)
    return default if value is None else value


def build_dataset(
    rows: Sequence[Dict[str, str]],
    fold_by_id: Mapping[str, Mapping[str, str]],
    model_payload: Mapping[str, object],
    training_args: Mapping[str, object],
    rbp_by_id: Mapping[str, np.ndarray],
    rbp_fields: Sequence[str],
    rbp_pos_by_id: Mapping[str, np.ndarray],
    rbp_pos_shape: Sequence[int],
):
    config = dict(model_payload.get("config", {}))
    rbp_dim = int(config.get("rbp_dim", len(rbp_fields) if rbp_fields else 0))
    seq_encoder = str(config.get("seq_encoder", namespace_get(training_args, "seq_encoder", "motif")))
    dataset_extra = {}
    if seq_encoder == "multi_length":
        multi_length_columns = tuple(config.get("multi_length_columns") or parse_csv(namespace_get(training_args, "multi_length_columns", "")))
        multi_length_lengths = tuple(config.get("multi_length_lengths") or parse_int_csv(namespace_get(training_args, "multi_length_lengths", "")))
        dataset_extra.update(
            {
                "multi_length_columns": multi_length_columns,
                "multi_length_lengths": multi_length_lengths,
            }
        )
        if bool(config.get("use_rbp_positional", False)):
            dataset_cls = v4.RMBaseFusionMultiLengthPositionalDataset
            dataset_extra.update({"rbp_pos_by_id": rbp_pos_by_id, "rbp_pos_shape": tuple(rbp_pos_shape)})
        else:
            dataset_cls = v4.RMBaseFusionMultiLengthDataset
    elif bool(config.get("use_rbp_positional", False)):
        dataset_cls = v4.RMBaseFusionPositionalDataset
        dataset_extra.update({"rbp_pos_by_id": rbp_pos_by_id, "rbp_pos_shape": tuple(rbp_pos_shape)})
    else:
        dataset_cls = v4.RMBaseFusionDataset

    return dataset_cls(
        rows,
        fold_by_id,
        int(config.get("input_length", namespace_get(training_args, "input_length", 41))),
        rbp_by_id=rbp_by_id,
        rbp_dim=rbp_dim,
        struct_stats=model_payload.get("struct_stats"),
        rbp_stats=model_payload.get("rbp_stats"),
        shuffle_structure=False,
        shuffle_rbp=False,
        seed=int(namespace_get(training_args, "seed", 42)),
        sequence_column=str(namespace_get(training_args, "sequence_column", "sequence_41nt")),
        **dataset_extra,
    )


def load_selected_rbp_positional_features(path: str, site_ids: Sequence[str]) -> tuple[Dict[str, np.ndarray], tuple[int, int, int], Dict[str, object]]:
    if not path:
        return {}, (0, 0, 0), {}
    rbp_path = Path(path)
    if not rbp_path.exists():
        raise FileNotFoundError(f"RBP positional feature file not found: {rbp_path}")
    selected = set(str(site_id) for site_id in site_ids)
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
    values: Dict[str, np.ndarray] = {}
    with rbp_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "site_id" not in reader.fieldnames:
            raise ValueError(f"RBP positional TSV must contain site_id: {rbp_path}")
        fields_in_file = [field for field in reader.fieldnames if field != "site_id"]
        if not rbps or not bins or not features:
            parsed = [field.split("__") for field in fields_in_file if field.startswith("rbp_pos__")]
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
        seen = 0
        for row_index, row in enumerate(reader, start=1):
            site_id = row.get("site_id", "")
            if site_id in selected:
                arr = np.asarray([float(row.get(column, 0.0) or 0.0) for column in columns], dtype=np.float32)
                values[site_id] = arr.reshape(shape)
                seen += 1
            if row_index % 200000 == 0:
                print(f"[load_rbp_pos] scanned={row_index} kept={seen}", flush=True)
    metadata.setdefault("rbps", rbps)
    metadata.setdefault("bins", bins)
    metadata.setdefault("features", features)
    metadata.setdefault("shape", list(shape))
    return values, shape, metadata


def build_config(model_payload: Mapping[str, object]) -> v4.MotifStructV4Config:
    raw_config = dict(model_payload.get("config", {}))
    allowed = {field.name for field in fields(v4.MotifStructV4Config)}
    return v4.MotifStructV4Config(**{key: value for key, value in raw_config.items() if key in allowed})


@torch.no_grad()
def predict_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    probs: List[List[float]] = []
    labels: List[int] = []
    site_ids: List[str] = []
    mod_types: List[str] = []
    for batch in loader:
        batch = v4.move_batch(batch, device)
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
        )
        probs.extend(torch.softmax(out["logits"], dim=-1).detach().cpu().tolist())
        labels.extend(batch["label"].detach().cpu().tolist())
        site_ids.extend(batch["site_id"])
        mod_types.extend(batch["mod_type"])
    return {
        "probs": np.asarray(probs, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "site_ids": site_ids,
        "mod_types": mod_types,
    }


def write_prediction_rows(path: Path, predictions: Mapping[str, object], append: bool) -> None:
    probs = predictions["probs"]
    labels = predictions["labels"]
    assert isinstance(probs, np.ndarray) and isinstance(labels, np.ndarray)
    pred = probs.argmax(axis=1)
    fields_out = ["site_id", "true_mod_type", "true_class_index", "pred_mod_type", "pred_class_index"]
    fields_out += [f"prob_{name}" for name in v4.CLASS_NAMES]
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_out, delimiter="\t", lineterminator="\n")
        if not append:
            writer.writeheader()
        for idx, site_id in enumerate(predictions["site_ids"]):  # type: ignore[index]
            row = {
                "site_id": site_id,
                "true_mod_type": predictions["mod_types"][idx],  # type: ignore[index]
                "true_class_index": int(labels[idx]),
                "pred_mod_type": v4.CLASS_NAMES[int(pred[idx])],
                "pred_class_index": int(pred[idx]),
            }
            row.update({f"prob_{name}": float(probs[idx, j]) for j, name in enumerate(v4.CLASS_NAMES)})
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "valid", "test", "all"])
    parser.add_argument("--output", default="")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--filter-classes", default="", help="Optional comma-separated true mod_type values to export.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_payload = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    training_config_path = model_dir / "training_config.json"
    training_args = dict(model_payload.get("training_args", {}))
    if training_config_path.exists():
        training_args.update(json.loads(training_config_path.read_text(encoding="utf-8")))

    class_names = [str(name) for name in model_payload.get("class_names", [])]
    v4.configure_classes(class_names)
    config = dict(model_payload.get("config", {}))
    if "struct_fields" in config:
        v4.STRUCT_FIELDS[:] = [str(x) for x in config.get("struct_fields", [])]
    if "node_feature_fields" in config:
        v4.NODE_FEATURE_FIELDS[:] = [str(x) for x in config.get("node_feature_fields", [])]

    rows = v4.read_tsv(Path(str(namespace_get(training_args, "dataset", ""))))
    fold_by_id = v4.load_rnafold(Path(str(namespace_get(training_args, "rnafold", ""))))
    rows = [row for row in rows if row.get("site_id") in fold_by_id]
    if int(namespace_get(training_args, "max_rows", 0) or 0):
        random.Random(int(namespace_get(training_args, "seed", 42))).shuffle(rows)
        rows = rows[: int(namespace_get(training_args, "max_rows", 0))]
    train_rows, valid_rows, test_rows = v4.split_rows(rows, str(namespace_get(training_args, "split", "split_cdhit80_cluster")))
    selected = {
        "train": train_rows,
        "valid": valid_rows,
        "test": test_rows,
        "all": list(train_rows) + list(valid_rows) + list(test_rows),
    }[args.split]
    filter_classes = set(parse_csv(args.filter_classes))
    if filter_classes:
        selected = [row for row in selected if str(row.get("mod_type", "")) in filter_classes]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = dict(model_payload.get("config", {}))
    print(f"[load] split={args.split} selected_rows={len(selected)} device={device}", flush=True)
    if bool(config.get("use_rbp", False)):
        rbp_by_id, rbp_fields = v4.load_rbp_features(str(namespace_get(training_args, "rbp_features", "")))
    else:
        rbp_by_id, rbp_fields = {}, []
    if bool(config.get("use_rbp_positional", False)):
        rbp_pos_by_id, rbp_pos_shape, _ = load_selected_rbp_positional_features(
            str(namespace_get(training_args, "rbp_positional_features", "")),
            [str(row.get("site_id", "")) for row in selected],
        )
    else:
        rbp_pos_by_id, rbp_pos_shape = {}, (0, 0, 0)
    print(
        f"[load] rbp_dim={len(rbp_fields)} rbp_pos_rows={len(rbp_pos_by_id)} rbp_pos_shape={tuple(rbp_pos_shape)}",
        flush=True,
    )
    model = v4.MotifStructFusionV4(build_config(model_payload)).to(device)
    model.load_state_dict(model_payload["model_state_dict"])

    output = Path(args.output) if args.output else model_dir / f"{args.split}_predictions.tsv"
    if output.exists():
        output.unlink()
    chunk_size = max(1, int(args.chunk_size))
    written = 0
    for start in range(0, len(selected), chunk_size):
        chunk_rows = selected[start : start + chunk_size]
        dataset = build_dataset(
            chunk_rows,
            fold_by_id,
            model_payload,
            training_args,
            rbp_by_id,
            rbp_fields,
            rbp_pos_by_id,
            rbp_pos_shape,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=v4.collate_batch_v4, num_workers=0)
        predictions = predict_probs(model, loader, device)
        write_prediction_rows(output, predictions, append=written > 0)
        written += len(chunk_rows)
        print(f"[predict] written={written}/{len(selected)} output={output}", flush=True)
    print(f"[done] split={args.split} rows={len(selected)} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
