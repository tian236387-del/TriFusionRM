#!/usr/bin/env python
"""Train a structure-interaction fusion model on RMBase11.

V2 fixes the gate-collapse issue observed in the first PairedGAT-RBP model.
Instead of allowing a softmax gate to ignore the structure branch, it uses
sequence-structure interaction features, residual refinement, auxiliary heads
and sequence-modality dropout.

The default main model is sequence + RNAfold structure. RBP is optional because
the current local RBP features are sample-level and were ignored by V1.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_paired_gat_rbp_fusion import (
    CLASS_NAMES,
    FocalCrossEntropy,
    RBPBranch,
    RMBaseFusionDataset,
    SequenceBranch,
    StructureBranch,
    class_weights_from_rows,
    collate_batch,
    evaluate,
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
class StructInteractConfig:
    input_length: int = 41
    num_classes: int = len(CLASS_NAMES)
    d_model: int = 128
    seq_layers: int = 2
    seq_heads: int = 4
    gat_layers: int = 2
    dropout: float = 0.3
    rbp_dim: int = 0
    use_structure: bool = True
    use_rbp: bool = False
    use_position: bool = True
    sequence_modality_dropout: float = 0.1


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StructInteractionFusion(nn.Module):
    """Sequence-structure interaction model without softmax gate collapse."""

    def __init__(self, cfg: StructInteractConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.seq_branch = SequenceBranch(d, cfg.input_length, cfg.seq_layers, cfg.seq_heads, cfg.dropout, cfg.use_position)
        self.struct_branch = StructureBranch(d, cfg.input_length, cfg.gat_layers, cfg.dropout, cfg.use_position)
        self.rbp_branch = RBPBranch(cfg.rbp_dim, d, cfg.dropout)
        self.seq_norm = nn.LayerNorm(d)
        self.struct_norm = nn.LayerNorm(d)
        self.rbp_norm = nn.LayerNorm(d)

        interaction_dim = d * 4
        self.seq_residual = MLPBlock(interaction_dim, d, cfg.dropout)
        self.struct_residual = MLPBlock(interaction_dim, d, cfg.dropout)
        self.fusion = nn.Sequential(
            nn.Linear(interaction_dim + (d if cfg.use_rbp and cfg.rbp_dim > 0 else 0), d * 2),
            nn.GELU(),
            nn.LayerNorm(d * 2),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )
        self.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d, cfg.num_classes))
        self.seq_head = nn.Linear(d, cfg.num_classes)
        self.struct_head = nn.Linear(d, cfg.num_classes)

    def _interaction(self, z_seq: torch.Tensor, z_struct: torch.Tensor) -> torch.Tensor:
        return torch.cat([z_seq, z_struct, z_seq * z_struct, torch.abs(z_seq - z_struct)], dim=-1)

    def _drop_sequence_modality(self, z_seq: torch.Tensor, z_struct: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.cfg.use_structure or self.cfg.sequence_modality_dropout <= 0:
            return z_seq
        keep = torch.rand(z_seq.shape[0], 1, device=z_seq.device) > self.cfg.sequence_modality_dropout
        # Keep a small residual instead of full zeroing so optimization stays stable.
        return torch.where(keep, z_seq, z_seq * 0.15 + z_struct * 0.85)

    def forward(
        self,
        seq: torch.Tensor,
        paired_flags: torch.Tensor,
        partner: torch.Tensor,
        struct: torch.Tensor,
        rbp: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch = seq.shape[0]
        device = seq.device
        z_seq = self.seq_norm(self.seq_branch(seq))
        if self.cfg.use_structure:
            z_struct, edge_attention = self.struct_branch(seq, paired_flags, partner, struct)
            z_struct = self.struct_norm(z_struct)
        else:
            z_struct = torch.zeros_like(z_seq)
            edge_attention = torch.zeros(batch, self.cfg.input_length, 4, device=device)

        z_seq_main = self._drop_sequence_modality(z_seq, z_struct)
        inter = self._interaction(z_seq_main, z_struct)
        z_seq_refined = self.seq_norm(z_seq_main + self.seq_residual(inter))
        z_struct_refined = self.struct_norm(z_struct + self.struct_residual(inter)) if self.cfg.use_structure else z_struct
        fusion_input = self._interaction(z_seq_refined, z_struct_refined)

        if self.cfg.use_rbp and self.cfg.rbp_dim > 0:
            z_rbp = self.rbp_norm(self.rbp_branch(rbp, batch, device))
            fusion_input = torch.cat([fusion_input, z_rbp], dim=-1)
            diagnostic_mix = torch.tensor([0.45, 0.45 if self.cfg.use_structure else 0.0, 0.10], device=device)
        else:
            diagnostic_mix = torch.tensor([0.50, 0.50 if self.cfg.use_structure else 0.0, 0.0], device=device)
        diagnostic_mix = diagnostic_mix / diagnostic_mix.sum().clamp_min(1.0)
        diagnostic_mix = diagnostic_mix.view(1, 3).expand(batch, 3)

        fused = self.fusion(fusion_input)
        return {
            "logits": self.classifier(fused),
            "seq_logits": self.seq_head(z_seq),
            "struct_logits": self.struct_head(z_struct) if self.cfg.use_structure else torch.zeros(batch, self.cfg.num_classes, device=device),
            "modality_gate": diagnostic_mix,
            "edge_attention": edge_attention,
        }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    aux_seq_weight: float,
    aux_struct_weight: float,
) -> Dict[str, float]:
    model.train()
    total = 0.0
    main_total = 0.0
    seq_aux_total = 0.0
    struct_aux_total = 0.0
    rows = 0
    mix_sum = np.zeros(3, dtype=np.float64)
    use_structure = bool(getattr(model, "cfg").use_structure)
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["seq"], batch["paired_flags"], batch["pair_partner"], batch["struct"], batch["rbp"])  # type: ignore[arg-type]
        labels = batch["label"]  # type: ignore[assignment]
        main_loss = criterion(out["logits"], labels)
        seq_aux = criterion(out["seq_logits"], labels) if aux_seq_weight > 0 else torch.zeros((), device=device)
        struct_aux = (
            criterion(out["struct_logits"], labels)
            if use_structure and aux_struct_weight > 0
            else torch.zeros((), device=device)
        )
        loss = main_loss + aux_seq_weight * seq_aux + aux_struct_weight * struct_aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size = int(labels.shape[0])
        total += float(loss.detach().cpu()) * batch_size
        main_total += float(main_loss.detach().cpu()) * batch_size
        seq_aux_total += float(seq_aux.detach().cpu()) * batch_size
        struct_aux_total += float(struct_aux.detach().cpu()) * batch_size
        rows += batch_size
        mix_sum += out["modality_gate"].detach().cpu().numpy().sum(axis=0)

    mix_mean = mix_sum / max(rows, 1)
    return {
        "loss": total / max(rows, 1),
        "main_loss": main_total / max(rows, 1),
        "seq_aux_loss": seq_aux_total / max(rows, 1),
        "struct_aux_loss": struct_aux_total / max(rows, 1),
        "gate_seq": float(mix_mean[0]),
        "gate_structure": float(mix_mean[1]),
        "gate_rbp": float(mix_mean[2]),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    probs, labels = [], []
    mix_sum = np.zeros(3, dtype=np.float64)
    rows = 0
    site_ids: List[str] = []
    mod_types: List[str] = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["seq"], batch["paired_flags"], batch["pair_partner"], batch["struct"], batch["rbp"])  # type: ignore[arg-type]
        batch_probs = torch.softmax(out["logits"], dim=-1)
        probs.append(batch_probs.detach().cpu().numpy())
        labels.append(batch["label"].detach().cpu().numpy())  # type: ignore[union-attr]
        batch_size = int(batch["label"].shape[0])  # type: ignore[index]
        mix_sum += out["modality_gate"].detach().cpu().numpy().sum(axis=0)
        rows += batch_size
        site_ids.extend(batch["site_id"])  # type: ignore[arg-type]
        mod_types.extend(batch["mod_type"])  # type: ignore[arg-type]
    return {
        "probs": np.vstack(probs),
        "labels": np.concatenate(labels),
        "site_ids": site_ids,
        "mod_types": mod_types,
        "gate_mean": (mix_sum / max(rows, 1)).tolist(),
    }


def write_ablation_note(path: Path, args: argparse.Namespace) -> None:
    lines = [
        "# StructInteract fusion model",
        "",
        "V2 model designed to fix gate collapse in the original PairedGAT-RBP fusion.",
        "",
        "Key changes:",
        "",
        "- no softmax modality gate in the main fusion path",
        "- sequence-structure interaction features: `[seq, struct, seq*struct, |seq-struct|]`",
        "- residual refinement for sequence and structure representations",
        "- auxiliary sequence and structure classification heads",
        "- sequence-modality dropout during training to force structure usage",
        "- RBP is optional and disabled by default because current RBP features are sample-level and noisy",
        "",
        f"Arguments: `{json.dumps(vars(args), ensure_ascii=False)}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"

    rows = read_tsv(Path(args.dataset))
    fold_by_id = load_rnafold(Path(args.rnafold))
    rows = [row for row in rows if row.get("site_id") in fold_by_id]
    if args.max_rows:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_rows]
    train_rows, valid_rows, test_rows = split_rows(rows, args.split)

    rbp_by_id, rbp_fields = load_rbp_features(args.rbp_features) if args.use_rbp else ({}, [])
    rbp_dim = len(rbp_fields) if args.use_rbp else 0

    train_ds = RMBaseFusionDataset(
        train_rows,
        fold_by_id,
        args.input_length,
        rbp_by_id=rbp_by_id,
        rbp_dim=rbp_dim,
        shuffle_structure=args.shuffle_structure,
        shuffle_rbp=args.shuffle_rbp,
        seed=args.seed,
    )
    valid_ds = RMBaseFusionDataset(
        valid_rows,
        fold_by_id,
        args.input_length,
        rbp_by_id=rbp_by_id,
        rbp_dim=rbp_dim,
        struct_stats=train_ds.struct_stats,
        rbp_stats=train_ds.rbp_stats,
        seed=args.seed,
    )
    test_ds = RMBaseFusionDataset(
        test_rows,
        fold_by_id,
        args.input_length,
        rbp_by_id=rbp_by_id,
        rbp_dim=rbp_dim,
        struct_stats=train_ds.struct_stats,
        rbp_stats=train_ds.rbp_stats,
        seed=args.seed,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    cfg = StructInteractConfig(
        input_length=args.input_length,
        d_model=args.d_model,
        seq_layers=args.seq_layers,
        seq_heads=args.seq_heads,
        gat_layers=args.gat_layers,
        dropout=args.dropout,
        rbp_dim=rbp_dim,
        use_structure=not args.disable_structure,
        use_rbp=args.use_rbp,
        use_position=not args.disable_position,
        sequence_modality_dropout=args.sequence_modality_dropout,
    )
    model = StructInteractionFusion(cfg).to(device)
    weights = class_weights_from_rows(train_rows, args.class_weight_power, device)
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
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("variant") != "StructInteractFusionV2":
            raise RuntimeError(f"Checkpoint at {checkpoint_path} is not a StructInteractFusionV2 checkpoint.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint.get("history", [])
        best_state = checkpoint.get("best_state_dict")
        best_score = float(checkpoint.get("best_score", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"resumed_from={checkpoint_path} start_epoch={start_epoch} best_score={best_score:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        losses = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            aux_seq_weight=args.aux_seq_weight,
            aux_struct_weight=args.aux_struct_weight,
        )
        scheduler.step()
        valid_pred = predict(model, valid_loader, device)
        valid_metrics = evaluate(valid_pred["probs"], valid_pred["labels"])  # type: ignore[arg-type]
        score = float(valid_metrics[args.selection_metric])
        history.append({"epoch": epoch, "train": losses, "validation": valid_metrics, "learning_rate": optimizer.param_groups[0]["lr"]})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save(
            {
                "variant": "StructInteractFusionV2",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_state_dict": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": history,
                "config": asdict(cfg),
            },
            checkpoint_path,
        )
        print(
            f"epoch={epoch} loss={losses['loss']:.4f} valid_{args.selection_metric}={score:.4f} "
            f"mix=({losses['gate_seq']:.2f},{losses['gate_structure']:.2f},{losses['gate_rbp']:.2f}) "
            f"main={losses['main_loss']:.4f} struct_aux={losses['struct_aux_loss']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_pred = predict(model, valid_loader, device)
    test_pred = predict(model, test_loader, device)
    valid_metrics = evaluate(valid_pred["probs"], valid_pred["labels"])  # type: ignore[arg-type]
    test_metrics = evaluate(test_pred["probs"], test_pred["labels"])  # type: ignore[arg-type]

    torch.save(
        {
            "variant": "StructInteractFusionV2",
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "class_names": CLASS_NAMES,
            "struct_stats": train_ds.struct_stats,
            "rbp_fields": rbp_fields,
            "rbp_stats": train_ds.rbp_stats,
            "training_args": vars(args),
        },
        out_dir / "model.pt",
    )
    metrics: Dict[str, object] = {
        "dataset": args.dataset,
        "rnafold": args.rnafold,
        "rbp_features": args.rbp_features if args.use_rbp else "",
        "variant": "StructInteractFusionV2",
        "task": "RMBase11_hg38_11_class_center_site_prediction",
        "classes": CLASS_NAMES,
        "device": str(device),
        "split": args.split,
        "row_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "config": asdict(cfg),
        "loss": args.loss,
        "aux_loss": {"seq": args.aux_seq_weight, "structure": args.aux_struct_weight},
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "validation": valid_metrics,
        "test": test_metrics,
        "validation_gate_mean": valid_pred["gate_mean"],
        "test_gate_mean": test_pred["gate_mean"],
        "history": history,
        "caveats": [
            "The reported gate/mix is diagnostic only; V2 does not use a softmax gate in the main fusion path.",
            "RBP is optional and disabled by default because current RBP inputs are sample-level features.",
            "The benchmark uses 41 nt windows for fair comparison with existing strict-split baselines.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions(out_dir / "valid_predictions.tsv", valid_pred)
    write_predictions(out_dir / "test_predictions.tsv", test_pred)
    write_per_class_metrics(out_dir / "valid_per_class_metrics.tsv", valid_metrics)
    write_per_class_metrics(out_dir / "test_per_class_metrics.tsv", test_metrics)
    write_confusion_matrix(out_dir / "test_confusion_matrix.tsv", test_metrics["confusion_matrix"])  # type: ignore[arg-type]
    write_gate_summary(out_dir / "test_modality_gate_summary.tsv", test_pred["gate_mean"])  # type: ignore[arg-type]
    write_ablation_note(out_dir / "README.md", args)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train StructInteractFusionV2 on RMBase11.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_mmseqs80_strict_splits.tsv")
    parser.add_argument("--rnafold", default="rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv")
    parser.add_argument("--rbp-features", default="rmbase11_full_work/rbp/rmbase11_rbp_features.tsv")
    parser.add_argument("--split", default="split_mmseqs80_cluster")
    parser.add_argument("--out-dir", default="rmbase11_full_work/models/struct_interact_fusion_v2/full")
    parser.add_argument("--input-length", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seq-layers", type=int, default=2)
    parser.add_argument("--seq-heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--aux-seq-weight", type=float, default=0.15)
    parser.add_argument("--aux-struct-weight", type=float, default=0.35)
    parser.add_argument("--sequence-modality-dropout", type=float, default=0.1)
    parser.add_argument("--selection-metric", choices=["macro_AUPR_ovr", "macro_F1", "MCC", "balanced_accuracy"], default="macro_AUPR_ovr")
    parser.add_argument("--disable-structure", action="store_true")
    parser.add_argument("--use-rbp", action="store_true")
    parser.add_argument("--disable-position", action="store_true")
    parser.add_argument("--shuffle-structure", action="store_true")
    parser.add_argument("--shuffle-rbp", action="store_true")
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
