#!/usr/bin/env python
"""Export per-class AUPR tables from official reproduction benchmark metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
DEFAULT_MODELS = ["multirm_official", "transrnam_official", "evormd_adapted"]


def load_metrics(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def per_class_rows(metrics: Dict[str, object], setting: str, model: str) -> List[Dict[str, object]]:
    test = metrics["test"]
    per_class = test["per_class"]
    rows: List[Dict[str, object]] = []
    for name in CLASS_NAMES:
        item = per_class[name]
        rows.append(
            {
                "setting": setting,
                "model": model,
                "modification": name,
                "support": item.get("support", ""),
                "AUPR_ovr": item.get("AUPR_ovr", ""),
                "AUC_ovr": item.get("AUC_ovr", ""),
                "F1": item.get("F1", ""),
                "Precision": item.get("Precision", ""),
                "Recall": item.get("Recall", ""),
                "MCC_ovr": item.get("MCC_ovr", ""),
            }
        )
    rows.append(
        {
            "setting": setting,
            "model": model,
            "modification": "macro_average",
            "support": sum(int(per_class[name].get("support", 0)) for name in CLASS_NAMES),
            "AUPR_ovr": test.get("macro_AUPR_ovr", ""),
            "AUC_ovr": test.get("macro_AUC_ovr", ""),
            "F1": test.get("macro_F1", ""),
            "Precision": "",
            "Recall": "",
            "MCC_ovr": test.get("MCC", ""),
        }
    )
    return rows


def wide_row(metrics: Dict[str, object], setting: str, model: str) -> Dict[str, object]:
    test = metrics["test"]
    per_class = test["per_class"]
    row: Dict[str, object] = {"setting": setting, "model": model}
    for name in CLASS_NAMES:
        row[name] = per_class[name].get("AUPR_ovr", "")
    row["macro_AUPR"] = test.get("macro_AUPR_ovr", "")
    return row


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export per-class AUPR tables.")
    parser.add_argument("--root", default="rmbase11_full_work/official_reproduction_benchmark")
    parser.add_argument("--setting", default="strict_mmseqs80_split")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--out-dir", default="")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / args.setting / "per_class_tables"
    all_long_rows: List[Dict[str, object]] = []
    wide_rows: List[Dict[str, object]] = []

    for model in args.models:
        metrics_path = root / args.setting / model / "metrics.json"
        metrics = load_metrics(metrics_path)
        rows = per_class_rows(metrics, args.setting, model)
        write_tsv(out_dir / f"{model}_per_class_AUPR.tsv", rows)
        all_long_rows.extend(rows)
        wide_rows.append(wide_row(metrics, args.setting, model))

    write_tsv(out_dir / "all_models_per_class_AUPR_long.tsv", all_long_rows)
    write_tsv(out_dir / "all_models_per_class_AUPR_wide.tsv", wide_rows)
    print(f"Wrote per-class tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
