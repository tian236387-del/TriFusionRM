#!/usr/bin/env python
"""Export frozen benchmark split files for external model comparisons."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


EXPORT_FIELDS = [
    "site_id",
    "chr",
    "transcript_id",
    "gene_id",
    "region_type",
    "relative_pos",
    "tx_len",
    "sequence",
    "label_m1A",
    "label_m6A",
    "split_random_seed42",
    "split_chromosome",
]


def assign_chromosome_split(chrom: str) -> str:
    chrom = str(chrom or "").strip()
    if chrom in {"chr17", "chr18", "chr19"}:
        return "valid"
    if chrom in {"chr20", "chr21", "chr22", "chrX"}:
        return "test"
    return "train"


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_fold_site_ids(path: Path | None) -> Set[str]:
    if path is None:
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["site_id"] for row in csv.DictReader(handle, delimiter="\t")}


def select_rows(rows: Sequence[Dict[str, str]], fold_site_ids: Set[str], max_rows: int, seed: int) -> List[Dict[str, str]]:
    selected = [row for row in rows if not fold_site_ids or row.get("site_id") in fold_site_ids]
    random.Random(seed).shuffle(selected)
    if max_rows:
        selected = selected[:max_rows]
    return sorted(selected, key=lambda row: row.get("site_id", ""))


def export_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "site_id": row.get("site_id", ""),
        "chr": row.get("chr", ""),
        "transcript_id": row.get("transcript_id", ""),
        "gene_id": row.get("gene_id", ""),
        "region_type": row.get("region_type", ""),
        "relative_pos": row.get("relative_pos", ""),
        "tx_len": row.get("tx_len", ""),
        "sequence": row.get("sequence_101nt", row.get("sequence", "")),
        "label_m1A": row.get("label_m1A", "0"),
        "label_m6A": row.get("label_m6A", "0"),
        "split_random_seed42": row.get("split_random", ""),
        "split_chromosome": assign_chromosome_split(row.get("chr", "")),
    }


def write_benchmark_split(dataset_path: Path, rnafold_path: Path | None, out_dir: Path, max_rows: int, seed: int) -> Path:
    rows = read_tsv(dataset_path)
    fold_site_ids = read_fold_site_ids(rnafold_path)
    selected = select_rows(rows, fold_site_ids, max_rows=max_rows, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "m1A_m6A_split_random_seed42.tsv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(export_row(row) for row in selected)
    return out_path


def summarize(path: Path) -> Dict[str, Dict[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    summary = {"split_random_seed42": {}, "split_chromosome": {}}
    for row in rows:
        for column in summary:
            value = row[column]
            summary[column][value] = summary[column].get(value, 0) + 1
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create frozen benchmark split files.")
    parser.add_argument("--dataset", default="05_datasets/dataset_joint_m1A_m6A_101nt.tsv")
    parser.add_argument("--rnafold", default="06_structure/rnafold_101nt.tsv")
    parser.add_argument("--out-dir", default="05_datasets/benchmark_splits")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_path = write_benchmark_split(Path(args.dataset), Path(args.rnafold), Path(args.out_dir), args.max_rows, args.seed)
    print(f"wrote {out_path}")
    print(summarize(out_path))


if __name__ == "__main__":
    main()
