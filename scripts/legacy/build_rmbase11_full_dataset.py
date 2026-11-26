#!/usr/bin/env python
"""Build the full RMBase hg38 11-class dataset without class-capped sampling.

This script is intentionally separate from build_rmbase11_dataset.py.  The
earlier script creates a balanced, class-capped benchmark.  This one keeps all
valid target sites after exact coordinate de-duplication so every available
RMBase hg38 site can be folded by RNAfold and used by the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


TARGET_CLASSES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
EXPECTED_CENTER = {
    "Am": "A",
    "m1A": "A",
    "m6A": "A",
    "Cm": "C",
    "m5C": "C",
    "Gm": "G",
    "m7G": "G",
    "Um": "T",
    "D": "T",
    "Y": "T",
    "m5U": "T",
}
FIELDS = [
    "site_id",
    "mod_type",
    "class_index",
    "chr",
    "start",
    "end",
    "strand",
    "support_num",
    "support_list",
    "pub_list",
    "cell_list",
    "seq_type_list",
    "gene_id",
    "transcript_id",
    "gene_name",
    "gene_type",
    "region_raw",
    "region_type",
    "sequence_41nt",
    "center_base",
    "source_file",
    "split_random",
]


def normalize_sequence(seq: str) -> str:
    return str(seq or "").upper().replace("U", "T")


def normalize_region(region: str) -> str:
    text = str(region or "").lower().split(",", 1)[0]
    if "utr5" in text or "5utr" in text:
        return "5UTR"
    if "utr3" in text or "3utr" in text:
        return "3UTR"
    if "cds" in text:
        return "CDS"
    if "intron" in text:
        return "intron"
    if "exon" in text:
        return "exon"
    if "intergenic" in text:
        return "intergenic"
    return "other"


def first_item(value: str) -> str:
    return str(value or "").split(",", 1)[0]


def parse_bed_row(parts: Sequence[str], source_file: str) -> Dict[str, str] | None:
    if len(parts) < 19:
        return None
    mod_type = parts[6]
    if mod_type not in CLASS_TO_ID:
        return None
    seq = normalize_sequence(parts[18])
    return {
        "site_id": parts[3],
        "mod_type": mod_type,
        "class_index": str(CLASS_TO_ID[mod_type]),
        "chr": parts[0],
        "start": parts[1],
        "end": parts[2],
        "strand": parts[5],
        "support_num": parts[7],
        "support_list": parts[8],
        "pub_list": parts[10],
        "cell_list": parts[11],
        "seq_type_list": parts[12],
        "gene_id": first_item(parts[13]),
        "transcript_id": first_item(parts[14]),
        "gene_name": first_item(parts[15]),
        "gene_type": first_item(parts[16]),
        "region_raw": first_item(parts[17]),
        "region_type": normalize_region(parts[17]),
        "sequence_41nt": seq,
        "center_base": seq[len(seq) // 2] if seq else "",
        "source_file": source_file,
        "split_random": "",
    }


def row_is_valid(row: Dict[str, str], window_length: int) -> tuple[bool, str]:
    seq = normalize_sequence(row["sequence_41nt"])
    if len(seq) != window_length:
        return False, "invalid_length"
    if set(seq) - set("ACGT"):
        return False, "invalid_base"
    expected = EXPECTED_CENTER[row["mod_type"]]
    if row["center_base"] != expected:
        return False, "center_base_mismatch"
    return True, "valid"


def assign_stratified_splits(rows: List[Dict[str, str]], seed: int) -> None:
    rng = random.Random(seed)
    by_class: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["mod_type"]].append(row)
    for group in by_class.values():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * 0.8))
        n_valid = max(1, round(n * 0.1)) if n >= 10 else max(0, n - n_train - 1)
        if n_train + n_valid >= n and n >= 3:
            n_train = n - 2
            n_valid = 1
        for idx, row in enumerate(group):
            if idx < n_train:
                row["split_random"] = "train"
            elif idx < n_train + n_valid:
                row["split_random"] = "valid"
            else:
                row["split_random"] = "test"


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build(args: argparse.Namespace) -> Dict[str, object]:
    extracted_dir = Path(args.extracted_dir)
    out_dir = Path(args.out_dir)
    dataset_path = out_dir / "datasets" / "rmbase11_hg38_41nt_all_sites.tsv"
    counts_path = out_dir / "tables" / "rmbase11_full_class_counts.tsv"
    summary_path = out_dir / "datasets" / "rmbase11_full_build_summary.json"
    readme_path = out_dir / "README.md"

    rows: List[Dict[str, str]] = []
    raw_counts = Counter()
    valid_counts = Counter()
    invalid_counts = Counter()
    exact_duplicate_counts = Counter()
    center_counts: Dict[str, Counter] = defaultdict(Counter)
    seen = set()

    for bed_path in sorted(extracted_dir.glob("*.bed")):
        with bed_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                row = parse_bed_row(line.rstrip("\n").split("\t"), bed_path.name)
                if row is None:
                    continue
                mod_type = row["mod_type"]
                raw_counts[mod_type] += 1
                ok, reason = row_is_valid(row, args.window_length)
                if not ok:
                    invalid_counts[(mod_type, reason)] += 1
                    continue
                valid_counts[mod_type] += 1
                center_counts[mod_type][row["center_base"]] += 1
                key = (row["chr"], row["start"], row["end"], row["strand"], row["mod_type"])
                if key in seen:
                    exact_duplicate_counts[mod_type] += 1
                    continue
                seen.add(key)
                rows.append(row)

    assign_stratified_splits(rows, args.seed)
    rows.sort(key=lambda row: (CLASS_TO_ID[row["mod_type"]], row["split_random"], row["chr"], int(row["start"])))
    write_tsv(dataset_path, rows, FIELDS)

    split_counts = Counter((row["mod_type"], row["split_random"]) for row in rows)
    retained_counts = Counter(row["mod_type"] for row in rows)
    count_rows = []
    for mod_type in TARGET_CLASSES:
        invalid_by_reason = {
            reason: invalid_counts[(mod_type, reason)]
            for cls, reason in invalid_counts
            if cls == mod_type
        }
        count_rows.append(
            {
                "mod_type": mod_type,
                "raw_target_rows": raw_counts[mod_type],
                "valid_41nt_center_checked_rows": valid_counts[mod_type],
                "invalid_rows": sum(invalid_counts[(mod_type, reason)] for cls, reason in invalid_counts if cls == mod_type),
                "invalid_by_reason": json.dumps(invalid_by_reason, ensure_ascii=False, sort_keys=True),
                "exact_duplicate_rows": exact_duplicate_counts[mod_type],
                "retained_all_site_rows": retained_counts[mod_type],
                "train_rows": split_counts[(mod_type, "train")],
                "valid_rows": split_counts[(mod_type, "valid")],
                "test_rows": split_counts[(mod_type, "test")],
                "center_base_distribution": json.dumps(dict(center_counts[mod_type]), ensure_ascii=False, sort_keys=True),
            }
        )
    write_tsv(
        counts_path,
        count_rows,
        [
            "mod_type",
            "raw_target_rows",
            "valid_41nt_center_checked_rows",
            "invalid_rows",
            "invalid_by_reason",
            "exact_duplicate_rows",
            "retained_all_site_rows",
            "train_rows",
            "valid_rows",
            "test_rows",
            "center_base_distribution",
        ],
    )

    summary = {
        "dataset": "RMBase v3 Homo sapiens hg38 11-class all-site dataset",
        "assembly": "hg38",
        "target_classes": TARGET_CLASSES,
        "window_length": args.window_length,
        "dataset_tsv": str(dataset_path),
        "counts_tsv": str(counts_path),
        "total_retained_rows": len(rows),
        "raw_counts": dict(raw_counts),
        "valid_counts": dict(valid_counts),
        "retained_counts": dict(retained_counts),
        "split_counts": {f"{mod}:{split}": count for (mod, split), count in sorted(split_counts.items())},
        "deduplication": "exact genomic coordinate plus strand plus modification type only; no class-capped sampling and no sequence-similarity removal",
        "note": "This all-site experiment keeps all valid RMBase hg38 target sites. It is intentionally imbalanced and should be trained with class weights or balanced sampling.",
        "seed": args.seed,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(
        "# RMBase hg38 11-class all-site experiment\n\n"
        "This directory contains the all-site version requested after the earlier balanced benchmark. "
        "It keeps every valid RMBase v3 hg38 site for the 11 target modification classes after length, "
        "center-base and exact-coordinate checks. No per-class cap is applied.\n\n"
        "Because m6A is much larger than all other classes, model training should use class weighting and/or "
        "balanced sampling.\n\n"
        f"- Dataset: `{dataset_path.relative_to(out_dir)}`\n"
        f"- Counts: `{counts_path.relative_to(out_dir)}`\n"
        f"- Build summary: `{summary_path.relative_to(out_dir)}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build full RMBase hg38 11-class all-site dataset.")
    parser.add_argument("--extracted-dir", default="rmbase11_work/extracted")
    parser.add_argument("--out-dir", default="rmbase11_full_work")
    parser.add_argument("--window-length", type=int, default=41)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
