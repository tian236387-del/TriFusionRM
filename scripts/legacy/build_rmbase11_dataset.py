#!/usr/bin/env python
"""Build an RMBase hg38 11-class RNA modification benchmark dataset.

The original StructDualRM workflow was built for a local hg19 m1A/m6A
dual-label task.  This script creates a separate RMBase v3 hg38 robustness
benchmark for 11 RNA modification classes.  It parses the RMBase 29-column
BED-like files, creates a balanced capped sample, applies a deterministic
CD-HIT-like 80% fixed-window identity filter, and writes stratified train,
validation and test splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


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
RAW_FIELDS = [
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
]
DATASET_FIELDS = RAW_FIELDS + ["split_random"]


def normalize_sequence(seq: str) -> str:
    return str(seq or "").upper().replace("U", "T")


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def normalize_region(region: str) -> str:
    text = str(region or "").lower()
    first = text.split(",", 1)[0]
    if "utr5" in first or "5utr" in first or first.startswith("utr-5"):
        return "5UTR"
    if "utr3" in first or "3utr" in first or first.startswith("utr-3"):
        return "3UTR"
    if "cds" in first:
        return "CDS"
    if "intron" in first:
        return "intron"
    if "exon" in first:
        return "exon"
    if "intergenic" in first:
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
    center = seq[len(seq) // 2] if seq else ""
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
        "center_base": center,
        "source_file": source_file,
    }


def valid_center_and_sequence(row: Dict[str, str], length: int) -> Tuple[bool, str]:
    seq = normalize_sequence(row.get("sequence_41nt", ""))
    if len(seq) != length:
        return False, "invalid_length"
    if set(seq) - set("ACGT"):
        return False, "invalid_base"
    expected = EXPECTED_CENTER.get(row["mod_type"])
    if expected and row.get("center_base") != expected:
        return False, "center_base_mismatch"
    return True, "valid"


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def make_blocks(length: int, max_mismatches: int) -> List[Tuple[int, int]]:
    block_count = max(1, min(length, max_mismatches + 1))
    cuts = [round(i * length / block_count) for i in range(block_count + 1)]
    return [(cuts[i], cuts[i + 1]) for i in range(block_count) if cuts[i] < cuts[i + 1]]


def hamming_mismatches(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def find_representative(
    seq: str,
    kept: Sequence[Dict[str, str]],
    block_index: Dict[int, Dict[str, List[int]]],
    blocks: Sequence[Tuple[int, int]],
    max_mismatches: int,
) -> Tuple[int | None, int | None]:
    candidate_ids = set()
    for block_id, (start, end) in enumerate(blocks):
        candidate_ids.update(block_index[block_id].get(seq[start:end], []))
    for kept_idx in sorted(candidate_ids):
        rep_seq = kept[kept_idx]["sequence_41nt"]
        mismatches = hamming_mismatches(seq, rep_seq)
        if mismatches <= max_mismatches:
            return kept_idx, mismatches
    return None, None


def add_to_block_index(
    row: Dict[str, str],
    kept_idx: int,
    block_index: Dict[int, Dict[str, List[int]]],
    blocks: Sequence[Tuple[int, int]],
) -> None:
    seq = row["sequence_41nt"]
    for block_id, (start, end) in enumerate(blocks):
        block_index[block_id].setdefault(seq[start:end], []).append(kept_idx)


def filter_one_class(
    rows: Sequence[Dict[str, str]],
    identity_threshold: float,
    length: int,
    mod_type: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, object]], Dict[str, object]]:
    max_mismatches = math.floor((1.0 - identity_threshold) * length + 1e-9)
    blocks = make_blocks(length, max_mismatches)
    kept: List[Dict[str, str]] = []
    membership: List[Dict[str, object]] = []
    block_index: Dict[int, Dict[str, List[int]]] = {idx: {} for idx in range(len(blocks))}
    removed = Counter()

    for input_order, row in enumerate(rows):
        rep_idx, mismatches = find_representative(row["sequence_41nt"], kept, block_index, blocks, max_mismatches)
        if rep_idx is None:
            rep_idx = len(kept)
            kept.append(row)
            add_to_block_index(row, rep_idx, block_index, blocks)
            membership.append(
                {
                    "mod_type": mod_type,
                    "input_order": input_order,
                    "site_id": row["site_id"],
                    "cluster_id": f"{mod_type}_{rep_idx}",
                    "representative_site_id": row["site_id"],
                    "kept": 1,
                    "reason": "representative",
                    "mismatches_to_representative": "",
                }
            )
        else:
            rep = kept[rep_idx]
            removed["sequence_identity"] += 1
            membership.append(
                {
                    "mod_type": mod_type,
                    "input_order": input_order,
                    "site_id": row["site_id"],
                    "cluster_id": f"{mod_type}_{rep_idx}",
                    "representative_site_id": rep["site_id"],
                    "kept": 0,
                    "reason": "sequence_identity",
                    "mismatches_to_representative": mismatches,
                }
            )

    return kept, membership, {
        "mod_type": mod_type,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(rows) - len(kept),
        "removed_by_reason": dict(removed),
        "max_mismatches_for_fixed_window": max_mismatches,
        "blocks": blocks,
    }


def assign_stratified_splits(rows: List[Dict[str, str]], seed: int, train_frac: float, valid_frac: float) -> None:
    rng = random.Random(seed)
    by_class: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["mod_type"]].append(row)

    for mod_type, group in by_class.items():
        rng.shuffle(group)
        n = len(group)
        if n >= 3:
            n_train = max(1, int(round(n * train_frac)))
            n_valid = max(1, int(round(n * valid_frac)))
            if n_train + n_valid >= n:
                n_train = max(1, n - 2)
                n_valid = 1
        elif n == 2:
            n_train, n_valid = 1, 0
        else:
            n_train, n_valid = n, 0

        for idx, row in enumerate(group):
            if idx < n_train:
                row["split_random"] = "train"
            elif idx < n_train + n_valid:
                row["split_random"] = "valid"
            else:
                row["split_random"] = "test"


def build_dataset(args: argparse.Namespace) -> Dict[str, object]:
    extracted_dir = Path(args.extracted_dir)
    out_dir = Path(args.out_dir)
    datasets_dir = out_dir / "datasets"
    tables_dir = out_dir / "tables"
    raw_out = datasets_dir / "rmbase11_hg38_raw_parsed.tsv"
    sample_out = datasets_dir / "rmbase11_hg38_balanced_sample_before_redundancy.tsv"
    filtered_out = datasets_dir / "rmbase11_hg38_41nt_cdhit80_like.tsv"
    membership_out = datasets_dir / "rmbase11_hg38_redundancy_cluster_membership.tsv"
    summary_out = datasets_dir / "rmbase11_hg38_build_summary.json"
    counts_out = tables_dir / "rmbase11_class_counts.tsv"
    readme_out = out_dir / "README.md"

    rng = random.Random(args.seed)
    raw_counts = Counter()
    valid_counts = Counter()
    invalid_counts = Counter()
    dedup_counts = Counter()
    duplicate_counts = Counter()
    center_counts: Dict[str, Counter] = defaultdict(Counter)
    seen_site_keys = set()
    reservoirs: Dict[str, List[Dict[str, str]]] = {mod: [] for mod in TARGET_CLASSES}
    reservoir_seen = Counter()

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    with raw_out.open("w", encoding="utf-8", newline="") as raw_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=RAW_FIELDS, delimiter="\t", lineterminator="\n")
        raw_writer.writeheader()
        for bed_path in sorted(extracted_dir.glob("*.bed")):
            with bed_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    row = parse_bed_row(parts, bed_path.name)
                    if row is None:
                        continue
                    mod_type = row["mod_type"]
                    raw_counts[mod_type] += 1
                    ok, reason = valid_center_and_sequence(row, args.window_length)
                    if not ok:
                        invalid_counts[(mod_type, reason)] += 1
                        continue
                    valid_counts[mod_type] += 1
                    center_counts[mod_type][row["center_base"]] += 1
                    raw_writer.writerow(row)

                    site_key = (row["chr"], row["start"], row["end"], row["strand"], row["mod_type"])
                    if site_key in seen_site_keys:
                        duplicate_counts[mod_type] += 1
                        continue
                    seen_site_keys.add(site_key)
                    dedup_counts[mod_type] += 1

                    reservoir_seen[mod_type] += 1
                    if args.max_per_class <= 0 or len(reservoirs[mod_type]) < args.max_per_class:
                        reservoirs[mod_type].append(dict(row))
                    else:
                        j = rng.randint(1, reservoir_seen[mod_type])
                        if j <= args.max_per_class:
                            reservoirs[mod_type][j - 1] = dict(row)

    sampled_rows: List[Dict[str, str]] = []
    for mod_type in TARGET_CLASSES:
        group = reservoirs[mod_type]
        for row in group:
            row["_sort_key"] = f"{rng.random():.12f}"
        group.sort(key=lambda row: (-parse_int(row.get("support_num")), row["_sort_key"], row["site_id"]))
        for row in group:
            row.pop("_sort_key", None)
        sampled_rows.extend(group)
    write_tsv(sample_out, sampled_rows, RAW_FIELDS)

    kept_rows: List[Dict[str, str]] = []
    membership_rows: List[Dict[str, object]] = []
    class_filter_summaries = []
    for mod_type in TARGET_CLASSES:
        group = [row for row in sampled_rows if row["mod_type"] == mod_type]
        kept, membership, filter_summary = filter_one_class(
            group,
            identity_threshold=args.identity_threshold,
            length=args.window_length,
            mod_type=mod_type,
        )
        kept_rows.extend(kept)
        membership_rows.extend(membership)
        class_filter_summaries.append(filter_summary)

    assign_stratified_splits(kept_rows, args.seed, args.train_frac, args.valid_frac)
    kept_rows.sort(key=lambda row: (CLASS_TO_ID[row["mod_type"]], row["split_random"], row["site_id"]))
    write_tsv(filtered_out, kept_rows, DATASET_FIELDS)
    write_tsv(
        membership_out,
        membership_rows,
        [
            "mod_type",
            "input_order",
            "site_id",
            "cluster_id",
            "representative_site_id",
            "kept",
            "reason",
            "mismatches_to_representative",
        ],
    )

    sampled_counts = Counter(row["mod_type"] for row in sampled_rows)
    filtered_counts = Counter(row["mod_type"] for row in kept_rows)
    split_counts = Counter((row["mod_type"], row["split_random"]) for row in kept_rows)
    count_rows = []
    for mod_type in TARGET_CLASSES:
        invalid_by_reason = {
            reason: invalid_counts[(mod_type, reason)]
            for _, reason in invalid_counts
            if _ == mod_type
        }
        count_rows.append(
            {
                "mod_type": mod_type,
                "raw_target_rows": raw_counts[mod_type],
                "valid_41nt_center_checked_rows": valid_counts[mod_type],
                "invalid_rows": sum(invalid_counts[(mod_type, reason)] for _, reason in invalid_counts if _ == mod_type),
                "invalid_by_reason": json.dumps(invalid_by_reason, ensure_ascii=False, sort_keys=True),
                "exact_deduplicated_rows": dedup_counts[mod_type],
                "exact_duplicate_rows": duplicate_counts[mod_type],
                "balanced_sample_rows": sampled_counts[mod_type],
                "redundancy_filtered_rows": filtered_counts[mod_type],
                "train_rows": split_counts[(mod_type, "train")],
                "valid_rows": split_counts[(mod_type, "valid")],
                "test_rows": split_counts[(mod_type, "test")],
                "center_base_distribution": json.dumps(dict(center_counts[mod_type]), ensure_ascii=False, sort_keys=True),
            }
        )
    write_tsv(
        counts_out,
        count_rows,
        [
            "mod_type",
            "raw_target_rows",
            "valid_41nt_center_checked_rows",
            "invalid_rows",
            "invalid_by_reason",
            "exact_deduplicated_rows",
            "exact_duplicate_rows",
            "balanced_sample_rows",
            "redundancy_filtered_rows",
            "train_rows",
            "valid_rows",
            "test_rows",
            "center_base_distribution",
        ],
    )

    summary = {
        "dataset": "RMBase v3 Homo sapiens hg38 11-class RNA modification benchmark",
        "target_classes": TARGET_CLASSES,
        "note_on_assembly": "RMBase v3 public single-site downloads used here are hg38. This benchmark is separate from the original local hg19 m1A/m6A workflow.",
        "raw_parsed_tsv": str(raw_out),
        "balanced_sample_tsv": str(sample_out),
        "filtered_dataset_tsv": str(filtered_out),
        "membership_tsv": str(membership_out),
        "class_counts_tsv": str(counts_out),
        "window_length": args.window_length,
        "max_per_class_before_redundancy_filter": args.max_per_class,
        "identity_threshold": args.identity_threshold,
        "filter_method": "python_fixed_window_greedy_cdhit_like_within_each_modification_class",
        "caveat": "No cd-hit-est executable was available locally; the redundancy filtering is CD-HIT-like and deterministic, not a direct CD-HIT run.",
        "raw_counts": dict(raw_counts),
        "valid_counts": dict(valid_counts),
        "deduplicated_counts": dict(dedup_counts),
        "sampled_counts": dict(sampled_counts),
        "filtered_counts": dict(filtered_counts),
        "split_counts": {f"{mod}:{split}": count for (mod, split), count in sorted(split_counts.items())},
        "class_filter_summaries": class_filter_summaries,
        "seed": args.seed,
    }
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_out.write_text(
        "# RMBase hg38 11-class StructDualRM robustness benchmark\n\n"
        "This directory contains a separate RMBase v3 hg38 benchmark for 11 RNA modification classes: "
        + ", ".join(TARGET_CLASSES)
        + ". It is not merged with the original local hg19 m1A/m6A workflow.\n\n"
        "Files:\n"
        f"- `{raw_out.relative_to(out_dir)}`: all parsed valid target rows from RMBase hg38 single-site files.\n"
        f"- `{sample_out.relative_to(out_dir)}`: balanced class-capped sample before redundancy filtering.\n"
        f"- `{filtered_out.relative_to(out_dir)}`: final 41 nt dataset with stratified train/valid/test split.\n"
        f"- `{membership_out.relative_to(out_dir)}`: CD-HIT-like cluster membership and removed sequences.\n"
        f"- `{counts_out.relative_to(out_dir)}`: class-level row counts and split counts.\n"
        f"- `{summary_out.relative_to(out_dir)}`: parameters and build summary.\n\n"
        "Important caveat: this run used a deterministic Python fixed-window 80% identity greedy filter because "
        "`cd-hit-est` was not available on this Windows environment. In the manuscript this should be described "
        "as CD-HIT-like redundancy filtering unless CD-HIT itself is installed and rerun.\n",
        encoding="utf-8",
    )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RMBase hg38 11-class dataset.")
    parser.add_argument("--extracted-dir", default="rmbase11_work/extracted")
    parser.add_argument("--out-dir", default="rmbase11_work")
    parser.add_argument("--window-length", type=int, default=41)
    parser.add_argument("--max-per-class", type=int, default=5000)
    parser.add_argument("--identity-threshold", type=float, default=0.8)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
