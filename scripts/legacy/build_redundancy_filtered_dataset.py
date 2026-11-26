#!/usr/bin/env python
"""Build a sequence-redundancy-filtered StructDualRM dataset.

The original workflow deduplicates genomic/transcript sites, but it does not
perform CD-HIT-style sequence similarity filtering.  This script creates a
drop-in 101 nt dataset after 80% identity filtering among RNAfold-covered
examples.  Because the project runs on Windows where cd-hit-est is often absent,
the default implementation is a deterministic CD-HIT-like greedy filter for
fixed-length windows, followed by transcript-proximity filtering to remove
shifted overlapping windows from the same transcript.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_sequence(seq: str) -> str:
    return str(seq or "").upper().replace("U", "T")


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def label_key(row: Dict[str, str]) -> str:
    m1a = int(float(row.get("label_m1A", 0) or 0))
    m6a = int(float(row.get("label_m6A", 0) or 0))
    if m1a and m6a:
        return "co_positive"
    if m1a:
        return "m1A_only"
    if m6a:
        return "m6A_only"
    return "negative"


def make_blocks(length: int, block_count: int) -> List[Tuple[int, int]]:
    block_count = max(1, min(block_count, length))
    cuts = [round(i * length / block_count) for i in range(block_count + 1)]
    return [(cuts[i], cuts[i + 1]) for i in range(block_count) if cuts[i] < cuts[i + 1]]


def hamming_mismatches(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def select_input_rows(dataset_path: Path, rnafold_path: Path, max_rows: int, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    fold_rows = read_tsv(rnafold_path)
    fold_site_ids = {row["site_id"] for row in fold_rows}
    rows = [row for row in read_tsv(dataset_path) if row.get("site_id") in fold_site_ids]
    random.Random(seed).shuffle(rows)
    if max_rows:
        rows = rows[:max_rows]
    return rows, fold_rows


def find_sequence_representative(
    seq: str,
    kept: Sequence[Dict[str, str]],
    block_index: Dict[int, Dict[str, List[int]]],
    blocks: Sequence[Tuple[int, int]],
    max_mismatches: int,
) -> Tuple[int | None, int]:
    candidate_ids = set()
    for block_id, (start, end) in enumerate(blocks):
        candidate_ids.update(block_index[block_id].get(seq[start:end], []))

    comparisons = 0
    for kept_idx in sorted(candidate_ids):
        comparisons += 1
        rep_seq = normalize_sequence(kept[kept_idx].get("sequence_101nt", kept[kept_idx].get("sequence", "")))
        if hamming_mismatches(seq, rep_seq) <= max_mismatches:
            return kept_idx, comparisons
    return None, comparisons


def find_proximity_representative(
    row: Dict[str, str],
    kept: Sequence[Dict[str, str]],
    positions_by_tx: Dict[str, List[Tuple[int, int]]],
    proximity_nt: int,
) -> int | None:
    if proximity_nt < 0:
        return None
    tx = row.get("transcript_id", "")
    pos = parse_int(row.get("tx_pos", row.get("pos_1based", 0)))
    entries = positions_by_tx.get(tx, [])
    if not entries:
        return None

    positions = [item[0] for item in entries]
    idx = bisect.bisect_left(positions, pos)
    candidates: List[Tuple[int, int]] = []
    if idx > 0:
        candidates.append(entries[idx - 1])
    if idx < len(entries):
        candidates.append(entries[idx])
    for rep_pos, kept_idx in sorted(candidates, key=lambda item: abs(item[0] - pos)):
        if abs(rep_pos - pos) <= proximity_nt:
            return kept_idx
    return None


def add_representative_indexes(
    row: Dict[str, str],
    kept_idx: int,
    block_index: Dict[int, Dict[str, List[int]]],
    positions_by_tx: Dict[str, List[Tuple[int, int]]],
    blocks: Sequence[Tuple[int, int]],
) -> None:
    seq = normalize_sequence(row.get("sequence_101nt", row.get("sequence", "")))
    for block_id, (start, end) in enumerate(blocks):
        block_index[block_id].setdefault(seq[start:end], []).append(kept_idx)

    tx = row.get("transcript_id", "")
    pos = parse_int(row.get("tx_pos", row.get("pos_1based", 0)))
    entries = positions_by_tx.setdefault(tx, [])
    bisect.insort(entries, (pos, kept_idx))


def greedy_filter(
    rows: Sequence[Dict[str, str]],
    identity_threshold: float,
    block_count: int,
    proximity_nt: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, object]], Dict[str, object]]:
    if not rows:
        return [], [], {}

    length = len(normalize_sequence(rows[0].get("sequence_101nt", rows[0].get("sequence", ""))))
    max_mismatches = math.floor((1.0 - identity_threshold) * length + 1e-9)
    blocks = make_blocks(length, block_count)

    kept: List[Dict[str, str]] = []
    membership: List[Dict[str, object]] = []
    block_index: Dict[int, Dict[str, List[int]]] = {idx: {} for idx in range(len(blocks))}
    positions_by_tx: Dict[str, List[Tuple[int, int]]] = {}
    comparisons = 0

    for order, row in enumerate(rows):
        seq = normalize_sequence(row.get("sequence_101nt", row.get("sequence", "")))
        rep_idx, n_comp = find_sequence_representative(seq, kept, block_index, blocks, max_mismatches)
        comparisons += n_comp
        reason = ""
        if rep_idx is not None:
            reason = "sequence_identity"
        else:
            rep_idx = find_proximity_representative(row, kept, positions_by_tx, proximity_nt)
            if rep_idx is not None:
                reason = "transcript_proximity"

        if rep_idx is None:
            rep_idx = len(kept)
            kept.append(row)
            add_representative_indexes(row, rep_idx, block_index, positions_by_tx, blocks)
            membership.append(
                {
                    "input_order": order,
                    "site_id": row.get("site_id", ""),
                    "cluster_id": rep_idx,
                    "representative_site_id": row.get("site_id", ""),
                    "kept": 1,
                    "reason": "representative",
                    "label_group": label_key(row),
                    "split_random": row.get("split_random", ""),
                    "transcript_id": row.get("transcript_id", ""),
                    "tx_pos": row.get("tx_pos", ""),
                }
            )
        else:
            rep = kept[rep_idx]
            membership.append(
                {
                    "input_order": order,
                    "site_id": row.get("site_id", ""),
                    "cluster_id": rep_idx,
                    "representative_site_id": rep.get("site_id", ""),
                    "kept": 0,
                    "reason": reason,
                    "label_group": label_key(row),
                    "split_random": row.get("split_random", ""),
                    "transcript_id": row.get("transcript_id", ""),
                    "tx_pos": row.get("tx_pos", ""),
                }
            )

    summary = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(rows) - len(kept),
        "identity_threshold": identity_threshold,
        "window_length": length,
        "max_mismatches_for_fixed_window": max_mismatches,
        "block_count": block_count,
        "blocks": blocks,
        "transcript_proximity_nt": proximity_nt,
        "candidate_pair_comparisons": comparisons,
        "removed_by_reason": dict(Counter(row["reason"] for row in membership if not row["kept"])),
        "input_by_label": dict(Counter(label_key(row) for row in rows)),
        "kept_by_label": dict(Counter(label_key(row) for row in kept)),
        "input_by_split_random": dict(Counter(row.get("split_random", "") for row in rows)),
        "kept_by_split_random": dict(Counter(row.get("split_random", "") for row in kept)),
    }
    return kept, membership, summary


def make_benchmark_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        out.append(
            {
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
                "split_chromosome": row.get("split_chromosome", ""),
            }
        )
    return out


def write_outputs(
    kept: Sequence[Dict[str, str]],
    membership: Sequence[Dict[str, object]],
    fold_rows: Sequence[Dict[str, str]],
    summary: Dict[str, object],
    out_dir: Path,
    dataset_name: str,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_out = out_dir / dataset_name
    rnafold_out = out_dir / "rnafold_101nt_redundancy_filtered.tsv"
    membership_out = out_dir / "redundancy_cluster_membership.tsv"
    summary_out = out_dir / "redundancy_filter_summary.json"
    benchmark_out = out_dir / "m1A_m6A_split_random_seed42_redundancy_filtered.tsv"
    readme_out = out_dir / "README.md"

    if kept:
        write_tsv(dataset_out, kept, list(kept[0].keys()))
    else:
        dataset_out.write_text("", encoding="utf-8")

    kept_ids = {row["site_id"] for row in kept}
    filtered_fold = [row for row in fold_rows if row.get("site_id") in kept_ids]
    if filtered_fold:
        write_tsv(rnafold_out, filtered_fold, list(filtered_fold[0].keys()))
    else:
        rnafold_out.write_text("", encoding="utf-8")

    membership_fields = [
        "input_order",
        "site_id",
        "cluster_id",
        "representative_site_id",
        "kept",
        "reason",
        "label_group",
        "split_random",
        "transcript_id",
        "tx_pos",
    ]
    write_tsv(membership_out, membership, membership_fields)

    benchmark_fields = [
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
    write_tsv(benchmark_out, make_benchmark_rows(kept), benchmark_fields)

    summary["cd_hit_est_on_path"] = bool(shutil.which("cd-hit-est"))
    summary["cd_hit_on_path"] = bool(shutil.which("cd-hit"))
    summary["method"] = "python_fixed_window_80pct_identity_plus_transcript_proximity"
    summary["note"] = (
        "The previous StructDualRM workflow did not call CD-HIT. This file was generated by a deterministic "
        "CD-HIT-like greedy filter for equal-length 101 nt windows, with an additional transcript-proximity "
        "step to remove shifted overlapping windows."
    )
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme_out.write_text(
        "# Redundancy-filtered StructDualRM dataset\n\n"
        "This directory contains a drop-in dataset generated after 80% identity sequence redundancy filtering. "
        "No `cd-hit-est` executable was available on PATH when this was generated, so the filtering used a "
        "deterministic CD-HIT-like greedy procedure for fixed-length 101 nt windows plus transcript-proximity "
        "filtering for highly overlapping windows from the same transcript.\n\n"
        "Files:\n"
        f"- `{dataset_out.name}`: filtered 101 nt dataset for StructDualRM training\n"
        f"- `{rnafold_out.name}`: matching RNAfold rows\n"
        f"- `{benchmark_out.name}`: filtered split file for external baselines\n"
        f"- `{membership_out.name}`: representative/removed cluster membership table\n"
        f"- `{summary_out.name}`: filtering parameters and row-count summary\n",
        encoding="utf-8",
    )

    return {
        "dataset": str(dataset_out),
        "rnafold": str(rnafold_out),
        "membership": str(membership_out),
        "summary": str(summary_out),
        "benchmark_split": str(benchmark_out),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create redundancy-filtered StructDualRM input files.")
    parser.add_argument("--dataset", default="05_datasets/dataset_joint_m1A_m6A_101nt.tsv")
    parser.add_argument("--rnafold", default="06_structure/rnafold_101nt.tsv")
    parser.add_argument("--out-dir", default="05_datasets/cdhit80_like")
    parser.add_argument("--dataset-name", default="dataset_joint_m1A_m6A_101nt_cdhit80_like.tsv")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identity-threshold", type=float, default=0.8)
    parser.add_argument("--block-count", type=int, default=5)
    parser.add_argument("--transcript-proximity-nt", type=int, default=20)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, fold_rows = select_input_rows(Path(args.dataset), Path(args.rnafold), args.max_rows, args.seed)
    kept, membership, summary = greedy_filter(
        rows,
        identity_threshold=args.identity_threshold,
        block_count=args.block_count,
        proximity_nt=args.transcript_proximity_nt,
    )
    outputs = write_outputs(kept, membership, fold_rows, summary, Path(args.out_dir), args.dataset_name)
    print(json.dumps({"outputs": outputs, "summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
