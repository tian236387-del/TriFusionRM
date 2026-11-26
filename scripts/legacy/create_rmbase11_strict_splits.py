#!/usr/bin/env python
"""Create reviewer-safe RMBase11 split columns.

The all-site RMBase11 benchmark originally uses a stratified random site split.
This script keeps the same rows but adds two stricter split columns:

* split_gene: no known gene_id is shared across train/valid/test.
* split_cdhit80_cluster: no 41 nt sequence cluster at >=80% identity is shared
  across train/valid/test. This is a deterministic CD-HIT-like greedy cluster
  assignment for fixed-length windows; it does not call cd-hit-est.
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


CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SPLITS = ("train", "valid", "test")


def read_tsv(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def write_tsv(path: Path, rows: Sequence[Dict[str, str]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def known_gene_key(row: Dict[str, str]) -> str:
    gene_id = (row.get("gene_id") or "").strip()
    if gene_id and gene_id.lower() not in {"na", "none", "intergenic", "."}:
        return f"gene:{gene_id}"
    return f"site:{row.get('site_id', '')}"


def class_counts(rows: Sequence[Dict[str, str]]) -> List[int]:
    counts = [0] * len(CLASS_NAMES)
    for row in rows:
        mod = row.get("mod_type", "")
        if mod in CLASS_TO_ID:
            counts[CLASS_TO_ID[mod]] += 1
    return counts


def split_grouped_rows(
    rows: Sequence[Dict[str, str]],
    group_key_by_site: Dict[str, str],
    seed: int,
    train_frac: float,
    valid_frac: float,
) -> Dict[str, str]:
    rng = random.Random(seed)
    target_fracs = {"train": train_frac, "valid": valid_frac, "test": max(0.0, 1.0 - train_frac - valid_frac)}
    total = class_counts(rows)
    targets = {
        split: [target_fracs[split] * total[idx] for idx in range(len(CLASS_NAMES))]
        for split in SPLITS
    }

    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[group_key_by_site[row["site_id"]]].append(row)

    group_items = []
    for key, group_rows in groups.items():
        counts = class_counts(group_rows)
        size = sum(counts)
        # Random jitter only breaks ties deterministically.
        group_items.append((key, group_rows, counts, size, rng.random()))
    group_items.sort(key=lambda item: (-item[3], item[4], item[0]))

    assigned_counts = {split: [0] * len(CLASS_NAMES) for split in SPLITS}
    group_to_split: Dict[str, str] = {}

    for key, _, counts, size, _ in group_items:
        best_split = "train"
        best_score = None
        for split in SPLITS:
            score = 0.0
            for idx, count in enumerate(counts):
                if count == 0:
                    continue
                target = max(targets[split][idx], 1.0)
                before = assigned_counts[split][idx]
                after = before + count
                # Penalize overshooting class targets and reward filling deficits.
                deficit_before = max(target - before, 0.0) / target
                overshoot_after = max(after - target, 0.0) / target
                score += count * (deficit_before - 2.0 * overshoot_after)
            # Keep split sizes near target even for mixed/rare groups.
            split_total = sum(assigned_counts[split])
            split_target_total = max(sum(targets[split]), 1.0)
            score -= 0.05 * abs((split_total + size) - split_target_total) / split_target_total
            if best_score is None or score > best_score:
                best_score = score
                best_split = split
        group_to_split[key] = best_split
        for idx, count in enumerate(counts):
            assigned_counts[best_split][idx] += count

    site_to_split = {}
    for row in rows:
        group = group_key_by_site[row["site_id"]]
        site_to_split[row["site_id"]] = group_to_split[group]
    return site_to_split


def make_blocks(length: int, max_mismatches: int) -> List[Tuple[int, int]]:
    block_count = max(1, min(length, max_mismatches + 1))
    cuts = [round(i * length / block_count) for i in range(block_count + 1)]
    return [(cuts[i], cuts[i + 1]) for i in range(block_count) if cuts[i] < cuts[i + 1]]


def hamming_mismatches(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def cluster_sequences(
    rows: Sequence[Dict[str, str]],
    identity_threshold: float,
    seed: int,
) -> tuple[Dict[str, str], Dict[str, object]]:
    if not rows:
        return {}, {}
    length = len(rows[0].get("sequence_41nt", ""))
    max_mismatches = math.floor((1.0 - identity_threshold) * length + 1e-9)
    blocks = make_blocks(length, max_mismatches)

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    representatives: List[Dict[str, str]] = []
    block_index: Dict[int, Dict[str, List[int]]] = {idx: {} for idx in range(len(blocks))}
    site_to_cluster: Dict[str, str] = {}
    cluster_sizes: Counter[str] = Counter()

    for idx, row in enumerate(shuffled, start=1):
        seq = row["sequence_41nt"]
        candidate_ids = set()
        for block_id, (start, end) in enumerate(blocks):
            candidate_ids.update(block_index[block_id].get(seq[start:end], []))
        rep_idx = None
        for candidate in sorted(candidate_ids):
            if hamming_mismatches(seq, representatives[candidate]["sequence_41nt"]) <= max_mismatches:
                rep_idx = candidate
                break
        if rep_idx is None:
            rep_idx = len(representatives)
            representatives.append(row)
            for block_id, (start, end) in enumerate(blocks):
                block_index[block_id].setdefault(seq[start:end], []).append(rep_idx)
        cluster_id = f"cdhit80_like_{rep_idx}"
        site_to_cluster[row["site_id"]] = cluster_id
        cluster_sizes[cluster_id] += 1
        if idx % 100000 == 0:
            print(f"clustered_rows={idx} representatives={len(representatives)}", flush=True)

    summary = {
        "identity_threshold": identity_threshold,
        "window_length": length,
        "max_mismatches": max_mismatches,
        "cluster_count": len(representatives),
        "largest_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
        "mean_cluster_size": sum(cluster_sizes.values()) / max(len(cluster_sizes), 1),
        "method": "deterministic_python_fixed_window_cdhit_like_greedy",
    }
    return site_to_cluster, summary


def summarize_splits(rows: Sequence[Dict[str, str]], split_columns: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for split_col in split_columns:
        counts = Counter((row.get("mod_type", ""), row.get(split_col, "")) for row in rows)
        out[split_col] = {f"{mod}:{split}": counts[(mod, split)] for mod in CLASS_NAMES for split in SPLITS}
    return out


def run(args: argparse.Namespace) -> Dict[str, object]:
    rows, fields = read_tsv(Path(args.dataset))
    site_to_gene_group = {row["site_id"]: known_gene_key(row) for row in rows}
    gene_site_to_split = split_grouped_rows(rows, site_to_gene_group, args.seed, args.train_frac, args.valid_frac)
    for row in rows:
        row["split_gene"] = gene_site_to_split[row["site_id"]]

    cluster_summary: Dict[str, object] = {}
    if args.cluster_split:
        site_to_cluster, cluster_summary = cluster_sequences(rows, args.identity_threshold, args.seed)
        cluster_site_to_split = split_grouped_rows(rows, site_to_cluster, args.seed, args.train_frac, args.valid_frac)
        for row in rows:
            row["cdhit80_like_cluster_id"] = site_to_cluster[row["site_id"]]
            row["split_cdhit80_cluster"] = cluster_site_to_split[row["site_id"]]

    out_fields = list(fields)
    for field in ["split_gene", "cdhit80_like_cluster_id", "split_cdhit80_cluster"]:
        if field in rows[0] and field not in out_fields:
            out_fields.append(field)
    write_tsv(Path(args.out), rows, out_fields)

    summary = {
        "input_dataset": args.dataset,
        "output_dataset": args.out,
        "row_count": len(rows),
        "class_counts": dict(Counter(row["mod_type"] for row in rows)),
        "gene_group_count": len(set(site_to_gene_group.values())),
        "cluster_split_created": bool(args.cluster_split),
        "cluster_summary": cluster_summary,
        "split_counts": summarize_splits(rows, [field for field in ["split_random", "split_gene", "split_cdhit80_cluster"] if field in rows[0]]),
        "seed": args.seed,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add strict RMBase11 split columns.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites.tsv")
    parser.add_argument("--out", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_strict_splits.tsv")
    parser.add_argument("--summary", default="rmbase11_full_work/datasets/rmbase11_strict_splits_summary.json")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--identity-threshold", type=float, default=0.8)
    parser.add_argument("--cluster-split", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
