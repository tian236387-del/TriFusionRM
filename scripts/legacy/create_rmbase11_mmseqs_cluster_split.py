#!/usr/bin/env python
"""Create a full RMBase11 MMseqs2 80% redundancy-aware cluster split.

The earlier RMBase11 all-site file contains a random split and, after
`create_rmbase11_strict_splits.py`, a gene-level split. This script adds a
full external-tool sequence-cluster split using MMseqs2 on the 41 nt windows.

MMseqs2 is used because the official CD-HIT release does not provide a native
Windows binary, while MMseqs2 provides an official win64 executable and supports
CD-HIT-style greedy clustering via `--cluster-mode 2`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


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


def write_fasta(rows: Sequence[Dict[str, str]], path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for row in rows:
            seq = (row.get("sequence_41nt") or "").upper().replace("U", "T")
            site_id = row["site_id"].replace(" ", "_")
            handle.write(f">{site_id}\n{seq}\n")


def run_command(command: Sequence[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(list(command), check=True)


def run_mmseqs(args: argparse.Namespace) -> List[List[str]] | List[str]:
    cluster_tsv = Path(f"{args.cluster_prefix}_cluster.tsv")
    if cluster_tsv.exists() and not args.force_mmseqs:
        return ["skipped_existing_mmseqs_cluster_tsv"]
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    prefix = Path(args.cluster_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    sequence_db = f"{args.cluster_prefix}_seqdb"
    cluster_db = f"{args.cluster_prefix}_clusterdb"
    commands: List[List[str]] = []
    commands.append(
        [
            args.mmseqs,
            "createdb",
            args.fasta,
            sequence_db,
            "--dbtype",
            "2",
            "--shuffle",
            "0",
            "--createdb-mode",
            "0",
            "--threads",
            str(args.threads),
        ]
    )
    commands.append(
        [
            args.mmseqs,
            "linclust",
            sequence_db,
            cluster_db,
            args.tmp_dir,
            "--min-seq-id",
            str(args.min_seq_id),
            "-c",
            str(args.coverage),
            "--cov-mode",
            str(args.cov_mode),
            "--cluster-mode",
            str(args.cluster_mode),
            "--similarity-type",
            "2",
            "--kmer-per-seq-scale",
            str(args.kmer_per_seq_scale),
            "--threads",
            str(args.threads),
            "--remove-tmp-files",
            "1",
        ]
    )
    commands.append(
        [
            args.mmseqs,
            "createtsv",
            sequence_db,
            sequence_db,
            cluster_db,
            str(cluster_tsv),
            "--threads",
            str(args.threads),
        ]
    )
    for command in commands:
        run_command(command)
    return commands


def run_mmseqs_easy(args: argparse.Namespace) -> List[str]:
    cluster_tsv = Path(f"{args.cluster_prefix}_cluster.tsv")
    if cluster_tsv.exists() and not args.force_mmseqs:
        return ["skipped_existing_mmseqs_cluster_tsv"]
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cluster_prefix).parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.mmseqs,
        "easy-linclust",
        args.fasta,
        args.cluster_prefix,
        args.tmp_dir,
        "--dbtype",
        "2",
        "--min-seq-id",
        str(args.min_seq_id),
        "-c",
        str(args.coverage),
        "--cov-mode",
        str(args.cov_mode),
        "--cluster-mode",
        str(args.cluster_mode),
        "--similarity-type",
        "2",
        "--kmer-per-seq-scale",
        str(args.kmer_per_seq_scale),
        "--threads",
        str(args.threads),
        "--remove-tmp-files",
        "1",
    ]
    run_command(command)
    return command


def parse_cluster_tsv(path: Path, site_ids: Sequence[str]) -> tuple[Dict[str, str], Dict[str, object]]:
    site_set = set(site_ids)
    site_to_cluster: Dict[str, str] = {}
    cluster_members: Dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for parts in reader:
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            if member not in site_set:
                continue
            cluster_id = f"mmseqs80_{rep}"
            site_to_cluster[member] = cluster_id
            cluster_members[cluster_id].add(member)
    for site_id in site_ids:
        if site_id not in site_to_cluster:
            cluster_id = f"mmseqs80_{site_id}"
            site_to_cluster[site_id] = cluster_id
            cluster_members[cluster_id].add(site_id)
    sizes = [len(members) for members in cluster_members.values()]
    summary = {
        "cluster_count": len(cluster_members),
        "largest_cluster_size": max(sizes) if sizes else 0,
        "mean_cluster_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "singleton_clusters": sum(1 for size in sizes if size == 1),
    }
    return site_to_cluster, summary


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
    targets = {split: [target_fracs[split] * total[idx] for idx in range(len(CLASS_NAMES))] for split in SPLITS}

    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[group_key_by_site[row["site_id"]]].append(row)

    group_items = []
    for key, group_rows in groups.items():
        counts = class_counts(group_rows)
        size = sum(counts)
        group_items.append((key, counts, size, rng.random()))
    group_items.sort(key=lambda item: (-item[2], item[3], item[0]))

    assigned_counts = {split: [0] * len(CLASS_NAMES) for split in SPLITS}
    group_to_split: Dict[str, str] = {}
    for key, counts, size, _ in group_items:
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
                deficit_before = max(target - before, 0.0) / target
                overshoot_after = max(after - target, 0.0) / target
                score += count * (deficit_before - 2.0 * overshoot_after)
            split_total = sum(assigned_counts[split])
            split_target_total = max(sum(targets[split]), 1.0)
            score -= 0.05 * abs((split_total + size) - split_target_total) / split_target_total
            if best_score is None or score > best_score:
                best_score = score
                best_split = split
        group_to_split[key] = best_split
        for idx, count in enumerate(counts):
            assigned_counts[best_split][idx] += count

    return {row["site_id"]: group_to_split[group_key_by_site[row["site_id"]]] for row in rows}


def summarize_splits(rows: Sequence[Dict[str, str]], split_columns: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for split_col in split_columns:
        counts = Counter((row.get("mod_type", ""), row.get(split_col, "")) for row in rows)
        out[split_col] = {f"{mod}:{split}": counts[(mod, split)] for mod in CLASS_NAMES for split in SPLITS}
    return out


def check_group_leakage(rows: Sequence[Dict[str, str]], group_col: str, split_col: str) -> Dict[str, object]:
    group_to_splits: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_to_splits[row[group_col]].add(row[split_col])
    leaked = {group: sorted(splits) for group, splits in group_to_splits.items() if len(splits) > 1}
    return {
        "group_column": group_col,
        "split_column": split_col,
        "leaked_group_count": len(leaked),
        "example_leaked_groups": dict(list(leaked.items())[:10]),
    }


def get_mmseqs_version(mmseqs: str) -> str:
    try:
        result = subprocess.run([mmseqs, "version"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as exc:  # pragma: no cover - best effort metadata only
        return f"unavailable: {exc}"


def run(args: argparse.Namespace) -> Dict[str, object]:
    rows, fields = read_tsv(Path(args.dataset))
    site_ids = [row["site_id"] for row in rows]
    write_fasta(rows, Path(args.fasta), force=args.force_fasta)
    command = run_mmseqs(args)
    cluster_tsv = Path(f"{args.cluster_prefix}_cluster.tsv")
    site_to_cluster, cluster_summary = parse_cluster_tsv(cluster_tsv, site_ids)
    split_by_site = split_grouped_rows(rows, site_to_cluster, args.seed, args.train_frac, args.valid_frac)

    for row in rows:
        row["mmseqs80_cluster_id"] = site_to_cluster[row["site_id"]]
        row["split_mmseqs80_cluster"] = split_by_site[row["site_id"]]

    out_fields = list(fields)
    for field in ["mmseqs80_cluster_id", "split_mmseqs80_cluster"]:
        if field not in out_fields:
            out_fields.append(field)
    write_tsv(Path(args.out), rows, out_fields)

    cluster_summary.update(
        {
            "method": "MMseqs2 easy-linclust, nucleotide mode, sequence identity clustering",
            "mmseqs_version": get_mmseqs_version(args.mmseqs),
            "min_seq_id": args.min_seq_id,
            "coverage": args.coverage,
            "cov_mode": args.cov_mode,
            "cluster_mode": args.cluster_mode,
            "similarity_type": 2,
            "kmer_per_seq_scale": args.kmer_per_seq_scale,
            "cluster_tsv": str(cluster_tsv),
            "fasta": args.fasta,
            "command": command,
        }
    )
    summary = {
        "input_dataset": args.dataset,
        "output_dataset": args.out,
        "row_count": len(rows),
        "class_counts": dict(Counter(row["mod_type"] for row in rows)),
        "cluster_summary": cluster_summary,
        "split_counts": summarize_splits(rows, [field for field in ["split_random", "split_gene", "split_mmseqs80_cluster"] if field in rows[0]]),
        "leakage_checks": [
            check_group_leakage(rows, "mmseqs80_cluster_id", "split_mmseqs80_cluster"),
        ],
        "seed": args.seed,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add full RMBase11 MMseqs2 80% cluster split.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_strict_splits.tsv")
    parser.add_argument("--out", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_mmseqs80_strict_splits.tsv")
    parser.add_argument("--summary", default="rmbase11_full_work/datasets/rmbase11_mmseqs80_strict_splits_summary.json")
    parser.add_argument("--fasta", default="rmbase11_full_work/cluster/rmbase11_hg38_41nt_all_sites.fa")
    parser.add_argument("--cluster-prefix", default="rmbase11_full_work/cluster/mmseqs80/rmbase11_mmseqs80")
    parser.add_argument("--tmp-dir", default="rmbase11_full_work/cluster/mmseqs80_tmp")
    parser.add_argument("--mmseqs", default="tools/mmseqs2/mmseqs/bin/mmseqs.exe")
    parser.add_argument("--min-seq-id", type=float, default=0.8)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--cluster-mode", type=int, default=2)
    parser.add_argument("--kmer-per-seq-scale", type=float, default=0.3)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-fasta", action="store_true")
    parser.add_argument("--force-mmseqs", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
