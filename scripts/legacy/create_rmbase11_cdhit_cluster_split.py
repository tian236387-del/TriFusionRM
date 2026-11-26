#!/usr/bin/env python
"""Create a full RMBase CD-HIT-est 80% cluster split.

This script adds a standard CD-HIT-est redundancy-aware split to an RMBase
all-site table.  Rows from the same CD-HIT-est sequence cluster are assigned to
only one of train/valid/test, reducing train-test leakage from highly similar
41-nt windows.

Typical server use:

    python -B scripts/create_rmbase11_cdhit_cluster_split.py \
      --cdhit-est "$(command -v cd-hit-est)" \
      --threads 32

The input file can already contain random, gene and MMseqs split columns; this
script preserves them and appends:

    cdhit80_cluster_id
    split_cdhit80_cluster
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
SPLITS = ("train", "valid", "test")


def read_tsv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
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


def normalized_sequence(row: Dict[str, str], sequence_column: str = "sequence_41nt") -> str:
    return (row.get(sequence_column) or "").upper().replace("U", "T")


def infer_class_names(rows: Sequence[Dict[str, str]], explicit: str = "") -> List[str]:
    if explicit.strip():
        names = [name.strip() for name in explicit.split(",") if name.strip()]
        if not names:
            raise ValueError("--class-names was provided but no class names were parsed")
        return names
    pairs = []
    fallback = []
    for idx, row in enumerate(rows):
        mod = row.get("mod_type", "")
        if not mod:
            continue
        fallback.append((idx, mod))
        try:
            class_idx = int(float(row.get("class_index", "")))
            pairs.append((class_idx, idx, mod))
        except ValueError:
            continue
    if pairs:
        by_index: Dict[int, str] = {}
        for class_idx, _, mod in sorted(pairs):
            by_index.setdefault(class_idx, mod)
        return [by_index[idx] for idx in sorted(by_index)]
    seen = set()
    names = []
    for _, mod in fallback:
        if mod not in seen:
            seen.add(mod)
            names.append(mod)
    if not names:
        raise ValueError("Could not infer class names from dataset")
    return names


def write_fasta(rows: Sequence[Dict[str, str]], path: Path, sequence_column: str = "sequence_41nt", force: bool = False) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for row in rows:
            site_id = row["site_id"].replace(" ", "_")
            seq = normalized_sequence(row, sequence_column)
            handle.write(f">{site_id}\n{seq}\n")


def write_unique_sequence_fasta(
    rows: Sequence[Dict[str, str]],
    path: Path,
    sequence_column: str = "sequence_41nt",
    force: bool = False,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    """Write one FASTA record per exact sequence and map sites to records."""
    seq_to_record: Dict[str, str] = {}
    site_to_record: Dict[str, str] = {}
    if path.exists() and not force:
        # Rebuild the mapping from rows so that an existing FASTA can be reused.
        for row in rows:
            seq = normalized_sequence(row, sequence_column)
            record_id = seq_to_record.setdefault(seq, f"seq{len(seq_to_record):09d}")
            site_to_record[row["site_id"]] = record_id
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii", newline="\n") as handle:
            for row in rows:
                seq = normalized_sequence(row, sequence_column)
                record_id = seq_to_record.get(seq)
                if record_id is None:
                    record_id = f"seq{len(seq_to_record):09d}"
                    seq_to_record[seq] = record_id
                    handle.write(f">{record_id}\n{seq}\n")
                site_to_record[row["site_id"]] = record_id

    duplicate_count = len(rows) - len(seq_to_record)
    summary = {
        "deduplicated_identical_sequences": True,
        "input_rows": len(rows),
        "unique_sequences": len(seq_to_record),
        "exact_duplicate_rows_removed_before_cdhit": duplicate_count,
        "unique_fasta": str(path),
    }
    return site_to_record, summary


def run_command(command: Sequence[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(list(command), check=True)


def run_cdhit(args: argparse.Namespace) -> List[str]:
    output_prefix = Path(args.cdhit_output_prefix)
    clstr = Path(f"{args.cdhit_output_prefix}.clstr")
    if clstr.exists() and not args.force_cdhit:
        return ["skipped_existing_cdhit_clstr"]

    executable = args.cdhit_est or shutil.which("cd-hit-est")
    if not executable:
        raise FileNotFoundError(
            "cd-hit-est was not found. Install it on the server, or pass --cdhit-est /path/to/cd-hit-est."
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "-i",
        args.fasta,
        "-o",
        args.cdhit_output_prefix,
        "-c",
        str(args.identity),
        "-n",
        str(args.word_size),
        "-G",
        "1",
        "-aS",
        str(args.coverage_shorter),
        "-aL",
        str(args.coverage_longer),
        "-d",
        "0",
        "-T",
        str(args.threads),
        "-M",
        str(args.memory_mb),
    ]
    run_command(command)
    return command


def parse_cdhit_clstr(path: Path, site_ids: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, object]]:
    site_set = set(site_ids)
    site_to_cluster: Dict[str, str] = {}
    cluster_members: Dict[str, set[str]] = defaultdict(set)
    current_cluster = ""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                current_cluster = "cdhit80_" + line.split()[-1]
                continue
            if ">" not in line:
                continue
            member = line.split(">", 1)[1].split("...", 1)[0].strip()
            if member not in site_set:
                continue
            site_to_cluster[member] = current_cluster
            cluster_members[current_cluster].add(member)

    # CD-HIT should report all input sequences, but keep the split robust.
    for site_id in site_ids:
        if site_id not in site_to_cluster:
            cluster = f"cdhit80_singleton_{site_id}"
            site_to_cluster[site_id] = cluster
            cluster_members[cluster].add(site_id)

    sizes = [len(members) for members in cluster_members.values()]
    summary = {
        "cluster_count": len(cluster_members),
        "largest_cluster_size": max(sizes) if sizes else 0,
        "mean_cluster_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "singleton_clusters": sum(1 for size in sizes if size == 1),
        "cluster_file": str(path),
    }
    return site_to_cluster, summary


def parse_cdhit_clstr_for_records(path: Path, record_ids: Iterable[str]) -> Dict[str, str]:
    record_set = set(record_ids)
    record_to_cluster: Dict[str, str] = {}
    current_cluster = ""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                current_cluster = "cdhit80_" + line.split()[-1]
                continue
            if ">" not in line:
                continue
            member = line.split(">", 1)[1].split("...", 1)[0].strip()
            if member in record_set:
                record_to_cluster[member] = current_cluster

    for record_id in record_set:
        if record_id not in record_to_cluster:
            record_to_cluster[record_id] = f"cdhit80_singleton_{record_id}"
    return record_to_cluster


def expand_record_clusters_to_sites(
    site_to_record: Dict[str, str],
    record_to_cluster: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, object]]:
    site_to_cluster = {site_id: record_to_cluster[record_id] for site_id, record_id in site_to_record.items()}
    cluster_members: Dict[str, set[str]] = defaultdict(set)
    for site_id, cluster in site_to_cluster.items():
        cluster_members[cluster].add(site_id)
    sizes = [len(members) for members in cluster_members.values()]
    summary = {
        "cluster_count": len(cluster_members),
        "largest_cluster_size": max(sizes) if sizes else 0,
        "mean_cluster_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "singleton_clusters": sum(1 for size in sizes if size == 1),
    }
    return site_to_cluster, summary


def parse_label(row: Mapping[str, str], class_to_id: Mapping[str, int]) -> int | None:
    if row.get("class_index", "") != "":
        try:
            idx = int(float(row["class_index"]))
            if 0 <= idx < len(class_to_id):
                return idx
        except ValueError:
            pass
    mod = row.get("mod_type", "")
    return class_to_id.get(mod)


def class_counts(rows: Sequence[Dict[str, str]], class_to_id: Mapping[str, int]) -> List[int]:
    counts = [0] * len(class_to_id)
    for row in rows:
        idx = parse_label(row, class_to_id)
        if idx is not None:
            counts[idx] += 1
    return counts


def split_grouped_rows(
    rows: Sequence[Dict[str, str]],
    group_key_by_site: Dict[str, str],
    class_names: Sequence[str],
    seed: int,
    train_frac: float,
    valid_frac: float,
) -> Dict[str, str]:
    rng = random.Random(seed)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    target_fracs = {"train": train_frac, "valid": valid_frac, "test": max(0.0, 1.0 - train_frac - valid_frac)}
    total = class_counts(rows, class_to_id)
    targets = {split: [target_fracs[split] * total[idx] for idx in range(len(class_names))] for split in SPLITS}

    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[group_key_by_site[row["site_id"]]].append(row)

    group_items = []
    for key, group_rows in groups.items():
        counts = class_counts(group_rows, class_to_id)
        size = sum(counts)
        group_items.append((key, counts, size, rng.random()))
    group_items.sort(key=lambda item: (-item[2], item[3], item[0]))

    assigned_counts = {split: [0] * len(class_names) for split in SPLITS}
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


def summarize_splits(
    rows: Sequence[Dict[str, str]],
    split_columns: Sequence[str],
    class_names: Sequence[str],
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for split_col in split_columns:
        counts = Counter((row.get("mod_type", ""), row.get(split_col, "")) for row in rows)
        out[split_col] = {f"{mod}:{split}": counts[(mod, split)] for mod in class_names for split in SPLITS}
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


def get_cdhit_version(cdhit_est: str | None) -> str:
    executable = cdhit_est or shutil.which("cd-hit-est")
    if not executable:
        return "unavailable"
    try:
        result = subprocess.run([executable], check=False, capture_output=True, text=True)
        text = (result.stdout + "\n" + result.stderr).strip()
        return text.splitlines()[0] if text else "unavailable"
    except Exception as exc:  # pragma: no cover - metadata only
        return f"unavailable: {exc}"


def run(args: argparse.Namespace) -> Dict[str, object]:
    rows, fields = read_tsv(Path(args.dataset))
    required = {"site_id", "mod_type", args.sequence_column}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")
    class_names = infer_class_names(rows, args.class_names)

    site_ids = [row["site_id"] for row in rows]
    dedup_summary: Dict[str, object] = {"deduplicated_identical_sequences": False}
    site_to_record: Dict[str, str] | None = None
    if args.deduplicate_identical_sequences:
        site_to_record, dedup_summary = write_unique_sequence_fasta(
            rows,
            Path(args.fasta),
            sequence_column=args.sequence_column,
            force=args.force_fasta,
        )
    else:
        write_fasta(rows, Path(args.fasta), sequence_column=args.sequence_column, force=args.force_fasta)
    command = run_cdhit(args)
    clstr = Path(f"{args.cdhit_output_prefix}.clstr")
    if site_to_record is not None:
        record_to_cluster = parse_cdhit_clstr_for_records(clstr, set(site_to_record.values()))
        site_to_cluster, cluster_summary = expand_record_clusters_to_sites(site_to_record, record_to_cluster)
        cluster_summary["cluster_file"] = str(clstr)
    else:
        site_to_cluster, cluster_summary = parse_cdhit_clstr(clstr, site_ids)
    split_by_site = split_grouped_rows(rows, site_to_cluster, class_names, args.seed, args.train_frac, args.valid_frac)

    for row in rows:
        row["cdhit80_cluster_id"] = site_to_cluster[row["site_id"]]
        row["split_cdhit80_cluster"] = split_by_site[row["site_id"]]

    out_fields = list(fields)
    for field in ["cdhit80_cluster_id", "split_cdhit80_cluster"]:
        if field not in out_fields:
            out_fields.append(field)
    write_tsv(Path(args.out), rows, out_fields)

    cluster_summary.update(
        {
            "method": "CD-HIT-est nucleotide sequence clustering",
            "cdhit_version": get_cdhit_version(args.cdhit_est),
            "identity": args.identity,
            "word_size": args.word_size,
            "coverage_shorter": args.coverage_shorter,
            "coverage_longer": args.coverage_longer,
            "global_identity": True,
            "fasta": args.fasta,
            "sequence_column": args.sequence_column,
            "representative_fasta": args.cdhit_output_prefix,
            "command": command,
            "deduplication": dedup_summary,
        }
    )
    split_columns = [
        field
        for field in ["split_random", "split_gene", "split_mmseqs80_cluster", "split_cdhit80_cluster"]
        if field in rows[0]
    ]
    summary = {
        "input_dataset": args.dataset,
        "output_dataset": args.out,
        "row_count": len(rows),
        "class_names": class_names,
        "class_counts": dict(Counter(row["mod_type"] for row in rows)),
        "cluster_summary": cluster_summary,
        "split_counts": summarize_splits(rows, split_columns, class_names),
        "leakage_checks": [
            check_group_leakage(rows, "cdhit80_cluster_id", "split_cdhit80_cluster"),
        ],
        "seed": args.seed,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add full RMBase CD-HIT-est 80% cluster split.")
    parser.add_argument(
        "--dataset",
        default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_mmseqs80_strict_splits.tsv",
    )
    parser.add_argument(
        "--out",
        default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv",
    )
    parser.add_argument(
        "--summary",
        default="rmbase11_full_work/datasets/rmbase11_cdhit80_strict_splits_summary.json",
    )
    parser.add_argument("--fasta", default="rmbase11_full_work/cluster/cdhit80/rmbase11_hg38_41nt_all_sites.fa")
    parser.add_argument("--cdhit-output-prefix", default="rmbase11_full_work/cluster/cdhit80/rmbase11_cdhit80")
    parser.add_argument("--sequence-column", default="sequence_41nt")
    parser.add_argument("--cdhit-est", default="")
    parser.add_argument("--identity", type=float, default=0.8)
    parser.add_argument("--word-size", type=int, default=5)
    parser.add_argument("--coverage-shorter", type=float, default=0.8)
    parser.add_argument("--coverage-longer", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-mb", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-names",
        default="",
        help="Optional comma-separated class order. If omitted, infer from class_index/mod_type columns.",
    )
    parser.add_argument("--force-fasta", action="store_true")
    parser.add_argument("--force-cdhit", action="store_true")
    parser.add_argument(
        "--deduplicate-identical-sequences",
        action="store_true",
        help="Cluster exact unique sequences first, then map clusters back to all original sites.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
