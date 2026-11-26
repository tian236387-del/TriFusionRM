#!/usr/bin/env python3
"""Prepare the StructDualRM V1 13-class hg38 site-detection dataset.

The positive input is the existing RMBase v3 hg38 all-site table.  This
workflow canonicalizes the 12 requested modification labels, removes
coordinate-level conflicts, builds a balanced set of matched unannotated
candidates through the existing negative builder, and assigns leakage-aware
splits after 201-nt clustering.

The script deliberately keeps intermediate tables and audit files.  The
negative class means "unannotated matched candidate", not experimentally
verified unmodified RNA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MOD_TYPES = [
    "m6A", "m1A", "m5C", "m5U", "m6Am", "m7G",
    "Psi", "I", "Am", "Cm", "Gm", "Um",
]
CLASS_NAMES = ["Unmodified"] + MOD_TYPES
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
LABEL_MAP = {
    "Y": "Psi", "Psi": "Psi", "pseudouridine": "Psi",
    "AtoI": "I", "A-I": "I", "I": "I",
}
EXPECTED_BASES = {
    "m6A": "A", "m1A": "A", "m6Am": "A", "Am": "A", "I": "A",
    "m5C": "C", "Cm": "C",
    "m7G": "G", "Gm": "G",
    "m5U": "T", "Psi": "T", "Um": "T",
}
SPLITS = ("train", "valid", "test")
TOKEN_SPLIT = ",;|"


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def write_tsv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    temp.replace(path)
    return count


def normalize_label(value: object) -> str:
    label = str(value or "").strip()
    return LABEL_MAP.get(label, label)


def normalize_base(value: object) -> str:
    return str(value or "").strip().upper().replace("U", "T")


def coordinate_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return str(row.get("chr", "")), str(row.get("start", "")), str(row.get("strand", "."))


def split_tokens(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"na", "nan", "."}:
        return []
    current = text
    for delimiter in ";|":
        current = current.replace(delimiter, ",")
    return [item.strip() for item in current.split(",") if item.strip()]


def canonicalize(args: argparse.Namespace) -> None:
    source_rows, source_fields = read_tsv(Path(args.source))
    required = {"mod_type", "chr", "start", "end", "strand", "sequence_201nt"}
    missing = required - set(source_fields)
    if missing:
        raise ValueError(f"source table is missing fields: {sorted(missing)}")

    known_by_coord: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    valid_by_coord_label: dict[tuple[tuple[str, str, str], str], list[dict[str, str]]] = defaultdict(list)
    excluded: list[dict[str, object]] = []
    invalid_center = Counter()

    for row in source_rows:
        label = normalize_label(row.get("mod_type"))
        if label not in MOD_TYPES:
            excluded.append({"site_id": row.get("site_id", ""), "mod_type": row.get("mod_type", ""), "reason": "outside_target_12_classes"})
            continue
        coord = coordinate_key(row)
        known_by_coord[coord].add(label)
        sequence = str(row.get("sequence_201nt", "") or "").upper().replace("U", "T")
        expected = EXPECTED_BASES[label]
        if len(sequence) != 201:
            invalid_center["sequence_not_201nt"] += 1
            excluded.append({"site_id": row.get("site_id", ""), "mod_type": label, "reason": "sequence_not_201nt"})
            continue
        if sequence[100] != expected:
            invalid_center["center_base_mismatch"] += 1
            excluded.append({"site_id": row.get("site_id", ""), "mod_type": label, "reason": "center_base_mismatch"})
            continue
        valid_by_coord_label[(coord, label)].append(row)

    conflict_coords = {coord for coord, labels in known_by_coord.items() if len(labels) > 1}
    valid_row_counts_by_coord = Counter(
        coord for (coord, _label), candidates in valid_by_coord_label.items() for _ in candidates
    )
    conflict_rows = []
    for coord in sorted(conflict_coords):
        conflict_rows.append({
            "chr": coord[0], "start": coord[1], "strand": coord[2],
            "label_set": "|".join(sorted(known_by_coord[coord], key=MOD_TYPES.index)),
            "source_row_count": valid_row_counts_by_coord.get(coord, 0),
            "reason": "coordinate_has_multiple_modification_labels",
        })

    positive_rows: list[dict[str, object]] = []
    duplicate_count = 0
    for (coord, label), candidates in sorted(valid_by_coord_label.items(), key=lambda item: (item[0][0], MOD_TYPES.index(item[0][1]))):
        if coord in conflict_coords:
            for row in candidates:
                excluded.append({"site_id": row.get("site_id", ""), "mod_type": label, "reason": "coordinate_has_multiple_modification_labels"})
            continue
        representative = max(candidates, key=lambda row: float(row.get("support_num", row.get("supportNum", "0")) or 0))
        duplicate_count += max(0, len(candidates) - 1)
        output = dict(representative)
        # Preserve the existing RMBase site ID so compatible RNAfold/RBP
        # intermediates can be reused without changing the biological record.
        output["site_id"] = representative.get("site_id", "")
        output["mod_type"] = label
        output["source_mod_type"] = representative.get("mod_type", "")
        output["class_index"] = CLASS_TO_ID[label]
        output["label"] = label
        output["det_label"] = 1
        output["detection_class"] = "modified"
        output["negative_status"] = ""
        output["matched_positive_site_id"] = ""
        output["matched_mod_type"] = ""
        output["match_level"] = ""
        output["source_site_ids"] = ",".join(str(row.get("site_id", "")) for row in candidates)
        output["center_base"] = normalize_base(str(output.get("sequence_201nt", ""))[100:101])
        output["sequence_101nt"] = str(output.get("sequence_201nt", ""))[50:151]
        output["sequence_41nt"] = str(output.get("sequence_201nt", ""))[80:121]
        for split_field in ["split_cluster201", "split_gene_holdout", "split_transcript_holdout", "split_study_holdout", "split_lockbox", "is_frozen_lockbox"]:
            output.setdefault(split_field, "")
        positive_rows.append(output)

    fields = list(source_fields)
    extra = [
        "site_id", "mod_type", "class_index", "label", "det_label", "detection_class",
        "negative_status", "matched_positive_site_id", "matched_mod_type", "match_level",
        "source_site_ids", "sequence_101nt", "sequence_41nt",
        "split_cluster201", "split_gene_holdout", "split_transcript_holdout", "split_study_holdout", "split_lockbox", "is_frozen_lockbox",
    ]
    fields = [field for field in fields if field not in extra]
    fields = extra + fields
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "positive_clean.tsv", positive_rows, fields)

    known_rows = []
    for coord, labels in sorted(known_by_coord.items()):
        known_rows.append({
            "site_id": f"KNOWN_{hashlib.sha1('|'.join(coord).encode()).hexdigest()[:16]}",
            "chr": coord[0], "start": coord[1], "end": str(int(float(coord[1])) + 1),
            "strand": coord[2], "known_label_set": "|".join(sorted(labels, key=MOD_TYPES.index)),
        })
    write_tsv(out_dir / "known_all_target_coordinates.tsv", known_rows, ["site_id", "chr", "start", "end", "strand", "known_label_set"])
    write_tsv(out_dir / "coordinate_conflicts.tsv", conflict_rows, ["chr", "start", "strand", "label_set", "source_row_count", "reason"])
    write_tsv(out_dir / "excluded_positive_records.tsv", excluded, ["site_id", "mod_type", "reason"])

    summary = {
        "source": str(args.source),
        "target_classes": MOD_TYPES,
        "class_names": CLASS_NAMES,
        "raw_source_rows": len(source_rows),
        "retained_positive_rows": len(positive_rows),
        "known_unique_coordinates": len(known_rows),
        "coordinate_conflict_count": len(conflict_coords),
        "duplicate_coordinate_label_rows_removed": duplicate_count,
        "excluded_rows": len(excluded),
        "invalid_reasons": dict(invalid_center),
        "positive_counts": dict(Counter(row["mod_type"] for row in positive_rows)),
        "label_policy": "Y->Psi; AtoI/A-I->I; only coordinate-level conflicts are removed; m6Am class is retained",
    }
    (out_dir / "canonicalization_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def combine(args: argparse.Namespace) -> None:
    positives, positive_fields = read_tsv(Path(args.positives))
    negatives, negative_fields = read_tsv(Path(args.negatives))
    fields = list(dict.fromkeys([
        "site_id", "label", "mod_type", "class_index", "det_label", "detection_class",
        "negative_status", "matched_positive_site_id", "matched_mod_type", "match_level",
        "chr", "start", "end", "strand", "gene_id", "transcript_id", "gene_name",
        "gene_type", "region_type", "support_num", "support_list", "pub_list", "cell_list",
        "seq_type_list", "source_file", "center_base", "sequence_41nt", "sequence_101nt",
        "sequence_201nt", "gc_fraction_201nt", *positive_fields, *negative_fields,
    ]))
    out_rows: list[dict[str, object]] = []
    coordinate_seen: set[tuple[str, str, str]] = set()
    for row in positives:
        copied = dict(row)
        copied["label"] = normalize_label(row.get("mod_type"))
        copied["mod_type"] = copied["label"]
        copied["class_index"] = CLASS_TO_ID[copied["label"]]
        copied["det_label"] = 1
        copied["detection_class"] = "modified"
        copied["negative_status"] = ""
        copied["center_base"] = normalize_base(str(copied.get("sequence_201nt", ""))[100:101])
        coordinate_seen.add(coordinate_key(copied))
        out_rows.append(copied)
    duplicate_negative = 0
    for row in negatives:
        copied = dict(row)
        coord = coordinate_key(copied)
        if coord in coordinate_seen:
            duplicate_negative += 1
            continue
        coordinate_seen.add(coord)
        copied["label"] = "Unmodified"
        copied["mod_type"] = "Unmodified"
        copied["class_index"] = 0
        copied["det_label"] = 0
        copied["detection_class"] = "unmodified"
        copied["negative_status"] = copied.get("negative_status", "unannotated_matched_candidate")
        copied["center_base"] = normalize_base(str(copied.get("sequence_201nt", ""))[100:101])
        out_rows.append(copied)

    for row in out_rows:
        row["sequence_201nt"] = str(row.get("sequence_201nt", "") or "").upper().replace("U", "T")
        sequence = str(row["sequence_201nt"])
        row["sequence_101nt"] = sequence[50:151]
        row["sequence_41nt"] = sequence[80:121]
        row["center_base"] = sequence[100:101]
        row["pair_id"] = row.get("matched_positive_site_id") or row.get("site_id", "")
        row["pair_role"] = "negative" if row.get("label") == "Unmodified" else "positive"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_fields = [field for field in fields + ["pair_id", "pair_role"] if field not in {"source_site_ids"}]
    dataset_fields = list(dict.fromkeys(dataset_fields))
    write_tsv(out_dir / "dataset_13class_unclustered.tsv", out_rows, dataset_fields)

    fasta_path = out_dir / "sequence_201nt_unique.fa"
    sequence_to_record: dict[str, str] = {}
    with fasta_path.open("w", encoding="ascii", newline="\n") as handle:
        for row in out_rows:
            sequence = str(row.get("sequence_201nt", ""))
            if len(sequence) != 201 or set(sequence) - set("ACGT"):
                continue
            record = sequence_to_record.get(sequence)
            if record is None:
                record = f"seq201_{len(sequence_to_record):09d}"
                sequence_to_record[sequence] = record
                handle.write(f">{record}\n{sequence}\n")
    site_to_record = []
    for row in out_rows:
        sequence = str(row.get("sequence_201nt", ""))
        site_to_record.append({"site_id": row["site_id"], "seq_record_id": sequence_to_record.get(sequence, "")})
    write_tsv(out_dir / "site_to_sequence_record.tsv", site_to_record, ["site_id", "seq_record_id"])
    summary = {
        "positive_rows": len(positives),
        "negative_rows_input": len(negatives),
        "negative_rows_removed_coordinate_duplicate": duplicate_negative,
        "combined_rows": len(out_rows),
        "class_names": CLASS_NAMES,
        "class_counts": dict(Counter(str(row["label"]) for row in out_rows)),
        "unique_201nt_sequences": len(sequence_to_record),
        "fasta": str(fasta_path),
        "negative_label_caveat": "Unmodified means unannotated matched candidate and is not experimentally verified negative.",
    }
    (out_dir / "combine_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


class UnionFind:
    def __init__(self, items: Iterable[str] = ()) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}
        for item in items:
            self.add(item)

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def read_cluster_members(path: Path) -> dict[str, str]:
    record_to_cluster: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 2:
                continue
            representative, member = fields[0].strip(), fields[1].strip()
            if representative and member:
                record_to_cluster[member] = representative
    return record_to_cluster


def class_vector(label: str) -> list[int]:
    vector = [0] * len(CLASS_NAMES)
    if label in CLASS_TO_ID:
        vector[CLASS_TO_ID[label]] = 1
    return vector


def allocate_groups(group_rows: Mapping[str, list[int]], row_count: Mapping[str, int], seed: int) -> dict[str, str]:
    totals = [0] * len(CLASS_NAMES)
    for counts in group_rows.values():
        for idx, count in enumerate(counts):
            totals[idx] += count
    targets = [[fraction * total for total in totals] for fraction in (0.8, 0.1, 0.1)]
    size_targets = [fraction * sum(row_count.values()) for fraction in (0.8, 0.1, 0.1)]
    assigned = [[0] * len(CLASS_NAMES) for _ in SPLITS]
    assigned_sizes = [0] * len(SPLITS)
    rarity = [1.0 / max(total, 1) for total in totals]
    rng = random.Random(seed)
    ordered = []
    for group, counts in group_rows.items():
        rarity_score = sum(count * rarity[idx] for idx, count in enumerate(counts))
        ordered.append((group, counts, row_count[group], rarity_score, rng.random()))
    ordered.sort(key=lambda item: (-item[3], -item[2], item[4], item[0]))

    def objective(split_idx: int, counts: Sequence[int], size: int) -> float:
        score = 0.0
        for current_split in range(3):
            for cls, total in enumerate(totals):
                value = assigned[current_split][cls] + (counts[cls] if current_split == split_idx else 0)
                score += ((value - targets[current_split][cls]) / max(total, 1)) ** 2
            current_size = assigned_sizes[current_split] + (size if current_split == split_idx else 0)
            score += 0.2 * ((current_size - size_targets[current_split]) / max(sum(row_count.values()), 1)) ** 2
        return score

    assignment = {}
    for group, counts, size, _, _ in ordered:
        split_idx = min(range(3), key=lambda idx: (objective(idx, counts, size), idx))
        assignment[group] = SPLITS[split_idx]
        assigned_sizes[split_idx] += size
        for cls, count in enumerate(counts):
            assigned[split_idx][cls] += count
    return assignment


def split_dataset(args: argparse.Namespace) -> None:
    rows, fields = read_tsv(Path(args.dataset))
    cluster_members = read_cluster_members(Path(args.cluster_tsv)) if args.cluster_tsv else {}
    record_rows, _ = read_tsv(Path(args.site_to_record))
    site_to_record = {row["site_id"]: row.get("seq_record_id", "") for row in record_rows}

    uf = UnionFind(row["site_id"] for row in rows)
    site_to_cluster: dict[str, str] = {}
    for row in rows:
        site_id = row["site_id"]
        record = site_to_record.get(site_id, "")
        cluster = cluster_members.get(record, f"singleton_{record or site_id}")
        site_to_cluster[site_id] = cluster
    cluster_first: dict[str, str] = {}
    for site_id, cluster in site_to_cluster.items():
        if cluster in cluster_first:
            uf.union(site_id, cluster_first[cluster])
        else:
            cluster_first[cluster] = site_id
    for row in rows:
        pair = row.get("pair_id", "")
        if pair and pair in site_to_cluster:
            uf.union(row["site_id"], pair)

    root_by_site = {row["site_id"]: uf.find(row["site_id"]) for row in rows}
    group_rows: dict[str, list[int]] = defaultdict(lambda: [0] * len(CLASS_NAMES))
    group_count: Counter[str] = Counter()
    for row in rows:
        root = root_by_site[row["site_id"]]
        label = row.get("label", "")
        if label in CLASS_TO_ID:
            group_rows[root][CLASS_TO_ID[label]] += 1
        group_count[root] += 1
    assignment = allocate_groups(group_rows, group_count, args.seed)
    output_rows = []
    for row in rows:
        copied = dict(row)
        root = root_by_site[row["site_id"]]
        copied["sequence_cluster201_id"] = site_to_cluster[row["site_id"]]
        copied["paired_cluster_component_id"] = f"paircluster201_{hashlib.sha1(root.encode()).hexdigest()[:16]}"
        copied["split_cluster201"] = assignment[root]
        copied["split"] = assignment[root]
        output_rows.append(copied)

    output_fields = list(fields)
    for field in ["sequence_cluster201_id", "paired_cluster_component_id", "split_cluster201", "split"]:
        if field not in output_fields:
            output_fields.append(field)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "dataset_13class_cluster201_splits.tsv", output_rows, output_fields)

    split_counts = Counter((row["label"], row["split_cluster201"]) for row in output_rows)
    leakage = {
        "sequence_clusters": len(set(site_to_cluster.values())),
        "paired_cluster_components": len(set(root_by_site.values())),
        "rows": len(output_rows),
        "class_counts": dict(Counter(row["label"] for row in output_rows)),
        "split_class_counts": {f"{label}:{split}": count for (label, split), count in sorted(split_counts.items())},
        "cluster_leaked_components": 0,
        "pair_leaked_components": 0,
        "split_policy": "union matched positive-negative pairs and all 201-nt sequence-cluster members before assignment",
    }
    (out_dir / "split_summary.json").write_text(json.dumps(leakage, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(leakage, indent=2, ensure_ascii=False), flush=True)


def audit(args: argparse.Namespace) -> None:
    rows, _ = read_tsv(Path(args.dataset))
    errors = Counter()
    positives = {row["site_id"]: row for row in rows if row.get("label") != "Unmodified"}
    seen_coords: dict[tuple[str, str, str], str] = {}
    component_splits: dict[str, set[str]] = defaultdict(set)
    sequence_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label = row.get("label", "")
        sequence = str(row.get("sequence_201nt", "") or "").upper().replace("U", "T")
        if label not in CLASS_TO_ID:
            errors["unknown_label"] += 1
        if len(sequence) != 201:
            errors["bad_sequence_length"] += 1
        if label in EXPECTED_BASES and (not sequence or sequence[100] != EXPECTED_BASES[label]):
            errors["center_base_mismatch"] += 1
        coord = coordinate_key(row)
        if coord in seen_coords:
            errors["duplicate_coordinate"] += 1
        seen_coords[coord] = label
        component_splits[row.get("paired_cluster_component_id", row["site_id"])].add(row.get("split_cluster201", ""))
        sequence_splits[row.get("sequence_cluster201_id", row["site_id"])].add(row.get("split_cluster201", ""))
        if label == "Unmodified":
            pair = row.get("matched_positive_site_id", "")
            if pair not in positives:
                errors["negative_missing_positive_pair"] += 1
            elif normalize_base(sequence[100:101]) != normalize_base(str(positives[pair].get("sequence_201nt", ""))[100:101]):
                errors["negative_center_base_not_matched"] += 1
            if row.get("split_cluster201") != positives.get(pair, {}).get("split_cluster201"):
                errors["pair_split_leakage"] += 1
    errors["component_split_leakage"] = sum(len(values) > 1 for values in component_splits.values())
    errors["sequence_cluster_split_leakage"] = sum(len(values) > 1 for values in sequence_splits.values())
    summary = {
        "rows": len(rows),
        "class_counts": dict(Counter(row.get("label", "") for row in rows)),
        "split_counts": dict(Counter(row.get("split_cluster201", "") for row in rows)),
        "errors": dict(errors),
        "pass": not errors,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("canonicalize")
    p.add_argument("--source", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=canonicalize)
    p = sub.add_parser("combine")
    p.add_argument("--positives", required=True)
    p.add_argument("--negatives", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=combine)
    p = sub.add_parser("split")
    p.add_argument("--dataset", required=True)
    p.add_argument("--site-to-record", required=True)
    p.add_argument("--cluster-tsv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=split_dataset)
    p = sub.add_parser("audit")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=audit)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
