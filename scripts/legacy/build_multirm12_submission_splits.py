#!/usr/bin/env python
"""Build frozen, reviewer-facing splits for the hg38 MultiRM12 study.

The workflow has two phases because CD-HIT-est is an external executable:

1. ``prepare`` removes ambiguous single-label records, emits an 11-class
   confirmatory table, constructs a coordinate-collapsed 12-label table, and
   writes one 201-nt FASTA record per exact sequence.
2. ``finalize`` parses either a CD-HIT ``.clstr`` file or an MMseqs2 cluster
   TSV and adds leakage-free 201-nt sequence-cluster, gene, transcript and
   study holdout assignments.

The confirmatory task excludes m6Am because all currently available m6Am
records overlap Am at the same genomic coordinates.  The 12-label table keeps
these biological co-occurrences as multi-hot targets instead of forcing an
incorrect mutually exclusive label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence


CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "Y", "AtoI"]
CONFIRMATORY_CLASSES = [name for name in CLASS_NAMES if name != "m6Am"]
SPLITS = ("train", "valid", "test")
TOKEN_SPLIT = re.compile(r"[;,|]+")


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def write_tsv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def normalize_sequence(value: str) -> str:
    return str(value or "").strip().upper().replace("U", "T")


def coordinate_key(row: Mapping[str, str]) -> tuple[str, str, str, str]:
    return tuple(str(row.get(field, "")).strip() for field in ("chr", "start", "end", "strand"))  # type: ignore[return-value]


def stable_key(parts: Sequence[str], prefix: str) -> str:
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def split_tokens(value: str) -> list[str]:
    ignored = {"", ".", "-", "na", "nan", "none", "null", "unknown"}
    tokens = []
    seen = set()
    for raw in TOKEN_SPLIT.split(str(value or "")):
        token = raw.strip()
        if token.lower() in ignored or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return sorted(tokens)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def token_component_groups(rows: Sequence[Mapping[str, str]], column: str, prefix: str) -> dict[str, str]:
    """Group rows by connected components of possibly multi-valued IDs."""
    union_find = UnionFind()
    row_tokens: dict[str, list[str]] = {}
    for row in rows:
        site_id = row["site_id"]
        tokens = split_tokens(row.get(column, ""))
        row_tokens[site_id] = tokens
        for token in tokens:
            union_find.add(token)
        for token in tokens[1:]:
            union_find.union(tokens[0], token)

    root_to_id: dict[str, str] = {}
    result: dict[str, str] = {}
    for row in rows:
        site_id = row["site_id"]
        tokens = row_tokens[site_id]
        if not tokens:
            result[site_id] = f"{prefix}_missing_{site_id}"
            continue
        root = union_find.find(tokens[0])
        result[site_id] = root_to_id.setdefault(root, f"{prefix}_{len(root_to_id):09d}")
    return result


def class_vector(row: Mapping[str, str], class_to_id: Mapping[str, int]) -> list[int]:
    vector = [0] * len(class_to_id)
    labels = split_tokens(row.get("label_set", "")) or split_tokens(row.get("mod_type", ""))
    for label in labels:
        if label in class_to_id:
            vector[class_to_id[label]] = 1
    return vector


def stratified_group_split(
    rows: Sequence[Mapping[str, str]],
    site_to_group: Mapping[str, str],
    class_names: Sequence[str],
    seed: int,
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict[str, str]:
    """Greedily allocate whole groups while minimizing class and size error."""
    if len(fractions) != 3 or not math.isclose(sum(fractions), 1.0, abs_tol=1e-8):
        raise ValueError("fractions must contain train/valid/test values summing to one")
    class_to_id = {name: index for index, name in enumerate(class_names)}
    groups: dict[str, list[int]] = {}
    group_sizes: Counter[str] = Counter()
    totals = [0] * len(class_names)
    for row in rows:
        key = site_to_group[row["site_id"]]
        if key not in groups:
            groups[key] = [0] * len(class_names)
        vector = class_vector(row, class_to_id)
        for index, value in enumerate(vector):
            groups[key][index] += value
            totals[index] += value
        group_sizes[key] += 1

    rng = random.Random(seed)
    rarity = [1.0 / max(total, 1) for total in totals]
    ordered = [
        (key, counts, group_sizes[key], sum(count * rarity[index] for index, count in enumerate(counts)), rng.random())
        for key, counts in groups.items()
    ]
    ordered.sort(key=lambda item: (-item[3], -item[2], item[4], item[0]))

    targets = [[fractions[s] * totals[c] for c in range(len(class_names))] for s in range(3)]
    total_rows = sum(group_sizes.values())
    size_targets = [fraction * total_rows for fraction in fractions]
    assigned = [[0] * len(class_names) for _ in range(3)]
    assigned_sizes = [0, 0, 0]
    group_split: dict[str, str] = {}

    def objective(candidate_split: int, counts: Sequence[int], size: int) -> float:
        score = 0.0
        for split_index in range(3):
            for class_index, total in enumerate(totals):
                value = assigned[split_index][class_index]
                if split_index == candidate_split:
                    value += counts[class_index]
                scale = max(total, 1)
                score += ((value - targets[split_index][class_index]) / scale) ** 2
            split_size = assigned_sizes[split_index] + (size if split_index == candidate_split else 0)
            score += 0.2 * ((split_size - size_targets[split_index]) / max(total_rows, 1)) ** 2
        return score

    for key, counts, size, _, _ in ordered:
        chosen = min(range(3), key=lambda split_index: (objective(split_index, counts, size), split_index))
        group_split[key] = SPLITS[chosen]
        assigned_sizes[chosen] += size
        for index, count in enumerate(counts):
            assigned[chosen][index] += count
    return {row["site_id"]: group_split[site_to_group[row["site_id"]]] for row in rows}


def parse_cdhit_clusters(path: Path) -> dict[str, str]:
    record_to_cluster: dict[str, str] = {}
    current = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">Cluster"):
                current = f"cdhit201_{int(line.split()[-1]):09d}"
            elif current and ">" in line:
                record = line.split(">", 1)[1].split("...", 1)[0].strip()
                record_to_cluster[record] = current
    return record_to_cluster


def parse_mmseqs_clusters(path: Path) -> dict[str, str]:
    """Map every MMseqs2 cluster member to a stable cluster identifier."""
    record_to_cluster: dict[str, str] = {}
    representative_to_cluster: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) < 2:
                continue
            representative, member = fields[0].strip(), fields[1].strip()
            if not representative or not member:
                continue
            cluster = representative_to_cluster.setdefault(
                representative,
                f"mmseqs201_{len(representative_to_cluster):09d}",
            )
            prior = record_to_cluster.setdefault(member, cluster)
            if prior != cluster:
                raise ValueError(f"MMseqs2 member {member} occurs in multiple clusters")
    return record_to_cluster


def leakage_audit(
    rows: Sequence[Mapping[str, str]],
    site_to_group: Mapping[str, str],
    split_by_site: Mapping[str, str],
) -> dict[str, object]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        site_id = row["site_id"]
        group_splits[site_to_group[site_id]].add(split_by_site[site_id])
    leaked = [group for group, values in group_splits.items() if len(values) > 1]
    sizes = Counter(site_to_group[row["site_id"]] for row in rows)
    return {
        "groups": len(group_splits),
        "leaked_groups": len(leaked),
        "largest_group_rows": max(sizes.values(), default=0),
        "singleton_groups": sum(value == 1 for value in sizes.values()),
        "example_leaked_groups": leaked[:10],
    }


def summarize_split(rows: Sequence[Mapping[str, str]], split_column: str, class_names: Sequence[str]) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        labels = split_tokens(row.get("label_set", "")) or split_tokens(row.get("mod_type", ""))
        for label in labels:
            counts[(label, row.get(split_column, ""))] += 1
    return [
        {"split_column": split_column, "mod_type": name, **{split: counts[(name, split)] for split in SPLITS}}
        for name in class_names
    ]


def prepare(args: argparse.Namespace) -> None:
    rows, fields = read_tsv(args.dataset)
    required = {"site_id", "mod_type", "chr", "start", "end", "strand", "sequence_41nt", "sequence_201nt"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    coordinate_labels: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    sequence_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        coordinate_labels[coordinate_key(row)].add(row["mod_type"])
        sequence_labels[normalize_sequence(row["sequence_201nt"])].add(row["mod_type"])
    coordinate_conflicts = {key for key, values in coordinate_labels.items() if len(values) > 1}
    sequence_conflicts = {key for key, values in sequence_labels.items() if len(values) > 1}

    excluded_rows: list[dict[str, object]] = []
    confirmatory: list[dict[str, str]] = []
    seen_coordinate_label: set[tuple[tuple[str, str, str, str], str]] = set()
    for row in rows:
        reasons = []
        if row["mod_type"] == "m6Am":
            reasons.append("m6Am_reserved_for_multilabel_supplement")
        if coordinate_key(row) in coordinate_conflicts:
            reasons.append("coordinate_has_multiple_labels")
        if normalize_sequence(row["sequence_201nt"]) in sequence_conflicts:
            reasons.append("identical_201nt_sequence_has_multiple_labels")
        duplicate_key = (coordinate_key(row), row["mod_type"])
        if duplicate_key in seen_coordinate_label:
            reasons.append("duplicate_coordinate_label")
        if reasons:
            excluded_rows.append({"site_id": row["site_id"], "mod_type": row["mod_type"], "reasons": ";".join(reasons)})
            continue
        seen_coordinate_label.add(duplicate_key)
        copied = dict(row)
        copied["class_index"] = str(CONFIRMATORY_CLASSES.index(row["mod_type"]))
        confirmatory.append(copied)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    confirmatory_path = args.output_dir / "multirm11_hg38_conflict_clean_41_101_201nt.tsv"
    write_tsv(confirmatory_path, confirmatory, fields)
    write_tsv(args.output_dir / "excluded_single_label_records.tsv", excluded_rows, ["site_id", "mod_type", "reasons"])

    coordinate_rows: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        coordinate_rows[coordinate_key(row)].append(row)
    multilabel_rows: list[dict[str, object]] = []
    multilabel_fields = list(fields) + ["source_site_id", "label_set", "label_count"] + [f"label_{name}" for name in CLASS_NAMES]
    for key, members in coordinate_rows.items():
        representative = max(members, key=lambda item: float(item.get("support_num", "0") or 0))
        merged: dict[str, object] = dict(representative)
        merged["source_site_id"] = representative["site_id"]
        labels = sorted({member["mod_type"] for member in members}, key=CLASS_NAMES.index)
        merged["site_id"] = stable_key(list(key), "coord")
        merged["mod_type"] = labels[0] if len(labels) == 1 else "|".join(labels)
        merged["class_index"] = ""
        merged["label_set"] = "|".join(labels)
        merged["label_count"] = len(labels)
        for name in CLASS_NAMES:
            merged[f"label_{name}"] = int(name in labels)
        multilabel_rows.append(merged)
    multilabel_path = args.output_dir / "multirm12_hg38_coordinate_multilabel_41_101_201nt.tsv"
    write_tsv(multilabel_path, multilabel_rows, multilabel_fields)

    sequence_to_record: dict[str, str] = {}
    site_record_rows = []
    multilabel_record_rows = []

    def record_for(sequence_value: object) -> str:
        sequence = normalize_sequence(str(sequence_value or ""))
        record = sequence_to_record.get(sequence)
        if record is None:
            record = f"seq201_{len(sequence_to_record):09d}"
            sequence_to_record[sequence] = record
        return record

    for row in confirmatory:
        site_record_rows.append({"site_id": row["site_id"], "cdhit_record_id": record_for(row["sequence_201nt"])})
    for row in multilabel_rows:
        multilabel_record_rows.append(
            {"site_id": row["site_id"], "cdhit_record_id": record_for(row["sequence_201nt"])}
        )

    fasta_path = args.output_dir / "cdhit201_unique_sequences.fa"
    with fasta_path.open("w", encoding="ascii", newline="\n") as handle:
        for sequence, record in sequence_to_record.items():
            handle.write(f">{record}\n{sequence}\n")
    write_tsv(args.output_dir / "site_to_cdhit201_record.tsv", site_record_rows, ["site_id", "cdhit_record_id"])
    write_tsv(
        args.output_dir / "multilabel_to_cdhit201_record.tsv",
        multilabel_record_rows,
        ["site_id", "cdhit_record_id"],
    )

    reason_counts = Counter(reason for row in excluded_rows for reason in str(row["reasons"]).split(";"))
    summary = {
        "phase": "prepare",
        "input_dataset": str(args.dataset),
        "input_rows": len(rows),
        "input_class_counts": dict(Counter(row["mod_type"] for row in rows)),
        "coordinate_conflict_keys": len(coordinate_conflicts),
        "identical_201nt_sequence_conflict_keys": len(sequence_conflicts),
        "confirmatory_classes": CONFIRMATORY_CLASSES,
        "confirmatory_rows": len(confirmatory),
        "confirmatory_class_counts": dict(Counter(row["mod_type"] for row in confirmatory)),
        "excluded_rows": len(excluded_rows),
        "exclusion_reason_occurrences": dict(reason_counts),
        "multilabel_coordinate_rows": len(multilabel_rows),
        "multilabel_rows_with_multiple_labels": sum(int(row["label_count"]) > 1 for row in multilabel_rows),
        "unique_confirmatory_201nt_sequences": len({row["cdhit_record_id"] for row in site_record_rows}),
        "unique_multilabel_201nt_sequences": len({row["cdhit_record_id"] for row in multilabel_record_rows}),
        "unique_union_201nt_sequences": len(sequence_to_record),
        "recommended_cdhit_command": (
            "cd-hit-est -i cdhit201_unique_sequences.fa -o cdhit201_c080 "
            "-c 0.80 -n 5 -G 1 -aS 0.80 -aL 0.80 -d 0 -T 32 -M 0"
        ),
        "recommended_mmseqs_command": (
            "mmseqs easy-linclust cdhit201_unique_sequences.fa mmseqs201_c080 mmseqs201_tmp "
            "--min-seq-id 0.80 -c 0.80 --cov-mode 0 --cluster-mode 0"
        ),
    }
    (args.output_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def finalize(args: argparse.Namespace) -> None:
    input_path = args.output_dir / "multirm11_hg38_conflict_clean_41_101_201nt.tsv"
    rows, fields = read_tsv(input_path)
    multilabel_input = args.output_dir / "multirm12_hg38_coordinate_multilabel_41_101_201nt.tsv"
    multilabel_rows, multilabel_fields = read_tsv(multilabel_input)
    record_rows, _ = read_tsv(args.output_dir / "site_to_cdhit201_record.tsv")
    multilabel_record_rows, _ = read_tsv(args.output_dir / "multilabel_to_cdhit201_record.tsv")
    site_to_record = {row["site_id"]: row["cdhit_record_id"] for row in record_rows}
    multilabel_site_to_record = {row["site_id"]: row["cdhit_record_id"] for row in multilabel_record_rows}
    if args.mmseqs_cluster_tsv is not None:
        cluster_path = args.mmseqs_cluster_tsv
        cluster_algorithm = "MMseqs2 easy-linclust"
        record_to_cluster = parse_mmseqs_clusters(cluster_path)
    else:
        cluster_path = args.cdhit_clstr
        cluster_algorithm = "CD-HIT-EST"
        if cluster_path is None:
            raise ValueError("A CD-HIT or MMseqs2 cluster file is required")
        record_to_cluster = parse_cdhit_clusters(cluster_path)
    expected_records = set(site_to_record.values()) | set(multilabel_site_to_record.values())
    missing_records = sorted(expected_records - set(record_to_cluster))
    if missing_records:
        raise ValueError(f"CD-HIT output is missing {len(missing_records)} records; examples: {missing_records[:5]}")

    group_maps = {
        "cluster201_id": {site: record_to_cluster[record] for site, record in site_to_record.items()},
        "gene_component_id": token_component_groups(rows, "gene_id", "gene"),
        "transcript_component_id": token_component_groups(rows, "transcript_id", "tx"),
        "study_component_id": token_component_groups(rows, "pub_list", "study"),
    }
    split_specs = {
        "split_cluster201": "cluster201_id",
        "split_gene_holdout": "gene_component_id",
        "split_transcript_holdout": "transcript_component_id",
        "split_study_holdout": "study_component_id",
    }
    split_maps: dict[str, dict[str, str]] = {}
    for split_column, group_column in split_specs.items():
        split_maps[split_column] = stratified_group_split(
            rows,
            group_maps[group_column],
            CONFIRMATORY_CLASSES,
            args.seed,
            (args.train_fraction, args.valid_fraction, 1.0 - args.train_fraction - args.valid_fraction),
        )

    for row in rows:
        site_id = row["site_id"]
        for column, mapping in group_maps.items():
            row[column] = mapping[site_id]
        for column, mapping in split_maps.items():
            row[column] = mapping[site_id]
        row["split_lockbox"] = split_maps["split_study_holdout"][site_id]
        row["is_frozen_lockbox"] = "1" if row["split_lockbox"] == "test" else "0"

    output_fields = list(fields)
    for field in [*group_maps, *split_specs, "split_lockbox", "is_frozen_lockbox"]:
        if field not in output_fields:
            output_fields.append(field)
    final_path = args.output_dir / "multirm11_hg38_conflict_clean_strict_splits.tsv"
    write_tsv(final_path, rows, output_fields)

    multilabel_group_maps = {
        "cluster201_id": {
            site: record_to_cluster[record] for site, record in multilabel_site_to_record.items()
        },
        "gene_component_id": token_component_groups(multilabel_rows, "gene_id", "ml_gene"),
        "transcript_component_id": token_component_groups(multilabel_rows, "transcript_id", "ml_tx"),
        "study_component_id": token_component_groups(multilabel_rows, "pub_list", "ml_study"),
    }
    multilabel_split_maps: dict[str, dict[str, str]] = {}
    for split_column, group_column in split_specs.items():
        multilabel_split_maps[split_column] = stratified_group_split(
            multilabel_rows,
            multilabel_group_maps[group_column],
            CLASS_NAMES,
            args.seed,
            (args.train_fraction, args.valid_fraction, 1.0 - args.train_fraction - args.valid_fraction),
        )
    for row in multilabel_rows:
        site_id = row["site_id"]
        for column, mapping in multilabel_group_maps.items():
            row[column] = mapping[site_id]
        for column, mapping in multilabel_split_maps.items():
            row[column] = mapping[site_id]
        row["split_lockbox"] = multilabel_split_maps["split_study_holdout"][site_id]
        row["is_frozen_lockbox"] = "1" if row["split_lockbox"] == "test" else "0"
    multilabel_output_fields = list(multilabel_fields)
    for field in [*multilabel_group_maps, *split_specs, "split_lockbox", "is_frozen_lockbox"]:
        if field not in multilabel_output_fields:
            multilabel_output_fields.append(field)
    multilabel_final_path = args.output_dir / "multirm12_hg38_coordinate_multilabel_strict_splits.tsv"
    write_tsv(multilabel_final_path, multilabel_rows, multilabel_output_fields)

    split_summary_rows = []
    leakage = {}
    for split_column, group_column in split_specs.items():
        split_summary_rows.extend(summarize_split(rows, split_column, CONFIRMATORY_CLASSES))
        leakage[split_column] = leakage_audit(rows, group_maps[group_column], split_maps[split_column])
    write_tsv(
        args.output_dir / "strict_split_class_support.tsv",
        split_summary_rows,
        ["split_column", "mod_type", "train", "valid", "test"],
    )
    multilabel_split_summary_rows = []
    multilabel_leakage = {}
    for split_column, group_column in split_specs.items():
        multilabel_split_summary_rows.extend(summarize_split(multilabel_rows, split_column, CLASS_NAMES))
        multilabel_leakage[split_column] = leakage_audit(
            multilabel_rows,
            multilabel_group_maps[group_column],
            multilabel_split_maps[split_column],
        )
    write_tsv(
        args.output_dir / "multilabel_strict_split_class_support.tsv",
        multilabel_split_summary_rows,
        ["split_column", "mod_type", "train", "valid", "test"],
    )
    lockbox_counts = Counter(row["mod_type"] for row in rows if row["split_lockbox"] == "test")
    lockbox_valid_counts = Counter(row["mod_type"] for row in rows if row["split_lockbox"] == "valid")
    lockbox_evaluable = [
        name
        for name in CONFIRMATORY_CLASSES
        if lockbox_valid_counts.get(name, 0) > 0 and lockbox_counts.get(name, 0) > 0
    ]
    multilabel_lockbox_counts: Counter[str] = Counter()
    for row in multilabel_rows:
        if row["split_lockbox"] == "test":
            multilabel_lockbox_counts.update(split_tokens(row.get("label_set", "")))
    summary = {
        "phase": "finalize",
        "dataset": str(final_path),
        "rows": len(rows),
        "seed": args.seed,
        "fractions": {
            "train": args.train_fraction,
            "valid": args.valid_fraction,
            "test_or_lockbox": 1.0 - args.train_fraction - args.valid_fraction,
        },
        "class_names": CONFIRMATORY_CLASSES,
        "clustering": {
            "algorithm": cluster_algorithm,
            "sequence_length_nt": 201,
            "identity": 0.80,
            "coverage_mode": "both query and target",
            "minimum_coverage_query": 0.80,
            "minimum_coverage_target": 0.80,
            "cluster_file": str(cluster_path),
            "records": len(record_to_cluster),
            "clusters": len(set(record_to_cluster.values())),
        },
        "leakage_audits": leakage,
        "lockbox_definition": "Unseen connected components of normalized RMBase pub_list study identifiers",
        "lockbox_rows": sum(row["split_lockbox"] == "test" for row in rows),
        "lockbox_class_counts": dict(lockbox_counts),
        "lockbox_evaluable_classes": lockbox_evaluable,
        "lockbox_missing_classes": [name for name in CONFIRMATORY_CLASSES if name not in lockbox_evaluable],
        "lockbox_limitation": (
            "The publication graph contains a giant connected component. The zero-leakage study lockbox therefore "
            "supports only classes represented in independent study components; unsupported classes are not scored."
        ),
        "multilabel": {
            "dataset": str(multilabel_final_path),
            "rows": len(multilabel_rows),
            "classes": CLASS_NAMES,
            "leakage_audits": multilabel_leakage,
            "lockbox_rows": sum(row["split_lockbox"] == "test" for row in multilabel_rows),
            "lockbox_class_counts": dict(multilabel_lockbox_counts),
        },
    }
    (args.output_dir / "strict_split_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dataset", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.set_defaults(func=prepare)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    cluster_group = finalize_parser.add_mutually_exclusive_group(required=True)
    cluster_group.add_argument("--cdhit-clstr", type=Path)
    cluster_group.add_argument("--mmseqs-cluster-tsv", type=Path)
    finalize_parser.add_argument("--seed", type=int, default=42)
    finalize_parser.add_argument("--train-fraction", type=float, default=0.8)
    finalize_parser.add_argument("--valid-fraction", type=float, default=0.1)
    finalize_parser.set_defaults(func=finalize)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
