#!/usr/bin/env python
"""Prepare and finalize strict splits for modified-vs-unannotated detection.

The detection split groups both 201-nt sequence clusters and matched
positive/negative pairs. Consequently, neither a homologous sequence cluster
nor the opposite member of a matched pair can cross train/validation/test.
Positive rows receive the same new assignments in a separate 11-class table so
the second-stage modification classifier can be retrained without leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from build_multirm12_submission_splits import (
    UnionFind,
    leakage_audit,
    parse_mmseqs_clusters,
    stratified_group_split,
    token_component_groups,
)


MODIFICATIONS = ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m7G", "Y", "AtoI"]
DETECTION_CLASSES = ["unmodified", "modified"]
SPLITS = ["train", "valid", "test"]
SOURCE_SPLITS = [
    "split_cluster201",
    "split_gene_holdout",
    "split_transcript_holdout",
    "split_study_holdout",
    "split_lockbox",
    "is_frozen_lockbox",
]
NEW_SPLITS = [
    "split_detection_cluster201",
    "split_detection_gene_holdout",
    "split_detection_transcript_holdout",
    "split_detection_study_holdout",
]


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def normalized_sequence(value: object) -> str:
    return str(value or "").strip().upper().replace("U", "T")


def detection_stratum(row: Mapping[str, str]) -> str:
    return f"{row['detection_class']}:{row['original_mod_type']}"


def prepare(args: argparse.Namespace) -> None:
    positives, positive_fields = read_tsv(args.positives)
    negatives, negative_fields = read_tsv(args.negatives)
    positive_by_id = {row["site_id"]: row for row in positives}
    negative_by_positive: dict[str, dict[str, str]] = {}
    for row in negatives:
        pair_id = row["matched_positive_site_id"]
        if pair_id in negative_by_positive:
            raise ValueError(f"Positive {pair_id} has more than one matched negative")
        if pair_id not in positive_by_id:
            raise ValueError(f"Negative {row['site_id']} has no positive pair")
        negative_by_positive[pair_id] = row
    rows: list[dict[str, object]] = []
    for source in positives:
        negative = negative_by_positive.get(source["site_id"])
        if negative is None:
            continue
        row: dict[str, object] = dict(source)
        row.update(
            {
                "mod_type": "modified",
                "class_index": "1",
                "det_label": "1",
                "detection_class": "modified",
                "original_mod_type": source["mod_type"],
                "matched_positive_site_id": source["site_id"],
                "negative_status": "experimentally_annotated_positive",
                "match_level": "positive",
            }
        )
        for column in SOURCE_SPLITS:
            row[f"source_{column}"] = source.get(column, "")
        rows.append(row)
        row = dict(negative)
        row.update(
            {
                "mod_type": "unmodified",
                "class_index": "0",
                "det_label": "0",
                "detection_class": "unmodified",
                "original_mod_type": negative["matched_mod_type"],
            }
        )
        for column in SOURCE_SPLITS:
            row[f"source_{column}"] = source.get(column, "")
        rows.append(row)

    fields = list(dict.fromkeys(
        [
            "site_id", "mod_type", "class_index", "det_label", "detection_class",
            "original_mod_type", "matched_positive_site_id", "negative_status", "match_level",
            "chr", "start", "end", "strand", "support_num", "support_list", "pub_list",
            "cell_list", "seq_type_list", "gene_id", "transcript_id", "gene_name", "gene_type",
            "region_type", "sequence_41nt", "sequence_101nt", "sequence_201nt", "center_base",
            "source_file", "gc_fraction_201nt", "distance_to_nearest_known_modification",
            *[f"source_{column}" for column in SOURCE_SPLITS],
        ]
    ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_path = args.output_dir / "detection_candidates_precluster.tsv"
    write_tsv(stage_path, rows, fields)

    sequence_to_record: dict[str, str] = {}
    site_to_record: list[dict[str, str]] = []
    for row in rows:
        sequence = normalized_sequence(row["sequence_201nt"])
        if len(sequence) != 201:
            raise ValueError(f"Invalid 201-nt sequence for {row['site_id']}")
        record = sequence_to_record.get(sequence)
        if record is None:
            record = f"detseq_{len(sequence_to_record):09d}"
            sequence_to_record[sequence] = record
        site_to_record.append({"site_id": str(row["site_id"]), "record_id": record})
    with (args.output_dir / "detection_unique_201nt.fa").open("w", encoding="ascii", newline="\n") as handle:
        for sequence, record in sequence_to_record.items():
            handle.write(f">{record}\n{sequence}\n")
    write_tsv(args.output_dir / "detection_site_to_record.tsv", site_to_record, ["site_id", "record_id"])
    summary = {
        "phase": "prepare",
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "matched_positive_rows": len(negative_by_positive),
        "unmatched_positive_rows_excluded": len(positives) - len(negative_by_positive),
        "combined_rows": len(rows),
        "unique_201nt_sequences": len(sequence_to_record),
        "detection_counts": dict(Counter(str(row["detection_class"]) for row in rows)),
        "stratum_counts": dict(Counter(detection_stratum(row) for row in rows)),
        "mmseqs_command": (
            "mmseqs easy-linclust detection_unique_201nt.fa detection201_c080 detection201_tmp "
            "--min-seq-id 0.80 -c 0.80 --cov-mode 0 --cluster-mode 0"
        ),
    }
    (args.output_dir / "detection_prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def stratified_detection_split(
    rows: Sequence[Mapping[str, str]],
    groups: Mapping[str, str],
    seed: int,
) -> dict[str, str]:
    copied = []
    strata = []
    for row in rows:
        item = dict(row)
        item["mod_type"] = detection_stratum(row)
        copied.append(item)
        strata.append(item["mod_type"])
    return stratified_group_split(copied, groups, sorted(set(strata)), seed, (0.8, 0.1, 0.1))


def paired_cluster_components(
    rows: Sequence[Mapping[str, str]],
    cluster_by_site: Mapping[str, str],
) -> dict[str, str]:
    union = UnionFind()
    pair_clusters: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cluster = cluster_by_site[row["site_id"]]
        union.add(cluster)
        pair_clusters[row["matched_positive_site_id"]].add(cluster)
    for clusters in pair_clusters.values():
        ordered = sorted(clusters)
        for cluster in ordered[1:]:
            union.union(ordered[0], cluster)
    root_to_group: dict[str, str] = {}
    return {
        row["site_id"]: root_to_group.setdefault(
            union.find(cluster_by_site[row["site_id"]]),
            f"detgroup_{len(root_to_group):09d}",
        )
        for row in rows
    }


def merged_site_components(
    rows: Sequence[Mapping[str, str]],
    mappings: Sequence[Mapping[str, str]],
    prefix: str,
) -> dict[str, str]:
    """Merge several site-level grouping constraints into components."""
    union = UnionFind()
    for row in rows:
        site_id = row["site_id"]
        groups = [mapping[site_id] for mapping in mappings]
        for group in groups:
            union.add(group)
        for group in groups[1:]:
            union.union(groups[0], group)
    root_to_group: dict[str, str] = {}
    return {
        row["site_id"]: root_to_group.setdefault(
            union.find(mappings[0][row["site_id"]]),
            f"{prefix}_{len(root_to_group):09d}",
        )
        for row in rows
    }


def support_rows(rows: Sequence[Mapping[str, str]], split_column: str) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        counts[(row["detection_class"], row["original_mod_type"], row[split_column])] += 1
    output = []
    for detection_class in DETECTION_CLASSES:
        for mod_type in MODIFICATIONS:
            output.append(
                {
                    "split_column": split_column,
                    "detection_class": detection_class,
                    "original_mod_type": mod_type,
                    **{split: counts[(detection_class, mod_type, split)] for split in SPLITS},
                }
            )
    return output


def pair_leakage(rows: Sequence[Mapping[str, str]], split_column: str) -> dict[str, object]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        values[row["matched_positive_site_id"]].add(row[split_column])
    leaked = [pair for pair, splits in values.items() if len(splits) > 1]
    return {"pairs": len(values), "leaked_pairs": len(leaked), "examples": leaked[:10]}


def finalize(args: argparse.Namespace) -> None:
    rows, fields = read_tsv(args.output_dir / "detection_candidates_precluster.tsv")
    site_records, _ = read_tsv(args.output_dir / "detection_site_to_record.tsv")
    site_to_record = {row["site_id"]: row["record_id"] for row in site_records}
    record_to_cluster = parse_mmseqs_clusters(args.mmseqs_cluster_tsv)
    missing = sorted(set(site_to_record.values()) - set(record_to_cluster))
    if missing:
        raise ValueError(f"MMseqs output misses {len(missing)} records; examples={missing[:5]}")
    cluster_by_site = {site: record_to_cluster[record] for site, record in site_to_record.items()}
    detection_groups = paired_cluster_components(rows, cluster_by_site)
    gene_tokens = token_component_groups(rows, "gene_id", "det_gene_token")
    transcript_tokens = token_component_groups(rows, "transcript_id", "det_tx_token")
    study_tokens = token_component_groups(rows, "pub_list", "det_study_token")
    # Every endpoint inherits the sequence-cluster and matched-pair constraints.
    # The additional metadata mapping then creates the requested holdout unit.
    gene_groups = merged_site_components(rows, [detection_groups, gene_tokens], "det_gene")
    transcript_groups = merged_site_components(rows, [detection_groups, transcript_tokens], "det_tx")
    study_groups = merged_site_components(rows, [detection_groups, study_tokens], "det_study")
    group_maps = {
        "detection_cluster201_id": detection_groups,
        "detection_gene_component_id": gene_groups,
        "detection_transcript_component_id": transcript_groups,
        "detection_study_component_id": study_groups,
    }
    split_to_group = {
        "split_detection_cluster201": "detection_cluster201_id",
        "split_detection_gene_holdout": "detection_gene_component_id",
        "split_detection_transcript_holdout": "detection_transcript_component_id",
        "split_detection_study_holdout": "detection_study_component_id",
    }
    split_maps = {
        split: stratified_detection_split(rows, group_maps[group], args.seed)
        for split, group in split_to_group.items()
    }
    for row in rows:
        site_id = row["site_id"]
        row["detection_sequence_cluster_id"] = cluster_by_site[site_id]
        for column, mapping in group_maps.items():
            row[column] = mapping[site_id]
        for column, mapping in split_maps.items():
            row[column] = mapping[site_id]
        row["split_detection_lockbox"] = row["split_detection_study_holdout"]
        row["is_detection_lockbox"] = "1" if row["split_detection_lockbox"] == "test" else "0"
    output_fields = list(fields)
    for column in [
        "detection_sequence_cluster_id", *group_maps, *NEW_SPLITS,
        "split_detection_lockbox", "is_detection_lockbox",
    ]:
        if column not in output_fields:
            output_fields.append(column)
    final_path = args.output_dir / "hg38_modified_vs_unannotated_strict_splits.tsv"
    write_tsv(final_path, rows, output_fields)

    positive_source, positive_fields = read_tsv(args.positives)
    assignment = {
        row["site_id"]: {column: row[column] for column in [*NEW_SPLITS, "split_detection_lockbox", "is_detection_lockbox"]}
        for row in rows
        if row["detection_class"] == "modified"
    }
    positive_output = []
    for row in positive_source:
        if row["site_id"] not in assignment:
            continue
        copied = dict(row)
        copied.update(assignment[row["site_id"]])
        positive_output.append(copied)
    positive_output_fields = list(positive_fields)
    for column in [*NEW_SPLITS, "split_detection_lockbox", "is_detection_lockbox"]:
        if column not in positive_output_fields:
            positive_output_fields.append(column)
    write_tsv(
        args.output_dir / "multirm11_detection_aligned_positive_strict_splits.tsv",
        positive_output,
        positive_output_fields,
    )

    support = []
    leakage = {}
    for split_column, group_column in split_to_group.items():
        support.extend(support_rows(rows, split_column))
        leakage[split_column] = {
            "component": leakage_audit(rows, group_maps[group_column], split_maps[split_column]),
            "sequence_cluster": leakage_audit(rows, cluster_by_site, split_maps[split_column]),
            "pair": pair_leakage(rows, split_column),
        }
    write_tsv(
        args.output_dir / "detection_split_support.tsv",
        support,
        ["split_column", "detection_class", "original_mod_type", *SPLITS],
    )
    exact_sequence_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        exact_sequence_labels[normalized_sequence(row["sequence_201nt"])].add(row["detection_class"])
    summary = {
        "phase": "finalize",
        "rows": len(rows),
        "positive_rows": sum(row["detection_class"] == "modified" for row in rows),
        "negative_rows": sum(row["detection_class"] == "unmodified" for row in rows),
        "mmseqs_cluster_file": str(args.mmseqs_cluster_tsv),
        "sequence_clusters": len(set(cluster_by_site.values())),
        "paired_sequence_components": len(set(detection_groups.values())),
        "exact_201nt_sequences_with_both_labels": sum(len(labels) > 1 for labels in exact_sequence_labels.values()),
        "leakage_audits": leakage,
        "seed": args.seed,
    }
    failures = []
    for split, audit in leakage.items():
        if audit["component"]["leaked_groups"] or audit["sequence_cluster"]["leaked_groups"] or audit["pair"]["leaked_pairs"]:
            failures.append(split)
    summary["overall_status"] = "PASS" if not failures else "FAIL"
    summary["failed_splits"] = failures
    (args.output_dir / "detection_split_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--positives", type=Path, required=True)
    prepare_parser.add_argument("--negatives", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--positives", type=Path, required=True)
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    finalize_parser.add_argument("--mmseqs-cluster-tsv", type=Path, required=True)
    finalize_parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
