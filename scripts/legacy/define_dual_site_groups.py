#!/usr/bin/env python
"""Define biologically interpretable m1A-m6A site and transcript groups."""

from __future__ import annotations

import argparse
import bisect
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def site_id(row: Dict[str, str]) -> str:
    return f"{row['transcript_id']}_{int(float(row['tx_pos']))}"


def nearest_distance(position: int, sorted_positions: Sequence[int]) -> Tuple[int | None, int | None]:
    if not sorted_positions:
        return None, None
    idx = bisect.bisect_left(sorted_positions, position)
    candidates = []
    if idx < len(sorted_positions):
        candidates.append(sorted_positions[idx])
    if idx > 0:
        candidates.append(sorted_positions[idx - 1])
    nearest = min(candidates, key=lambda value: abs(value - position))
    return abs(nearest - position), nearest


def classify_m1a_site(region: str, distance: int | None) -> str:
    if distance is None:
        return "m1A_no_m6A_same_transcript"
    if distance == 0:
        return "m1A_exact_overlap_m6A"
    if region == "3UTR":
        if distance <= 25:
            return "m1A_3UTR_prox25"
        if distance > 50:
            return "m1A_3UTR_distal50"
        return "m1A_3UTR_intermediate26_50"
    if region == "CDS":
        if distance <= 25:
            return "m1A_CDS_prox25"
        if distance > 50:
            return "m1A_CDS_distal50"
        return "m1A_CDS_intermediate26_50"
    if distance <= 25:
        return "m1A_other_region_prox25"
    if distance > 50:
        return "m1A_other_region_distal50"
    return "m1A_other_region_intermediate26_50"


def read_unique_rows(path: Path) -> List[Dict[str, str]]:
    rows_by_id: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("transcript_id") and row.get("tx_pos"):
                rows_by_id.setdefault(site_id(row), row)
    return list(rows_by_id.values())


def read_dataset_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def m6a_positions_by_transcript(m6a_rows: Sequence[Dict[str, str]]) -> Dict[str, List[int]]:
    positions: Dict[str, List[int]] = defaultdict(list)
    for row in m6a_rows:
        positions[row["transcript_id"]].append(int(float(row["tx_pos"])))
    for values in positions.values():
        values.sort()
    return positions


def write_site_groups(
    m1a_rows: Sequence[Dict[str, str]],
    m6a_rows: Sequence[Dict[str, str]],
    dataset_rows: Sequence[Dict[str, str]],
    out_path: Path,
    seed: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    m6a_by_tx = m6a_positions_by_transcript(m6a_rows)
    m1a_by_tx = m6a_positions_by_transcript(m1a_rows)
    rows: List[Dict[str, object]] = []

    for row in m1a_rows:
        tx_pos = int(float(row["tx_pos"]))
        distance, nearest = nearest_distance(tx_pos, m6a_by_tx.get(row["transcript_id"], []))
        group = classify_m1a_site(row.get("region_type", ""), distance)
        rows.append(
            {
                "site_id": site_id(row),
                "site_group": group,
                "mod_type": "m1A",
                "transcript_id": row.get("transcript_id", ""),
                "gene_id": row.get("gene_id", ""),
                "gene_name": row.get("gene_name", ""),
                "region_type": row.get("region_type", ""),
                "tx_pos": tx_pos,
                "tx_len": row.get("tx_len", ""),
                "nearest_partner_mod": "m6A" if distance is not None else "",
                "nearest_partner_tx_pos": nearest if nearest is not None else "",
                "nearest_partner_distance": distance if distance is not None else "",
            }
        )

    for row in m6a_rows:
        tx_pos = int(float(row["tx_pos"]))
        distance, nearest = nearest_distance(tx_pos, m1a_by_tx.get(row["transcript_id"], []))
        if distance is None or distance > 25:
            rows.append(
                {
                    "site_id": site_id(row),
                    "site_group": "m6A_only_same_transcript",
                    "mod_type": "m6A",
                    "transcript_id": row.get("transcript_id", ""),
                    "gene_id": row.get("gene_id", ""),
                    "gene_name": row.get("gene_name", ""),
                    "region_type": row.get("region_type", ""),
                    "tx_pos": tx_pos,
                    "tx_len": row.get("tx_len", ""),
                    "nearest_partner_mod": "m1A" if distance is not None else "",
                    "nearest_partner_tx_pos": nearest if nearest is not None else "",
                    "nearest_partner_distance": distance if distance is not None else "",
                }
            )

    target_regions = [row["region_type"] for row in rows if str(row["site_group"]).startswith("m1A_")]
    target_count_by_region = Counter(target_regions)
    negatives_by_region: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in dataset_rows:
        if row.get("label_m1A") == "0" and row.get("label_m6A") == "0":
            negatives_by_region[row.get("region_type", "")].append(row)
    for region, count in target_count_by_region.items():
        candidates = negatives_by_region.get(region, [])
        if not candidates:
            continue
        sample_size = min(count, len(candidates))
        for row in rng.sample(candidates, sample_size):
            rows.append(
                {
                    "site_id": row["site_id"],
                    "site_group": "matched_unmodified_A",
                    "mod_type": "unmodified_A",
                    "transcript_id": row.get("transcript_id", ""),
                    "gene_id": row.get("gene_id", ""),
                    "gene_name": row.get("gene_name", ""),
                    "region_type": row.get("region_type", ""),
                    "tx_pos": int(float(row["tx_pos"])),
                    "tx_len": row.get("tx_len", ""),
                    "nearest_partner_mod": "",
                    "nearest_partner_tx_pos": "",
                    "nearest_partner_distance": "",
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_transcript_membership(site_rows: Sequence[Dict[str, object]], path: Path) -> None:
    groups_by_tx: Dict[str, set[str]] = defaultdict(set)
    meta_by_tx: Dict[str, Dict[str, object]] = {}
    for row in site_rows:
        tx = str(row["transcript_id"])
        groups_by_tx[tx].add(str(row["site_group"]))
        meta_by_tx.setdefault(tx, {"transcript_id": tx, "gene_id": row.get("gene_id", ""), "gene_name": row.get("gene_name", "")})
    labels = sorted({group for groups in groups_by_tx.values() for group in groups})
    fields = ["transcript_id", "gene_id", "gene_name", *labels]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for tx, meta in sorted(meta_by_tx.items()):
            out = dict(meta)
            for label in labels:
                out[label] = int(label in groups_by_tx[tx])
            writer.writerow(out)


def write_qc(site_rows: Sequence[Dict[str, object]], path: Path) -> None:
    by_group: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in site_rows:
        by_group[str(row["site_group"])].append(row)
    fields = ["site_group", "n_sites", "n_transcripts", "median_tx_len", "region_counts", "chromosome_counts"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for group, rows in sorted(by_group.items()):
            tx_lens = sorted(float(row["tx_len"] or 0) for row in rows)
            median = tx_lens[len(tx_lens) // 2] if tx_lens else 0
            writer.writerow(
                {
                    "site_group": group,
                    "n_sites": len(rows),
                    "n_transcripts": len({row["transcript_id"] for row in rows}),
                    "median_tx_len": median,
                    "region_counts": ";".join(f"{k}:{v}" for k, v in Counter(row["region_type"] for row in rows).items()),
                    "chromosome_counts": "",
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Define m1A-m6A biological site groups.")
    parser.add_argument("--m1a", default="04_transcript_space/m1A_refseq_mapped.tsv")
    parser.add_argument("--m6a", default="04_transcript_space/m6A_refseq_mapped.tsv")
    parser.add_argument("--dataset", default="05_datasets/dataset_joint_m1A_m6A_101nt.tsv")
    parser.add_argument("--out-dir", default="11_tables/biology_summary")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    site_rows = write_site_groups(
        read_unique_rows(Path(args.m1a)),
        read_unique_rows(Path(args.m6a)),
        read_dataset_rows(Path(args.dataset)),
        out_dir / "dual_site_groups.tsv",
        args.seed,
    )
    write_transcript_membership(site_rows, out_dir / "transcript_group_membership.tsv")
    write_qc(site_rows, out_dir / "dual_site_group_qc.tsv")
    print(f"wrote {len(site_rows)} rows to {out_dir / 'dual_site_groups.tsv'}")


if __name__ == "__main__":
    main()
