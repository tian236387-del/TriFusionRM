#!/usr/bin/env python
"""Run public m6A-reader annotation enrichment for dual-site transcript groups."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


KNOWN_M6A_READERS = {
    "YTHDF1",
    "YTHDF2",
    "YTHDF3",
    "YTHDC1",
    "YTHDC2",
    "IGF2BP1",
    "IGF2BP2",
    "IGF2BP3",
    "HNRNPC",
    "HNRNPA2B1",
    "EIF3A",
    "EIF3B",
    "ELAVL1",
    "FMR1",
    "LRPPRC",
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_table(rows: Sequence[Dict[str, object]], path: Path, fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_symbol(value: str) -> str:
    return value.strip().upper()


def normalize_reader_name(value: str) -> str:
    tokens = []
    for raw in value.replace("|", ";").replace(",", ";").replace("/", ";").split(";"):
        token = raw.strip().upper().replace("-", "")
        if token:
            tokens.append(token)
    known = [token for token in tokens if token in KNOWN_M6A_READERS]
    return ";".join(dict.fromkeys(known))


def row_value(row: Dict[str, str], candidates: Sequence[str]) -> str:
    for key in candidates:
        if key in row and row[key] not in {"", None}:
            return str(row[key])
    lower_map = {key.lower(): key for key in row}
    for key in candidates:
        mapped = lower_map.get(key.lower())
        if mapped and row.get(mapped):
            return str(row[mapped])
    return ""


def mod_is_m6a(row: Dict[str, str]) -> bool:
    value = row_value(row, ["RNAmodType", "mod_type", "modType", "mod", "modification"])
    return "M6A" in value.upper().replace("-", "")


def aggregate_reader_annotations(rows: Sequence[Dict[str, str]], key_field: str = "gene_name") -> Dict[str, Dict[str, object]]:
    by_key: Dict[str, Dict[str, object]] = {}
    for row in rows:
        if not mod_is_m6a(row):
            continue
        reader_names = normalize_reader_name(row_value(row, ["rbpName", "RBP", "rbp", "rbp_name", "protein"]))
        if not reader_names:
            continue
        entity = normalize_symbol(row_value(row, [key_field, "gene_name", "geneName", "Gene", "gene", "transcript_id", "transcriptID"]))
        if not entity:
            continue
        current = by_key.setdefault(entity, {"reader_site_count": 0, "unique_reader_count": 0, "readers": ""})
        current["reader_site_count"] = int(current["reader_site_count"]) + 1
        reader_set = set(str(current["readers"]).split(";")) if current["readers"] else set()
        reader_set.update(reader_names.split(";"))
        current["readers"] = ";".join(sorted(reader for reader in reader_set if reader))
        current["unique_reader_count"] = len(reader_set)
    return by_key


def truthy(value: object) -> bool:
    return str(value).strip() in {"1", "true", "True", "yes", "Y"}


def collapse_memberships(rows: Sequence[Dict[str, str]], key_field: str) -> List[Dict[str, str]]:
    collapsed: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = normalize_symbol(row_value(row, [key_field, "gene_name", "geneName", "transcript_id", "transcriptID"]))
        if not key:
            continue
        out = collapsed.setdefault(key, {key_field: key})
        for col, value in row.items():
            if col == key_field:
                continue
            if truthy(value):
                out[col] = "1"
            elif col not in out:
                out[col] = value
    return list(collapsed.values())


def fisher_exact(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
    try:
        from scipy.stats import fisher_exact as scipy_fisher

        odds, p_value = scipy_fisher([[a, b], [c, d]])
        return float(odds), float(p_value)
    except Exception:
        odds = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        return odds, float("nan")


def bh_fdr(p_values: Sequence[float]) -> List[float]:
    valid = [(idx, p) for idx, p in enumerate(p_values) if not math.isnan(p)]
    ranked = sorted(valid, key=lambda item: item[1])
    adjusted = [float("nan")] * len(p_values)
    running = 1.0
    total = len(ranked)
    for rank, (idx, p_value) in reversed(list(enumerate(ranked, start=1))):
        running = min(running, p_value * total / rank)
        adjusted[idx] = running
    return adjusted


def entities_for_group(rows: Sequence[Dict[str, str]], group: str, key_field: str) -> set[str]:
    return {normalize_symbol(row.get(key_field, "")) for row in rows if truthy(row.get(group)) and row.get(key_field)}


def compare_reader_enrichment(
    memberships: Sequence[Dict[str, str]],
    reader_summary: Dict[str, Dict[str, object]],
    comparisons: Sequence[Tuple[str, str]],
    key_field: str = "gene_name",
) -> List[Dict[str, object]]:
    membership_rows = collapse_memberships(memberships, key_field)
    reader_positive = {key for key, value in reader_summary.items() if int(value.get("reader_site_count", 0)) > 0}
    rows = []
    for left, right in comparisons:
        left_entities = entities_for_group(membership_rows, left, key_field)
        right_entities = entities_for_group(membership_rows, right, key_field) - left_entities
        left_pos = len(left_entities & reader_positive)
        right_pos = len(right_entities & reader_positive)
        left_total = len(left_entities)
        right_total = len(right_entities)
        left_neg = left_total - left_pos
        right_neg = right_total - right_pos
        odds, p_value = fisher_exact(left_pos, left_neg, right_pos, right_neg)
        left_frac = left_pos / left_total if left_total else 0.0
        right_frac = right_pos / right_total if right_total else 0.0
        rows.append(
            {
                "comparison": f"{left} vs {right}",
                "left_group": left,
                "right_group": right,
                "left_total": left_total,
                "right_total": right_total,
                "left_reader_positive": left_pos,
                "right_reader_positive": right_pos,
                "left_reader_fraction": left_frac,
                "right_reader_fraction": right_frac,
                "risk_difference": left_frac - right_frac,
                "odds_ratio": odds,
                "p_value": p_value,
            }
        )
    for row, q_value in zip(rows, bh_fdr([float(row["p_value"]) for row in rows])):
        row["fdr_bh"] = q_value
    return rows


def default_comparisons() -> List[Tuple[str, str]]:
    return [
        ("m1A_3UTR_prox25", "m1A_3UTR_distal50"),
        ("m1A_CDS_prox25", "m1A_CDS_distal50"),
        ("m1A_3UTR_prox25", "m6A_only_same_transcript"),
        ("m1A_CDS_prox25", "m6A_only_same_transcript"),
    ]


def write_overlap(reader_summary: Dict[str, Dict[str, object]], memberships: Sequence[Dict[str, str]], key_field: str, out: Path) -> None:
    groups = [col for col in memberships[0].keys() if col not in {key_field, "gene_id", "transcript_id", "gene_name"}] if memberships else []
    membership_by_key = {normalize_symbol(row.get(key_field, "")): row for row in collapse_memberships(memberships, key_field)}
    rows = []
    for key, summary in sorted(reader_summary.items()):
        member = membership_by_key.get(key, {})
        rows.append(
            {
                key_field: key,
                "reader_site_count": summary["reader_site_count"],
                "unique_reader_count": summary["unique_reader_count"],
                "readers": summary["readers"],
                "positive_groups": ";".join(group for group in groups if truthy(member.get(group))),
            }
        )
    write_table(rows, out, [key_field, "reader_site_count", "unique_reader_count", "readers", "positive_groups"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run m6A-reader annotation enrichment by dual-site groups.")
    parser.add_argument("--membership", default="11_tables/biology_summary/transcript_group_membership.tsv")
    parser.add_argument("--reader-annotation", required=True, help="RMBase modRBP-like TSV with rbpName/RNAmodType/geneName columns.")
    parser.add_argument("--out-dir", default="11_tables/biology_summary")
    parser.add_argument("--key-field", default="gene_name")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    memberships = read_rows(Path(args.membership))
    annotations = read_rows(Path(args.reader_annotation))
    reader_summary = aggregate_reader_annotations(annotations, args.key_field)
    comparisons = compare_reader_enrichment(memberships, reader_summary, default_comparisons(), args.key_field)
    out_dir = Path(args.out_dir)
    write_table(comparisons, out_dir / "reader_enrichment.tsv")
    write_overlap(reader_summary, memberships, args.key_field, out_dir / "reader_annotation_overlap.tsv")
    print(f"wrote {len(comparisons)} reader enrichment rows and {len(reader_summary)} annotated entities")


if __name__ == "__main__":
    main()
