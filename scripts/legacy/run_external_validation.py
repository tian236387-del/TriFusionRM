#!/usr/bin/env python
"""Summarize external public m1A/m6A annotations against current dual-site groups."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


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


def normalize(value: str) -> str:
    return value.strip().upper()


def row_value(row: Dict[str, str], candidates: Sequence[str]) -> str:
    for key in candidates:
        if row.get(key):
            return str(row[key])
    lower_map = {key.lower(): key for key in row}
    for key in candidates:
        mapped = lower_map.get(key.lower())
        if mapped and row.get(mapped):
            return str(row[mapped])
    return ""


def normalize_mod(value: str) -> str:
    clean = value.upper().replace("-", "")
    if "M1A" in clean:
        return "m1A"
    if "M6A" in clean:
        return "m6A"
    return ""


def truthy(value: object) -> bool:
    return str(value).strip() in {"1", "true", "True", "yes", "Y"}


def summarize_external_modifications(rows: Sequence[Dict[str, str]], key_field: str = "gene_name") -> Dict[str, Dict[str, object]]:
    by_key: Dict[str, Dict[str, object]] = {}
    for row in rows:
        mod = normalize_mod(row_value(row, ["mod_type", "RNAmodType", "modType", "mod", "modification"]))
        if mod not in {"m1A", "m6A"}:
            continue
        entity = normalize(row_value(row, [key_field, "gene_name", "geneName", "Gene", "gene", "transcript_id", "transcriptID"]))
        if not entity:
            continue
        current = by_key.setdefault(
            entity,
            {
                "external_m1A_count": 0,
                "external_m6A_count": 0,
                "external_dual_mod": 0,
                "external_regions": set(),
            },
        )
        current[f"external_{mod}_count"] = int(current[f"external_{mod}_count"]) + 1
        region = row_value(row, ["region_type", "Region", "region"])
        if region:
            current["external_regions"].add(region)
    for value in by_key.values():
        value["external_dual_mod"] = int(int(value["external_m1A_count"]) > 0 and int(value["external_m6A_count"]) > 0)
        value["external_regions"] = ";".join(sorted(value["external_regions"]))
    return by_key


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


def collapse_memberships(rows: Sequence[Dict[str, str]], key_field: str) -> List[Dict[str, str]]:
    collapsed: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = normalize(row_value(row, [key_field, "gene_name", "geneName", "transcript_id", "transcriptID"]))
        if not key:
            continue
        out = collapsed.setdefault(key, {key_field: key})
        for col, value in row.items():
            if truthy(value):
                out[col] = "1"
            elif col not in out:
                out[col] = value
    return list(collapsed.values())


def entities_for_group(rows: Sequence[Dict[str, str]], group: str, key_field: str) -> set[str]:
    return {normalize(row.get(key_field, "")) for row in rows if truthy(row.get(group)) and row.get(key_field)}


def compare_external_annotation_groups(
    memberships: Sequence[Dict[str, str]],
    external_summary: Dict[str, Dict[str, object]],
    comparisons: Sequence[Tuple[str, str]],
    key_field: str = "gene_name",
) -> List[Dict[str, object]]:
    membership_rows = collapse_memberships(memberships, key_field)
    external_dual = {key for key, value in external_summary.items() if int(value.get("external_dual_mod", 0)) == 1}
    rows = []
    for left, right in comparisons:
        left_entities = entities_for_group(membership_rows, left, key_field)
        right_entities = entities_for_group(membership_rows, right, key_field) - left_entities
        left_pos = len(left_entities & external_dual)
        right_pos = len(right_entities & external_dual)
        left_total = len(left_entities)
        right_total = len(right_entities)
        odds, p_value = fisher_exact(left_pos, left_total - left_pos, right_pos, right_total - right_pos)
        left_frac = left_pos / left_total if left_total else 0.0
        right_frac = right_pos / right_total if right_total else 0.0
        rows.append(
            {
                "comparison": f"{left} vs {right}",
                "left_group": left,
                "right_group": right,
                "left_total": left_total,
                "right_total": right_total,
                "left_external_dual_positive": left_pos,
                "right_external_dual_positive": right_pos,
                "left_external_dual_fraction": left_frac,
                "right_external_dual_fraction": right_frac,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run external annotation consistency checks.")
    parser.add_argument("--membership", default="11_tables/biology_summary/transcript_group_membership.tsv")
    parser.add_argument("--external-annotation", required=True, action="append", help="External m1A/m6A TSV. Can be passed multiple times.")
    parser.add_argument("--out-dir", default="11_tables/biology_summary")
    parser.add_argument("--key-field", default="gene_name")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    memberships = read_rows(Path(args.membership))
    external_rows: List[Dict[str, str]] = []
    for path_text in args.external_annotation:
        external_rows.extend(read_rows(Path(path_text)))
    summary = summarize_external_modifications(external_rows, args.key_field)
    comparisons = compare_external_annotation_groups(memberships, summary, default_comparisons(), args.key_field)
    summary_rows = [{args.key_field: key, **value} for key, value in sorted(summary.items())]
    out_dir = Path(args.out_dir)
    write_table(summary_rows, out_dir / "external_annotation_overlap.tsv")
    write_table(comparisons, out_dir / "external_validation_summary.tsv")
    print(f"wrote {len(comparisons)} external validation rows and {len(summary)} annotated entities")


if __name__ == "__main__":
    main()
