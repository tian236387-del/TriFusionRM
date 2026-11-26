#!/usr/bin/env python
"""Compare sequence and RNAfold structure features across biological site groups."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


def centered_kmer(seq: str, width: int) -> str:
    seq = seq.upper().replace("U", "T")
    center = len(seq) // 2
    half = width // 2
    return seq[center - half : center + half + 1]


def is_drach(kmer: str) -> bool:
    kmer = kmer.upper().replace("U", "T")
    if len(kmer) != 5:
        return False
    d, r, a, c, h = kmer
    return d in "AGT" and r in "AG" and a == "A" and c == "C" and h in "ACT"


def gc_fraction(seq: str) -> float:
    seq = seq.upper().replace("U", "T")
    valid = [base for base in seq if base in "ACGT"]
    return (valid.count("G") + valid.count("C")) / len(valid) if valid else 0.0


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def index_rows(rows: Sequence[Dict[str, str]], key: str) -> Dict[str, Dict[str, str]]:
    return {row[key]: row for row in rows if row.get(key)}


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def feature_rows(group_rows: Sequence[Dict[str, str]], dataset_by_id: Dict[str, Dict[str, str]], fold_by_id: Dict[str, Dict[str, str]]) -> List[Dict[str, object]]:
    rows = []
    for group in group_rows:
        sid = group["site_id"]
        data = dataset_by_id.get(sid)
        if not data:
            continue
        fold = fold_by_id.get(sid, {})
        has_rnafold = int(bool(fold))
        seq = data.get("sequence_101nt") or data.get("sequence") or ""
        k5 = centered_kmer(seq, 5)
        k7 = centered_kmer(seq, 7)
        central = seq[len(seq) // 2 - 50 : len(seq) // 2 + 51] if len(seq) >= 101 else seq
        rows.append(
            {
                "site_id": sid,
                "site_group": group["site_group"],
                "region_type": data.get("region_type", group.get("region_type", "")),
                "centered_5mer": k5,
                "centered_7mer": k7,
                "drach_centered": int(is_drach(k5)),
                "ggac_like": int("GGAC" in k7 or "AGAC" in k7),
                "gc_fraction_101nt": gc_fraction(central),
                "A_fraction_101nt": central.upper().count("A") / len(central) if central else 0.0,
                "C_fraction_101nt": central.upper().count("C") / len(central) if central else 0.0,
                "G_fraction_101nt": central.upper().count("G") / len(central) if central else 0.0,
                "T_fraction_101nt": central.upper().replace("U", "T").count("T") / len(central) if central else 0.0,
                "has_rnafold": has_rnafold,
                "MFE": parse_float(fold.get("MFE")),
                "MFE_per_nt": parse_float(fold.get("MFE_per_nt")),
                "paired_fraction_101nt": parse_float(fold.get("paired_fraction_101nt")),
                "center_paired": parse_float(fold.get("center_paired")),
                "central_11nt_paired_fraction": parse_float(fold.get("central_11nt_paired_fraction")),
                "central_21nt_paired_fraction": parse_float(fold.get("central_21nt_paired_fraction")),
                "loop_span_around_center": parse_float(fold.get("loop_span_around_center")),
            }
        )
    return rows


def mann_whitney_p(x: Sequence[float], y: Sequence[float]) -> float:
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def fisher_p(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    try:
        from scipy.stats import fisher_exact

        odds, p_value = fisher_exact([[a, b], [c, d]])
        return float(odds), float(p_value)
    except Exception:
        return float("nan"), float("nan")


def bh_fdr(p_values: Sequence[float]) -> List[float]:
    indexed = [(idx, p) for idx, p in enumerate(p_values)]
    valid = [(idx, p) for idx, p in indexed if not math.isnan(p)]
    ranked = sorted(valid, key=lambda item: item[1])
    adjusted = [float("nan")] * len(p_values)
    running = 1.0
    m = len(ranked)
    for rank, (idx, p) in reversed(list(enumerate(ranked, start=1))):
        running = min(running, p * m / rank)
        adjusted[idx] = running
    return adjusted


def cliff_delta(x: Sequence[float], y: Sequence[float]) -> float:
    if not x or not y:
        return 0.0
    greater = less = 0
    for xv in x:
        for yv in y:
            greater += xv > yv
            less += xv < yv
    return (greater - less) / (len(x) * len(y))


def compare_groups(rows: Sequence[Dict[str, object]], comparisons: Sequence[tuple[str, str]]) -> List[Dict[str, object]]:
    by_group: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["site_group"])].append(row)
    sequence_continuous = ["gc_fraction_101nt"]
    structure_continuous = [
        "MFE",
        "MFE_per_nt",
        "paired_fraction_101nt",
        "center_paired",
        "central_11nt_paired_fraction",
        "central_21nt_paired_fraction",
        "loop_span_around_center",
    ]
    binary = ["drach_centered", "ggac_like"]
    results = []
    for left, right in comparisons:
        left_rows = by_group.get(left, [])
        right_rows = by_group.get(right, [])
        for feature in sequence_continuous + structure_continuous:
            if feature in structure_continuous:
                left_feature_rows = [row for row in left_rows if int(row.get("has_rnafold", 0)) == 1]
                right_feature_rows = [row for row in right_rows if int(row.get("has_rnafold", 0)) == 1]
            else:
                left_feature_rows = left_rows
                right_feature_rows = right_rows
            x = [float(row[feature]) for row in left_feature_rows]
            y = [float(row[feature]) for row in right_feature_rows]
            results.append(
                {
                    "comparison": f"{left} vs {right}",
                    "feature": feature,
                    "test": "Mann-Whitney U",
                    "left_n": len(x),
                    "right_n": len(y),
                    "left_median": float(np.median(x)) if x else "",
                    "right_median": float(np.median(y)) if y else "",
                    "effect_size": cliff_delta(x, y),
                    "odds_ratio": "",
                    "p_value": mann_whitney_p(x, y) if x and y else float("nan"),
                }
            )
        for feature in binary:
            a = sum(int(row[feature]) == 1 for row in left_rows)
            b = len(left_rows) - a
            c = sum(int(row[feature]) == 1 for row in right_rows)
            d = len(right_rows) - c
            odds, p_value = fisher_p(a, b, c, d)
            results.append(
                {
                    "comparison": f"{left} vs {right}",
                    "feature": feature,
                    "test": "Fisher exact",
                    "left_n": len(left_rows),
                    "right_n": len(right_rows),
                    "left_median": a / len(left_rows) if left_rows else "",
                    "right_median": c / len(right_rows) if right_rows else "",
                    "effect_size": (a / len(left_rows) if left_rows else 0) - (c / len(right_rows) if right_rows else 0),
                    "odds_ratio": odds,
                    "p_value": p_value,
                }
            )
    fdr = bh_fdr([float(row["p_value"]) for row in results])
    for row, q_value in zip(results, fdr):
        row["fdr_bh"] = q_value
    return results


def write_table(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    group_rows = read_rows(Path(args.groups))
    dataset_by_id = index_rows(read_rows(Path(args.dataset)), "site_id")
    fold_by_id = index_rows(read_rows(Path(args.rnafold)), "site_id")
    features = feature_rows(group_rows, dataset_by_id, fold_by_id)
    comparisons = [
        ("m1A_3UTR_prox25", "m1A_3UTR_distal50"),
        ("m1A_CDS_prox25", "m1A_CDS_distal50"),
        ("m1A_3UTR_prox25", "matched_unmodified_A"),
        ("m1A_CDS_prox25", "matched_unmodified_A"),
    ]
    stats = compare_groups(features, comparisons)
    out_dir = Path(args.out_dir)
    write_table(features, out_dir / "sequence_structure_site_features.tsv")
    write_table([row for row in stats if row["test"] == "Fisher exact"], out_dir / "sequence_feature_comparison.tsv")
    write_table([row for row in stats if row["test"] == "Mann-Whitney U"], out_dir / "rnafold_structure_comparison.tsv")
    print(f"wrote {len(features)} feature rows and {len(stats)} comparison rows")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sequence and RNAfold structure comparisons for biological site groups.")
    parser.add_argument("--groups", default="11_tables/biology_summary/dual_site_groups.tsv")
    parser.add_argument("--dataset", default="05_datasets/dataset_joint_m1A_m6A_101nt.tsv")
    parser.add_argument("--rnafold", default="06_structure/rnafold_101nt.tsv")
    parser.add_argument("--out-dir", default="11_tables/biology_summary")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
