#!/usr/bin/env python
"""Run transcript-aware m1A-m6A spacing permutation tests."""

from __future__ import annotations

import argparse
import bisect
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def summarize_distances(distances: Sequence[int]) -> Dict[str, float]:
    values = list(distances)
    if not values:
        return {
            "median_nearest_distance": 0.0,
            "mean_nearest_distance": 0.0,
            "fraction_exact_overlap": 0.0,
            "fraction_le_10nt": 0.0,
            "fraction_le_25nt": 0.0,
            "fraction_le_50nt": 0.0,
        }
    return {
        "median_nearest_distance": float(statistics.median(values)),
        "mean_nearest_distance": float(sum(values) / len(values)),
        "fraction_exact_overlap": sum(d == 0 for d in values) / len(values),
        "fraction_le_10nt": sum(d <= 10 for d in values) / len(values),
        "fraction_le_25nt": sum(d <= 25 for d in values) / len(values),
        "fraction_le_50nt": sum(d <= 50 for d in values) / len(values),
    }


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def unique_site_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    out: Dict[Tuple[str, int], Dict[str, str]] = {}
    for row in rows:
        if row.get("transcript_id") and row.get("tx_pos"):
            out.setdefault((row["transcript_id"], int(float(row["tx_pos"]))), row)
    return list(out.values())


def positions_by_key(rows: Sequence[Dict[str, str]], include_region: bool) -> Dict[Tuple[str, str], List[int]]:
    out: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for row in rows:
        key = (row["transcript_id"], row.get("region_type", "") if include_region else "ALL")
        out[key].append(int(float(row["tx_pos"])))
    for values in out.values():
        values.sort()
    return out


def nearest_distance(position: int, sorted_positions: Sequence[int]) -> int | None:
    if not sorted_positions:
        return None
    idx = bisect.bisect_left(sorted_positions, position)
    candidates = []
    if idx < len(sorted_positions):
        candidates.append(sorted_positions[idx])
    if idx > 0:
        candidates.append(sorted_positions[idx - 1])
    return min(abs(position - candidate) for candidate in candidates)


def observed_distances(m1a_rows: Sequence[Dict[str, str]], m6a_by_key: Dict[Tuple[str, str], List[int]], include_region: bool) -> List[int]:
    distances = []
    for row in m1a_rows:
        key = (row["transcript_id"], row.get("region_type", "") if include_region else "ALL")
        distance = nearest_distance(int(float(row["tx_pos"])), m6a_by_key.get(key, []))
        if distance is not None:
            distances.append(distance)
    return distances


def simulate_distances(
    m1a_rows: Sequence[Dict[str, str]],
    m6a_by_key: Dict[Tuple[str, str], List[int]],
    candidates_by_key: Dict[Tuple[str, str], List[int]],
    include_region: bool,
    rng: random.Random,
) -> List[int]:
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in m1a_rows:
        key = (row["transcript_id"], row.get("region_type", "") if include_region else "ALL")
        counts[key] += 1
    distances = []
    for key, count in counts.items():
        candidates = candidates_by_key.get(key, [])
        if not candidates:
            continue
        sampled = rng.sample(candidates, min(count, len(candidates)))
        if len(sampled) < count:
            sampled.extend(rng.choices(candidates, k=count - len(sampled)))
        for pos in sampled:
            distance = nearest_distance(pos, m6a_by_key.get(key, []))
            if distance is not None:
                distances.append(distance)
    return distances


def empirical_p(metric: str, observed: float, null_values: Sequence[float]) -> float:
    if metric in {"median_nearest_distance", "mean_nearest_distance"}:
        extreme = sum(value <= observed for value in null_values)
    else:
        extreme = sum(value >= observed for value in null_values)
    return (extreme + 1) / (len(null_values) + 1)


def run_permutation(args: argparse.Namespace) -> List[Dict[str, object]]:
    rng = random.Random(args.seed)
    m1a_rows = unique_site_rows(read_rows(Path(args.m1a)))
    m6a_rows = unique_site_rows(read_rows(Path(args.m6a)))
    dataset_rows = read_rows(Path(args.dataset))
    rows_out = []
    for design, include_region in [("global_transcript", False), ("region_stratified", True)]:
        m6a_by_key = positions_by_key(m6a_rows, include_region)
        candidates_by_key = positions_by_key(dataset_rows, include_region)
        obs = summarize_distances(observed_distances(m1a_rows, m6a_by_key, include_region))
        null_by_metric: Dict[str, List[float]] = {metric: [] for metric in obs}
        for _ in range(args.n_permutations):
            sim = summarize_distances(simulate_distances(m1a_rows, m6a_by_key, candidates_by_key, include_region, rng))
            for metric, value in sim.items():
                null_by_metric[metric].append(value)
        for metric, observed_value in obs.items():
            null_values = null_by_metric[metric]
            null_mean = sum(null_values) / len(null_values)
            rows_out.append(
                {
                    "design": design,
                    "metric": metric,
                    "observed": observed_value,
                    "null_mean": null_mean,
                    "observed_minus_null": observed_value - null_mean,
                    "observed_over_null": observed_value / null_mean if null_mean else "",
                    "empirical_p_one_sided": empirical_p(metric, observed_value, null_values),
                    "n_permutations": args.n_permutations,
                }
            )
    return rows_out


def write_table(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run m1A-m6A spacing permutation tests.")
    parser.add_argument("--m1a", default="04_transcript_space/m1A_refseq_mapped.tsv")
    parser.add_argument("--m6a", default="04_transcript_space/m6A_refseq_mapped.tsv")
    parser.add_argument("--dataset", default="05_datasets/dataset_joint_m1A_m6A_101nt.tsv")
    parser.add_argument("--out", default="11_tables/biology_summary/spacing_permutation_results.tsv")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = run_permutation(args)
    write_table(rows, Path(args.out))
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
