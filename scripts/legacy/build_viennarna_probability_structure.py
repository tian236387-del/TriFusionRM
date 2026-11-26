#!/usr/bin/env python
"""Build probability-style ViennaRNA structure features for RMBase windows.

The training code expects the same schema as ``run_rmbase11_rnafold.py``:
paired_flags, base_pair_list and a small set of scalar structure features.
This script keeps that schema but uses partition-function base-pair
probabilities instead of hard MFE paired flags.

Requires the ViennaRNA Python bindings (``import RNA``).  On conda this is
usually provided by the ``viennarna`` package.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


STRUCT_FIELDS = [
    "site_id",
    "sequence",
    "dot_bracket",
    "MFE",
    "paired_flags",
    "base_pair_list",
    "center_paired",
    "central_11nt_paired_fraction",
    "central_21nt_paired_fraction",
    "paired_fraction_window",
    "MFE_per_nt",
    "loop_span_around_center",
    "ensemble_free_energy",
    "ensemble_diversity",
    "center_unpaired_probability",
    "mean_unpaired_probability",
    "unpaired_probability_flags",
    "center_pair_entropy",
    "central_11nt_pair_entropy",
    "central_21nt_pair_entropy",
    "mean_pair_entropy",
    "pair_entropy_flags",
    "center_expected_partner_distance",
    "central_11nt_expected_partner_distance",
    "central_21nt_expected_partner_distance",
    "mean_expected_partner_distance",
    "expected_partner_distance_flags",
    "pair_edge_probability",
    "pair_probability_list",
    "structure_feature_type",
]


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def iter_tsv(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_existing(path: str) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return {row["site_id"]: row for row in read_tsv(p)}


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_dot_bracket(dot: str) -> List[Tuple[int, int]]:
    stack: List[int] = []
    pairs: List[Tuple[int, int]] = []
    for idx, char in enumerate(dot, start=1):
        if char == "(":
            stack.append(idx)
        elif char == ")" and stack:
            left = stack.pop()
            pairs.append((left, idx))
    pairs.sort()
    return pairs


def pairs_to_dot(length: int, pairs: Sequence[Tuple[int, int]]) -> str:
    chars = ["."] * length
    for left, right in pairs:
        if 1 <= left < right <= length:
            chars[left - 1] = "("
            chars[right - 1] = ")"
    return "".join(chars)


def threshold_loop_span(paired_probability: Sequence[float], center_idx: int, threshold: float) -> int:
    left = center_idx
    right = center_idx
    while left - 1 >= 0 and paired_probability[left - 1] < threshold:
        left -= 1
    while right + 1 < len(paired_probability) and paired_probability[right + 1] < threshold:
        right += 1
    return right - left + 1


def reciprocal_probability_pairs(
    pair_probability: Sequence[Sequence[float]],
    length: int,
    threshold: float,
) -> List[Tuple[int, int]]:
    best_partner = [-1] * length
    best_prob = [0.0] * length
    for i in range(length):
        row = pair_probability[i]
        for j in range(i + 1, length):
            p = float(row[j])
            if p > best_prob[i]:
                best_prob[i] = p
                best_partner[i] = j
            if p > best_prob[j]:
                best_prob[j] = p
                best_partner[j] = i
    pairs: List[Tuple[int, int]] = []
    used = set()
    for i, j in enumerate(best_partner):
        if j < 0 or i in used or j in used:
            continue
        if best_partner[j] == i and min(best_prob[i], best_prob[j]) >= threshold:
            left, right = sorted((i + 1, j + 1))
            pairs.append((left, right))
            used.add(i)
            used.add(j)
    pairs.sort()
    return pairs


def sparse_probability_pairs(
    pair_probability: Sequence[Sequence[float]],
    length: int,
    threshold: float,
    max_candidates: int,
) -> List[Tuple[int, int, float]]:
    """Keep each nucleotide's strongest ensemble partners as sparse directed evidence.

    The serialized edge list is undirected, while the downstream parser applies
    the top-k limit independently for every source nucleotide.  Retaining the
    union here prevents a strong partner selected from only one endpoint from
    being dropped prematurely.
    """

    selected: Dict[Tuple[int, int], float] = {}
    for i in range(length):
        candidates = sorted(
            ((j, float(pair_probability[i][j])) for j in range(length) if j != i),
            key=lambda item: item[1],
            reverse=True,
        )
        kept = 0
        for j, probability in candidates:
            if probability < threshold:
                break
            left, right = sorted((i + 1, j + 1))
            selected[(left, right)] = max(selected.get((left, right), 0.0), probability)
            kept += 1
            if kept >= max(1, max_candidates):
                break
    return [(left, right, selected[(left, right)]) for left, right in sorted(selected)]


def safe_matrix_value(matrix: object, i: int, j: int) -> float:
    try:
        return float(matrix[i][j])  # type: ignore[index]
    except Exception:
        return 0.0


def compute_probability_features(
    seq: str,
    pair_threshold: float,
    candidate_threshold: float = 0.05,
    max_pair_candidates: int = 3,
) -> Dict[str, object]:
    try:
        import RNA  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on server env
        raise RuntimeError(
            "ViennaRNA Python bindings are required. Install/activate an environment "
            "where `python -c \"import RNA\"` works."
        ) from exc

    seq_u = str(seq or "").upper().replace("T", "U")
    length = len(seq_u)
    if length == 0:
        raise ValueError("empty sequence")

    fold_compound = RNA.fold_compound(seq_u)
    mfe_struct, mfe = fold_compound.mfe()
    ensemble_free_energy = math.nan
    try:
        pf_result = fold_compound.pf()
        if isinstance(pf_result, tuple) and len(pf_result) >= 2:
            ensemble_free_energy = float(pf_result[1])
        elif isinstance(pf_result, (int, float)):
            ensemble_free_energy = float(pf_result)
    except Exception:
        fold_compound.pf()

    bpp_matrix = fold_compound.bpp()
    pair_probability = [[0.0 for _ in range(length)] for _ in range(length)]
    paired_probability = [0.0 for _ in range(length)]
    expected_partner_distance = [0.0 for _ in range(length)]
    for i in range(1, length + 1):
        for j in range(i + 1, length + 1):
            p = max(0.0, min(1.0, safe_matrix_value(bpp_matrix, i, j)))
            if p <= 0.0:
                continue
            pair_probability[i - 1][j - 1] = p
            pair_probability[j - 1][i - 1] = p
            paired_probability[i - 1] += p
            paired_probability[j - 1] += p
            expected_partner_distance[i - 1] += p * abs(i - j)
            expected_partner_distance[j - 1] += p * abs(i - j)
    paired_probability = [max(0.0, min(1.0, p)) for p in paired_probability]

    # Entropy is calculated over the complete partner-state distribution
    # (all possible partners plus the unpaired state), not merely a paired /
    # unpaired Bernoulli variable.  Division by log(L + 1) maps it to [0, 1].
    entropy_normalizer = max(math.log(length + 1.0), 1e-12)
    pair_entropy: List[float] = []
    for i in range(length):
        unpaired = max(0.0, 1.0 - paired_probability[i])
        entropy = -unpaired * math.log(max(unpaired, 1e-12))
        entropy -= sum(
            probability * math.log(max(probability, 1e-12))
            for probability in pair_probability[i]
            if probability > 0.0
        )
        pair_entropy.append(max(0.0, min(1.0, entropy / entropy_normalizer)))

    prob_pairs = reciprocal_probability_pairs(pair_probability, length, pair_threshold)
    if not prob_pairs:
        prob_pairs = parse_dot_bracket(mfe_struct)
    dot = pairs_to_dot(length, prob_pairs)
    sparse_pairs = sparse_probability_pairs(
        pair_probability,
        length,
        candidate_threshold,
        max_pair_candidates,
    )
    selected_edge_probability = [0.0] * length
    for left, right in prob_pairs:
        probability = pair_probability[left - 1][right - 1]
        selected_edge_probability[left - 1] = probability
        selected_edge_probability[right - 1] = probability

    center_idx = length // 2
    central_11 = paired_probability[max(0, center_idx - 5) : min(length, center_idx + 6)]
    central_21 = paired_probability[max(0, center_idx - 10) : min(length, center_idx + 11)]
    central_11_entropy = pair_entropy[max(0, center_idx - 5) : min(length, center_idx + 6)]
    central_21_entropy = pair_entropy[max(0, center_idx - 10) : min(length, center_idx + 11)]
    central_11_distance = expected_partner_distance[max(0, center_idx - 5) : min(length, center_idx + 6)]
    central_21_distance = expected_partner_distance[max(0, center_idx - 10) : min(length, center_idx + 11)]
    unpaired_probability = [1.0 - p for p in paired_probability]

    def mean(values: Sequence[float]) -> float:
        return float(sum(values) / max(len(values), 1))

    try:
        ensemble_diversity = float(fold_compound.mean_bp_distance())
    except Exception:
        ensemble_diversity = math.nan
    if not math.isfinite(ensemble_free_energy):
        ensemble_free_energy = float(mfe)
    if not math.isfinite(ensemble_diversity):
        ensemble_diversity = 0.0

    return {
        "sequence": seq_u.replace("U", "T"),
        "dot_bracket": dot,
        "MFE": float(mfe),
        "paired_flags": ",".join(f"{p:.6g}" for p in paired_probability),
        "base_pair_list": ";".join(f"{left}-{right}" for left, right in prob_pairs),
        "center_paired": paired_probability[center_idx],
        "central_11nt_paired_fraction": sum(central_11) / max(len(central_11), 1),
        "central_21nt_paired_fraction": sum(central_21) / max(len(central_21), 1),
        "paired_fraction_window": sum(paired_probability) / max(length, 1),
        "MFE_per_nt": float(mfe) / max(length, 1),
        "loop_span_around_center": threshold_loop_span(paired_probability, center_idx, pair_threshold),
        "ensemble_free_energy": ensemble_free_energy,
        "ensemble_diversity": ensemble_diversity,
        "center_unpaired_probability": unpaired_probability[center_idx],
        "mean_unpaired_probability": mean(unpaired_probability),
        "unpaired_probability_flags": ",".join(f"{value:.6g}" for value in unpaired_probability),
        "center_pair_entropy": pair_entropy[center_idx],
        "central_11nt_pair_entropy": mean(central_11_entropy),
        "central_21nt_pair_entropy": mean(central_21_entropy),
        "mean_pair_entropy": mean(pair_entropy),
        "pair_entropy_flags": ",".join(f"{value:.6g}" for value in pair_entropy),
        "center_expected_partner_distance": expected_partner_distance[center_idx],
        "central_11nt_expected_partner_distance": mean(central_11_distance),
        "central_21nt_expected_partner_distance": mean(central_21_distance),
        "mean_expected_partner_distance": mean(expected_partner_distance),
        "expected_partner_distance_flags": ",".join(
            f"{value / max(length - 1, 1):.6g}" for value in expected_partner_distance
        ),
        "pair_edge_probability": ",".join(f"{value:.6g}" for value in selected_edge_probability),
        "pair_probability_list": ";".join(
            f"{left}-{right}:{probability:.6g}" for left, right, probability in sparse_pairs
        ),
        "structure_feature_type": "viennarna_partition_probability",
    }


def cache_key(seq: str, args: argparse.Namespace) -> str:
    seq_u = str(seq or "").upper().replace("T", "U")
    return (
        f"prob-v2|pair={args.pair_threshold:.8g}|candidate={args.candidate_threshold:.8g}|"
        f"topk={args.max_pair_candidates}|{seq_u}"
    )


def compute_probability_job(job: Tuple[str, float, float, int]) -> Tuple[str, Dict[str, object]]:
    seq, pair_threshold, candidate_threshold, max_pair_candidates = job
    return seq, compute_probability_features(
        seq,
        pair_threshold,
        candidate_threshold,
        max_pair_candidates,
    )


def chunked_rows(rows: Iterator[Dict[str, str]], chunk_size: int) -> Iterator[List[Dict[str, str]]]:
    chunk: List[Dict[str, str]] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def cache_lookup(connection: sqlite3.Connection, keys: Sequence[str]) -> Dict[str, Dict[str, object]]:
    output: Dict[str, Dict[str, object]] = {}
    for start in range(0, len(keys), 500):
        subset = keys[start : start + 500]
        placeholders = ",".join("?" for _ in subset)
        query = f"SELECT cache_key, payload FROM structure_cache WHERE cache_key IN ({placeholders})"
        for key, payload in connection.execute(query, subset):
            output[str(key)] = json.loads(str(payload))
    return output


def build(args: argparse.Namespace) -> Dict[str, object]:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_db) if args.cache_db else out_path.with_suffix(".cache.sqlite3")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS structure_cache (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
    )

    partial_path = out_path.with_suffix(out_path.suffix + ".partial")
    input_rows = 0
    rows_written = 0
    reused = 0
    computed = 0
    failed = 0
    workers = max(1, int(args.workers))
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        with partial_path.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=STRUCT_FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            source_rows = iter_tsv(Path(args.dataset))
            for chunk in chunked_rows(source_rows, max(1, int(args.chunk_size))):
                if args.max_rows:
                    remaining = args.max_rows - input_rows
                    if remaining <= 0:
                        break
                    chunk = chunk[:remaining]
                input_rows += len(chunk)
                sequence_by_key: Dict[str, str] = {}
                key_by_row: List[str] = []
                for row in chunk:
                    seq = row.get(args.sequence_column, "")
                    key = cache_key(seq, args)
                    key_by_row.append(key)
                    sequence_by_key.setdefault(key, seq)

                cached = cache_lookup(connection, list(sequence_by_key))
                missing_keys = [key for key in sequence_by_key if key not in cached]
                jobs = [
                    (
                        sequence_by_key[key],
                        args.pair_threshold,
                        args.candidate_threshold,
                        args.max_pair_candidates,
                    )
                    for key in missing_keys
                ]
                try:
                    if executor is None:
                        computed_items = map(compute_probability_job, jobs)
                    else:
                        computed_items = executor.map(compute_probability_job, jobs, chunksize=max(1, args.worker_chunksize))
                    by_sequence = {seq: item for seq, item in computed_items}
                    additions = []
                    for key in missing_keys:
                        item = by_sequence[sequence_by_key[key]]
                        cached[key] = item
                        additions.append((key, json.dumps(item, separators=(",", ":"), ensure_ascii=True)))
                    if additions:
                        connection.executemany(
                            "INSERT OR REPLACE INTO structure_cache(cache_key, payload) VALUES (?, ?)",
                            additions,
                        )
                        connection.commit()
                    computed += len(additions)
                except Exception as exc:
                    failed += len(missing_keys)
                    raise RuntimeError(f"failed to fold a chunk ending at input row {input_rows}: {exc}") from exc

                reused += len(sequence_by_key) - len(missing_keys)
                for row, key in zip(chunk, key_by_row):
                    item = dict(cached[key])
                    item["site_id"] = row["site_id"]
                    writer.writerow({field: item.get(field, "") for field in STRUCT_FIELDS})
                    rows_written += 1
                if args.progress_every and rows_written % args.progress_every < len(chunk):
                    print(
                        f"rows={rows_written} unique_computed={computed} cache_hits={reused} failed={failed}",
                        flush=True,
                    )
        os.replace(partial_path, out_path)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        connection.close()

    summary = {
        "dataset": args.dataset,
        "out": args.out,
        "cache_db": str(cache_path),
        "feature_type": "viennarna_partition_probability",
        "input_rows": input_rows,
        "rows_written": rows_written,
        "unique_rows_written": rows_written,
        "cache_hits": reused,
        "reused_rows": reused,
        "unique_sequences_computed": computed,
        "computed_rows": computed,
        "failed_rows": failed,
        "pair_threshold": args.pair_threshold,
        "candidate_threshold": args.candidate_threshold,
        "max_pair_candidates": args.max_pair_candidates,
        "workers": workers,
        "chunk_size": args.chunk_size,
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ViennaRNA probability-style structure features.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv")
    parser.add_argument("--out", default="rmbase11_full_work/structure/rnafold_prob_41nt_all_sites.tsv")
    parser.add_argument("--sequence-column", default="sequence_41nt")
    parser.add_argument("--pair-threshold", type=float, default=0.5)
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.05,
        help="Minimum partition-function probability retained in pair_probability_list.",
    )
    parser.add_argument(
        "--max-pair-candidates",
        type=int,
        default=3,
        help="Maximum strongest partner candidates retained per nucleotide before edge union.",
    )
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--worker-chunksize", type=int, default=32)
    parser.add_argument(
        "--cache-db",
        default="",
        help="Optional SQLite cache keyed by sequence and folding parameters.",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
