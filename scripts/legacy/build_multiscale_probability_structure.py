#!/usr/bin/env python
"""Merge aligned ViennaRNA probability features across sequence window lengths.

Input files must contain the same site IDs in the same order.  The script is
streaming so that million-row structure tables do not need to be loaded into
memory.  Longer-window node profiles are center-cropped to the target window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import ExitStack
from itertools import chain, islice, zip_longest
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


NODE_INPUT_FIELDS = {
    "paired": "paired_flags",
    "unpaired": "unpaired_probability_flags",
    "entropy": "pair_entropy_flags",
    "distance": "expected_partner_distance_flags",
}

NODE_OUTPUT_FIELDS = [
    "multiscale_mean_paired_flags",
    "multiscale_mean_unpaired_flags",
    "multiscale_mean_entropy_flags",
    "multiscale_mean_expected_partner_distance_flags",
    "multiscale_pair_variance_flags",
]

SCALAR_OUTPUT_FIELDS = [
    "multiscale_center_paired_mean",
    "multiscale_center_paired_variance",
    "multiscale_center_unpaired_mean",
    "multiscale_center_entropy_mean",
    "multiscale_center_entropy_max",
    "multiscale_profile_jsd",
    "multiscale_pairing_agreement",
    "multiscale_structure_confidence",
    "multiscale_window_count",
]


def parse_vector(value: object, expected_length: int, field: str, site_id: str) -> List[float]:
    values = [float(item) for item in str(value or "").split(",") if item != ""]
    if len(values) != expected_length:
        raise ValueError(
            f"{site_id}: {field} length={len(values)} does not match declared window={expected_length}"
        )
    return values


def center_crop(values: Sequence[float], target_length: int) -> List[float]:
    if len(values) < target_length:
        raise ValueError(f"cannot crop length {len(values)} to larger target {target_length}")
    start = (len(values) - target_length) // 2
    return list(values[start : start + target_length])


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def variance(values: Sequence[float]) -> float:
    center = mean(values)
    return mean([(value - center) ** 2 for value in values])


def binary_entropy(probability: float) -> float:
    p = max(0.0, min(1.0, probability))
    q = 1.0 - p
    return -p * math.log(max(p, 1e-12)) - q * math.log(max(q, 1e-12))


def generalized_bernoulli_jsd(probabilities: Sequence[float]) -> float:
    """Jensen-Shannon divergence among scale-specific paired probabilities."""

    if len(probabilities) < 2:
        return 0.0
    mixture = mean(probabilities)
    divergence = binary_entropy(mixture) - mean([binary_entropy(value) for value in probabilities])
    return max(0.0, divergence / math.log(2.0))


def format_vector(values: Sequence[float]) -> str:
    return ",".join(f"{value:.6g}" for value in values)


def merge_rows(
    rows: Sequence[Mapping[str, str]],
    lengths: Sequence[int],
    target_length: int,
    validate_site_ids: bool = True,
) -> Dict[str, object]:
    site_ids = [str(row.get("site_id", "")) for row in rows]
    if not site_ids[0] or (validate_site_ids and any(site_id != site_ids[0] for site_id in site_ids[1:])):
        raise ValueError(f"structure files are not aligned: site_ids={site_ids}")
    profiles: Dict[str, List[List[float]]] = {name: [] for name in NODE_INPUT_FIELDS}
    for row, length in zip(rows, lengths):
        for name, field in NODE_INPUT_FIELDS.items():
            profiles[name].append(center_crop(parse_vector(row.get(field), length, field, site_ids[0]), target_length))

    paired_by_position = list(zip(*profiles["paired"]))
    mean_paired = [mean(values) for values in paired_by_position]
    pair_variance = [variance(values) for values in paired_by_position]
    mean_unpaired = [mean(values) for values in zip(*profiles["unpaired"])]
    mean_entropy = [mean(values) for values in zip(*profiles["entropy"])]
    mean_distance = [mean(values) for values in zip(*profiles["distance"])]
    jsd = [generalized_bernoulli_jsd(values) for values in paired_by_position]
    center = target_length // 2
    local_slice = slice(max(0, center - 10), min(target_length, center + 11))
    local_variance = mean(pair_variance[local_slice])
    local_entropy = mean(mean_entropy[local_slice])
    agreement = max(0.0, min(1.0, 1.0 - 4.0 * local_variance))
    confidence = max(0.0, min(1.0, agreement * (1.0 - local_entropy)))

    output: Dict[str, object] = dict(rows[0])
    output.update(
        {
            "multiscale_mean_paired_flags": format_vector(mean_paired),
            "multiscale_mean_unpaired_flags": format_vector(mean_unpaired),
            "multiscale_mean_entropy_flags": format_vector(mean_entropy),
            "multiscale_mean_expected_partner_distance_flags": format_vector(mean_distance),
            "multiscale_pair_variance_flags": format_vector(pair_variance),
            "multiscale_center_paired_mean": mean_paired[center],
            "multiscale_center_paired_variance": pair_variance[center],
            "multiscale_center_unpaired_mean": mean_unpaired[center],
            "multiscale_center_entropy_mean": mean_entropy[center],
            "multiscale_center_entropy_max": max(profile[center] for profile in profiles["entropy"]),
            "multiscale_profile_jsd": mean(jsd),
            "multiscale_pairing_agreement": agreement,
            "multiscale_structure_confidence": confidence,
            "multiscale_window_count": len(rows),
            "structure_feature_type": "viennarna_multiscale_partition_probability",
        }
    )
    return output


def rotate_reader(reader: Iterable[Dict[str, str]], offset: int) -> Iterable[Dict[str, str]]:
    """Cyclically shift a streaming reader while buffering only ``offset`` rows."""

    iterator = iter(reader)
    prefix = list(islice(iterator, max(1, int(offset))))
    return chain(iterator, prefix)


def build(args: argparse.Namespace) -> Dict[str, object]:
    paths = [Path(value) for value in args.structure_files.split(",") if value.strip()]
    lengths = [int(value) for value in args.window_lengths.split(",") if value.strip()]
    if len(paths) != len(lengths) or len(paths) < 2:
        raise ValueError("--structure-files and --window-lengths must contain the same 2+ entries")
    if args.target_length not in lengths:
        raise ValueError("--target-length must be one of --window-lengths")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with ExitStack() as stack:
        handles = [stack.enter_context(path.open("r", encoding="utf-8", newline="")) for path in paths]
        readers = [csv.DictReader(handle, delimiter="\t") for handle in handles]
        base_fields = list(readers[lengths.index(args.target_length)].fieldnames or [])
        output_fields = base_fields + [
            field for field in NODE_OUTPUT_FIELDS + SCALAR_OUTPUT_FIELDS if field not in base_fields
        ]
        out_handle = stack.enter_context(out_path.open("w", encoding="utf-8", newline=""))
        writer = csv.DictWriter(out_handle, fieldnames=output_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        target_index = lengths.index(args.target_length)
        ordered_readers: List[Iterable[Dict[str, str]]] = []
        for index, reader in enumerate(readers):
            if args.shuffle_non_target_scales and index != target_index:
                ordered_readers.append(rotate_reader(reader, args.shuffle_offset))
            else:
                ordered_readers.append(reader)
        for row_group in zip_longest(*ordered_readers):
            if any(row is None for row in row_group):
                raise ValueError("structure files contain different row counts")
            rows = list(row_group)  # type: ignore[arg-type]
            merged = merge_rows(
                rows,
                lengths,
                args.target_length,
                validate_site_ids=not args.shuffle_non_target_scales,
            )
            # Keep graph/scalar metadata from the target-length view, not simply
            # from the first input path supplied by the caller.
            target_row = rows[target_index]
            for field in base_fields:
                if field not in NODE_OUTPUT_FIELDS and field not in SCALAR_OUTPUT_FIELDS:
                    merged[field] = target_row.get(field, merged.get(field, ""))
            writer.writerow({field: merged.get(field, "") for field in output_fields})
            rows_written += 1
            if args.progress_every and rows_written % args.progress_every == 0:
                print(f"merged={rows_written}", flush=True)

    summary = {
        "structure_files": [str(path) for path in paths],
        "window_lengths": lengths,
        "target_length": args.target_length,
        "rows_written": rows_written,
        "shuffle_non_target_scales": bool(args.shuffle_non_target_scales),
        "shuffle_offset": args.shuffle_offset if args.shuffle_non_target_scales else 0,
        "out": str(out_path),
    }
    out_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge multi-scale ViennaRNA probability features.")
    parser.add_argument("--structure-files", required=True, help="Comma-separated probability TSVs.")
    parser.add_argument("--window-lengths", default="41,101,201")
    parser.add_argument("--target-length", type=int, default=41)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument(
        "--shuffle-non-target-scales",
        action="store_true",
        help="Cyclically misalign longer-window profiles while preserving their marginal distributions.",
    )
    parser.add_argument("--shuffle-offset", type=int, default=997)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    build(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
