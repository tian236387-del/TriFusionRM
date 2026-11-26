#!/usr/bin/env python
"""Build RNAplfold local-accessibility structure features.

This is an optional, slower feature generator for structure ablation.  It
converts RNAplfold unpaired probabilities into the same TSV schema consumed by
``train_motif_struct_fusion_v4.py``.  For each sequence, ``paired_flags`` are
stored as local paired probabilities: ``1 - P(unpaired length 1)``.

Because RNAplfold writes per-sequence output files, this script is intended for
server use and supports ``--resume``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
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
    "mean_unpaired_probability",
    "center_unpaired_probability",
    "structure_feature_type",
]

UBOX_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+([0-9.eE+-]+)\s+ubox")


def iter_tsv(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_by_id(path: str) -> Dict[str, Dict[str, str]]:
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


def parse_paired_flags(value: str, length: int) -> List[float]:
    flags = [parse_float(item) for item in str(value or "").split(",") if item != ""]
    return (flags[:length] + [0.0] * max(0, length - len(flags)))[:length]


def parse_pairs(value: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for item in str(value or "").split(";"):
        if not item or "-" not in item:
            continue
        left, right = item.split("-", 1)
        try:
            pairs.append((int(float(left)), int(float(right))))
        except ValueError:
            continue
    return pairs


def pairs_to_dot(length: int, pairs: Sequence[Tuple[int, int]]) -> str:
    chars = ["."] * length
    for left, right in pairs:
        if 1 <= left < right <= length:
            chars[left - 1] = "("
            chars[right - 1] = ")"
    return "".join(chars)


def loop_span_around_center(paired_probability: Sequence[float], center_idx: int, threshold: float) -> int:
    left = center_idx
    right = center_idx
    while left - 1 >= 0 and paired_probability[left - 1] < threshold:
        left -= 1
    while right + 1 < len(paired_probability) and paired_probability[right + 1] < threshold:
        right += 1
    return right - left + 1


def parse_lunp(path: Path, length: int) -> List[float]:
    values = [1.0] * length
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            idx = int(float(parts[0])) - 1
            prob = float(parts[1])
        except ValueError:
            continue
        if 0 <= idx < length:
            values[idx] = max(0.0, min(1.0, prob))
    return values


def parse_dp_ps(path: Path, length: int, threshold: float) -> List[Tuple[int, int]]:
    best_partner = [-1] * length
    best_prob = [0.0] * length
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = UBOX_RE.match(line)
        if not match:
            continue
        i = int(match.group(1)) - 1
        j = int(match.group(2)) - 1
        try:
            prob = float(match.group(3)) ** 2
        except ValueError:
            continue
        if not (0 <= i < j < length):
            continue
        if prob > best_prob[i]:
            best_prob[i] = prob
            best_partner[i] = j
        if prob > best_prob[j]:
            best_prob[j] = prob
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


def run_rnaplfold_one(
    seq: str,
    rnaplfold_bin: str,
    window: int,
    max_pair_span: int,
    unpaired_length: int,
    pair_threshold: float,
) -> Tuple[List[float], List[Tuple[int, int]]]:
    seq_u = seq.upper().replace("T", "U")
    length = len(seq_u)
    with tempfile.TemporaryDirectory(prefix="rnaplfold_") as tmp:
        tmp_path = Path(tmp)
        cmd = [
            rnaplfold_bin,
            "-W",
            str(min(window, length)),
            "-L",
            str(min(max_pair_span, max(1, length - 1))),
            "-u",
            str(unpaired_length),
        ]
        subprocess.run(
            cmd,
            input=seq_u + "\n",
            text=True,
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )
        lunp_files = list(tmp_path.glob("*_lunp")) + list(tmp_path.glob("plfold_lunp"))
        dp_files = list(tmp_path.glob("*_dp.ps")) + list(tmp_path.glob("plfold_dp.ps"))
        unpaired = parse_lunp(lunp_files[0], length) if lunp_files else [1.0] * length
        pairs = parse_dp_ps(dp_files[0], length, pair_threshold) if dp_files else []
    return unpaired, pairs


def build_feature_row(
    site_id: str,
    seq: str,
    unpaired_probability: Sequence[float],
    pairs: Sequence[Tuple[int, int]],
    mfe_reference: Optional[Mapping[str, str]],
    pair_threshold: float,
) -> Dict[str, object]:
    seq_t = seq.upper().replace("U", "T")
    length = len(seq_t)
    paired_probability = [1.0 - max(0.0, min(1.0, p)) for p in unpaired_probability]
    if not pairs and mfe_reference:
        pairs = parse_pairs(str(mfe_reference.get("base_pair_list", "")))
    dot = pairs_to_dot(length, pairs)
    center_idx = length // 2
    central_11 = paired_probability[max(0, center_idx - 5) : min(length, center_idx + 6)]
    central_21 = paired_probability[max(0, center_idx - 10) : min(length, center_idx + 11)]
    mfe = parse_float(mfe_reference.get("MFE", 0.0) if mfe_reference else 0.0)
    return {
        "site_id": site_id,
        "sequence": seq_t,
        "dot_bracket": dot,
        "MFE": mfe,
        "paired_flags": ",".join(f"{p:.6g}" for p in paired_probability),
        "base_pair_list": ";".join(f"{left}-{right}" for left, right in pairs),
        "center_paired": paired_probability[center_idx] if paired_probability else 0.0,
        "central_11nt_paired_fraction": sum(central_11) / max(len(central_11), 1),
        "central_21nt_paired_fraction": sum(central_21) / max(len(central_21), 1),
        "paired_fraction_window": sum(paired_probability) / max(length, 1),
        "MFE_per_nt": mfe / max(length, 1),
        "loop_span_around_center": loop_span_around_center(paired_probability, center_idx, pair_threshold),
        "mean_unpaired_probability": sum(unpaired_probability) / max(length, 1),
        "center_unpaired_probability": unpaired_probability[center_idx] if unpaired_probability else 1.0,
        "structure_feature_type": "rnaplfold_local_accessibility",
    }


def build(args: argparse.Namespace) -> Dict[str, object]:
    if shutil.which(args.rnaplfold_bin) is None:
        raise FileNotFoundError(f"RNAplfold executable not found: {args.rnaplfold_bin}")

    existing_by_id = load_by_id(args.out) if args.resume and Path(args.out).exists() else {}
    mfe_by_id = load_by_id(args.mfe_reference)

    rows_out: List[Dict[str, object]] = []
    input_rows = 0
    reused = 0
    computed = 0
    failed = 0
    seen = set()
    for row in iter_tsv(Path(args.dataset)):
        input_rows += 1
        if args.max_rows and input_rows > args.max_rows:
            break
        site_id = row["site_id"]
        if site_id in seen:
            continue
        seen.add(site_id)
        existing = existing_by_id.get(site_id)
        if existing:
            rows_out.append(existing)
            reused += 1
            continue
        seq = row.get(args.sequence_column, "")
        try:
            unpaired, pairs = run_rnaplfold_one(
                seq,
                args.rnaplfold_bin,
                args.window,
                args.max_pair_span,
                args.unpaired_length,
                args.pair_threshold,
            )
            rows_out.append(build_feature_row(site_id, seq, unpaired, pairs, mfe_by_id.get(site_id), args.pair_threshold))
            computed += 1
        except Exception as exc:
            failed += 1
            if failed <= 10:
                print(f"warning: failed {site_id}: {exc}", flush=True)
        if computed and computed % args.progress_every == 0:
            print(f"computed={computed} reused={reused} failed={failed}", flush=True)

    out_path = Path(args.out)
    write_tsv(out_path, rows_out, STRUCT_FIELDS)
    summary = {
        "dataset": args.dataset,
        "out": args.out,
        "feature_type": "rnaplfold_local_accessibility",
        "input_rows": input_rows,
        "unique_rows_written": len(rows_out),
        "reused_rows": reused,
        "computed_rows": computed,
        "failed_rows": failed,
        "window": args.window,
        "max_pair_span": args.max_pair_span,
        "unpaired_length": args.unpaired_length,
        "pair_threshold": args.pair_threshold,
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RNAplfold local-accessibility structure features.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv")
    parser.add_argument("--out", default="rmbase11_full_work/structure/rnaplfold_accessibility_41nt_all_sites.tsv")
    parser.add_argument("--mfe-reference", default="rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv")
    parser.add_argument("--sequence-column", default="sequence_41nt")
    parser.add_argument("--rnaplfold-bin", default="RNAplfold")
    parser.add_argument("--window", type=int, default=41)
    parser.add_argument("--max-pair-span", type=int, default=40)
    parser.add_argument("--unpaired-length", type=int, default=1)
    parser.add_argument("--pair-threshold", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
