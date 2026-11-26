#!/usr/bin/env python
"""Build RNAstructure Fold MFE structure features for RMBase windows.

RNAstructure ``Fold`` writes CT files rather than dot-bracket output.  This
script runs Fold per sequence, parses the first CT structure, and writes the
same structure TSV schema consumed by ``train_motif_struct_fusion_v4.py``.

The script is intended for server use and supports ``--resume``.  New rows are
appended as they finish, so an interrupted feature-generation job can continue
without losing all completed Fold calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
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
    "structure_feature_type",
]

ENERGY_RE = re.compile(r"(?:ENERGY|dG)\s*=\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)


def iter_tsv(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def read_existing(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            site_id = str(row.get("site_id", "")).strip()
            if site_id:
                rows[site_id] = row
    return rows


def append_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: object, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def binary_available(name: str) -> bool:
    return bool(shutil.which(name) or Path(name).exists())


def write_seq_file(path: Path, site_id: str, seq: str) -> None:
    seq_u = str(seq or "").upper().replace("T", "U")
    # RNAstructure .seq format: comments, title, then sequence terminated by 1.
    path.write_text(f"; {site_id}\n{site_id}\n{seq_u}1\n", encoding="ascii")


def parse_ct_header_energy(header: str) -> float:
    match = ENERGY_RE.search(header)
    if match:
        return parse_float(match.group(1), 0.0)
    return 0.0


def pairs_to_dot(length: int, pairs: Sequence[Tuple[int, int]]) -> str:
    chars = ["."] * length
    for left, right in pairs:
        if 1 <= left < right <= length:
            chars[left - 1] = "("
            chars[right - 1] = ")"
    return "".join(chars)


def loop_span_around_center(flags: Sequence[float], center_idx: int) -> int:
    left = center_idx
    right = center_idx
    while left - 1 >= 0 and flags[left - 1] < 0.5:
        left -= 1
    while right + 1 < len(flags) and flags[right + 1] < 0.5:
        right += 1
    return right - left + 1


def parse_ct(path: Path, structure_index: int = 1) -> Tuple[str, float, List[Tuple[int, int]]]:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    idx = 0
    current = 0
    while idx < len(lines):
        header = lines[idx].strip()
        parts = header.split()
        length = parse_int(parts[0] if parts else "", 0)
        if length <= 0:
            idx += 1
            continue
        current += 1
        nuc_lines = lines[idx + 1 : idx + 1 + length]
        if len(nuc_lines) < length:
            raise ValueError(f"incomplete CT structure in {path}")
        if current != structure_index:
            idx += 1 + length
            continue
        seq_chars: List[str] = []
        pairs: List[Tuple[int, int]] = []
        for raw in nuc_lines:
            cols = raw.split()
            if len(cols) < 6:
                continue
            pos = parse_int(cols[0], 0)
            base = cols[1].upper().replace("U", "T")
            partner = parse_int(cols[4], 0)
            seq_chars.append(base)
            if partner > pos > 0:
                pairs.append((pos, partner))
        pairs.sort()
        return "".join(seq_chars), parse_ct_header_energy(header), pairs
    raise ValueError(f"no CT structure #{structure_index} found in {path}")


def build_feature_row(
    site_id: str,
    seq: str,
    dot: str,
    mfe: float,
    pairs: Sequence[Tuple[int, int]],
) -> Dict[str, object]:
    seq_t = seq.upper().replace("U", "T")
    length = len(seq_t) or len(dot) or 41
    flags = [0.0] * length
    for left, right in pairs:
        if 1 <= left <= length:
            flags[left - 1] = 1.0
        if 1 <= right <= length:
            flags[right - 1] = 1.0
    center_idx = length // 2
    central_11 = flags[max(0, center_idx - 5) : min(length, center_idx + 6)]
    central_21 = flags[max(0, center_idx - 10) : min(length, center_idx + 11)]
    return {
        "site_id": site_id,
        "sequence": seq_t,
        "dot_bracket": dot,
        "MFE": mfe,
        "paired_flags": ",".join(f"{value:.6g}" for value in flags),
        "base_pair_list": ";".join(f"{left}-{right}" for left, right in pairs),
        "center_paired": flags[center_idx] if flags else 0.0,
        "central_11nt_paired_fraction": sum(central_11) / max(len(central_11), 1),
        "central_21nt_paired_fraction": sum(central_21) / max(len(central_21), 1),
        "paired_fraction_window": sum(flags) / max(length, 1),
        "MFE_per_nt": mfe / max(length, 1),
        "loop_span_around_center": loop_span_around_center(flags, center_idx),
        "structure_feature_type": "rnastructure_fold_mfe",
    }


def run_fold_one(task: Tuple[str, str, str, Tuple[str, ...], int]) -> Tuple[str, Optional[Dict[str, object]], str]:
    site_id, seq, fold_bin, fold_extra_args, structure_index = task
    seq_t = str(seq or "").upper().replace("U", "T")
    if not seq_t:
        return site_id, None, "empty sequence"
    with tempfile.TemporaryDirectory(prefix="rnastructure_fold_") as tmp:
        tmp_path = Path(tmp)
        seq_path = tmp_path / "input.seq"
        ct_path = tmp_path / "output.ct"
        write_seq_file(seq_path, site_id, seq_t)
        cmd = [fold_bin, str(seq_path), str(ct_path), *fold_extra_args]
        proc = subprocess.run(cmd, text=True, cwd=tmp_path, capture_output=True)
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")
            return site_id, None, f"Fold failed with code {proc.returncode}: {message[:400]}"
        if not ct_path.exists():
            return site_id, None, "Fold did not create a CT file"
        try:
            parsed_seq, mfe, pairs = parse_ct(ct_path, structure_index=structure_index)
            if not parsed_seq:
                parsed_seq = seq_t
            dot = pairs_to_dot(len(parsed_seq), pairs)
            return site_id, build_feature_row(site_id, parsed_seq, dot, mfe, pairs), ""
        except Exception as exc:
            return site_id, None, str(exc)


def build_tasks(args: argparse.Namespace, existing_by_id: Mapping[str, Mapping[str, str]]) -> Tuple[List[Tuple[str, str, str, Tuple[str, ...], int]], int, int]:
    tasks: List[Tuple[str, str, str, Tuple[str, ...], int]] = []
    seen = set()
    input_rows = 0
    reused = 0
    fold_extra_args = tuple(shlex.split(args.fold_extra_args))
    for row in iter_tsv(Path(args.dataset)):
        input_rows += 1
        if args.max_rows and input_rows > args.max_rows:
            break
        site_id = row.get("site_id", "")
        if not site_id or site_id in seen:
            continue
        seen.add(site_id)
        if site_id in existing_by_id:
            reused += 1
            continue
        seq = row.get(args.sequence_column, "") or row.get("sequence", "")
        tasks.append((site_id, seq, args.fold_bin, fold_extra_args, args.structure_index))
    return tasks, input_rows, reused


def build(args: argparse.Namespace) -> Dict[str, object]:
    if not binary_available(args.fold_bin):
        raise FileNotFoundError(
            f"RNAstructure Fold executable not found: {args.fold_bin}. "
            "Install RNAstructure, add Fold to PATH, or set --fold-bin /path/to/Fold."
        )
    out_path = Path(args.out)
    if out_path.exists() and not args.resume:
        out_path.unlink()
    existing_by_id = read_existing(out_path) if args.resume and out_path.exists() else {}
    tasks, input_rows, reused = build_tasks(args, existing_by_id)

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    if write_header:
        append_tsv(out_path, [], STRUCT_FIELDS, write_header=True)

    computed = 0
    failed = 0
    first_errors: List[str] = []
    pending: List[Mapping[str, object]] = []

    def flush() -> None:
        nonlocal pending
        append_tsv(out_path, pending, STRUCT_FIELDS, write_header=False)
        pending = []

    if tasks:
        print(f"RNAstructure Fold tasks={len(tasks)} reused={reused} workers={args.workers}", flush=True)
    if args.workers <= 1:
        iterator = map(run_fold_one, tasks)
        close_pool = None
    else:
        pool = Pool(processes=args.workers)
        iterator = pool.imap_unordered(run_fold_one, tasks, chunksize=max(1, args.chunk_size))
        close_pool = pool
    try:
        for site_id, row, error in iterator:
            if row is None:
                failed += 1
                if len(first_errors) < 10:
                    first_errors.append(f"{site_id}: {error}")
                    print(f"warning: {site_id}: {error}", flush=True)
            else:
                pending.append(row)
                computed += 1
            if len(pending) >= args.write_batch_size:
                flush()
            if (computed + failed) and (computed + failed) % args.progress_every == 0:
                print(f"processed={computed + failed}/{len(tasks)} computed={computed} failed={failed} reused={reused}", flush=True)
        if pending:
            flush()
    finally:
        if close_pool is not None:
            close_pool.close()
            close_pool.join()

    summary = {
        "dataset": args.dataset,
        "out": args.out,
        "feature_type": "rnastructure_fold_mfe",
        "fold_bin": args.fold_bin,
        "fold_extra_args": args.fold_extra_args,
        "input_rows": input_rows,
        "reused_rows": reused,
        "new_rows_to_fold": len(tasks),
        "newly_folded_rows": computed,
        "failed_rows": failed,
        "workers": args.workers,
        "first_errors": first_errors,
        "resume": args.resume,
    }
    out_path.with_suffix(out_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RNAstructure Fold MFE structure features.")
    parser.add_argument("--dataset", default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv")
    parser.add_argument("--out", default="rmbase11_full_work/structure/rnastructure_fold_41nt_all_sites.tsv")
    parser.add_argument("--sequence-column", default="sequence_41nt")
    parser.add_argument("--fold-bin", default="Fold")
    parser.add_argument("--fold-extra-args", default="", help="Optional extra arguments passed to RNAstructure Fold.")
    parser.add_argument("--structure-index", type=int, default=1, help="Which CT structure to parse; default is the first MFE structure.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--write-batch-size", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
