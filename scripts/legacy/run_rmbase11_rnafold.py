#!/usr/bin/env python
"""Run RNAfold for the RMBase hg38 11-class 41 nt benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


RNAFOLD_STRUCT_RE = re.compile(r"^(?P<dot>[().]+)\s+\(\s*(?P<mfe>[-+]?\d+(?:\.\d+)?)\s*\)")
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
]


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_dot_bracket(dot: str) -> Dict[str, object]:
    stack: List[int] = []
    pairs: List[Tuple[int, int]] = []
    flags = [0] * len(dot)
    for idx, char in enumerate(dot, start=1):
        if char == "(":
            stack.append(idx)
        elif char == ")" and stack:
            left = stack.pop()
            pairs.append((left, idx))
            flags[left - 1] = 1
            flags[idx - 1] = 1
    pairs.sort()
    return {"base_pair_list": pairs, "paired_flags": flags}


def parse_rnafold_structure_line(line: str) -> Tuple[str, float]:
    match = RNAFOLD_STRUCT_RE.match(line.strip())
    if not match:
        raise ValueError(f"cannot parse RNAfold structure line: {line}")
    return match.group("dot"), float(match.group("mfe"))


def loop_span_around_center(flags: Sequence[int], center_idx: int) -> int:
    left = center_idx
    right = center_idx
    while left - 1 >= 0 and flags[left - 1] == 0:
        left -= 1
    while right + 1 < len(flags) and flags[right + 1] == 0:
        right += 1
    return right - left + 1


def make_fasta(rows: Sequence[Dict[str, str]]) -> str:
    lines = []
    for row in rows:
        seq = row["sequence_41nt"].upper().replace("T", "U")
        lines.append(f">{row['site_id']}")
        lines.append(seq)
    return "\n".join(lines) + "\n"


def run_rnafold_batch(rows: Sequence[Dict[str, str]], rnafold_bin: str) -> List[Dict[str, object]]:
    if not rows:
        return []
    proc = subprocess.run(
        [rnafold_bin, "--noPS"],
        input=make_fasta(rows),
        text=True,
        capture_output=True,
        check=True,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    results: List[Dict[str, object]] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">") or i + 2 >= len(lines):
            i += 1
            continue
        site_id = lines[i][1:]
        seq = lines[i + 1]
        struct_line = lines[i + 2]
        try:
            dot, mfe = parse_rnafold_structure_line(struct_line)
        except ValueError:
            i += 3
            continue
        parsed = parse_dot_bracket(dot)
        flags: List[int] = parsed["paired_flags"]  # type: ignore[assignment]
        center_idx = len(flags) // 2
        central_11 = flags[max(0, center_idx - 5) : min(len(flags), center_idx + 6)]
        central_21 = flags[max(0, center_idx - 10) : min(len(flags), center_idx + 11)]
        results.append(
            {
                "site_id": site_id,
                "sequence": seq.replace("U", "T"),
                "dot_bracket": dot,
                "MFE": mfe,
                "paired_flags": ",".join(map(str, flags)),
                "base_pair_list": ";".join(f"{left}-{right}" for left, right in parsed["base_pair_list"]),
                "center_paired": flags[center_idx],
                "central_11nt_paired_fraction": sum(central_11) / max(len(central_11), 1),
                "central_21nt_paired_fraction": sum(central_21) / max(len(central_21), 1),
                "paired_fraction_window": sum(flags) / max(len(flags), 1),
                "MFE_per_nt": mfe / max(len(seq), 1),
                "loop_span_around_center": loop_span_around_center(flags, center_idx),
            }
        )
        i += 3
    return results


def run(args: argparse.Namespace) -> Dict[str, object]:
    rows = read_tsv(Path(args.dataset))
    if args.max_rows:
        rows = rows[: args.max_rows]
    existing_by_id: Dict[str, Dict[str, str]] = {}
    if args.existing_rnafold:
        existing_path = Path(args.existing_rnafold)
        if existing_path.exists():
            existing_by_id = {row["site_id"]: row for row in read_tsv(existing_path)}
        else:
            raise FileNotFoundError(f"--existing-rnafold not found: {existing_path}")

    if args.resume and Path(args.out).exists():
        resumed = {row["site_id"]: row for row in read_tsv(Path(args.out))}
        existing_by_id.update(resumed)

    reused_results: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, str]] = []
    seen_site_ids = set()
    for row in rows:
        site_id = row["site_id"]
        if site_id in seen_site_ids:
            continue
        seen_site_ids.add(site_id)
        existing = existing_by_id.get(site_id)
        if existing:
            reused_results.append(existing)
        else:
            missing_rows.append(row)

    all_results: List[Dict[str, object]] = []
    for start in range(0, len(missing_rows), args.batch_size):
        batch = missing_rows[start : start + args.batch_size]
        all_results.extend(run_rnafold_batch(batch, args.rnafold_bin))
        print(f"RNAfold new {min(start + args.batch_size, len(missing_rows))}/{len(missing_rows)}", flush=True)

    out_path = Path(args.out)
    result_by_id = {str(row["site_id"]): row for row in reused_results}
    result_by_id.update({str(row["site_id"]): row for row in all_results})
    ordered_results = [result_by_id[row["site_id"]] for row in rows if row["site_id"] in result_by_id]
    write_tsv(out_path, ordered_results, STRUCT_FIELDS)
    summary = {
        "dataset": args.dataset,
        "out": args.out,
        "rnafold_bin": args.rnafold_bin,
        "input_rows": len(rows),
        "reused_rows": len(reused_results),
        "new_rows_to_fold": len(missing_rows),
        "newly_folded_rows": len(all_results),
        "folded_rows": len(ordered_results),
        "batch_size": args.batch_size,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RNAfold on RMBase 11-class 41 nt windows.")
    parser.add_argument("--dataset", default="rmbase11_work/datasets/rmbase11_hg38_41nt_cdhit80_like.tsv")
    parser.add_argument("--out", default="rmbase11_work/structure/rnafold_41nt_rmbase11.tsv")
    parser.add_argument("--rnafold-bin", default="RNAfold")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--existing-rnafold", default="", help="Optional existing RNAfold TSV to reuse by site_id.")
    parser.add_argument("--resume", action="store_true", help="If --out exists, reuse its folded rows and fold only missing rows.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
