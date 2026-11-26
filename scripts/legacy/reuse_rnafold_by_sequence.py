#!/usr/bin/env python3
"""Reuse existing RNAfold rows by exact 41-nt sequence, folding only misses."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable


STRUCT_FIELDS = [
    "site_id", "sequence", "dot_bracket", "MFE", "paired_flags", "base_pair_list",
    "center_paired", "central_11nt_paired_fraction", "central_21nt_paired_fraction",
    "paired_fraction_window", "MFE_per_nt", "loop_span_around_center",
]
RNAFOLD_STRUCT_RE = re.compile(r"^(?P<dot>[().]+)\s+\(\s*(?P<mfe>[-+]?\d+(?:\.\d+)?)\s*\)")


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def normalize(value: object) -> str:
    return str(value or "").upper().replace("U", "T")


def parse_structure(dot: str) -> tuple[list[int], list[tuple[int, int]]]:
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    flags = [0] * len(dot)
    for index, char in enumerate(dot, start=1):
        if char == "(":
            stack.append(index)
        elif char == ")" and stack:
            left = stack.pop()
            pairs.append((left, index))
            flags[left - 1] = 1
            flags[index - 1] = 1
    return flags, sorted(pairs)


def parse_batch(stdout: str) -> list[dict[str, object]]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    output: list[dict[str, object]] = []
    index = 0
    while index + 2 < len(lines):
        if not lines[index].startswith(">"):
            index += 1
            continue
        site_id = lines[index][1:]
        sequence = lines[index + 1].replace("U", "T")
        match = RNAFOLD_STRUCT_RE.match(lines[index + 2])
        if not match:
            index += 3
            continue
        dot = match.group("dot")
        mfe = float(match.group("mfe"))
        flags, pairs = parse_structure(dot)
        center = len(flags) // 2
        central_11 = flags[max(0, center - 5):min(len(flags), center + 6)]
        central_21 = flags[max(0, center - 10):min(len(flags), center + 11)]
        left = center
        right = center
        while left > 0 and flags[left - 1] == 0:
            left -= 1
        while right + 1 < len(flags) and flags[right + 1] == 0:
            right += 1
        output.append({
            "site_id": site_id,
            "sequence": sequence,
            "dot_bracket": dot,
            "MFE": mfe,
            "paired_flags": ",".join(map(str, flags)),
            "base_pair_list": ";".join(f"{a}-{b}" for a, b in pairs),
            "center_paired": flags[center] if flags else 0,
            "central_11nt_paired_fraction": sum(central_11) / max(len(central_11), 1),
            "central_21nt_paired_fraction": sum(central_21) / max(len(central_21), 1),
            "paired_fraction_window": sum(flags) / max(len(flags), 1),
            "MFE_per_nt": mfe / max(len(sequence), 1),
            "loop_span_around_center": right - left + 1,
        })
        index += 3
    return output


def fold(rows: list[dict[str, str]], rnafold_bin: str, batch_size: int) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        fasta = "\n".join(
            f">{row['site_id']}\n{normalize(row['sequence_41nt']) .replace('T', 'U')}"
            for row in batch
        ) + "\n"
        process = subprocess.run(
            [rnafold_bin, "--noPS"], input=fasta, text=True,
            capture_output=True, check=True,
        )
        output.extend(parse_batch(process.stdout))
        print(f"RNAfold missing {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return output


def write_tsv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRUCT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--existing-rnafold", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rnafold-bin", required=True)
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args(argv)

    dataset_rows, _ = read_tsv(Path(args.dataset))
    existing_rows, _ = read_tsv(Path(args.existing_rnafold))
    by_sequence: dict[str, dict[str, str]] = {}
    for row in existing_rows:
        sequence = normalize(row.get("sequence", ""))
        if len(sequence) == 41:
            by_sequence.setdefault(sequence, row)

    reused: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in dataset_rows:
        site_id = row["site_id"]
        if site_id in seen:
            continue
        seen.add(site_id)
        sequence = normalize(row.get("sequence_41nt", ""))
        prior = by_sequence.get(sequence)
        if prior is not None:
            copied = dict(prior)
            copied["site_id"] = site_id
            reused.append(copied)
        else:
            missing.append(row)

    new_rows = fold(missing, args.rnafold_bin, args.batch_size)
    result_by_id = {str(row["site_id"]): row for row in reused}
    result_by_id.update({str(row["site_id"]): row for row in new_rows})
    ordered = [result_by_id[row["site_id"]] for row in dataset_rows if row["site_id"] in result_by_id]
    write_tsv(Path(args.out), ordered)
    summary = {
        "dataset": args.dataset,
        "existing_rnafold": args.existing_rnafold,
        "output": args.out,
        "input_rows": len(dataset_rows),
        "existing_structure_rows": len(existing_rows),
        "reused_by_exact_41nt_sequence": len(reused),
        "new_rows_folded": len(new_rows),
        "output_rows": len(ordered),
        "unique_existing_sequences": len(by_sequence),
    }
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
