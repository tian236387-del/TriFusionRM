#!/usr/bin/env python
"""Add odd-length genomic windows to an RMBase site table.

The RMBase col29 files already contain 41 nt windows, but official MultiRM and
TransRNAm scripts usually train on 51 nt or 101 nt inputs.  This utility rebuilds
centered windows from an hg38 FASTA using the genomic coordinate and strand in
the RMBase-derived TSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BASES = set("ACGT")
RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def normalize_seq(seq: str) -> str:
    return str(seq or "").upper().replace("U", "T")


def reverse_complement(seq: str) -> str:
    return seq.translate(RC_TABLE)[::-1].upper()


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def build_fai(fasta: Path, fai: Path) -> None:
    """Create a minimal .fai for an uncompressed FASTA."""
    if fasta.suffix == ".gz":
        raise ValueError("Random access requires an uncompressed FASTA. Gunzip the FASTA first.")
    records: List[Tuple[str, int, int, int, int]] = []
    current_name: Optional[str] = None
    current_len = 0
    seq_offset = 0
    line_bases = 0
    line_width = 0
    with fasta.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.startswith(b">"):
                if current_name is not None:
                    records.append((current_name, current_len, seq_offset, line_bases, line_width))
                current_name = line[1:].split()[0].decode("ascii", errors="replace")
                current_len = 0
                seq_offset = handle.tell()
                line_bases = 0
                line_width = 0
                continue
            stripped = line.rstrip(b"\r\n")
            if not stripped:
                continue
            if line_bases == 0:
                line_bases = len(stripped)
                line_width = len(line)
            current_len += len(stripped)
        if current_name is not None:
            records.append((current_name, current_len, seq_offset, line_bases, line_width))
    with fai.open("w", encoding="utf-8", newline="") as out:
        for name, length, offset, bases, width in records:
            out.write(f"{name}\t{length}\t{offset}\t{bases}\t{width}\n")


class IndexedFasta:
    def __init__(self, fasta: Path, fai: Optional[Path] = None):
        self.fasta = fasta
        self.fai = fai or fasta.with_suffix(fasta.suffix + ".fai")
        if not self.fai.exists():
            build_fai(fasta, self.fai)
        self.index: Dict[str, Tuple[int, int, int, int]] = {}
        with self.fai.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                name = parts[0]
                self.index[name] = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        self.handle = fasta.open("rb")

    def close(self) -> None:
        self.handle.close()

    def resolve_chrom(self, chrom: str) -> Optional[str]:
        chrom = str(chrom)
        if chrom in self.index:
            return chrom
        if chrom.startswith("chr") and chrom[3:] in self.index:
            return chrom[3:]
        prefixed = "chr" + chrom
        if prefixed in self.index:
            return prefixed
        return None

    def fetch(self, chrom: str, start0: int, end0: int) -> Optional[str]:
        resolved = self.resolve_chrom(chrom)
        if resolved is None:
            return None
        length, offset, line_bases, line_width = self.index[resolved]
        if start0 < 0 or end0 > length or start0 >= end0:
            return None
        start_byte = offset + (start0 // line_bases) * line_width + (start0 % line_bases)
        approx = (end0 - start0) + ((end0 - start0) // line_bases + 2) * (line_width - line_bases)
        self.handle.seek(start_byte)
        chunk = self.handle.read(approx)
        seq = chunk.replace(b"\n", b"").replace(b"\r", b"")[: end0 - start0]
        if len(seq) != end0 - start0:
            return None
        return seq.decode("ascii", errors="replace").upper()


def centered_window(fasta: IndexedFasta, row: Mapping[str, str], length: int) -> Tuple[str, str]:
    radius = length // 2
    center0 = safe_int(row.get("start", "0"))
    seq = fasta.fetch(str(row.get("chr", "")), center0 - radius, center0 + radius + 1)
    if seq is None:
        return "", "missing_or_out_of_bounds"
    if str(row.get("strand", "+")) == "-":
        seq = reverse_complement(seq)
    seq = normalize_seq(seq)
    if len(seq) != length:
        return "", "invalid_length"
    if set(seq) - BASES:
        return seq, "invalid_base"
    return seq, "ok"


def parse_lengths(text: str) -> List[int]:
    lengths = []
    for item in text.replace(",", " ").split():
        value = int(item)
        if value <= 0 or value % 2 == 0:
            raise ValueError(f"Window length must be a positive odd integer: {value}")
        lengths.append(value)
    if not lengths:
        raise ValueError("No window lengths were provided")
    return sorted(set(lengths))


def run(args: argparse.Namespace) -> Dict[str, object]:
    lengths = parse_lengths(args.lengths)
    dataset = Path(args.dataset)
    out = Path(args.out)
    summary_out = Path(args.summary_out) if args.summary_out else out.with_suffix(out.suffix + ".summary.json")
    fasta = IndexedFasta(Path(args.fasta), Path(args.fai) if args.fai else None)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    status_counts: Dict[str, int] = {}
    validation = {"checked": 0, "matched": 0, "mismatched": 0, "missing_existing": 0}

    try:
        with dataset.open("r", encoding="utf-8", newline="") as handle, out.open("w", encoding="utf-8", newline="") as out_handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"Input dataset has no header: {dataset}")
            added_cols = [f"sequence_{length}nt" for length in lengths]
            fieldnames = list(reader.fieldnames)
            for col in added_cols:
                if col not in fieldnames:
                    fieldnames.append(col)
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in reader:
                total += 1
                output = dict(row)
                row_ok = True
                if args.validate_existing_41 and "sequence_41nt" in row:
                    seq41, status41 = centered_window(fasta, row, 41)
                    validation["checked"] += 1
                    existing = normalize_seq(row.get("sequence_41nt", ""))
                    if not existing:
                        validation["missing_existing"] += 1
                    elif status41 == "ok" and seq41 == existing:
                        validation["matched"] += 1
                    else:
                        validation["mismatched"] += 1
                        if args.drop_mismatched_41:
                            row_ok = False
                for length in lengths:
                    col = f"sequence_{length}nt"
                    if col in row and row[col]:
                        seq = normalize_seq(row[col])
                        status = "ok" if len(seq) == length and not (set(seq) - BASES) else "preexisting_invalid"
                    else:
                        seq, status = centered_window(fasta, row, length)
                    status_counts[f"{length}:{status}"] = status_counts.get(f"{length}:{status}", 0) + 1
                    output[col] = seq
                    if status != "ok" and args.drop_invalid:
                        row_ok = False
                if not row_ok:
                    continue
                writer.writerow({field: output.get(field, "") for field in fieldnames})
                written += 1
    finally:
        fasta.close()

    summary = {
        "input_dataset": str(dataset),
        "output_dataset": str(out),
        "fasta": str(args.fasta),
        "lengths": lengths,
        "total_rows": total,
        "written_rows": written,
        "status_counts": status_counts,
        "validation_existing_41": validation,
        "drop_invalid": bool(args.drop_invalid),
        "drop_mismatched_41": bool(args.drop_mismatched_41),
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract 51/101 nt RMBase windows from hg38 FASTA.")
    parser.add_argument("--dataset", required=True, help="RMBase TSV with chr/start/end/strand columns.")
    parser.add_argument("--fasta", required=True, help="Uncompressed hg38 FASTA.")
    parser.add_argument("--fai", default="", help="Optional FASTA .fai index. Created if absent.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--lengths", default="51,101", help="Comma/space separated odd window lengths.")
    parser.add_argument("--validate-existing-41", action="store_true", help="Compare FASTA-derived 41nt windows to sequence_41nt.")
    parser.add_argument("--drop-mismatched-41", action="store_true", help="Drop rows whose FASTA 41nt sequence disagrees with sequence_41nt.")
    parser.add_argument("--drop-invalid", action="store_true", help="Drop rows with missing/out-of-bound/ambiguous generated windows.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
