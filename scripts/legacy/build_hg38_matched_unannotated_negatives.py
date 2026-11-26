#!/usr/bin/env python
"""Build matched hg38 unannotated candidates for true site detection.

These candidates are not claimed to be experimentally verified negatives.
They are sites absent from the complete RMBase positive-coordinate index and
matched to positives by central base, transcript/region where possible, and
local genomic context otherwise. Every record retains its matched positive and
frozen split assignments so downstream leakage audits can keep each pair in the
same partition.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DNA = set("ACGT")
COMPLEMENT = bytes.maketrans(b"ACGTN", b"TGCAN")
SPLIT_COLUMNS = [
    "split_cluster201",
    "split_gene_holdout",
    "split_transcript_holdout",
    "split_study_holdout",
    "split_lockbox",
    "is_frozen_lockbox",
]
OUTPUT_FIELDS = [
    "site_id",
    "det_label",
    "detection_class",
    "negative_status",
    "matched_positive_site_id",
    "matched_mod_type",
    "match_level",
    "chr",
    "start",
    "end",
    "strand",
    "gene_id",
    "transcript_id",
    "gene_name",
    "gene_type",
    "region_type",
    "support_num",
    "support_list",
    "pub_list",
    "cell_list",
    "seq_type_list",
    "source_file",
    "center_base",
    "sequence_41nt",
    "sequence_101nt",
    "sequence_201nt",
    "gc_fraction_201nt",
    "distance_to_nearest_known_modification",
    *SPLIT_COLUMNS,
]


def strip_version(value: str) -> str:
    return value.split(".", 1)[0]


def parse_attributes(raw: str) -> dict[str, str]:
    return {key: value for key, value in re.findall(r'(\S+) "([^"]*)"', raw)}


def reverse_complement(sequence: bytes) -> bytes:
    return sequence.translate(COMPLEMENT)[::-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fasta_gz(path: Path) -> dict[str, bytes]:
    sequences: dict[str, bytearray] = {}
    current = ""
    with gzip.open(path, "rb") as handle:
        for raw in handle:
            if raw.startswith(b">"):
                current = raw[1:].split(None, 1)[0].decode("ascii")
                sequences[current] = bytearray()
            elif current:
                sequences[current].extend(raw.strip().upper())
    return {name: bytes(value) for name, value in sequences.items()}


@dataclass
class TranscriptIntervals:
    chromosome: str = ""
    strand: str = "+"
    gene_id: str = ""
    exons: list[tuple[int, int]] = field(default_factory=list)
    cds: list[tuple[int, int]] = field(default_factory=list)

    def merged(self, intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        output: list[list[int]] = []
        for start, end in sorted(intervals):
            if not output or start > output[-1][1]:
                output.append([start, end])
            else:
                output[-1][1] = max(output[-1][1], end)
        return [(start, end) for start, end in output]

    def introns(self) -> list[tuple[int, int]]:
        exons = self.merged(self.exons)
        return [(left[1], right[0]) for left, right in zip(exons, exons[1:]) if right[0] > left[1]]

    def region_intervals(self, region: str) -> list[tuple[int, int]]:
        exons = self.merged(self.exons)
        cds = self.merged(self.cds)
        if region == "CDS" and cds:
            return cds
        if region == "intron":
            return self.introns()
        if region in {"5UTR", "3UTR"} and cds:
            cds_left = min(start for start, _ in cds)
            cds_right = max(end for _, end in cds)
            left = [(start, min(end, cds_left)) for start, end in exons if start < cds_left]
            right = [(max(start, cds_right), end) for start, end in exons if end > cds_right]
            left = [(start, end) for start, end in left if end > start]
            right = [(start, end) for start, end in right if end > start]
            if self.strand == "+":
                return left if region == "5UTR" else right
            return right if region == "5UTR" else left
        if region in {"exon", "5UTR", "3UTR"}:
            return exons
        return []


def parse_relevant_gtf(path: Path, transcript_ids: set[str]) -> dict[str, TranscriptIntervals]:
    transcripts: dict[str, TranscriptIntervals] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            if not raw or raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] not in {"exon", "CDS"}:
                continue
            attributes = parse_attributes(fields[8])
            transcript_id = strip_version(attributes.get("transcript_id", ""))
            if transcript_id not in transcript_ids:
                continue
            record = transcripts.setdefault(transcript_id, TranscriptIntervals())
            record.chromosome = fields[0]
            record.strand = fields[6]
            record.gene_id = strip_version(attributes.get("gene_id", ""))
            interval = (int(fields[3]) - 1, int(fields[4]))
            (record.exons if fields[2] == "exon" else record.cds).append(interval)
    return transcripts


class KnownIndex:
    def __init__(self, positions: Mapping[str, Sequence[int]]):
        self.positions = {chrom: sorted(set(values)) for chrom, values in positions.items()}

    def nearest_distance(self, chrom: str, position: int) -> int:
        values = self.positions.get(chrom, [])
        index = bisect.bisect_left(values, position)
        candidates = []
        if index < len(values):
            candidates.append(abs(values[index] - position))
        if index:
            candidates.append(abs(values[index - 1] - position))
        return min(candidates) if candidates else 10**12


def oriented_base(genome: Mapping[str, bytes], chrom: str, position: int, strand: str) -> str:
    sequence = genome.get(chrom, b"")
    if position < 0 or position >= len(sequence):
        return "N"
    base = sequence[position : position + 1]
    if strand == "-":
        base = base.translate(COMPLEMENT)
    return base.decode("ascii")


def oriented_window(genome: Mapping[str, bytes], chrom: str, position: int, strand: str, radius: int) -> str:
    sequence = genome.get(chrom, b"")
    if position - radius < 0 or position + radius + 1 > len(sequence):
        return ""
    window = sequence[position - radius : position + radius + 1]
    if strand == "-":
        window = reverse_complement(window)
    text = window.decode("ascii")
    return text if set(text) <= DNA else ""


def candidate_positions(
    intervals: Sequence[tuple[int, int]],
    genome: Mapping[str, bytes],
    chrom: str,
    strand: str,
    expected_base: str,
    known: KnownIndex,
    exclusion_radius: int,
) -> list[int]:
    values: list[int] = []
    for start, end in intervals:
        start = max(start, 100)
        end = min(end, len(genome.get(chrom, b"")) - 100)
        for position in range(start, end):
            if known.nearest_distance(chrom, position) <= exclusion_radius:
                continue
            if oriented_base(genome, chrom, position, strand) != expected_base:
                continue
            values.append(position)
    return values


def sample_candidate_positions(
    intervals: Sequence[tuple[int, int]],
    target_count: int,
    genome: Mapping[str, bytes],
    chrom: str,
    strand: str,
    expected_base: str,
    known: KnownIndex,
    exclusion_radius: int,
    used: set[tuple[str, int, str]],
    rng: random.Random,
) -> list[int]:
    sequence_length = len(genome.get(chrom, b""))
    eligible: list[tuple[int, int]] = []
    cumulative: list[int] = []
    total = 0
    for start, end in intervals:
        start = max(start, 100)
        end = min(end, sequence_length - 100)
        if end <= start:
            continue
        eligible.append((start, end))
        total += end - start
        cumulative.append(total)
    if total <= 0:
        return []
    values: list[int] = []
    seen: set[int] = set()
    max_attempts = max(256, target_count * 20)
    for _ in range(max_attempts):
        offset = rng.randrange(total)
        interval_index = bisect.bisect_right(cumulative, offset)
        prior = cumulative[interval_index - 1] if interval_index else 0
        start, _ = eligible[interval_index]
        position = start + offset - prior
        if position in seen or (chrom, position, strand) in used:
            continue
        seen.add(position)
        if known.nearest_distance(chrom, position) <= exclusion_radius:
            continue
        if oriented_base(genome, chrom, position, strand) != expected_base:
            continue
        if not oriented_window(genome, chrom, position, strand, 100):
            continue
        values.append(position)
        if len(values) >= target_count:
            break
    return values


def choose_local_candidate(
    positive: Mapping[str, str],
    genome: Mapping[str, bytes],
    known: KnownIndex,
    exclusion_radius: int,
    local_radius: int,
    used: set[tuple[str, int, str]],
    rng: random.Random,
    attempts: int = 512,
) -> int | None:
    chrom = positive["chr"]
    center = int(positive["start"])
    strand = positive["strand"]
    expected = positive["center_base"].replace("U", "T")
    target_gc = gc_fraction(positive["sequence_201nt"])
    sequence_length = len(genome.get(chrom, b""))
    low = max(100, center - local_radius)
    high = min(sequence_length - 101, center + local_radius)
    if high <= low:
        return None
    candidates: list[tuple[float, int]] = []
    for _ in range(attempts):
        position = rng.randint(low, high)
        key = (chrom, position, strand)
        if key in used or known.nearest_distance(chrom, position) <= exclusion_radius:
            continue
        if oriented_base(genome, chrom, position, strand) != expected:
            continue
        window = oriented_window(genome, chrom, position, strand, 100)
        if not window:
            continue
        candidates.append((abs(gc_fraction(window) - target_gc), position))
    return min(candidates)[1] if candidates else None


def gc_fraction(sequence: str) -> float:
    return (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)


def gc_matched_candidate_buckets(
    positions: Sequence[int],
    genome: Mapping[str, bytes],
    chrom: str,
    strand: str,
    rng: random.Random,
) -> dict[int, list[int]]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for position in positions:
        window = oriented_window(genome, chrom, position, strand, 100)
        if window:
            buckets[window.count("G") + window.count("C")].append(position)
    for values in buckets.values():
        rng.shuffle(values)
    return buckets


def pop_gc_matched_candidate(
    buckets: Mapping[int, list[int]],
    target_gc_count: int,
    chrom: str,
    strand: str,
    used: set[tuple[str, int, str]],
    rng: random.Random,
) -> int | None:
    for delta in range(202):
        counts = [target_gc_count - delta]
        if delta:
            counts.append(target_gc_count + delta)
            rng.shuffle(counts)
        for count in counts:
            values = buckets.get(count)
            while values:
                position = values.pop()
                if (chrom, position, strand) not in used:
                    return position
    return None


def write_tsv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_negative_row(
    positive: Mapping[str, str],
    position: int,
    match_level: str,
    genome: Mapping[str, bytes],
    known: KnownIndex,
) -> dict[str, object]:
    chrom, strand = positive["chr"], positive["strand"]
    sequence_201 = oriented_window(genome, chrom, position, strand, 100)
    if len(sequence_201) != 201:
        raise ValueError(
            f"Candidate window is invalid for {positive['site_id']}: {chrom}:{position}:{strand}"
        )
    sequence_101 = sequence_201[50:151]
    sequence_41 = sequence_201[80:121]
    identifier = hashlib.sha1(
        f"{chrom}:{position}:{strand}:{positive['site_id']}".encode("utf-8")
    ).hexdigest()[:16]
    row: dict[str, object] = {
        "site_id": f"NEG_{identifier}",
        "det_label": 0,
        "detection_class": "unmodified",
        "negative_status": "unannotated_matched_candidate",
        "matched_positive_site_id": positive["site_id"],
        "matched_mod_type": positive["mod_type"],
        "match_level": match_level,
        "chr": chrom,
        "start": position,
        "end": position + 1,
        "strand": strand,
        "gene_id": positive["gene_id"],
        "transcript_id": positive["transcript_id"],
        "gene_name": positive["gene_name"],
        "gene_type": positive["gene_type"],
        "region_type": positive["region_type"],
        # Do not copy experimental support metadata from the matched positive.
        # It would let a classifier identify the annotation source instead of
        # learning sequence/structure/RBP evidence.  The matched-positive ID
        # remains available for pair-level leakage auditing.
        "support_num": "",
        "support_list": "",
        "pub_list": "",
        "cell_list": "",
        "seq_type_list": "",
        "source_file": "matched_unannotated_candidate",
        "center_base": sequence_201[100],
        "sequence_41nt": sequence_41,
        "sequence_101nt": sequence_101,
        "sequence_201nt": sequence_201,
        "gc_fraction_201nt": gc_fraction(sequence_201),
        "distance_to_nearest_known_modification": known.nearest_distance(chrom, position),
    }
    row.update({column: positive.get(column, "") for column in SPLIT_COLUMNS})
    return row


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    positives: list[dict[str, str]] = []
    transcript_ids: set[str] = set()
    with args.positives.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "site_id", "mod_type", "chr", "start", "strand", "gene_id", "transcript_id",
            "gene_name", "gene_type", "region_type", "center_base", "sequence_201nt", *SPLIT_COLUMNS,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Positive dataset is missing fields: {sorted(missing)}")
        for row in reader:
            positives.append(row)
            if row["transcript_id"].startswith("ENST"):
                transcript_ids.add(strip_version(row["transcript_id"]))
    known_path = args.known_modifications or args.positives
    known_positions: dict[str, list[int]] = defaultdict(list)
    with known_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = {"chr", "start"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Known-modification table is missing fields: {sorted(missing)}")
        for row in reader:
            known_positions[row["chr"]].append(int(row["start"]))
    known = KnownIndex(known_positions)
    print(
        f"loaded_positives={len(positives)} known_modification_coordinates="
        f"{sum(len(values) for values in known.positions.values())} "
        f"relevant_transcripts={len(transcript_ids)}",
        flush=True,
    )
    genome = read_fasta_gz(args.genome)
    print(f"loaded_contigs={len(genome)} genome_bases={sum(map(len, genome.values()))}", flush=True)
    transcripts = parse_relevant_gtf(args.gtf, transcript_ids)
    print(f"matched_gtf_transcripts={len(transcripts)}", flush=True)

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in positives:
        grouped[
            (
                row["chr"],
                row["strand"],
                strip_version(row["transcript_id"]),
                row["region_type"],
                row["center_base"].replace("U", "T"),
            )
        ].append(row)

    used: set[tuple[str, int, str]] = set()
    output_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    match_counts: Counter[str] = Counter()
    mod_requested: Counter[str] = Counter(row["mod_type"] for row in positives)
    mod_matched: Counter[str] = Counter()
    rng = random.Random(args.seed)

    for group_index, ((group_chrom, group_strand, transcript_id, region, expected_base), group_rows) in enumerate(
        grouped.items(), start=1
    ):
        transcript = transcripts.get(transcript_id)
        exact: list[int] = []
        exact_buckets: dict[int, list[int]] = {}
        if transcript and transcript.chromosome == group_chrom and transcript.strand == group_strand:
            intervals = transcript.region_intervals(region)
            if intervals:
                exact = sample_candidate_positions(
                    intervals,
                    max(16, len(group_rows) * 2),
                    genome,
                    transcript.chromosome,
                    transcript.strand,
                    expected_base,
                    known,
                    args.exclusion_radius,
                    used,
                    rng,
                )
                exact_buckets = gc_matched_candidate_buckets(
                    exact,
                    genome,
                    transcript.chromosome,
                    transcript.strand,
                    rng,
                )
        for positive in group_rows:
            target_sequence = positive["sequence_201nt"].upper().replace("U", "T")
            target_gc_count = target_sequence.count("G") + target_sequence.count("C")
            chosen = pop_gc_matched_candidate(
                exact_buckets,
                target_gc_count,
                group_chrom,
                group_strand,
                used,
                rng,
            )
            level = ""
            if chosen is not None:
                level = "same_transcript_same_region_gc_matched"
            if chosen is None:
                chosen = choose_local_candidate(
                    positive,
                    genome,
                    known,
                    args.exclusion_radius,
                    args.local_radius,
                    used,
                    rng,
                )
                if chosen is not None:
                    level = "local_genomic_fallback"
            if chosen is None:
                failure_rows.append(
                    {
                        "site_id": positive["site_id"],
                        "mod_type": positive["mod_type"],
                        "transcript_id": positive["transcript_id"],
                        "region_type": region,
                        "reason": "no_eligible_candidate",
                    }
                )
                continue
            used.add((positive["chr"], chosen, positive["strand"]))
            output_rows.append(build_negative_row(positive, chosen, level, genome, known))
            match_counts[level] += 1
            mod_matched[positive["mod_type"]] += 1
        if group_index % 10000 == 0:
            print(
                f"processed_groups={group_index}/{len(grouped)} matched={len(output_rows)} failures={len(failure_rows)}",
                flush=True,
            )

    write_tsv(args.output, output_rows, OUTPUT_FIELDS)
    write_tsv(
        args.failures,
        failure_rows,
        ["site_id", "mod_type", "transcript_id", "region_type", "reason"],
    )
    summary = {
        "positive_dataset": str(args.positives),
        "known_modification_dataset": str(known_path),
        "known_modification_dataset_sha256": sha256(known_path),
        "known_modification_coordinates": sum(len(values) for values in known.positions.values()),
        "genome": str(args.genome),
        "genome_sha256": sha256(args.genome),
        "gtf": str(args.gtf),
        "gtf_sha256": sha256(args.gtf),
        "seed": args.seed,
        "exclusion_radius_nt": args.exclusion_radius,
        "local_fallback_radius_nt": args.local_radius,
        "positive_rows": len(positives),
        "matched_negative_rows": len(output_rows),
        "failed_positive_rows": len(failure_rows),
        "match_rate": len(output_rows) / max(len(positives), 1),
        "match_levels": dict(match_counts),
        "requested_by_modification": dict(mod_requested),
        "matched_by_modification": dict(mod_matched),
        "matched_gtf_transcripts": len(transcripts),
        "relevant_positive_transcripts": len(transcript_ids),
        "negative_label_caveat": (
            "Candidates are absent from the complete RMBase coordinate index but are not experimentally verified "
            "as unmodified; results must be described as positive-versus-unannotated candidate detection."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument(
        "--known-modifications",
        type=Path,
        help="Complete hg38 RMBase coordinate table used only for exclusion; defaults to --positives.",
    )
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--exclusion-radius", type=int, default=25)
    parser.add_argument("--local-radius", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
