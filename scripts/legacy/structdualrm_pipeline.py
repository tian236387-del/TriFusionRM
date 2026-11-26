#!/usr/bin/env python
"""Local StructDualRM data workflow for m1A/m6A candidate sites.

The implementation intentionally uses only the Python standard library plus the
installed RNAfold executable so the data-preparation path can run in this local
environment without downloading additional packages.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


SCHEMA = [
    "mod_type",
    "chr",
    "pos_1based",
    "strand",
    "assembly",
    "source",
    "technology",
    "confidence",
    "supportNum",
    "gene_name",
    "extra_info",
]

PRIMARY_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
CHR_TO_NC = {f"chr{i}": f"NC_{i:06d}.10" for i in range(1, 23)}
CHR_TO_NC.update({"chrX": "NC_000023.10", "chrY": "NC_000024.9"})
NC_TO_CHR = {v: k for k, v in CHR_TO_NC.items()}
RNAFOLD_STRUCT_RE = re.compile(r"^([().]+)\s+\(\s*([-+0-9.eE]+)\)")


def ensure_dirs(out: Path) -> None:
    for rel in [
        "00_raw",
        "01_reference",
        "02_parsed_sites",
        "03_hg19_sites",
        "04_transcript_space",
        "05_datasets",
        "06_structure",
        "07_graphs",
        "08_models/M0_transformer_only",
        "08_models/M1_transformer_global_structure",
        "08_models/M2_transformer_gnn",
        "08_models/M3_structdualrm_gate",
        "08_models/M4_structdualrm_full",
        "09_results/metrics",
        "09_results/calibration",
        "09_results/ablation",
        "09_results/structure_analysis",
        "10_figures",
        "11_tables",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace", newline="")


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def normalize_chr(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if value.startswith("NC_"):
        return NC_TO_CHR.get(value, value)
    if value.lower().startswith("chr"):
        return "chr" + value[3:]
    if value in {"X", "Y"} or value.isdigit():
        return "chr" + value
    return value


def infer_strand_from_path(path: Path, fallback: str = ".") -> str:
    name = path.name.lower()
    if "_plus" in name or "plus." in name or name.endswith("_plus.bed"):
        return "+"
    if "_minus" in name or "minus." in name or name.endswith("_minus.bed"):
        return "-"
    return fallback


def infer_assembly_from_path(path: Path, default: str = "hg19_assumed") -> str:
    text = str(path).lower()
    if "hg19" in text or "grch37" in text:
        return "hg19"
    if "hg18" in text:
        return "hg18"
    if "hg38" in text or "grch38" in text:
        return "hg38"
    if "mm9" in text:
        return "mm9"
    if "mm10" in text:
        return "mm10"
    return default


def parse_bed_fields(
    fields: Sequence[str],
    mod_type: str,
    source: str,
    assembly: str,
    fallback_strand: str = ".",
) -> Dict[str, str]:
    chrom = normalize_chr(fields[0])
    start0 = int(float(fields[1]))
    pos_1based = start0 + 1
    support = fields[4] if len(fields) > 4 and fields[4] not in {"", "."} else "1"
    strand = fields[5] if len(fields) > 5 and fields[5] in {"+", "-"} else fallback_strand
    name = fields[3] if len(fields) > 3 else ""
    return {
        "mod_type": mod_type,
        "chr": chrom,
        "pos_1based": str(pos_1based),
        "strand": strand,
        "assembly": assembly,
        "source": source,
        "technology": "BED-like",
        "confidence": "medium",
        "supportNum": str(support),
        "gene_name": "",
        "extra_info": name,
    }


def is_probably_bed_line(fields: Sequence[str]) -> bool:
    if len(fields) < 3:
        return False
    if not normalize_chr(fields[0]).startswith("chr"):
        return False
    try:
        int(float(fields[1]))
        int(float(fields[2]))
    except ValueError:
        return False
    return True


def iter_non_archive_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower.endswith((".tar", ".zip")):
            continue
        if lower.endswith(".gz"):
            # Prefer extracted files when present; most provided archives already
            # have a neighboring extracted copy.
            continue
        yield path


def parse_gse70485(path: Path) -> List[Dict[str, str]]:
    rows = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            chrom = normalize_chr(raw.get("Chromosome", ""))
            pos = raw.get("m1A peak middle position", "")
            if not chrom or not str(pos).isdigit():
                continue
            rows.append(
                {
                    "mod_type": "m1A",
                    "chr": chrom,
                    "pos_1based": str(pos),
                    "strand": ".",
                    "assembly": "hg19_assumed",
                    "source": "GSE70485_human_peaks",
                    "technology": "m1A peak",
                    "confidence": "medium",
                    "supportNum": "1",
                    "gene_name": "",
                    "extra_info": f"refseq={raw.get('RefSeq ID','')};segment={raw.get('Segment','')};cell={raw.get('Cell','')}",
                }
            )
    return rows


def xlsx_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings = []
    for si in root.findall("a:si", ns):
        strings.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
    return strings


def iter_xlsx_rows(path: Path, sheet_name: str) -> Iterator[List[str]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        strings = xlsx_shared_strings(zf)
        root = ET.fromstring(zf.read(sheet_name))
        for row in root.findall(".//a:row", ns):
            values = []
            for cell in row.findall("a:c", ns):
                value_node = cell.find("a:v", ns)
                value = "" if value_node is None else value_node.text or ""
                if cell.get("t") == "s" and value:
                    value = strings[int(value)]
                values.append(value)
            yield values


def parse_gse90963_xlsx(path: Path) -> List[Dict[str, str]]:
    rows = []
    # Sheet2 is the candidate-site sheet in this workbook. It is transcript-like,
    # so it is retained in raw parsed output but excluded from genomic hg19 later.
    for idx, values in enumerate(iter_xlsx_rows(path, "xl/worksheets/sheet2.xml")):
        if idx == 0:
            continue
        if len(values) < 6:
            continue
        coord, comments, gene, biotype, strand, base = values[:6]
        if base != "A" or ":" not in coord:
            continue
        tx_id, tx_pos = coord.split(":", 1)
        rows.append(
            {
                "mod_type": "m1A",
                "chr": "",
                "pos_1based": tx_pos,
                "strand": strand if strand in {"+", "-"} else ".",
                "assembly": "transcript_space",
                "source": "GSE90963_Table_S3_candidate_sites",
                "technology": "RBS-seq",
                "confidence": "high" if comments == "OK" else "medium",
                "supportNum": "2",
                "gene_name": gene,
                "extra_info": f"transcript_id={tx_id};biotype={biotype};comments={comments}",
            }
        )
    return rows


def parse_bed_file(path: Path, mod_type: str, source: str, default_assembly: str) -> List[Dict[str, str]]:
    rows = []
    fallback_strand = infer_strand_from_path(path)
    assembly = infer_assembly_from_path(path, default_assembly)
    with open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            if not is_probably_bed_line(fields):
                continue
            rows.append(parse_bed_fields(fields, mod_type, source, assembly, fallback_strand))
    return rows


def should_parse_m1a_bed(filename: str, include_m1a_bed: bool = False) -> bool:
    lower = filename.lower()
    return include_m1a_bed and lower.endswith(".bed") and "ip" in lower and "input" not in lower


def parse_raw_sites(
    m1a_root: Path,
    m6a_root: Path,
    include_m1a_bed: bool = False,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    m1a_rows: List[Dict[str, str]] = []
    m6a_rows: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []

    for path in iter_non_archive_files(m1a_root):
        lower = path.name.lower()
        try:
            if lower == "gse70485_human_peaks.txt":
                m1a_rows.extend(parse_gse70485(path))
            elif lower.endswith(".xlsx") and "gse90963" in lower:
                m1a_rows.extend(parse_gse90963_xlsx(path))
            elif should_parse_m1a_bed(path.name, include_m1a_bed):
                m1a_rows.extend(parse_bed_file(path, "m1A", path.stem, "hg19_assumed"))
            elif lower.endswith(".bed") and "ip" in lower and "input" not in lower:
                skipped.append({"path": str(path), "reason": "m1A_bed_skipped_first_pass_use_include_m1a_bed"})
            else:
                skipped.append({"path": str(path), "reason": "unsupported_or_non_candidate_m1A_input"})
        except Exception as exc:  # keep the full run moving while recording the problem
            skipped.append({"path": str(path), "reason": f"parse_error:{exc}"})

    for path in iter_non_archive_files(m6a_root):
        lower_path = str(path).lower()
        lower_name = path.name.lower()
        if not lower_name.endswith((".bed", ".sites")):
            skipped.append({"path": str(path), "reason": "unsupported_m6A_extension"})
            continue
        if "peak_region" in lower_path:
            skipped.append({"path": str(path), "reason": "peak_region_not_precise_site"})
            continue
        assembly = infer_assembly_from_path(path, "hg19_assumed")
        if assembly in {"hg18", "mm9", "mm10"}:
            skipped.append({"path": str(path), "reason": f"excluded_non_hg19_or_requires_liftover:{assembly}"})
            continue
        if "precise" not in lower_path and "uniq.sites" not in lower_path and "cits.m6a" not in lower_path:
            skipped.append({"path": str(path), "reason": "m6A_file_not_marked_precise_site"})
            continue
        try:
            m6a_rows.extend(parse_bed_file(path, "m6A", path.stem, assembly))
        except Exception as exc:
            skipped.append({"path": str(path), "reason": f"parse_error:{exc}"})

    return m1a_rows, m6a_rows, skipped


def parse_gtf_attributes(text: str) -> Dict[str, str]:
    attrs = {}
    for key, value in re.findall(r'(\S+) "([^"]*)";', text):
        attrs[key] = value
    return attrs


@dataclass
class TranscriptModel:
    transcript_id: str
    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    exons: List[Tuple[int, int]] = field(default_factory=list)
    cds_ranges: List[Tuple[int, int]] = field(default_factory=list)
    transcript_biotype: str = ""
    _ordered_exons: List[Tuple[int, int]] = field(default_factory=list, init=False, repr=False)
    _tx_len: int = field(default=0, init=False, repr=False)
    _cds_tx_min: Optional[int] = field(default=None, init=False, repr=False)
    _cds_tx_max: Optional[int] = field(default=None, init=False, repr=False)

    def finalize(self) -> None:
        self._ordered_exons = sorted(self.exons, reverse=self.strand == "-")
        self._tx_len = sum(end - start + 1 for start, end in self._ordered_exons)
        cds_positions = []
        for start, end in self.cds_ranges:
            for genomic_pos in (start, end):
                tx_pos = self._genomic_to_tx_pos(genomic_pos)
                if tx_pos is not None:
                    cds_positions.append(tx_pos)
        if cds_positions:
            self._cds_tx_min = min(cds_positions)
            self._cds_tx_max = max(cds_positions)

    @property
    def tx_len(self) -> int:
        if not self._ordered_exons:
            self.finalize()
        return self._tx_len

    def _genomic_to_tx_pos(self, pos: int) -> Optional[int]:
        if not self._ordered_exons:
            self._ordered_exons = sorted(self.exons, reverse=self.strand == "-")
        offset = 0
        for start, end in self._ordered_exons:
            if start <= pos <= end:
                if self.strand == "+":
                    return offset + (pos - start + 1)
                return offset + (end - pos + 1)
            offset += end - start + 1
        return None

    def region_for_tx_pos(self, tx_pos: int) -> str:
        if self._cds_tx_min is None or self._cds_tx_max is None:
            return "UTR"
        if self._cds_tx_min <= tx_pos <= self._cds_tx_max:
            return "CDS"
        if tx_pos < self._cds_tx_min:
            return "5UTR"
        return "3UTR"

    def map_genomic_pos(self, pos: int) -> Optional[Dict[str, object]]:
        if not self._ordered_exons:
            self.finalize()
        tx_pos = self._genomic_to_tx_pos(pos)
        if tx_pos is None:
            return None
        tx_len = self.tx_len
        return {
            "transcript_id": self.transcript_id,
            "gene_id": self.gene_id,
            "gene_name": self.gene_name,
            "tx_pos": tx_pos,
            "tx_len": tx_len,
            "relative_pos": tx_pos / tx_len if tx_len else "",
            "region_type": self.region_for_tx_pos(tx_pos),
            "transcript_biotype": self.transcript_biotype,
        }


class TranscriptIndex:
    def __init__(self, transcripts: Dict[str, TranscriptModel]):
        self.transcripts = transcripts
        self.by_chrom_bin: Dict[str, Dict[int, List[TranscriptModel]]] = defaultdict(lambda: defaultdict(list))
        self.by_id_base: Dict[str, TranscriptModel] = {}
        for tx in transcripts.values():
            tx.finalize()
            self.by_id_base[tx.transcript_id.split(".")[0]] = tx
            seen_bins = set()
            for start, end in tx.exons:
                for bin_id in range(start // 100000, end // 100000 + 1):
                    key = (tx.chrom, bin_id, tx.transcript_id)
                    if key not in seen_bins:
                        self.by_chrom_bin[tx.chrom][bin_id].append(tx)
                        seen_bins.add(key)

    def find_mappings(self, chrom: str, pos: int, strand: str = ".") -> List[Tuple[TranscriptModel, Dict[str, object]]]:
        mappings = []
        for tx in self.by_chrom_bin.get(chrom, {}).get(pos // 100000, []):
            if strand in {"+", "-"} and tx.strand != strand:
                continue
            mapped = tx.map_genomic_pos(pos)
            if mapped is not None:
                mappings.append((tx, mapped))
        mappings.sort(key=lambda item: (not item[0].transcript_id.startswith("NM_"), -item[0].tx_len, item[0].transcript_id))
        return mappings

    def by_transcript_id(self, tx_id: str) -> Optional[TranscriptModel]:
        return self.transcripts.get(tx_id) or self.by_id_base.get(tx_id.split(".")[0])


def load_gtf(gtf_path: Path) -> TranscriptIndex:
    records: Dict[str, TranscriptModel] = {}
    with open_text(gtf_path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            seqid, _source, feature, start, end, _score, strand, _phase, attr_text = fields
            attrs = parse_gtf_attributes(attr_text)
            tx_id = attrs.get("transcript_id", "")
            if not tx_id or not (tx_id.startswith("NM_") or tx_id.startswith("NR_")):
                continue
            chrom = normalize_chr(seqid)
            if chrom not in PRIMARY_CHROMS:
                continue
            tx = records.setdefault(
                tx_id,
                TranscriptModel(
                    transcript_id=tx_id,
                    gene_id=attrs.get("gene_id", ""),
                    gene_name=attrs.get("gene_name", attrs.get("gene", attrs.get("gene_id", ""))),
                    chrom=chrom,
                    strand=strand,
                    transcript_biotype=attrs.get("transcript_biotype", attrs.get("gbkey", "")),
                ),
            )
            if feature == "exon":
                tx.exons.append((int(start), int(end)))
            elif feature == "CDS":
                tx.cds_ranges.append((int(start), int(end)))
    records = {tx_id: tx for tx_id, tx in records.items() if tx.exons}
    return TranscriptIndex(records)


def load_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, List[str]] = {}
    current = ""
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                seqs[current] = []
            elif current:
                seqs[current].append(line.upper().replace("U", "T"))
    out = {}
    for ident, parts in seqs.items():
        seq = "".join(parts)
        out[ident] = seq
        out.setdefault(ident.split(".")[0], seq)
    return out


class IndexedFasta:
    def __init__(self, fasta_path: Path, index_path: Path):
        self.fasta_path = fasta_path
        self.index_path = index_path
        if index_path.exists():
            self.index = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            self.index = self._build_index()
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps(self.index, indent=2), encoding="utf-8")

    def _build_index(self) -> Dict[str, Dict[str, int]]:
        index: Dict[str, Dict[str, int]] = {}
        with open(self.fasta_path, "rb") as handle:
            current = None
            seq_len = 0
            seq_offset = 0
            line_bases = 0
            line_bytes = 0
            while True:
                pos = handle.tell()
                line = handle.readline()
                if not line:
                    if current:
                        index[current]["length"] = seq_len
                    break
                if line.startswith(b">"):
                    if current:
                        index[current]["length"] = seq_len
                    current = line[1:].decode("utf-8", errors="replace").strip().split()[0]
                    seq_len = 0
                    seq_offset = handle.tell()
                    line_bases = 0
                    line_bytes = 0
                    index[current] = {
                        "offset": seq_offset,
                        "line_bases": 0,
                        "line_bytes": 0,
                        "length": 0,
                    }
                    continue
                stripped = line.rstrip(b"\r\n")
                if current and stripped:
                    if line_bases == 0:
                        line_bases = len(stripped)
                        line_bytes = len(line)
                        index[current]["line_bases"] = line_bases
                        index[current]["line_bytes"] = line_bytes
                    seq_len += len(stripped)
        return index

    def fetch(self, chrom: str, start_1based: int, end_1based: int) -> str:
        if chrom not in self.index:
            return ""
        meta = self.index[chrom]
        if start_1based < 1 or end_1based > meta["length"] or end_1based < start_1based:
            return ""
        line_bases = meta["line_bases"]
        line_bytes = meta["line_bytes"]
        start0 = start_1based - 1
        remaining = end_1based - start_1based + 1
        parts = []
        with open(self.fasta_path, "rb") as handle:
            while remaining > 0:
                line_idx = start0 // line_bases
                col = start0 % line_bases
                take = min(remaining, line_bases - col)
                handle.seek(meta["offset"] + line_idx * line_bytes + col)
                parts.append(handle.read(take).decode("ascii").upper())
                start0 += take
                remaining -= take
        return "".join(parts)


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTUNacgtun", "TGCAANtgcaan"))[::-1].upper().replace("U", "T")


def fetch_genome_sequence(genome: object, chrom: str, start_1based: int, end_1based: int) -> str:
    if genome is None:
        return ""
    if isinstance(genome, dict):
        seq = genome.get(chrom, "")
        if start_1based < 1 or end_1based > len(seq):
            return ""
        return seq[start_1based - 1 : end_1based].upper()
    if hasattr(genome, "fetch"):
        return genome.fetch(chrom, start_1based, end_1based)
    return ""


def extract_genomic_centered_window(genome: object, chrom: str, pos_1based: int, strand: str, flank: int) -> Dict[str, object]:
    seq = fetch_genome_sequence(genome, chrom, pos_1based - flank, pos_1based + flank)
    if not seq:
        return {"sequence": "", "center_base": "", "window_N_fraction": "", "drop_reason": "incomplete_window"}
    if strand == "-":
        seq = reverse_complement(seq)
    center = seq[flank]
    n_fraction = seq.count("N") / len(seq)
    drop_reason = ""
    if center != "A":
        drop_reason = "center_not_A"
    elif n_fraction > 0.05:
        drop_reason = "high_N_fraction"
    return {"sequence": seq, "center_base": center, "window_N_fraction": n_fraction, "drop_reason": drop_reason}


def filter_hg19_rows(rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    kept = []
    dropped = []
    for row in rows:
        chrom = normalize_chr(row.get("chr", ""))
        assembly = row.get("assembly", "")
        reason = ""
        if chrom not in PRIMARY_CHROMS:
            reason = "non_primary_or_missing_chr"
        elif assembly not in {"hg19", "GRCh37", "hg19_assumed"}:
            reason = f"not_hg19:{assembly}"
        elif not str(row.get("pos_1based", "")).isdigit():
            reason = "invalid_position"
        if reason:
            dropped.append({**row, "drop_reason": reason})
            continue
        kept.append({**row, "chr": chrom, "assembly": "hg19" if assembly == "GRCh37" else assembly})
    return dedup_site_rows(kept), dropped


def dedup_site_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str, str, str, str], Dict[str, str]] = {}
    for row in rows:
        key = (row["mod_type"], row["chr"], row["pos_1based"], row.get("strand", "."), row.get("source", ""))
        if key not in grouped:
            grouped[key] = dict(row)
        else:
            prev = grouped[key]
            prev["supportNum"] = str(safe_int(prev.get("supportNum", "1")) + safe_int(row.get("supportNum", "1")))
            prev["extra_info"] = ";".join(x for x in [prev.get("extra_info", ""), row.get("extra_info", "")] if x)
    return list(grouped.values())


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def map_rows_to_refseq(rows: Sequence[Dict[str, str]], index: TranscriptIndex) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    mapped_rows = []
    dropped = []
    for row in rows:
        chrom = normalize_chr(row.get("chr", ""))
        pos = safe_int(row.get("pos_1based", ""))
        mappings = index.find_mappings(chrom, pos, row.get("strand", "."))
        if not mappings:
            dropped.append({**row, "drop_reason": "no_refseq_exonic_mapping"})
            continue
        _tx, mapped = mappings[0]
        mapped_rows.append({**row, **mapped, "map_count": len(mappings)})
    return mapped_rows, dropped


def map_transcript_space_m1a(rows: Sequence[Dict[str, str]], index: TranscriptIndex) -> List[Dict[str, object]]:
    mapped = []
    for row in rows:
        if row.get("assembly") != "transcript_space":
            continue
        match = re.search(r"transcript_id=([^;]+)", row.get("extra_info", ""))
        if not match:
            continue
        tx = index.by_transcript_id(match.group(1))
        tx_pos = safe_int(row.get("pos_1based", ""))
        if not tx or tx_pos < 1 or tx_pos > tx.tx_len:
            continue
        mapped.append(
            {
                **row,
                "chr": tx.chrom,
                "strand": tx.strand,
                "transcript_id": tx.transcript_id,
                "gene_id": tx.gene_id,
                "gene_name": row.get("gene_name") or tx.gene_name,
                "tx_pos": tx_pos,
                "tx_len": tx.tx_len,
                "relative_pos": tx_pos / tx.tx_len,
                "region_type": tx.region_for_tx_pos(tx_pos),
                "map_count": 1,
            }
        )
    return mapped


def extract_centered_window(seq: str, tx_pos: int, flank: int) -> Dict[str, object]:
    center_idx = tx_pos - 1
    start = center_idx - flank
    end = center_idx + flank + 1
    if start < 0 or end > len(seq):
        return {"sequence": "", "center_base": "", "window_N_fraction": "", "drop_reason": "incomplete_window"}
    window = seq[start:end].upper().replace("U", "T")
    center = window[flank]
    n_fraction = window.count("N") / len(window)
    drop_reason = ""
    if center != "A":
        drop_reason = "center_not_A"
    elif n_fraction > 0.05:
        drop_reason = "high_N_fraction"
    return {
        "sequence": window,
        "center_base": center,
        "window_N_fraction": n_fraction,
        "drop_reason": drop_reason,
    }


def add_windows(
    rows: Sequence[Dict[str, object]],
    transcript_fasta: Dict[str, str],
    genome: object = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    kept = []
    dropped = []
    for row in rows:
        tx_id = str(row.get("transcript_id", ""))
        tx_pos = safe_int(row.get("tx_pos", ""))
        use_genome = bool(genome) and row.get("assembly") != "transcript_space" and str(row.get("pos_1based", "")).isdigit()
        if use_genome:
            chrom = str(row.get("chr", ""))
            pos = safe_int(row.get("pos_1based", ""))
            strand = str(row.get("strand", "."))
            w51 = extract_genomic_centered_window(genome, chrom, pos, strand, 25)
            w101 = extract_genomic_centered_window(genome, chrom, pos, strand, 50)
            w201 = extract_genomic_centered_window(genome, chrom, pos, strand, 100)
            window_source = "genome_hg19"
        else:
            seq = transcript_fasta.get(tx_id) or transcript_fasta.get(tx_id.split(".")[0])
            if not seq:
                dropped.append({**row, "drop_reason": "missing_transcript_sequence"})
                continue
            w51 = extract_centered_window(seq, tx_pos, 25)
            w101 = extract_centered_window(seq, tx_pos, 50)
            w201 = extract_centered_window(seq, tx_pos, 100)
            window_source = "transcript_fasta"
        reason = w101["drop_reason"] or w201["drop_reason"] or w51["drop_reason"]
        out = {
            **row,
            "center_base": w101["center_base"],
            "sequence_51nt": w51["sequence"],
            "sequence_101nt": w101["sequence"],
            "sequence_201nt": w201["sequence"],
            "window_N_fraction": w201["window_N_fraction"] if w201["window_N_fraction"] != "" else w101["window_N_fraction"],
            "drop_reason": reason,
            "window_source": window_source,
        }
        if reason:
            dropped.append(out)
        else:
            kept.append(out)
    return kept, dropped


JOINT_FIELDS = [
    "site_id",
    "chr",
    "pos_1based",
    "strand",
    "gene_id",
    "gene_name",
    "transcript_id",
    "tx_pos",
    "tx_len",
    "relative_pos",
    "region_type",
    "sequence_51nt",
    "sequence_101nt",
    "sequence_201nt",
    "label_m1A",
    "label_m6A",
    "source_m1A",
    "source_m6A",
    "confidence_m1A",
    "confidence_m6A",
    "supportNum_m1A",
    "supportNum_m6A",
    "center_base",
    "window_N_fraction",
    "split_random",
    "split_gene",
    "split_chromosome",
]


def merge_joint_positives(m1a: Sequence[Dict[str, object]], m6a: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    for mod, rows in (("m1A", m1a), ("m6A", m6a)):
        for row in rows:
            site_id = f"{row['transcript_id']}_{row['tx_pos']}"
            current = grouped.setdefault(
                site_id,
                {
                    "site_id": site_id,
                    "chr": row.get("chr", ""),
                    "pos_1based": row.get("pos_1based", ""),
                    "strand": row.get("strand", ""),
                    "gene_id": row.get("gene_id", ""),
                    "gene_name": row.get("gene_name", ""),
                    "transcript_id": row.get("transcript_id", ""),
                    "tx_pos": row.get("tx_pos", ""),
                    "tx_len": row.get("tx_len", ""),
                    "relative_pos": row.get("relative_pos", ""),
                    "region_type": row.get("region_type", ""),
                    "sequence_51nt": row.get("sequence_51nt", ""),
                    "sequence_101nt": row.get("sequence_101nt", ""),
                    "sequence_201nt": row.get("sequence_201nt", ""),
                    "label_m1A": 0,
                    "label_m6A": 0,
                    "source_m1A": "",
                    "source_m6A": "",
                    "confidence_m1A": "",
                    "confidence_m6A": "",
                    "supportNum_m1A": "",
                    "supportNum_m6A": "",
                    "center_base": row.get("center_base", ""),
                    "window_N_fraction": row.get("window_N_fraction", ""),
                },
            )
            current[f"label_{mod}"] = 1
            current[f"source_{mod}"] = append_unique(current.get(f"source_{mod}", ""), str(row.get("source", "")))
            current[f"confidence_{mod}"] = append_unique(current.get(f"confidence_{mod}", ""), str(row.get("confidence", "")))
            current[f"supportNum_{mod}"] = str(safe_int(current.get(f"supportNum_{mod}", "0")) + safe_int(row.get("supportNum", "1")))
    return list(grouped.values())


def append_unique(existing: str, value: str) -> str:
    values = [x for x in existing.split(";") if x]
    if value and value not in values:
        values.append(value)
    return ";".join(values)


def build_negative_pool(
    positives: Sequence[Dict[str, object]],
    index: TranscriptIndex,
    transcript_fasta: Dict[str, str],
    ratio: int = 3,
    seed: int = 42,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    by_tx: Dict[str, List[int]] = defaultdict(list)
    meta_by_tx: Dict[str, Dict[str, object]] = {}
    for row in positives:
        tx_id = str(row["transcript_id"])
        by_tx[tx_id].append(safe_int(row["tx_pos"]))
        meta_by_tx[tx_id] = row
    negatives = []
    for tx_id, known_positions in by_tx.items():
        tx = index.by_transcript_id(tx_id)
        seq = transcript_fasta.get(tx_id) or transcript_fasta.get(tx_id.split(".")[0])
        if not tx or not seq:
            continue
        known = set(known_positions)
        candidates = []
        for i, base in enumerate(seq, start=1):
            if base != "A":
                continue
            if min((abs(i - k) for k in known), default=999999) <= 10:
                continue
            w101 = extract_centered_window(seq, i, 50)
            w201 = extract_centered_window(seq, i, 100)
            if w101["drop_reason"] or w201["drop_reason"]:
                continue
            candidates.append((i, w101, w201))
        rng.shuffle(candidates)
        take = min(len(candidates), max(1, len(known_positions) * ratio))
        template = meta_by_tx[tx_id]
        for tx_pos, w101, w201 in candidates[:take]:
            w51 = extract_centered_window(seq, tx_pos, 25)
            negatives.append(
                {
                    "site_id": f"{tx_id}_{tx_pos}",
                    "chr": tx.chrom,
                    "pos_1based": "",
                    "strand": tx.strand,
                    "gene_id": tx.gene_id,
                    "gene_name": tx.gene_name or template.get("gene_name", ""),
                    "transcript_id": tx_id,
                    "tx_pos": tx_pos,
                    "tx_len": tx.tx_len,
                    "relative_pos": tx_pos / tx.tx_len,
                    "region_type": tx.region_for_tx_pos(tx_pos),
                    "sequence_51nt": w51["sequence"],
                    "sequence_101nt": w101["sequence"],
                    "sequence_201nt": w201["sequence"],
                    "label_m1A": 0,
                    "label_m6A": 0,
                    "source_m1A": "",
                    "source_m6A": "",
                    "confidence_m1A": "",
                    "confidence_m6A": "",
                    "supportNum_m1A": "",
                    "supportNum_m6A": "",
                    "center_base": "A",
                    "window_N_fraction": w201["window_N_fraction"],
                }
            )
    return negatives


def add_splits(rows: List[Dict[str, object]], seed: int = 42) -> None:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    for idx, row in enumerate(shuffled):
        row["split_random"] = split_by_fraction(idx, n)

    def assign_group_split(key_name: str, out_name: str) -> None:
        groups = sorted({str(row.get(key_name, "")) for row in rows})
        rng.shuffle(groups)
        group_split = {group: split_by_fraction(idx, len(groups)) for idx, group in enumerate(groups)}
        for row in rows:
            row[out_name] = group_split.get(str(row.get(key_name, "")), "train")

    assign_group_split("gene_id", "split_gene")

    chroms = sorted({str(row.get("chr", "")) for row in rows if row.get("chr")})
    chrom_split = {chrom: split_by_fraction(idx, len(chroms)) for idx, chrom in enumerate(chroms)}
    for row in rows:
        row["split_chromosome"] = chrom_split.get(str(row.get("chr", "")), "train")


def split_by_fraction(idx: int, total: int) -> str:
    frac = idx / max(total, 1)
    if frac < 0.70:
        return "train"
    if frac < 0.85:
        return "valid"
    return "test"


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
    return match.group(1), float(match.group(2))


def run_rnafold(rows: Sequence[Dict[str, object]], out: Path) -> List[Dict[str, object]]:
    if not rows:
        return []
    fasta = []
    for row in rows:
        fasta.append(f">{row['site_id']}")
        fasta.append(str(row["sequence_101nt"]).replace("T", "U"))
    proc = subprocess.run(
        ["RNAfold", "--noPS"],
        input="\n".join(fasta) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    results = []
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
        flags = parsed["paired_flags"]
        center_idx = len(flags) // 2
        central_11 = flags[center_idx - 5 : center_idx + 6]
        central_21 = flags[center_idx - 10 : center_idx + 11]
        loop_span = loop_span_around_center(flags, center_idx)
        results.append(
            {
                "site_id": site_id,
                "dot_bracket": dot,
                "MFE": mfe,
                "paired_flags": ",".join(map(str, flags)),
                "base_pair_list": ";".join(f"{a}-{b}" for a, b in parsed["base_pair_list"]),
                "center_paired": flags[center_idx],
                "central_11nt_paired_fraction": sum(central_11) / len(central_11),
                "central_21nt_paired_fraction": sum(central_21) / len(central_21),
                "paired_fraction_101nt": sum(flags) / len(flags),
                "MFE_per_nt": mfe / len(seq),
                "loop_span_around_center": loop_span,
            }
        )
        i += 3
    write_tsv(
        out / "06_structure/rnafold_101nt.tsv",
        results,
        [
            "site_id",
            "dot_bracket",
            "MFE",
            "paired_flags",
            "base_pair_list",
            "center_paired",
            "central_11nt_paired_fraction",
            "central_21nt_paired_fraction",
            "paired_fraction_101nt",
            "MFE_per_nt",
            "loop_span_around_center",
        ],
    )
    return results


def rnafold_compatible_rows(rows: Sequence[Dict[str, object]], limit: int = 0) -> List[Dict[str, object]]:
    selected = []
    for row in rows:
        seq = str(row.get("sequence_101nt", "")).upper()
        if len(seq) != 101 or set(seq) - set("ACGT"):
            continue
        selected.append(row)
        if limit and len(selected) >= limit:
            break
    return selected


def loop_span_around_center(flags: Sequence[int], center_idx: int) -> int:
    left = center_idx
    right = center_idx
    while left - 1 >= 0 and flags[left - 1] == 0:
        left -= 1
    while right + 1 < len(flags) and flags[right + 1] == 0:
        right += 1
    return right - left + 1


def write_graphs(rows: Sequence[Dict[str, object]], rnafold_rows: Sequence[Dict[str, object]], out: Path) -> None:
    fold_by_id = {row["site_id"]: row for row in rnafold_rows}
    graph_path = out / "07_graphs/graph_data_101nt.jsonl"
    edge_path = out / "06_structure/rnafold_graph_edges_101nt.jsonl"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    with open(graph_path, "w", encoding="utf-8") as graph_handle, open(edge_path, "w", encoding="utf-8") as edge_handle:
        for row in rows:
            site_id = row["site_id"]
            fold = fold_by_id.get(site_id)
            if not fold:
                continue
            seq = str(row["sequence_101nt"])
            edges = []
            edge_types = []
            for idx in range(1, len(seq)):
                edges.extend([[idx, idx + 1], [idx + 1, idx]])
                edge_types.extend([0, 0])
            for pair in str(fold["base_pair_list"]).split(";"):
                if not pair:
                    continue
                a, b = [int(x) for x in pair.split("-")]
                edges.extend([[a, b], [b, a]])
                edge_types.extend([1, 1])
            payload = {
                "site_id": site_id,
                "sequence_101nt": seq,
                "edge_index": edges,
                "edge_type": edge_types,
                "label_m1A": int(row["label_m1A"]),
                "label_m6A": int(row["label_m6A"]),
            }
            graph_handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            edge_handle.write(json.dumps({"site_id": site_id, "edge_index": edges, "edge_type": edge_types}, separators=(",", ":")) + "\n")


def summarize_counts(rows: Sequence[Dict[str, object]]) -> Dict[str, int]:
    return {
        "rows": len(rows),
        "m1A_positive": sum(1 for row in rows if int(row.get("label_m1A", 0)) == 1),
        "m6A_positive": sum(1 for row in rows if int(row.get("label_m6A", 0)) == 1),
        "co_positive": sum(1 for row in rows if int(row.get("label_m1A", 0)) == 1 and int(row.get("label_m6A", 0)) == 1),
        "negative": sum(1 for row in rows if int(row.get("label_m1A", 0)) == 0 and int(row.get("label_m6A", 0)) == 0),
    }


def write_model_dependency_note(out: Path, dependency_status: Dict[str, bool]) -> None:
    note = (
        "StructDualRM model training was not executed in this run.\n\n"
        "Reason: this local Python environment is missing required ML packages "
        f"(torch={dependency_status.get('torch')}, sklearn={dependency_status.get('sklearn')}).\n"
        "Data, RNAfold features, and graph JSONL files were generated so training can be run after installing PyTorch/PyG or an equivalent implementation.\n"
    )
    for rel in [
        "08_models/M0_transformer_only/README.md",
        "08_models/M1_transformer_global_structure/README.md",
        "08_models/M2_transformer_gnn/README.md",
        "08_models/M3_structdualrm_gate/README.md",
        "08_models/M4_structdualrm_full/README.md",
    ]:
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(note, encoding="utf-8")


def dependency_status() -> Dict[str, bool]:
    import importlib.util

    status = {name: importlib.util.find_spec(name) is not None for name in ["pandas", "numpy", "sklearn", "torch"]}
    status["RNAfold"] = shutil_which("RNAfold") is not None
    return status


def shutil_which(cmd: str) -> Optional[str]:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    exts = [""] + os.environ.get("PATHEXT", "").split(os.pathsep)
    for folder in paths:
        for ext in exts:
            candidate = Path(folder) / f"{cmd}{ext}"
            if candidate.exists():
                return str(candidate)
    return None


def write_summary(out: Path, summary_rows: Sequence[Dict[str, object]]) -> None:
    write_tsv(out / "11_tables/workflow_summary.tsv", summary_rows, ["stage", "rows", "m1A_positive", "m6A_positive", "co_positive", "negative", "notes"])


def run_pipeline(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    ensure_dirs(out)
    m1a_root = Path(args.m1a)
    m6a_root = Path(args.m6a)
    default_ref = Path(r"D:\dataset\test1\01_reference")
    gtf_path = Path(args.gtf) if args.gtf else default_ref / "refseq_hg19.gtf"
    fasta_path = Path(args.transcripts_fasta) if args.transcripts_fasta else default_ref / "refseq_hg19.transcripts.fa"
    genome_path = Path(args.genome_fasta) if args.genome_fasta else default_ref / "hg19.fa"

    status = dependency_status()
    manifest = {
        "m1a_root": str(m1a_root),
        "m6a_root": str(m6a_root),
        "out": str(out),
        "gtf_path": str(gtf_path),
        "transcripts_fasta": str(fasta_path),
        "genome_fasta": str(genome_path) if genome_path.exists() else "",
        "dependency_status": status,
        "notes": [
            "hg19.fa is used for genomic strand-aware center-A/window QC when genomic coordinates are available.",
            "hg18/mm9/mm10 inputs are excluded rather than lifted over.",
            "Graph outputs are JSONL because torch is not installed, so .pt serialization is unavailable.",
            "m1A BED IP files are skipped by default in the first pass; rerun with --include-m1a-bed to include them.",
        ],
    }

    m1a_raw, m6a_raw, skipped = parse_raw_sites(m1a_root, m6a_root, include_m1a_bed=args.include_m1a_bed)
    write_tsv(out / "02_parsed_sites/m1A_raw_parsed.tsv", m1a_raw, SCHEMA)
    write_tsv(out / "02_parsed_sites/m6A_raw_parsed.tsv", m6a_raw, SCHEMA)
    write_tsv(out / "11_tables/skipped_inputs.tsv", skipped, ["path", "reason"])

    m1a_hg19, m1a_hg19_drop = filter_hg19_rows(m1a_raw)
    m6a_hg19, m6a_hg19_drop = filter_hg19_rows(m6a_raw)
    write_tsv(out / "03_hg19_sites/m1A_hg19.tsv", m1a_hg19, SCHEMA)
    write_tsv(out / "03_hg19_sites/m6A_hg19.tsv", m6a_hg19, SCHEMA)
    qc_rows = [
        {"mod_type": "m1A", "raw_sites": len(m1a_raw), "hg19_retained": len(m1a_hg19), "removed": len(m1a_hg19_drop)},
        {"mod_type": "m6A", "raw_sites": len(m6a_raw), "hg19_retained": len(m6a_hg19), "removed": len(m6a_hg19_drop)},
    ]
    write_tsv(out / "11_tables/hg19_filter_qc.tsv", qc_rows, ["mod_type", "raw_sites", "hg19_retained", "removed"])

    index = load_gtf(gtf_path)
    transcript_fasta = load_fasta(fasta_path)
    genome = IndexedFasta(genome_path, out / "01_reference/hg19.fa.index.json") if genome_path.exists() else None

    m1a_mapped_genomic, m1a_map_drop = map_rows_to_refseq(m1a_hg19, index)
    m1a_mapped_tx = map_transcript_space_m1a(m1a_raw, index)
    m1a_mapped = m1a_mapped_genomic + m1a_mapped_tx
    m6a_mapped, m6a_map_drop = map_rows_to_refseq(m6a_hg19, index)
    mapped_fields = SCHEMA + ["transcript_id", "gene_id", "tx_pos", "tx_len", "relative_pos", "region_type", "transcript_biotype", "map_count"]
    write_tsv(out / "04_transcript_space/m1A_refseq_mapped.tsv", m1a_mapped, mapped_fields)
    write_tsv(out / "04_transcript_space/m6A_refseq_mapped.tsv", m6a_mapped, mapped_fields)

    m1a_center, m1a_center_drop = add_windows(m1a_mapped, transcript_fasta, genome)
    m6a_center, m6a_center_drop = add_windows(m6a_mapped, transcript_fasta, genome)
    center_fields = mapped_fields + ["center_base", "sequence_51nt", "sequence_101nt", "sequence_201nt", "window_N_fraction", "drop_reason", "window_source"]
    write_tsv(out / "03_hg19_sites/m1A_hg19_centerA.tsv", m1a_center, center_fields)
    write_tsv(out / "03_hg19_sites/m6A_hg19_centerA.tsv", m6a_center, center_fields)

    positives = merge_joint_positives(m1a_center, m6a_center)
    write_tsv(out / "04_transcript_space/m1A_m6A_joint_positive_sites.tsv", positives, JOINT_FIELDS)
    negatives = build_negative_pool(positives, index, transcript_fasta, ratio=args.negative_ratio)
    write_tsv(out / "05_datasets/negative_pool_refseq_matched.tsv", negatives, JOINT_FIELDS)

    dataset = positives + negatives
    add_splits(dataset)
    write_tsv(out / "05_datasets/dataset_joint_m1A_m6A_101nt.tsv", dataset, JOINT_FIELDS)
    write_tsv(out / "05_datasets/dataset_joint_m1A_m6A_201nt.tsv", dataset, JOINT_FIELDS)

    rnafold_rows = []
    if status["RNAfold"]:
        rnafold_input = rnafold_compatible_rows(dataset, args.rnafold_max)
        if args.rnafold_max:
            manifest["notes"].append(
                f"RNAfold was run on the first {len(rnafold_input)} ACGT-only 101nt dataset rows "
                f"out of {len(dataset)} total rows, with requested limit {args.rnafold_max}."
            )
        elif len(rnafold_input) != len(dataset):
            manifest["notes"].append(f"RNAfold skipped {len(dataset) - len(rnafold_input)} rows containing ambiguous bases or invalid length.")
        rnafold_rows = run_rnafold(rnafold_input, out)
        write_graphs(rnafold_input, rnafold_rows, out)
    else:
        manifest["notes"].append("RNAfold executable was not found; structure files were not generated.")

    write_model_dependency_note(out, status)

    summary = [
        {"stage": "m1A_raw_parsed", "rows": len(m1a_raw), "notes": ""},
        {"stage": "m6A_raw_parsed", "rows": len(m6a_raw), "notes": ""},
        {"stage": "m1A_hg19", "rows": len(m1a_hg19), "notes": ""},
        {"stage": "m6A_hg19", "rows": len(m6a_hg19), "notes": ""},
        {"stage": "m1A_refseq_mapped", "rows": len(m1a_mapped), "notes": f"dropped={len(m1a_map_drop)}"},
        {"stage": "m6A_refseq_mapped", "rows": len(m6a_mapped), "notes": f"dropped={len(m6a_map_drop)}"},
        {"stage": "m1A_centerA", "rows": len(m1a_center), "notes": f"dropped={len(m1a_center_drop)}"},
        {"stage": "m6A_centerA", "rows": len(m6a_center), "notes": f"dropped={len(m6a_center_drop)}"},
        {**summarize_counts(positives), "stage": "joint_positive_sites", "notes": ""},
        {**summarize_counts(negatives), "stage": "negative_pool", "notes": ""},
        {**summarize_counts(dataset), "stage": "joint_dataset", "notes": ""},
        {"stage": "rnafold_101nt", "rows": len(rnafold_rows), "notes": ""},
    ]
    write_summary(out, summary)

    manifest["row_counts"] = {row["stage"]: row["rows"] for row in summary}
    with open(out / "09_results/workflow_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run StructDualRM m1A/m6A data workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--m1a", required=True)
    run.add_argument("--m6a", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--gtf")
    run.add_argument("--transcripts-fasta")
    run.add_argument("--genome-fasta")
    run.add_argument("--negative-ratio", type=int, default=3)
    run.add_argument("--include-m1a-bed", action="store_true")
    run.add_argument("--rnafold-max", type=int, default=0, help="Limit RNAfold/graph generation to the first N dataset rows; 0 means all rows.")
    args = parser.parse_args(argv)
    if args.command == "run":
        run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
