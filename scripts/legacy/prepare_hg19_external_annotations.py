#!/usr/bin/env python
"""Prepare hg19-compatible external m1A/m6A and reader annotations."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence


XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_table(rows: Sequence[Dict[str, object]], path: Path, fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_region(value: str) -> str:
    clean = value.strip().replace("-", "").replace("_", "").upper()
    if clean in {"3UTR", "UTR3"}:
        return "3UTR"
    if clean in {"5UTR", "UTR5"}:
        return "5UTR"
    if clean == "CDS":
        return "CDS"
    if clean == "NCRNA":
        return "ncRNA"
    return value.strip()


def parse_miclip_description(value: str) -> Dict[str, str]:
    parts = [part.strip() for part in str(value).split("|")]
    return {
        "transcript_id": parts[0] if len(parts) > 0 else "",
        "gene_name": parts[1] if len(parts) > 1 else "",
        "region_type": normalize_region(parts[2]) if len(parts) > 2 else "",
        "tx_pos": parts[3] if len(parts) > 3 else "",
    }


def _xlsx_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() == ".gz":
        return gzip.decompress(data)
    return data


def _shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values = []
    for si in root.findall("a:si", XLSX_NS):
        values.append("".join(t.text or "" for t in si.findall(".//a:t", XLSX_NS)))
    return values


def _sheet_target_by_name(zf: zipfile.ZipFile) -> Dict[str, str]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    out = {}
    for sheet in workbook.findall("a:sheets/a:sheet", XLSX_NS):
        rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = relmap[rid].lstrip("/")
        out[sheet.attrib["name"]] = target if target.startswith("xl/") else f"xl/{target}"
    return out


def iter_xlsx_rows(path: Path, sheet_name: str) -> Iterator[List[str]]:
    with zipfile.ZipFile(io.BytesIO(_xlsx_bytes(path))) as zf:
        sheet_map = _sheet_target_by_name(zf)
        if sheet_name not in sheet_map:
            raise ValueError(f"sheet {sheet_name!r} not found in {path}")
        shared = _shared_strings(zf)
        sheet = ET.fromstring(zf.read(sheet_map[sheet_name]))
        for row in sheet.findall("a:sheetData/a:row", XLSX_NS):
            cells = []
            for cell in row.findall("a:c", XLSX_NS):
                value_node = cell.find("a:v", XLSX_NS)
                value = "" if value_node is None else value_node.text or ""
                if cell.attrib.get("t") == "s" and value:
                    value = shared[int(value)]
                cells.append(value)
            yield cells


def rows_from_xlsx_sheet(path: Path, sheet_name: str, skip_rows: int = 0) -> List[Dict[str, str]]:
    rows = list(iter_xlsx_rows(path, sheet_name))
    rows = rows[skip_rows:]
    if not rows:
        return []
    header = [str(col).strip() for col in rows[0]]
    out = []
    for values in rows[1:]:
        padded = list(values) + [""] * max(0, len(header) - len(values))
        out.append({key: str(value).strip() for key, value in zip(header, padded)})
    return out


def _miclip_rows_from_sheet(path: Path, sheet_name: str, evidence_tier: str) -> List[Dict[str, object]]:
    rows = rows_from_xlsx_sheet(path, sheet_name)
    out = []
    for row in rows:
        desc = parse_miclip_description(row.get("description", row.get("decription", "")))
        if not desc["gene_name"]:
            continue
        out.append(
            {
                "mod_type": "m1A",
                "chr": row.get("chrom", ""),
                "start": int(float(row.get("position", "0"))) - 1 if row.get("position") else "",
                "end": int(float(row.get("position", "0"))) if row.get("position") else "",
                "pos_1based": row.get("position", ""),
                "strand": row.get("strand", ""),
                "gene_name": desc["gene_name"],
                "transcript_id": desc["transcript_id"],
                "region_type": desc["region_type"],
                "source": f"NatureCommunications2019_m1A_miCLIP_{sheet_name.replace(' ', '')}_hg19",
                "evidence_tier": evidence_tier,
            }
        )
    return out


def prepare_m1a_miclip(path: Path) -> List[Dict[str, object]]:
    broad = _miclip_rows_from_sheet(path, "Table 1", "broad_candidate_misincorporation_ge1pct_cov500")
    high = _miclip_rows_from_sheet(path, "Table 2", "high_confidence_multi_transition_ge5pct")
    return broad + high


def parse_bed_gz(path: Path) -> List[Dict[str, str]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            rows.append(
                {
                    "chr": parts[0],
                    "start": parts[1],
                    "end": parts[2],
                    "name": parts[3] if len(parts) > 3 else "",
                    "score": parts[4] if len(parts) > 4 else "",
                    "strand": parts[5] if len(parts) > 5 else "",
                }
            )
    return rows


def map_bed_peaks_to_sites(
    peaks: Sequence[Dict[str, str]],
    sites: Sequence[Dict[str, str]],
    mod_type: str,
    source: str,
) -> List[Dict[str, object]]:
    sites_by_chr: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for site in sites:
        if site.get("chr") and site.get("pos_1based"):
            sites_by_chr[site["chr"]].append(site)
    rows = []
    seen = set()
    for peak in peaks:
        chrom = peak.get("chr", "")
        try:
            start = int(float(peak.get("start", "")))
            end = int(float(peak.get("end", "")))
        except ValueError:
            continue
        for site in sites_by_chr.get(chrom, []):
            try:
                pos0 = int(float(site["pos_1based"])) - 1
            except ValueError:
                continue
            if start <= pos0 < end:
                key = (peak.get("name", ""), site.get("gene_name", ""), site.get("transcript_id", ""), site.get("pos_1based", ""))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "mod_type": mod_type,
                        "chr": chrom,
                        "start": start,
                        "end": end,
                        "pos_1based": site.get("pos_1based", ""),
                        "strand": peak.get("strand", ""),
                        "gene_name": site.get("gene_name", ""),
                        "transcript_id": site.get("transcript_id", ""),
                        "region_type": site.get("region_type", ""),
                        "source": source,
                        "external_peak_id": peak.get("name", ""),
                    }
                )
    return rows


def prepare_reader_ythdf2(path: Path) -> List[Dict[str, object]]:
    rows = rows_from_xlsx_sheet(path, "High confident peaks-2671")
    out = []
    for row in rows:
        gene = row.get("Gene", "")
        if not gene:
            continue
        out.append(
            {
                "gene_name": gene,
                "transcript_id": "",
                "rbpName": "YTHDF2",
                "RNAmodType": "m6A",
                "chr": row.get("Location", ""),
                "start": row.get("Cluster_start", ""),
                "end": row.get("Cluster_end", ""),
                "Region": normalize_region(row.get("Region", "")),
                "source": "GSE49339_YTHDF2_PARCLIP_high_confident_peaks_hg19",
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare hg19 external annotation TSVs for downstream biology analyses.")
    parser.add_argument("--m1a-xlsx", default="external/rmbase/nc2019_m1a_miclip_suppdata.xlsx")
    parser.add_argument("--m6a-bed-gz", default="external/rmbase/GSM2300426_m6A_peak_region_HeLa_CA_hg19.bed.gz")
    parser.add_argument("--reader-xlsx-gz", default="external/rmbase/GSE49339_A-PARCLIP.xlsx.gz")
    parser.add_argument("--m6a-sites", default="04_transcript_space/m6A_refseq_mapped.tsv")
    parser.add_argument("--out-dir", default="external/hg19_public")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    m1a_rows = prepare_m1a_miclip(Path(args.m1a_xlsx))
    m6a_rows = map_bed_peaks_to_sites(
        parse_bed_gz(Path(args.m6a_bed_gz)),
        read_rows(Path(args.m6a_sites)),
        mod_type="m6A",
        source="GSM2300426_HeLa_CA_m6A_peak_region_hg19_mapped_to_current_sites",
    )
    reader_rows = prepare_reader_ythdf2(Path(args.reader_xlsx_gz))

    external_fieldnames = [
        "mod_type",
        "chr",
        "start",
        "end",
        "pos_1based",
        "strand",
        "gene_name",
        "transcript_id",
        "region_type",
        "source",
        "evidence_tier",
        "external_peak_id",
    ]
    write_table(m1a_rows, out_dir / "nc2019_m1a_miclip_hg19.tsv", external_fieldnames)
    write_table(m6a_rows, out_dir / "gsm2300426_m6a_hg19_mapped.tsv", external_fieldnames)
    write_table(m1a_rows + m6a_rows, out_dir / "hg19_external_m1a_m6a_combined.tsv", external_fieldnames)
    write_table(reader_rows, out_dir / "gse49339_ythdf2_reader_hg19.tsv")
    print(f"wrote {len(m1a_rows)} m1A rows, {len(m6a_rows)} mapped m6A rows, {len(reader_rows)} reader rows")


if __name__ == "__main__":
    main()
