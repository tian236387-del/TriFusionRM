#!/usr/bin/env python
"""Build supplementary RMBase datasets aligned to MultiRM label coverage.

The primary project benchmark remains the strict RMBase11 dataset.  This script
adds a reviewer-facing supplementary benchmark by incorporating RMBase RNA
editing (A-to-I) and m6Am from otherMod while applying the same 41-nt sequence,
center-base and exact-coordinate de-duplication checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import ssl
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


PRESET_CLASSES = {
    "multirm12": ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "Y", "AtoI"],
    "extended13": ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "AtoI"],
}

SOURCE_ARCHIVES = {
    "m1A": "https://bioinformaticsscience.cn/rmbase/download/modSingleSiteFiles/hg38/hg38.m1A.tar.gz",
    "m5C": "https://bioinformaticsscience.cn/rmbase/download/modSingleSiteFiles/hg38/hg38.m5C.tar.gz",
    "m6A": "https://bioinformaticsscience.cn/rmbase/download/modSingleSiteFiles/hg38/hg38.m6A.tar.gz",
    "m7G": "https://bioinformaticsscience.cn/rmbase/download/modSingleSiteFiles/hg38/hg38.m7G.tar.gz",
    "Nm": "https://bioinformaticsscience.cn/rmbase/download/modSingleSiteFiles/hg38/hg38.Nm.tar.gz",
    "Pseudo": "https://bioinformaticsscience.cn/rmbase/download/modSingleSiteFiles/hg38/hg38.Pseudo.tar.gz",
    "otherMod": "https://bioinformaticsscience.cn/rmbase/download/modSingleSiteFiles/hg38/hg38.otherMod.tar.gz",
    "RNA-editing": "https://bioinformaticsscience.cn/rmbase/download/modSingleSiteFiles/hg38/hg38.RNA-editing.tar.gz",
}

REQUIRED_SOURCE_GROUPS = ["m1A", "m5C", "m6A", "m7G", "Nm", "Pseudo", "otherMod", "RNA-editing"]

RAW_MOD_TO_CLASS = {
    "Am": "Am",
    "Cm": "Cm",
    "Gm": "Gm",
    "Um": "Um",
    "Tm": "Um",
    "D": "D",
    "Y": "Y",
    "Psi": "Y",
    "Pseudo": "Y",
    "m1A": "m1A",
    "m5C": "m5C",
    "m5U": "m5U",
    "m6A": "m6A",
    "m6Am": "m6Am",
    "m7G": "m7G",
    "A-I": "AtoI",
    "A-to-I": "AtoI",
    "AtoI": "AtoI",
}

EXPECTED_CENTER = {
    "Am": "A",
    "m1A": "A",
    "m6A": "A",
    "m6Am": "A",
    "AtoI": "A",
    "Cm": "C",
    "m5C": "C",
    "Gm": "G",
    "m7G": "G",
    "Um": "T",
    "D": "T",
    "Y": "T",
    "m5U": "T",
}

FIELDS = [
    "site_id",
    "mod_type",
    "class_index",
    "chr",
    "start",
    "end",
    "strand",
    "support_num",
    "support_list",
    "pub_list",
    "cell_list",
    "seq_type_list",
    "gene_id",
    "transcript_id",
    "gene_name",
    "gene_type",
    "region_raw",
    "region_type",
    "sequence_41nt",
    "center_base",
    "source_mod_type",
    "source_file",
    "split_random",
]


def normalize_sequence(seq: str) -> str:
    return str(seq or "").upper().replace("U", "T")


def normalize_region(region: str) -> str:
    text = str(region or "").lower().split(",", 1)[0]
    if "utr5" in text or "5utr" in text:
        return "5UTR"
    if "utr3" in text or "3utr" in text:
        return "3UTR"
    if "cds" in text:
        return "CDS"
    if "intron" in text:
        return "intron"
    if "exon" in text:
        return "exon"
    if "intergenic" in text:
        return "intergenic"
    return "other"


def first_item(value: str) -> str:
    return str(value or "").split(",", 1)[0]


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def parse_class_names(args: argparse.Namespace) -> List[str]:
    if args.target_classes:
        names = [name.strip() for name in args.target_classes.split(",") if name.strip()]
        if not names:
            raise ValueError("--target-classes was provided but no class names were parsed")
        return names
    return list(PRESET_CLASSES[args.preset])


def source_group_from_path(path: Path) -> str:
    name = path.name
    for group in REQUIRED_SOURCE_GROUPS:
        if f".{group}." in name:
            return group
    return "unknown"


def find_bed_files(source_dirs: Sequence[Path]) -> List[Path]:
    files: Dict[str, Path] = {}
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.glob("human.hg38.*.result.col29.bed")):
            files[str(path.resolve())] = path
    return sorted(files.values(), key=lambda p: p.name)


def safe_extract_tar(path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_resolved = out_dir.resolve()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(out_resolved)):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        archive.extractall(out_dir)


def download_url(url: str, out_path: Path, insecure: bool = False) -> None:
    context = ssl._create_unverified_context() if insecure else None
    request = urllib.request.Request(url, headers={"User-Agent": "StructDualRM-RMBase-downloader/1.0"})
    with urllib.request.urlopen(request, context=context) as response, out_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def download_missing_archives(
    raw_dir: Path,
    extracted_dir: Path,
    required_groups: Sequence[str],
    existing_source_dirs: Sequence[Path],
    insecure: bool = False,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    existing_groups = {source_group_from_path(path) for path in find_bed_files([*existing_source_dirs, extracted_dir])}
    missing = [group for group in required_groups if group not in existing_groups]
    for group in missing:
        url = SOURCE_ARCHIVES[group]
        archive_path = raw_dir / f"hg38.{group}.tar.gz"
        if not archive_path.exists():
            print(f"downloading {group}: {url}", flush=True)
            download_url(url, archive_path, insecure=insecure)
        print(f"extracting {archive_path}", flush=True)
        safe_extract_tar(archive_path, extracted_dir)


def parse_bed_row(
    parts: Sequence[str],
    source_file: str,
    class_to_id: Mapping[str, int],
) -> Dict[str, str] | None:
    if len(parts) < 19:
        return None
    raw_mod = parts[6]
    mod_type = RAW_MOD_TO_CLASS.get(raw_mod, raw_mod)
    if mod_type not in class_to_id:
        return None
    seq = normalize_sequence(parts[18])
    return {
        "site_id": parts[3],
        "mod_type": mod_type,
        "class_index": str(class_to_id[mod_type]),
        "chr": parts[0],
        "start": parts[1],
        "end": parts[2],
        "strand": parts[5],
        "support_num": parts[7],
        "support_list": parts[8],
        "pub_list": parts[10],
        "cell_list": parts[11],
        "seq_type_list": parts[12],
        "gene_id": first_item(parts[13]),
        "transcript_id": first_item(parts[14]),
        "gene_name": first_item(parts[15]),
        "gene_type": first_item(parts[16]),
        "region_raw": first_item(parts[17]),
        "region_type": normalize_region(parts[17]),
        "sequence_41nt": seq,
        "center_base": seq[len(seq) // 2] if seq else "",
        "source_mod_type": raw_mod,
        "source_file": source_file,
        "split_random": "",
    }


def row_is_valid(row: Mapping[str, str], window_length: int) -> tuple[bool, str]:
    seq = normalize_sequence(row["sequence_41nt"])
    if len(seq) != window_length:
        return False, "invalid_length"
    if set(seq) - set("ACGT"):
        return False, "invalid_base"
    expected = EXPECTED_CENTER[row["mod_type"]]
    if row["center_base"] != expected:
        return False, "center_base_mismatch"
    return True, "valid"


def assign_stratified_splits(rows: List[Dict[str, str]], seed: int) -> None:
    rng = random.Random(seed)
    by_class: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["mod_type"]].append(row)
    for group in by_class.values():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * 0.8))
        n_valid = max(1, round(n * 0.1)) if n >= 10 else max(0, n - n_train - 1)
        if n_train + n_valid >= n and n >= 3:
            n_train = n - 2
            n_valid = 1
        for idx, row in enumerate(group):
            if idx < n_train:
                row["split_random"] = "train"
            elif idx < n_train + n_valid:
                row["split_random"] = "valid"
            else:
                row["split_random"] = "test"


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def default_output_paths(out_dir: Path, preset: str) -> tuple[Path, Path, Path]:
    stem = f"rmbase_{preset}_hg38_41nt_all_sites"
    return (
        out_dir / "datasets" / f"{stem}.tsv",
        out_dir / "tables" / f"{stem}_class_counts.tsv",
        out_dir / "datasets" / f"{stem}_build_summary.json",
    )


def build(args: argparse.Namespace) -> Dict[str, object]:
    class_names = parse_class_names(args)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    unknown_expected = [name for name in class_names if name not in EXPECTED_CENTER]
    if unknown_expected:
        raise ValueError(f"Missing expected-center mapping for classes: {unknown_expected}")

    source_dirs = [Path(path) for path in args.source_dirs]
    if args.download_missing:
        download_missing_archives(
            Path(args.raw_dir),
            Path(args.download_extract_dir),
            REQUIRED_SOURCE_GROUPS,
            source_dirs,
            insecure=args.insecure_download,
        )
        source_dirs.append(Path(args.download_extract_dir))

    out_dir = Path(args.out_dir)
    dataset_path, counts_path, summary_path = default_output_paths(out_dir, args.preset)
    if args.out_dataset:
        dataset_path = Path(args.out_dataset)
    if args.counts_out:
        counts_path = Path(args.counts_out)
    if args.summary_out:
        summary_path = Path(args.summary_out)

    bed_files = find_bed_files(source_dirs)
    available_groups = sorted({source_group_from_path(path) for path in bed_files})
    missing_groups = [group for group in REQUIRED_SOURCE_GROUPS if group not in available_groups]
    if missing_groups:
        raise FileNotFoundError(
            "Missing RMBase source groups: "
            + ", ".join(missing_groups)
            + ". Re-run with --download-missing or provide source directories containing the extracted BED files."
        )

    rows: List[Dict[str, str]] = []
    raw_counts = Counter()
    valid_counts = Counter()
    invalid_counts = Counter()
    duplicate_counts = Counter()
    center_counts: Dict[str, Counter] = defaultdict(Counter)
    source_group_counts = Counter()
    seen = set()

    for bed_path in bed_files:
        print(f"reading {bed_path}", flush=True)
        source_group_counts[source_group_from_path(bed_path)] += 1
        with bed_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                row = parse_bed_row(line.rstrip("\n").split("\t"), bed_path.name, class_to_id)
                if row is None:
                    continue
                mod_type = row["mod_type"]
                raw_counts[mod_type] += 1
                ok, reason = row_is_valid(row, args.window_length)
                if not ok:
                    invalid_counts[(mod_type, reason)] += 1
                    continue
                valid_counts[mod_type] += 1
                center_counts[mod_type][row["center_base"]] += 1
                key = (row["chr"], row["start"], row["end"], row["strand"], row["mod_type"])
                if key in seen:
                    duplicate_counts[mod_type] += 1
                    continue
                seen.add(key)
                rows.append(row)

    assign_stratified_splits(rows, args.seed)
    rows.sort(
        key=lambda row: (
            class_to_id[row["mod_type"]],
            row["split_random"],
            row["chr"],
            safe_int(row["start"]),
            row["site_id"],
        )
    )
    write_tsv(dataset_path, rows, FIELDS)

    split_counts = Counter((row["mod_type"], row["split_random"]) for row in rows)
    retained_counts = Counter(row["mod_type"] for row in rows)
    count_rows = []
    for mod_type in class_names:
        invalid_by_reason = {
            reason: invalid_counts[(mod_type, reason)]
            for cls, reason in invalid_counts
            if cls == mod_type
        }
        count_rows.append(
            {
                "mod_type": mod_type,
                "class_index": class_to_id[mod_type],
                "raw_target_rows": raw_counts[mod_type],
                "valid_41nt_center_checked_rows": valid_counts[mod_type],
                "invalid_rows": sum(invalid_by_reason.values()),
                "invalid_by_reason": json.dumps(invalid_by_reason, ensure_ascii=False, sort_keys=True),
                "exact_duplicate_rows": duplicate_counts[mod_type],
                "retained_all_site_rows": retained_counts[mod_type],
                "train_rows": split_counts[(mod_type, "train")],
                "valid_rows": split_counts[(mod_type, "valid")],
                "test_rows": split_counts[(mod_type, "test")],
                "center_base_distribution": json.dumps(dict(center_counts[mod_type]), ensure_ascii=False, sort_keys=True),
            }
        )
    write_tsv(
        counts_path,
        count_rows,
        [
            "mod_type",
            "class_index",
            "raw_target_rows",
            "valid_41nt_center_checked_rows",
            "invalid_rows",
            "invalid_by_reason",
            "exact_duplicate_rows",
            "retained_all_site_rows",
            "train_rows",
            "valid_rows",
            "test_rows",
            "center_base_distribution",
        ],
    )

    summary = {
        "dataset": f"RMBase v3 Homo sapiens hg38 {args.preset} supplementary all-site dataset",
        "assembly": "hg38",
        "preset": args.preset,
        "target_classes": class_names,
        "window_length": args.window_length,
        "dataset_tsv": str(dataset_path),
        "counts_tsv": str(counts_path),
        "total_retained_rows": len(rows),
        "raw_counts": dict(raw_counts),
        "valid_counts": dict(valid_counts),
        "retained_counts": dict(retained_counts),
        "split_counts": {f"{mod}:{split}": count for (mod, split), count in sorted(split_counts.items())},
        "source_dirs": [str(path) for path in source_dirs],
        "source_files": [str(path) for path in bed_files],
        "source_archive_urls": SOURCE_ARCHIVES,
        "label_mapping": {
            "A-I": "AtoI",
            "Y": "Y/Psi",
            "Tm": "Um",
            "m6Am": "m6Am from otherMod",
        },
        "deduplication": "exact genomic coordinate plus strand plus mapped modification type",
        "note": (
            "This is a supplementary fairness/robustness dataset aligned to MultiRM label coverage. "
            "It should not replace the primary RMBase11 strict benchmark because the label set and class distribution change."
        ),
        "seed": args.seed,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MultiRM-aligned supplementary RMBase hg38 datasets.")
    parser.add_argument("--preset", choices=sorted(PRESET_CLASSES), default="multirm12")
    parser.add_argument("--target-classes", default="", help="Optional comma-separated class list overriding --preset.")
    parser.add_argument("--source-dirs", nargs="+", default=["rmbase11_work/extracted", "rmbase12_work/extracted"])
    parser.add_argument("--out-dir", default="rmbase12_work")
    parser.add_argument("--out-dataset", default="")
    parser.add_argument("--counts-out", default="")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--window-length", type=int, default=41)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument(
        "--insecure-download",
        action="store_true",
        help="Skip TLS certificate verification for RMBase downloads. Use only when the official site certificate is expired.",
    )
    parser.add_argument("--raw-dir", default="rmbase12_work/raw")
    parser.add_argument("--download-extract-dir", default="rmbase12_work/extracted")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
