#!/usr/bin/env python
"""Create an ENCODE-like manifest from already downloaded local RBP peak files."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path
from typing import Iterable, Optional


def parse_peak_name(path: Path) -> dict[str, object]:
    stem = path.name
    if stem.endswith(".bed.gz"):
        stem = stem[:-7]
    elif stem.endswith(".bed"):
        stem = stem[:-4]
    match = re.match(r"(?P<target>.+?)_(?P<biosample>.+?)_(?P<accession>ENCFF[A-Z0-9]+)$", stem)
    if match:
        parsed = match.groupdict()
    else:
        parsed = {"target": stem.split("_", 1)[0], "biosample": "unknown", "accession": stem}
    parsed.update(
        {
            "status": "released_or_previously_downloaded",
            "preferred_default": "",
            "dataset": "",
            "assembly": "GRCh38",
            "file_type": "bed narrowPeak",
            "file_size": path.stat().st_size,
            "href": "",
            "download_url": "",
            "download_path": str(path),
            "downloaded_bytes": path.stat().st_size,
        }
    )
    return parsed


def is_readable_peak(path: Path) -> tuple[bool, str]:
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:  # type: ignore[arg-type]
            for _ in handle:
                pass
        return True, ""
    except Exception as exc:  # noqa: BLE001 - report malformed downloaded files clearly.
        return False, f"{type(exc).__name__}: {exc}"


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Create a manifest for local RBP peak BED files.")
    parser.add_argument("--peaks-dir", default="rmbase11_full_work/rbp/peaks")
    parser.add_argument("--out", default="rmbase11_full_work/rbp/encode_eclip_file_manifest.tsv")
    parser.add_argument("--skip-unreadable", action="store_true")
    args = parser.parse_args(argv)

    peaks_dir = Path(args.peaks_dir)
    files = sorted(list(peaks_dir.glob("*.bed.gz")) + list(peaks_dir.glob("*.bed")))
    if not files:
        raise SystemExit(f"No .bed or .bed.gz peak files found under {peaks_dir}")

    fields = [
        "target",
        "biosample",
        "accession",
        "status",
        "preferred_default",
        "dataset",
        "assembly",
        "file_type",
        "file_size",
        "href",
        "download_url",
        "download_path",
        "downloaded_bytes",
    ]
    rows = []
    skipped = []
    for path in files:
        ok, reason = is_readable_peak(path)
        if ok or not args.skip_unreadable:
            rows.append(parse_peak_name(path))
        else:
            skipped.append({"path": str(path), "reason": reason, "bytes": path.stat().st_size})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "manifest": str(out_path),
        "peaks_dir": str(peaks_dir),
        "peak_files": len(rows),
        "targets": sorted({str(row["target"]) for row in rows}),
        "skipped_unreadable": skipped,
        "note": "Created from local files because additional ENCODE downloads were not available in the current run.",
    }
    summary_path = out_path.with_name("local_rbp_manifest_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
