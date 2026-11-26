#!/usr/bin/env python
"""Download selected ENCODE eCLIP GRCh38 RBP peak files.

The script intentionally uses ENCODE's public JSON search API and records a
manifest, so the RBP feature branch can be reproduced from accession IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import ssl
import shutil
import socket
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ENCODE_BASE = "https://www.encodeproject.org"
DEFAULT_TARGETS = [
    "ELAVL1",
    "HNRNPC",
    "HNRNPA1",
    "HNRNPK",
    "HNRNPL",
    "IGF2BP1",
    "IGF2BP2",
    "IGF2BP3",
    "RBM15",
    "FTO",
    "NSUN2",
    "METTL1",
    "UPF1",
    "QKI",
    "PTBP1",
    "RBFOX2",
    "TIA1",
    "TIAL1",
    "PUM1",
    "PUM2",
]
BIOSAMPLE_PRIORITY = {"K562": 0, "HepG2": 1}


def parse_static_resolve(value: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for item in re.split(r"[,;]\s*", str(value or "").strip()):
        if not item or "=" not in item:
            continue
        host, ips_text = item.split("=", 1)
        host = host.strip().lower().rstrip(".")
        ips = [ip.strip() for ip in re.split(r"[| ]+", ips_text) if ip.strip()]
        if host and ips:
            mapping[host] = ips
    return mapping


def install_static_resolver(mapping: Mapping[str, List[str]]) -> None:
    if not mapping:
        return
    original_getaddrinfo = socket.getaddrinfo

    def static_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):  # type: ignore[no-untyped-def]
        key = str(host or "").lower().rstrip(".")
        ips = mapping.get(key)
        if ips:
            rows = []
            for ip in ips:
                address_family = socket.AF_INET6 if ":" in ip else socket.AF_INET
                if family not in (0, socket.AF_UNSPEC, address_family):
                    continue
                socket_type = type or socket.SOCK_STREAM
                protocol = proto or socket.IPPROTO_TCP
                rows.append((address_family, socket_type, protocol, "", (ip, port)))
            if rows:
                return rows
        return original_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = static_getaddrinfo  # type: ignore[assignment]


def run_curl(url: str, output: Optional[Path] = None) -> bytes:
    """Fetch a URL using only Python stdlib.

    The function name is kept for backward compatibility with older versions of
    this script, but it no longer requires the external curl executable.  Set
    ENCODE_INSECURE_SSL=1 only if the server environment has an obsolete CA
    bundle and HTTPS certificate verification fails.
    """

    headers = {
        "User-Agent": "StructDualRM-RBP-download/1.0 (+https://www.encodeproject.org)",
        "Accept": "application/json" if output is None else "*/*",
    }
    context = ssl._create_unverified_context() if os.environ.get("ENCODE_INSECURE_SSL") == "1" else None
    last_error: Optional[BaseException] = None
    for attempt in range(1, 4):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=120, context=context) as response:  # noqa: S310 - public ENCODE URLs.
                if output is None:
                    return response.read()
                output.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = output.with_suffix(output.suffix + ".part")
                with tmp_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
                tmp_path.replace(output)
                return b""
        except Exception as exc:  # noqa: BLE001 - include original failure in retry summary.
            last_error = exc
            if attempt < 3:
                time.sleep(2 * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error


def encode_search(target: Optional[str] = None, limit: str = "all") -> Dict[str, object]:
    params = {
        "type": "File",
        "assay_title": "eCLIP",
        "file_type": "bed narrowPeak",
        "assembly": "GRCh38",
        "output_type": "peaks",
        "status": "released",
        "format": "json",
        "limit": limit,
    }
    if target:
        params["target.label"] = target
    url = f"{ENCODE_BASE}/search/?{urlencode(params)}"
    return json.loads(run_curl(url).decode("utf-8"))


def clean_name(value: object) -> str:
    text = str(value or "unknown")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def biosample_name(file_record: Dict[str, object]) -> str:
    ontology = file_record.get("biosample_ontology")
    if isinstance(ontology, dict):
        return str(ontology.get("term_name") or "unknown")
    return "unknown"


def target_label(file_record: Dict[str, object]) -> str:
    target = file_record.get("target")
    if isinstance(target, dict):
        return str(target.get("label") or target.get("name") or "")
    return str(file_record.get("target.label") or target or "")


def score_file(record: Dict[str, object]) -> tuple:
    biosample = biosample_name(record)
    return (
        0 if record.get("preferred_default") else 1,
        BIOSAMPLE_PRIORITY.get(biosample, 10),
        0 if record.get("status") == "released" else 1,
        int(record.get("file_size") or 0),
    )


def select_files(records: List[Dict[str, object]], max_files_per_target: int) -> List[Dict[str, object]]:
    records = [
        rec
        for rec in records
        if rec.get("status") == "released"
        and rec.get("assembly") == "GRCh38"
        and rec.get("file_type") == "bed narrowPeak"
        and rec.get("output_type") == "peaks"
        and rec.get("href")
    ]
    preferred = [rec for rec in records if rec.get("preferred_default")]
    pool = preferred or records
    by_biosample: Dict[str, List[Dict[str, object]]] = {}
    for rec in sorted(pool, key=score_file):
        by_biosample.setdefault(biosample_name(rec), []).append(rec)

    selected: List[Dict[str, object]] = []
    for biosample in sorted(by_biosample, key=lambda name: BIOSAMPLE_PRIORITY.get(name, 10)):
        selected.append(by_biosample[biosample][0])
        if len(selected) >= max_files_per_target:
            break
    return selected


def write_rows(path: Path, fields: List[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    out_dir: Path,
    targets: List[str],
    manifest_rows: List[Dict[str, object]],
    missing_rows: List[Dict[str, object]],
    manifest_path: Path,
    peaks_dir: Path,
) -> None:
    summary = {
        "targets_requested": targets,
        "targets_with_files": sorted({str(row["target"]) for row in manifest_rows}),
        "missing_targets": missing_rows,
        "downloaded_files": len(manifest_rows),
        "manifest": str(manifest_path),
        "peaks_dir": str(peaks_dir),
        "source": "ENCODE public search API, eCLIP, GRCh38, bed narrowPeak peak files",
    }
    (out_dir / "encode_eclip_download_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download ENCODE eCLIP GRCh38 RBP peak files.")
    parser.add_argument("--out-dir", default="rmbase11_full_work/rbp")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help="Query all released ENCODE eCLIP GRCh38 bed narrowPeak peak files instead of the --targets list.",
    )
    parser.add_argument("--max-files-per-target", type=int, default=2)
    parser.add_argument("--limit", default="all", help="ENCODE search limit; keep 'all' for production runs.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifests without downloading peak files.")
    parser.add_argument(
        "--resolve",
        default=os.environ.get("ENCODE_RESOLVE", ""),
        help=(
            "Static DNS mapping for servers with broken DNS, e.g. "
            "'www.encodeproject.org=54.213.250.9,encode-public.s3.amazonaws.com=52.92.200.137'."
        ),
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable HTTPS certificate verification. Use only if the server CA bundle is broken.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.insecure:
        os.environ["ENCODE_INSECURE_SSL"] = "1"
    static_resolve = parse_static_resolve(args.resolve)
    if static_resolve:
        install_static_resolver(static_resolve)
        print(f"[ENCODE] using static DNS mapping for: {', '.join(sorted(static_resolve))}", flush=True)

    out_dir = Path(args.out_dir)
    peaks_dir = out_dir / "peaks"
    manifest_path = out_dir / "encode_eclip_file_manifest.tsv"
    missing_path = out_dir / "encode_eclip_missing_targets.tsv"
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]

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
    manifest_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []
    if args.all_targets:
        print("[ENCODE] querying all released GRCh38 eCLIP narrowPeak peak files", flush=True)
        payload = encode_search(None, args.limit)
        records = payload.get("@graph", [])
        if not isinstance(records, list):
            records = []
        by_target: Dict[str, List[Dict[str, object]]] = {}
        for rec in records:
            label = target_label(rec)
            if label:
                by_target.setdefault(label, []).append(rec)
        target_records = [(target, by_target[target]) for target in sorted(by_target)]
        targets = [target for target, _ in target_records]
    else:
        target_records = []
        for target in targets:
            print(f"[ENCODE] querying {target}", flush=True)
            payload = encode_search(target, args.limit)
            records = payload.get("@graph", [])
            if not isinstance(records, list):
                records = []
            target_records.append((target, records))

    for target, records in target_records:
        selected = select_files(records, args.max_files_per_target)
        if not selected:
            missing_rows.append({"target": target, "reason": "no released preferred/default GRCh38 bed narrowPeak eCLIP peak file"})
        else:
            for rec in selected:
                accession = str(rec.get("accession") or rec.get("title"))
                biosample = biosample_name(rec)
                href = str(rec["href"])
                url = href if href.startswith("http") else f"{ENCODE_BASE}{href}"
                filename = f"{clean_name(target)}_{clean_name(biosample)}_{clean_name(accession)}.bed.gz"
                out_path = peaks_dir / filename
                if not args.dry_run and (args.force or not out_path.exists() or out_path.stat().st_size == 0):
                    print(f"[ENCODE] downloading {target} {biosample} {accession}", flush=True)
                    run_curl(url, output=out_path)
                manifest_rows.append(
                    {
                        "target": target,
                        "biosample": biosample,
                        "accession": accession,
                        "status": rec.get("status", ""),
                        "preferred_default": bool(rec.get("preferred_default")),
                        "dataset": rec.get("dataset", ""),
                        "assembly": rec.get("assembly", ""),
                        "file_type": rec.get("file_type", ""),
                        "file_size": rec.get("file_size", ""),
                        "href": href,
                        "download_url": url,
                        "download_path": str(out_path),
                        "downloaded_bytes": out_path.stat().st_size if out_path.exists() else 0,
                    }
                )
        write_rows(manifest_path, fields, manifest_rows)
        write_rows(missing_path, ["target", "reason"], missing_rows)
        write_summary(out_dir, targets, manifest_rows, missing_rows, manifest_path, peaks_dir)

    summary = json.loads((out_dir / "encode_eclip_download_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
