#!/usr/bin/env python
"""Convert a large RBP positional TSV into a reusable NumPy memmap cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def expected_columns(metadata: dict) -> list[str]:
    return [
        f"rbp_pos__{rbp}__{bin_info['label']}__{feature}"
        for rbp in metadata["rbps"]
        for bin_info in metadata["bins"]
        for feature in metadata["features"]
    ]


def convert(source: Path, output: Path) -> dict:
    source_meta_path = source.with_suffix(source.suffix + ".json")
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    shape = tuple(int(value) for value in source_meta["shape"])
    site_count = int(source_meta["site_count"])
    columns = expected_columns(source_meta)
    if len(columns) != int(np.prod(shape)):
        raise ValueError(f"Metadata shape {shape} does not match {len(columns)} feature columns")

    source_stat = source.stat()
    output_meta_path = output / "metadata.json"
    if (output / "complete").is_file() and output_meta_path.is_file():
        prior = json.loads(output_meta_path.read_text(encoding="utf-8"))
        if (
            int(prior.get("source_size_bytes", -1)) == source_stat.st_size
            and int(prior.get("source_mtime_ns", -1)) == source_stat.st_mtime_ns
            and (output / "values.npy").is_file()
            and (output / "site_ids.txt").is_file()
        ):
            print(json.dumps(prior, indent=2), flush=True)
            return prior

    output.mkdir(parents=True, exist_ok=True)
    complete = output / "complete"
    if complete.exists():
        complete.unlink()
    values_path = output / "values.npy"
    ids_path = output / "site_ids.txt"
    values = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float32, shape=(site_count, *shape))
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as source_handle, ids_path.open("w", encoding="utf-8", newline="\n") as id_handle:
        header_raw = source_handle.readline()
        digest.update(header_raw)
        header = header_raw.decode("utf-8-sig").rstrip("\r\n").split("\t")
        if header != ["site_id", *columns]:
            mismatch = next((idx for idx, pair in enumerate(zip(header, ["site_id", *columns])) if pair[0] != pair[1]), None)
            raise ValueError(f"RBP TSV header/order mismatch at column {mismatch}: {source}")
        for count, raw in enumerate(source_handle, start=1):
            digest.update(raw)
            site_raw, separator, feature_raw = raw.rstrip(b"\r\n").partition(b"\t")
            if not separator:
                raise ValueError(f"Malformed RBP row {count}")
            array = np.fromstring(feature_raw.decode("ascii"), dtype=np.float32, sep="\t")
            if array.size != len(columns):
                raise ValueError(f"RBP row {count} has {array.size} values; expected {len(columns)}")
            if count > site_count:
                raise ValueError(f"RBP TSV contains more than declared {site_count} rows")
            values[count - 1] = array.reshape(shape)
            id_handle.write(site_raw.decode("utf-8") + "\n")
            if count % 100000 == 0:
                values.flush()
                print(f"converted_sites={count}", flush=True)
    values.flush()
    del values
    if count != site_count:
        raise ValueError(f"RBP TSV contains {count} rows; metadata declares {site_count}")

    payload = dict(source_meta)
    payload.update(
        {
            "storage": "numpy_memmap",
            "source_tsv": str(source),
            "source_metadata": str(source_meta_path),
            "source_size_bytes": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "source_sha256": digest.hexdigest(),
            "values_file": "values.npy",
            "site_ids_file": "site_ids.txt",
            "dtype": "float32",
            "array_shape": [site_count, *shape],
        }
    )
    temporary_meta = output / "metadata.json.tmp"
    temporary_meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_meta, output_meta_path)
    complete.write_text(payload["source_sha256"] + "\n", encoding="ascii")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    convert(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
