#!/usr/bin/env python
"""Convert an RMBase TSV into official-style MultiRM/TransRNAm inputs.

Outputs:
  - a pandas HDF5 file with train/valid/test in/out and 3-mer matrices
  - optional TransRNAm JSON files named *_encodingmatrix_index.json
  - row metadata files for mapping predictions back to RMBase sites
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


BASE_ORDER = "ACGT"
SPLIT_ORDER = ("train", "valid", "test")


def normalize_seq(seq: str) -> str:
    return str(seq or "").upper().replace("U", "T")


def class_names_from_rows(rows: Sequence[Mapping[str, str]], explicit: str = "") -> List[str]:
    if explicit.strip():
        return [name.strip() for name in explicit.split(",") if name.strip()]
    by_index: Dict[int, str] = {}
    fallback: List[str] = []
    seen = set()
    for row in rows:
        mod = row.get("mod_type", "")
        if mod and mod not in seen:
            fallback.append(mod)
            seen.add(mod)
        try:
            idx = int(str(row.get("class_index", "")))
        except ValueError:
            continue
        if mod:
            by_index[idx] = mod
    if by_index:
        return [by_index[idx] for idx in sorted(by_index)]
    return fallback


def load_rows(path: Path, max_rows: int = 0) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if max_rows:
        rows = rows[:max_rows]
    return rows


def load_embedding_vocab(path: Path) -> Dict[str, int]:
    with path.open("rb") as handle:
        embedding_dict = pickle.load(handle)
    return {str(word).upper().replace("U", "T"): idx for idx, word in enumerate(embedding_dict.keys())}


def encode_trimers(seq: str, vocab: Mapping[str, int]) -> List[int]:
    seq = normalize_seq(seq)
    encoded = []
    for idx in range(len(seq) - 2):
        token = seq[idx : idx + 3]
        if token not in vocab:
            raise ValueError(f"3-mer token not present in embedding vocabulary: {token}")
        encoded.append(int(vocab[token]))
    return encoded


def encode_onehot_flat(seq: str) -> List[int]:
    seq = normalize_seq(seq)
    out: List[int] = []
    for base in seq:
        out.extend([1 if base == candidate else 0 for candidate in BASE_ORDER])
    return out


def onehot_label(mod_type: str, class_to_id: Mapping[str, int]) -> List[int]:
    y = [0] * len(class_to_id)
    y[class_to_id[mod_type]] = 1
    return y


def split_rows(rows: Sequence[Dict[str, str]], split_col: str) -> Dict[str, List[Dict[str, str]]]:
    out = {name: [] for name in SPLIT_ORDER}
    for row in rows:
        split = str(row.get(split_col, "")).lower()
        if split in out:
            out[split].append(row)
    return out


def build_split_frames(
    rows: Sequence[Dict[str, str]],
    class_names: Sequence[str],
    seq_col: str,
    vocab: Mapping[str, int],
) -> Dict[str, pd.DataFrame]:
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    seqs = []
    trimers = []
    onehot = []
    labels = []
    class_indices = []
    expected_len: Optional[int] = None
    for row in rows:
        mod_type = row.get("mod_type", "")
        if mod_type not in class_to_id:
            continue
        seq = normalize_seq(row.get(seq_col, ""))
        if not seq:
            continue
        if set(seq) - set(BASE_ORDER):
            continue
        if expected_len is None:
            expected_len = len(seq)
        if len(seq) != expected_len:
            raise ValueError(f"Mixed sequence lengths in {seq_col}: expected {expected_len}, got {len(seq)}")
        seqs.append(seq)
        trimers.append(encode_trimers(seq, vocab))
        onehot.append(encode_onehot_flat(seq))
        labels.append(onehot_label(mod_type, class_to_id))
        class_indices.append(class_to_id[mod_type])
    if expected_len is None:
        expected_len = 0
    # Keep official-code HDF tables purely numeric with RangeIndex columns.
    # Several server environments mix old pandas/PyTables builds with newer
    # numpy; object/string indexes or column names can then fail even before
    # training starts.  The official MultiRM/TransRNAm code only consumes the
    # values, so dropping site IDs and string column labels is behaviorally
    # neutral for model training/evaluation.
    trimer_width = max(expected_len - 2, 0)
    onehot_width = expected_len * len(BASE_ORDER)
    return {
        "in_3_mers": pd.DataFrame(np.asarray(trimers, dtype=np.int32).reshape((-1, trimer_width))),
        "in": pd.DataFrame(np.asarray(onehot, dtype=np.int8).reshape((-1, onehot_width))),
        "in_nucleo": pd.DataFrame({"sequence_length": np.full(len(seqs), expected_len, dtype=np.int16)}),
        "out": pd.DataFrame(np.asarray(labels, dtype=np.int8).reshape((-1, len(class_names)))),
        "metadata": pd.DataFrame(
            {
                "row_id": np.arange(len(seqs), dtype=np.int64),
                "class_index": np.asarray(class_indices, dtype=np.int16),
                "sequence_length": np.full(len(seqs), expected_len, dtype=np.int16),
            }
        ),
    }


def write_transrnam_json(frames_by_split: Mapping[str, Mapping[str, pd.DataFrame]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_code = {"train": 0, "valid": 1, "test": 2}
    for split in SPLIT_ORDER:
        matrix = frames_by_split[split]["in_3_mers"].reset_index(drop=True)
        index_df = pd.DataFrame({"ind": np.arange(len(matrix), dtype=np.int64), "class": split_code[split]})
        index_df.to_json(out_dir / f"{split}_encodingmatrix_index.json")
        matrix.to_json(out_dir / f"{split}_Word2vec_encodingmatrix.json")
        if split == "train":
            # The released TransRNAm training loop looks for this legacy name in
            # the training phase.
            matrix.to_json(out_dir / "train_test_Word2vec_encodingmatrix.json")


def write_hdf(frames_by_split: Mapping[str, Mapping[str, pd.DataFrame]], out_h5: Path, complevel: int) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    if out_h5.exists():
        out_h5.unlink()
    with pd.HDFStore(out_h5, mode="w", complevel=complevel, complib="zlib") as store:
        for split in SPLIT_ORDER:
            frames = frames_by_split[split]
            # The released MultiRM/TransRNAm code does not use the DataFrame
            # index.  Keeping string site IDs as a fixed-HDF object index can
            # trigger pandas/numpy ABI-sensitive read failures in old server
            # environments, so official-consumed matrices are stored with a
            # plain RangeIndex.
            store.put(f"{split}_in_3_mers", frames["in_3_mers"].reset_index(drop=True), format="fixed")
            store.put(f"{split}_in", frames["in"].reset_index(drop=True), format="fixed")
            store.put(f"{split}_in_nucleo", frames["in_nucleo"].reset_index(drop=True), format="fixed")
            store.put(f"{split}_out", frames["out"].reset_index(drop=True), format="fixed")
            store.put(f"{split}_metadata", frames["metadata"].reset_index(drop=True), format="fixed")


def run(args: argparse.Namespace) -> Dict[str, object]:
    rows = load_rows(Path(args.dataset), args.max_rows)
    class_names = class_names_from_rows(rows, args.class_names)
    if not class_names:
        raise ValueError("Could not infer class names")
    seq_col = args.sequence_column
    if not any(seq_col in row for row in rows):
        raise ValueError(f"Sequence column not found: {seq_col}")
    vocab = load_embedding_vocab(Path(args.embedding_path))
    by_split = split_rows(rows, args.split)
    frames_by_split: Dict[str, Dict[str, pd.DataFrame]] = {}
    for split in SPLIT_ORDER:
        frames_by_split[split] = build_split_frames(by_split[split], class_names, seq_col, vocab)
    out_h5 = Path(args.out_h5)
    write_hdf(frames_by_split, out_h5, args.compression_level)
    if args.transrnam_json_dir:
        write_transrnam_json(frames_by_split, Path(args.transrnam_json_dir))
    summary = {
        "dataset": args.dataset,
        "out_h5": str(out_h5),
        "transrnam_json_dir": args.transrnam_json_dir,
        "split_column": args.split,
        "sequence_column": seq_col,
        "class_names": class_names,
        "row_counts": {split: int(frames_by_split[split]["out"].shape[0]) for split in SPLIT_ORDER},
        "sequence_length": {
            split: int(frames_by_split[split]["in_nucleo"]["sequence_length"].iloc[0])
            if len(frames_by_split[split]["in_nucleo"])
            else 0
            for split in SPLIT_ORDER
        },
        "hdf_keys": [
            f"{split}_{suffix}"
            for split in SPLIT_ORDER
            for suffix in ["in", "in_3_mers", "in_nucleo", "out", "metadata"]
        ],
    }
    summary_path = Path(args.summary_out) if args.summary_out else out_h5.with_suffix(out_h5.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert RMBase strict split TSV to official-style MultiRM HDF.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="split_cdhit80_cluster")
    parser.add_argument("--sequence-column", default="sequence_101nt")
    parser.add_argument("--class-names", default="")
    parser.add_argument("--embedding-path", default="external/MultiRM-master/Embeddings/embeddings_12RM.pkl")
    parser.add_argument("--out-h5", required=True)
    parser.add_argument("--transrnam-json-dir", default="", help="Write train/valid/test_encodingmatrix_index.json files here.")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--compression-level", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
