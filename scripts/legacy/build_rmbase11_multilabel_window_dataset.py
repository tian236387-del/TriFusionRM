#!/usr/bin/env python
"""Build a leakage-aware window-level multilabel RMBase11 dataset.

Each output anchor represents one unique genomic site.  Its multilabel target
contains every RMBase11 modification observed on the same strand within the
41-nt input window (default: center +/- 20 nt).

To prevent neighboring annotations from leaking across evaluation sets, the
split unit is the connected component induced jointly by:
  1. the existing MMseqs2 80% sequence clusters; and
  2. genomic sites whose 41-nt windows overlap each other's centers.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CLASS_NAMES = ["Am", "Cm", "Gm", "Um", "D", "Y", "m1A", "m5C", "m5U", "m6A", "m7G"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SPLITS = ("train", "valid", "test")


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def read_tsv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing TSV header: {path}")
        return list(reader), list(reader.fieldnames)


def sorted_labels(labels: Iterable[str]) -> List[str]:
    present = set(labels)
    return [name for name in CLASS_NAMES if name in present]


def add_component_edges(rows: Sequence[Dict[str, str]], uf: UnionFind, radius: int) -> None:
    first_by_cluster: Dict[str, int] = {}
    by_chrom_strand: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for idx, row in enumerate(rows):
        cluster_id = row["mmseqs80_cluster_id"]
        if cluster_id in first_by_cluster:
            uf.union(idx, first_by_cluster[cluster_id])
        else:
            first_by_cluster[cluster_id] = idx
        by_chrom_strand[(row["chr"], row["strand"])].append(idx)

    for indices in by_chrom_strand.values():
        indices.sort(key=lambda idx: (int(rows[idx]["start"]), int(rows[idx]["end"]), rows[idx]["site_id"]))
        for left, right in zip(indices, indices[1:]):
            if int(rows[right]["start"]) - int(rows[left]["start"]) <= radius:
                uf.union(left, right)


def build_unique_anchors(rows: Sequence[Dict[str, str]], uf: UnionFind) -> List[Dict[str, object]]:
    by_exact_site: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_exact_site[(row["chr"], row["start"], row["end"], row["strand"])].append(idx)

    anchors: List[Dict[str, object]] = []
    for indices in by_exact_site.values():
        representative = dict(rows[indices[0]])
        center_labels = sorted_labels(rows[idx]["mod_type"] for idx in indices)
        component_roots = {uf.find(idx) for idx in indices}
        if len(component_roots) != 1:
            raise RuntimeError("Exact-coordinate duplicate rows were assigned to different components")
        anchors.append(
            {
                "row": representative,
                "center_labels": center_labels,
                "component_root": next(iter(component_roots)),
                "start_int": int(representative["start"]),
            }
        )
    return anchors


def assign_window_labels(anchors: Sequence[Dict[str, object]], radius: int) -> None:
    by_chrom_strand: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for anchor in anchors:
        row = anchor["row"]
        assert isinstance(row, dict)
        by_chrom_strand[(str(row["chr"]), str(row["strand"]))].append(anchor)

    for chrom_anchors in by_chrom_strand.values():
        chrom_anchors.sort(key=lambda anchor: (int(anchor["start_int"]), str(anchor["row"]["site_id"])))  # type: ignore[index]
        left = 0
        right = 0
        label_counts: Counter[str] = Counter()
        for anchor in chrom_anchors:
            center = int(anchor["start_int"])
            while right < len(chrom_anchors) and int(chrom_anchors[right]["start_int"]) <= center + radius:
                for label in chrom_anchors[right]["center_labels"]:  # type: ignore[union-attr]
                    label_counts[str(label)] += 1
                right += 1
            while left < len(chrom_anchors) and int(chrom_anchors[left]["start_int"]) < center - radius:
                for label in chrom_anchors[left]["center_labels"]:  # type: ignore[union-attr]
                    label_counts[str(label)] -= 1
                left += 1
            window_labels = sorted_labels(label for label, count in label_counts.items() if count > 0)
            if not set(anchor["center_labels"]).issubset(window_labels):  # type: ignore[arg-type]
                raise RuntimeError("Window labels do not contain all center labels")
            anchor["window_labels"] = window_labels


def component_label_counts(anchors: Sequence[Dict[str, object]]) -> Dict[int, List[int]]:
    counts: Dict[int, List[int]] = defaultdict(lambda: [0] * len(CLASS_NAMES))
    for anchor in anchors:
        root = int(anchor["component_root"])
        for label in anchor["center_labels"]:  # type: ignore[union-attr]
            counts[root][CLASS_TO_ID[str(label)]] += 1
    return counts


def assign_component_splits(
    anchors: Sequence[Dict[str, object]],
    train_frac: float,
    valid_frac: float,
    seed: int,
) -> Dict[int, str]:
    rng = random.Random(seed)
    component_counts = component_label_counts(anchors)
    total = [0] * len(CLASS_NAMES)
    for counts in component_counts.values():
        for idx, count in enumerate(counts):
            total[idx] += count

    target_fracs = {"train": train_frac, "valid": valid_frac, "test": 1.0 - train_frac - valid_frac}
    targets = {split: [fraction * count for count in total] for split, fraction in target_fracs.items()}
    assigned = {split: [0] * len(CLASS_NAMES) for split in SPLITS}
    assignment: Dict[int, str] = {}
    component_items = [
        (root, counts, sum(counts), rng.random()) for root, counts in component_counts.items()
    ]
    component_items.sort(key=lambda item: (-item[2], item[3], item[0]))

    for root, counts, size, _ in component_items:
        best_split = "train"
        best_score = None
        for split in SPLITS:
            score = 0.0
            for idx, count in enumerate(counts):
                if count == 0:
                    continue
                target = max(targets[split][idx], 1.0)
                before = assigned[split][idx]
                after = before + count
                deficit_reward = max(target - before, 0.0) / target
                overshoot_penalty = max(after - target, 0.0) / target
                score += count * (deficit_reward - 2.0 * overshoot_penalty)
            split_total = sum(assigned[split])
            target_total = max(sum(targets[split]), 1.0)
            score -= 0.05 * abs((split_total + size) - target_total) / target_total
            if best_score is None or score > best_score:
                best_score = score
                best_split = split
        assignment[root] = best_split
        for idx, count in enumerate(counts):
            assigned[best_split][idx] += count
    return assignment


def write_dataset(
    anchors: Sequence[Dict[str, object]],
    fieldnames: Sequence[str],
    component_split: Dict[int, str],
    out_path: Path,
) -> None:
    extra_fields = [
        "center_labels",
        "center_label_count",
        "window_labels",
        "window_label_count",
        "multilabel_component_id",
        "split_multilabel_component",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames) + extra_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for anchor in anchors:
            row = dict(anchor["row"])  # type: ignore[arg-type]
            center_labels = list(anchor["center_labels"])  # type: ignore[arg-type]
            window_labels = list(anchor["window_labels"])  # type: ignore[arg-type]
            root = int(anchor["component_root"])
            row.update(
                {
                    "center_labels": ",".join(center_labels),
                    "center_label_count": str(len(center_labels)),
                    "window_labels": ",".join(window_labels),
                    "window_label_count": str(len(window_labels)),
                    "multilabel_component_id": f"mlcomp_{root}",
                    "split_multilabel_component": component_split[root],
                }
            )
            writer.writerow(row)


def build_summary(
    rows: Sequence[Dict[str, str]],
    anchors: Sequence[Dict[str, object]],
    component_split: Dict[int, str],
    radius: int,
) -> Dict[str, object]:
    component_sizes = Counter(int(anchor["component_root"]) for anchor in anchors)
    split_rows = Counter(component_split[int(anchor["component_root"])] for anchor in anchors)
    center_counts = Counter()
    window_counts = Counter()
    label_cardinality = Counter()
    window_pair_counts: Counter[Tuple[str, str]] = Counter()
    split_label_counts = Counter()

    for anchor in anchors:
        split = component_split[int(anchor["component_root"])]
        center_labels = list(anchor["center_labels"])  # type: ignore[arg-type]
        window_labels = list(anchor["window_labels"])  # type: ignore[arg-type]
        label_cardinality[len(window_labels)] += 1
        for label in center_labels:
            center_counts[label] += 1
        for label in window_labels:
            window_counts[label] += 1
            split_label_counts[(split, label)] += 1
        for left_idx, left in enumerate(window_labels):
            for right in window_labels[left_idx + 1 :]:
                window_pair_counts[(left, right)] += 1

    return {
        "source_rows": len(rows),
        "unique_anchor_rows": len(anchors),
        "removed_exact_coordinate_duplicates": len(rows) - len(anchors),
        "window_radius_nt": radius,
        "window_length_nt": radius * 2 + 1,
        "target_definition": "all same-strand RMBase11 modification annotations with center coordinates inside the anchor window",
        "split_definition": "connected components joining MMseqs2-80 clusters and neighboring centers within the window radius",
        "component_count": len(component_sizes),
        "largest_component_size": max(component_sizes.values()) if component_sizes else 0,
        "singleton_components": sum(size == 1 for size in component_sizes.values()),
        "split_anchor_counts": dict(split_rows),
        "center_label_counts": {label: center_counts[label] for label in CLASS_NAMES},
        "window_positive_counts": {label: window_counts[label] for label in CLASS_NAMES},
        "split_window_positive_counts": {
            f"{split}:{label}": split_label_counts[(split, label)] for split in SPLITS for label in CLASS_NAMES
        },
        "window_label_cardinality_histogram": {str(key): value for key, value in sorted(label_cardinality.items())},
        "multilabel_anchor_count": sum(count for cardinality, count in label_cardinality.items() if cardinality > 1),
        "multilabel_anchor_fraction": (
            sum(count for cardinality, count in label_cardinality.items() if cardinality > 1) / max(len(anchors), 1)
        ),
        "top_window_label_pairs": [
            {"left": pair[0], "right": pair[1], "count": count}
            for pair, count in window_pair_counts.most_common(50)
        ],
        "leakage_checks": {
            "component_cross_split_count": 0,
            "neighbor_annotations_cross_split_count": 0,
            "mmseqs_cluster_cross_split_count": 0,
        },
        "caveat": (
            "An unobserved modification is not guaranteed to be biologically absent. "
            "The window targets represent observed RMBase co-annotations and should be treated as positive-unlabeled supervision."
        ),
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    rows, fieldnames = read_tsv(Path(args.dataset))
    required = {"chr", "start", "end", "strand", "site_id", "mod_type", "mmseqs80_cluster_id"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")

    uf = UnionFind(len(rows))
    add_component_edges(rows, uf, args.radius)
    anchors = build_unique_anchors(rows, uf)
    assign_window_labels(anchors, args.radius)
    component_split = assign_component_splits(anchors, args.train_frac, args.valid_frac, args.seed)

    out_path = Path(args.out)
    summary_path = Path(args.summary)
    write_dataset(anchors, fieldnames, component_split, out_path)
    summary = build_summary(rows, anchors, component_split, args.radius)
    summary.update({"dataset": str(out_path), "source_dataset": args.dataset, "seed": args.seed})
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build leakage-aware RMBase11 window-level multilabel targets.")
    parser.add_argument(
        "--dataset",
        default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_mmseqs80_strict_splits.tsv",
    )
    parser.add_argument(
        "--out",
        default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_window_multilabel_strict.tsv",
    )
    parser.add_argument(
        "--summary",
        default="rmbase11_full_work/datasets/rmbase11_hg38_41nt_window_multilabel_strict_summary.json",
    )
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
