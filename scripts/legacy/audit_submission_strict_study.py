#!/usr/bin/env python
"""Programmatic audit of the frozen hg38 StructDualRM submission study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence


CLASSES = ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m7G", "Y", "AtoI"]
SEEDS = [42, 17, 73, 101, 202]
MODELS = {
    "StructDualRM": "main_controls/full",
    "MultiRM": "baselines/multirm_official",
    "TransRNAm": "baselines/transrnam_official",
    "EvoRMD": "baselines/evormd_safe",
    "RNA-FM sequence-only": "baselines/rnafm_sequence_only",
}
CONTROLS = [
    "sequence_only",
    "sequence_structure",
    "sequence_rbp",
    "full",
    "shuffled_structure",
    "shuffled_rbp",
    "shuffled_rbp_bins",
    "explicit_pair_edges",
    "capacity_matched_structure_cnn",
    "parameter_matched_rbp_mlp",
]
HOLDOUTS = ["gene_holdout", "transcript_holdout", "study_holdout"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(rows[0]) if rows else ["item", "status", "evidence", "detail"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prediction_ids(path: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            ids.append(row["site_id"])
            labels.append(row.get("true_mod_type", row.get("center_labels", "")))
    return ids, labels


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def check(self, item: str, condition: bool, evidence: str, detail: str, warning: bool = False) -> None:
        status = "PASS" if condition else "WARN" if warning else "FAIL"
        self.rows.append({"item": item, "status": status, "evidence": evidence, "detail": detail})

    def guarded(self, item: str, evidence: str, function: Callable[[], tuple[bool, str]], warning: bool = False) -> None:
        try:
            condition, detail = function()
            self.check(item, condition, evidence, detail, warning=warning)
        except Exception as exc:
            self.check(item, False, evidence, f"{type(exc).__name__}: {exc}", warning=warning)


def complete_seed_tree(root: Path) -> tuple[bool, str]:
    missing = []
    for seed in SEEDS:
        seed_root = root / f"seed{seed}"
        for name in ("metrics.json", "valid_predictions.tsv", "test_predictions.tsv", "training_config.json"):
            path = seed_root / name
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    return not missing, f"missing={len(missing)}" + (f" examples={missing[:3]}" if missing else "")


def finite_efficiency(result_root: Path) -> tuple[bool, str]:
    rows = read_tsv(result_root / "tables" / "efficiency_five_seed_summary.tsv")
    required = {
        "parameter_count",
        "trainable_parameter_count",
        "flops_per_sample",
        "training_seconds",
        "training_peak_memory_bytes",
        "samples_per_second",
        "latency_ms_per_sample",
    }
    missing = []
    for model in MODELS:
        by_metric = {row["metric"]: row for row in rows if row["model"] == model}
        for metric in required:
            try:
                value = float(by_metric[metric]["mean"])
            except (KeyError, TypeError, ValueError):
                value = float("nan")
            if not math.isfinite(value):
                missing.append(f"{model}:{metric}")
    return not missing, f"nonfinite={missing}"


def run(args: argparse.Namespace) -> dict:
    work, runs, results = args.work, args.run_root, args.result_root
    audit = Audit()
    strict = read_json(work / "strict_split_audit.json")
    prepare = read_json(work / "prepare_summary.json")

    audit.check(
        "1_coherent_hg38_confirmatory_study",
        strict.get("class_names") == CLASSES and int(strict.get("rows", 0)) == 1081174,
        str(work / "strict_split_audit.json"),
        f"classes={strict.get('class_names')} rows={strict.get('rows')}",
    )
    audit.check(
        "2_center_and_within_family_controls",
        all(
            (results / "tables" / name).is_file()
            for name in ("within_family_metrics.tsv", "within_family_confusion.tsv", "ensemble_per_class_metrics.tsv")
        )
        and "Center-base only" in read_json(results / "ensemble_metrics.json"),
        str(results / "tables"),
        "Center-base prior, within-family metrics and family confusion matrices are present",
    )
    audit.check(
        "3_conflicts_removed_and_12_label_supplement",
        int(prepare.get("coordinate_conflict_keys", 0)) == 429
        and int(prepare.get("excluded_rows", 0)) == 870
        and "m6Am" not in strict.get("class_names", [])
        and "m6Am" in strict.get("multilabel", {}).get("classes", []),
        str(work / "prepare_summary.json"),
        (
            f"coordinate_conflicts={prepare.get('coordinate_conflict_keys')} excluded={prepare.get('excluded_rows')} "
            f"multilabel_rows={strict.get('multilabel', {}).get('rows')}"
        ),
    )

    leakage = []
    for section in (strict, strict.get("multilabel", {})):
        leakage.extend(int(value.get("leaked_groups", -1)) for value in section.get("leakage_audits", {}).values())
    clustering = strict.get("clustering", {})
    audit.check(
        "4_201nt_clustering_and_group_holdouts",
        clustering.get("sequence_length_nt") == 201
        and float(clustering.get("identity", 0)) == 0.8
        and leakage
        and max(leakage) == 0,
        str(work / "strict_split_audit.json"),
        (
            f"algorithm={clustering.get('algorithm')} clusters={clustering.get('clusters')} leaked_groups={leakage}; "
            f"study_evaluable={strict.get('lockbox_evaluable_classes', [])}"
        ),
    )
    audit.check(
        "4b_study_lockbox_coverage_limitation_declared",
        len(strict.get("lockbox_evaluable_classes", [])) >= 2 and bool(strict.get("lockbox_limitation")),
        str(work / "strict_split_audit.json"),
        f"evaluable={strict.get('lockbox_evaluable_classes')} missing={strict.get('lockbox_missing_classes')}",
        warning=False,
    )

    rbp_labels = ("cluster201", "gene_holdout", "transcript_holdout", "study_holdout")
    rbp_manifests = [work / "rbp_train_only" / f"{label}_top30_train_manifest.tsv" for label in rbp_labels]
    rbp_details = []
    rbp_valid = True
    for label, path in zip(rbp_labels, rbp_manifests):
        if not path.is_file():
            rbp_valid = False
            rbp_details.append(f"missing:{path.name}")
            continue
        targets = {row["target"] for row in read_tsv(path)}
        selection_metadata_path = path.with_suffix(path.suffix + ".json")
        selection_metadata = read_json(selection_metadata_path) if selection_metadata_path.is_file() else {}
        expected_split = "split_cluster201" if label == "cluster201" else f"split_{label}"
        selection_valid = (
            len(targets) == 30
            and selection_metadata.get("split_column") == expected_split
            and selection_metadata.get("split_value") == "train"
            and int(selection_metadata.get("top_n", 0)) == 30
            and set(selection_metadata.get("selected_targets", [])) == targets
        )
        rbp_valid &= selection_valid
        rbp_details.append(
            f"{path.name}:{len(targets)}:split={selection_metadata.get('split_column')}:"
            f"value={selection_metadata.get('split_value')}"
        )
    metadata_path = work / "rbp_train_only" / "multirm11_top30_train_rbp_positional_features.tsv.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    metadata_manifest = metadata.get("training_only_manifest", metadata.get("manifest"))
    primary_manifest = str(rbp_manifests[0])
    audit.check(
        "5_training_only_top30_rbp_selection",
        rbp_valid and metadata_manifest == primary_manifest,
        ";".join(str(path) for path in rbp_manifests),
        f"manifests={rbp_details} metadata_manifest={metadata_manifest}",
    )

    for model, relative in MODELS.items():
        ok, detail = complete_seed_tree(runs / relative)
        audit.check(f"6_five_seeds_{model}", ok, str(runs / relative), detail)
    controls_ok = []
    for variant in CONTROLS:
        ok, detail = complete_seed_tree(runs / "main_controls" / variant)
        controls_ok.append(ok)
        if not ok:
            audit.rows.append(
                {"item": f"7_control_{variant}", "status": "FAIL", "evidence": str(runs), "detail": detail}
            )
    audit.check(
        "7_multiseed_negative_and_matched_controls",
        all(controls_ok),
        str(runs / "main_controls"),
        f"variants={len(CONTROLS)} complete={sum(controls_ok)}",
    )
    for holdout in HOLDOUTS:
        ok, detail = complete_seed_tree(runs / "holdouts" / holdout)
        audit.check(f"4_holdout_five_seeds_{holdout}", ok, str(runs / "holdouts" / holdout), detail)
    ok, detail = complete_seed_tree(runs / "multilabel")
    audit.check("3_multilabel_five_seeds", ok, str(runs / "multilabel"), detail)

    provenance_path = results / "bootstrap" / "bootstrap_provenance.json"
    provenance = read_json(provenance_path) if provenance_path.is_file() else {}
    audit.check(
        "8_cluster_paired_bootstrap_2000",
        int(provenance.get("iterations", 0)) >= 2000
        and provenance.get("group_column") == "cluster201_id"
        and (results / "bootstrap" / "cluster_paired_bootstrap.tsv").is_file(),
        str(provenance_path),
        f"iterations={provenance.get('iterations')} unit={provenance.get('resampling_unit')}",
    )
    lockbox_path = results / "lockbox_metrics.json"
    lockbox = read_json(lockbox_path) if lockbox_path.is_file() else {}
    calibration = lockbox.get("calibrated_calibration", {})
    audit.check(
        "9_frozen_lockbox_and_calibration",
        int(lockbox.get("test_rows", 0)) > 0
        and all(math.isfinite(float(calibration.get(key, float("nan")))) for key in ("ECE_15", "Brier_multiclass", "NLL"))
        and (results / "figures" / "reliability_curves.png").is_file(),
        str(lockbox_path),
        f"test_rows={lockbox.get('test_rows')} calibration={calibration}",
    )
    audit.guarded(
        "10_modern_baseline_and_efficiency",
        str(results / "tables" / "efficiency_five_seed_summary.tsv"),
        lambda: finite_efficiency(results),
    )

    reference_ids, reference_labels = prediction_ids(
        results / "ensemble_predictions" / "structdualrm_test_predictions_raw.tsv"
    )
    mismatched = []
    for filename in (
        "multirm_test_predictions_raw.tsv",
        "transrnam_test_predictions_raw.tsv",
        "evormd_test_predictions_raw.tsv",
        "rna_fm_sequence_only_test_predictions_raw.tsv",
    ):
        ids, labels = prediction_ids(results / "ensemble_predictions" / filename)
        if ids != reference_ids or labels != reference_labels:
            mismatched.append(filename)
    audit.check(
        "paired_test_rows_identical_across_models",
        not mismatched,
        str(results / "ensemble_predictions"),
        f"test_rows={len(reference_ids)} mismatched={mismatched}",
    )

    failures = [row for row in audit.rows if row["status"] == "FAIL"]
    warnings = [row for row in audit.rows if row["status"] == "WARN"]
    payload = {
        "overall_status": "PASS" if not failures else "FAIL",
        "checks": len(audit.rows),
        "passed": sum(row["status"] == "PASS" for row in audit.rows),
        "warnings": len(warnings),
        "failed": len(failures),
        "failure_items": [row["item"] for row in failures],
        "warning_items": [row["item"] for row in warnings],
        "scientific_limitations": [strict.get("lockbox_limitation", "")],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output / "submission_audit.tsv", audit.rows)
    (args.output / "submission_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if args.strict and failures:
        raise SystemExit(1)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("rmbase12_work/submission_strict"))
    parser.add_argument("--run-root", type=Path, default=Path("rmbase12_work/submission_strict_runs_20260806"))
    parser.add_argument("--result-root", type=Path, default=Path("rmbase12_work/submission_strict_final_20260806"))
    parser.add_argument("--output", type=Path, default=Path("rmbase12_work/submission_strict_final_20260806/audit"))
    parser.add_argument("--strict", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
