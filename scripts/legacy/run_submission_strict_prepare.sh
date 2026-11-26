#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-${HOME}/summer/m1A_m6A}"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
WORK="${WORK:-rmbase12_work/submission_strict}"
STATE="${STATE:-${WORK}/state}"
LOG_DIR="${LOG_DIR:-${WORK}/logs}"
SOURCE_DATASET="${SOURCE_DATASET:-rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv}"
CLUSTER_PID="${CLUSTER_PID:-2360890}"
MMSEQS_CLUSTER_TSV="${MMSEQS_CLUSTER_TSV:-${WORK}/mmseqs201_c080_cluster.tsv}"
RNAFOLD="${RNAFOLD:-rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv}"
MANIFEST="${MANIFEST:-rmbase11_full_work/rbp_encode_eclip_all/encode_eclip_file_manifest.tsv}"
RBP_DIR="${RBP_DIR:-${WORK}/rbp_train_only}"
RBP_FEATURES="${RBP_FEATURES:-${RBP_DIR}/multirm11_top30_train_rbp_positional_features.tsv}"
RBP_CACHE="${RBP_CACHE:-${RBP_DIR}/multirm11_top30_train_rbp_positional.memmap}"

mkdir -p "${STATE}" "${LOG_DIR}" "${RBP_DIR}"
rm -f "${STATE}/prepare.failed" "${STATE}/rbp.failed"
trap 'status=$?; printf "%s prepare_failed exit=%s line=%s\n" "$(date -Iseconds)" "$status" "$LINENO" | tee "${STATE}/prepare.failed"; exit "$status"' ERR

while kill -0 "${CLUSTER_PID}" 2>/dev/null; do
  printf '%s waiting_for_cluster201 pid=%s\n' "$(date -Iseconds)" "${CLUSTER_PID}" >> "${LOG_DIR}/submission_prepare.log"
  sleep 120
done

if [[ ! -s "${MMSEQS_CLUSTER_TSV}" ]]; then
  printf '%s missing_or_empty=%s\n' "$(date -Iseconds)" "${MMSEQS_CLUSTER_TSV}" | tee "${STATE}/prepare.failed"
  exit 2
fi

if [[ ! -s "${STATE}/splits.ready" ]]; then
  if [[ ! -s "${WORK}/multirm11_hg38_conflict_clean_strict_splits.tsv" || \
        ! -s "${WORK}/multirm12_hg38_coordinate_multilabel_strict_splits.tsv" || \
        ! -s "${WORK}/strict_split_audit.json" ]]; then
  "${PYTHON_BIN}" -B scripts/build_multirm12_submission_splits.py prepare \
    --dataset "${SOURCE_DATASET}" \
    --output-dir "${WORK}" \
    >> "${LOG_DIR}/submission_prepare.log" 2>&1
  "${PYTHON_BIN}" -B scripts/build_multirm12_submission_splits.py finalize \
    --output-dir "${WORK}" \
    --mmseqs-cluster-tsv "${MMSEQS_CLUSTER_TSV}" \
    --seed 42 \
    >> "${LOG_DIR}/submission_prepare.log" 2>&1
  fi

  "${PYTHON_BIN}" - "${WORK}" >> "${LOG_DIR}/submission_prepare.log" 2>&1 <<'PY'
import csv
import json
import sys
from pathlib import Path

work = Path(sys.argv[1])
audit = json.loads((work / "strict_split_audit.json").read_text(encoding="utf-8"))
for task_name, section in (("confirmatory", audit), ("multilabel", audit["multilabel"])):
    for split_name, values in section["leakage_audits"].items():
        if int(values["leaked_groups"]) != 0:
            raise SystemExit(f"{task_name} {split_name} leaks {values['leaked_groups']} groups")
support_audit = {}
for filename in ("strict_split_class_support.tsv", "multilabel_strict_split_class_support.tsv"):
    with (work / filename).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    missing = [
        (row["split_column"], row["mod_type"], split)
        for row in rows
        for split in ("train", "valid", "test")
        if row["split_column"] != "split_study_holdout" and int(row[split]) == 0
    ]
    if missing:
        raise SystemExit(f"{filename} has zero-support development partitions: {missing}")
    study_rows = [row for row in rows if row["split_column"] == "split_study_holdout"]
    evaluable = [row["mod_type"] for row in study_rows if int(row["valid"]) > 0 and int(row["test"]) > 0]
    if len(evaluable) < 2:
        raise SystemExit(f"{filename} study lockbox has fewer than two evaluable classes: {evaluable}")
    support_audit[filename] = {
        "study_lockbox_evaluable_classes": evaluable,
        "study_lockbox_missing_classes": [
            row["mod_type"] for row in study_rows if row["mod_type"] not in evaluable
        ],
    }
confirmatory_lockbox = support_audit["strict_split_class_support.tsv"]
audit["lockbox_evaluable_classes"] = confirmatory_lockbox["study_lockbox_evaluable_classes"]
audit["lockbox_missing_classes"] = confirmatory_lockbox["study_lockbox_missing_classes"]
audit["lockbox_limitation"] = (
    "The publication graph contains a giant connected component. The zero-leakage study lockbox therefore "
    "supports only classes represented in independent study components; unsupported classes are not scored."
)
(work / "strict_split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps({
    "confirmatory_rows": audit["rows"],
    "confirmatory_lockbox_rows": audit["lockbox_rows"],
    "multilabel_rows": audit["multilabel"]["rows"],
    "multilabel_lockbox_rows": audit["multilabel"]["lockbox_rows"],
    "cluster_algorithm": audit["clustering"]["algorithm"],
    "cluster201_clusters": audit["clustering"]["clusters"],
    "support_audit": support_audit,
}, indent=2))
PY
  date -Iseconds > "${STATE}/splits.ready"
fi

DATASET="${WORK}/multirm11_hg38_conflict_clean_strict_splits.tsv"
for split_column in split_cluster201 split_gene_holdout split_transcript_holdout split_study_holdout; do
  label="${split_column#split_}"
  out_manifest="${RBP_DIR}/${label}_top30_train_manifest.tsv"
  out_coverage="${RBP_DIR}/${label}_target_train_overlap.tsv"
  if [[ ! -s "${out_manifest}" ]]; then
    "${PYTHON_BIN}" -B scripts/select_rbp_manifest_by_site_overlap.py \
      --dataset "${DATASET}" \
      --manifest "${MANIFEST}" \
      --out-manifest "${out_manifest}" \
      --out-coverage "${out_coverage}" \
      --top-n 30 \
      --window 200 \
      --split-column "${split_column}" \
      --split-value train \
      >> "${LOG_DIR}/rbp_train_only.log" 2>&1
  fi
done

PRIMARY_MANIFEST="${RBP_DIR}/cluster201_top30_train_manifest.tsv"
REUSE_FEATURES="${REUSE_FEATURES:-rmbase12_work/rbp_encode_eclip_train_only_20260803/rmbase_multirm12_train_selected_top30_rbp_positional_features.tsv}"
REUSE_MANIFEST="${REUSE_MANIFEST:-rmbase12_work/rbp_encode_eclip_train_only_20260803/encode_eclip_top30_train_manifest.tsv}"

same_targets="$(${PYTHON_BIN} - "${PRIMARY_MANIFEST}" "${REUSE_MANIFEST}" <<'PY'
import csv
import sys

def targets(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return sorted({row["target"] for row in csv.DictReader(handle, delimiter="\t")})

print("yes" if targets(sys.argv[1]) == targets(sys.argv[2]) else "no")
PY
)"

if [[ "${same_targets}" == "yes" && -s "${REUSE_FEATURES}" ]]; then
  ln -sfn "$(realpath "${REUSE_FEATURES}")" "${RBP_FEATURES}"
  cp "${REUSE_FEATURES}.json" "${RBP_FEATURES}.json"
  "${PYTHON_BIN}" - "${RBP_FEATURES}.json" "${DATASET}" "${PRIMARY_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload.update({
    "strict_dataset": sys.argv[2],
    "training_only_manifest": sys.argv[3],
    "feature_reuse_reason": "The training-only Top-30 target set is identical; genomic peak-derived features are split-independent and reused without recomputation.",
})
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
  printf '%s reused=%s\n' "$(date -Iseconds)" "${REUSE_FEATURES}" >> "${LOG_DIR}/rbp_train_only.log"
else
  "${PYTHON_BIN}" -B scripts/build_rbp_positional_features.py \
    --dataset "${DATASET}" \
    --manifest "${PRIMARY_MANIFEST}" \
    --out "${RBP_FEATURES}" \
    >> "${LOG_DIR}/rbp_train_only.log" 2>&1
fi

[[ -s "${RBP_FEATURES}" && -s "${RBP_FEATURES}.json" ]]
"${PYTHON_BIN}" -B scripts/convert_rbp_positional_tsv_to_memmap.py \
  --source "${RBP_FEATURES}" \
  --output "${RBP_CACHE}" \
  >> "${LOG_DIR}/rbp_train_only.log" 2>&1
[[ -s "${RBP_CACHE}/complete" && -s "${RBP_CACHE}/metadata.json" ]]

for label in gene_holdout transcript_holdout study_holdout; do
  holdout_manifest="${RBP_DIR}/${label}_top30_train_manifest.tsv"
  holdout_features="${RBP_DIR}/multirm11_${label}_top30_train_rbp_positional_features.tsv"
  holdout_cache="${RBP_DIR}/multirm11_${label}_top30_train_rbp_positional.memmap"
  same_as_primary="$(${PYTHON_BIN} - "${PRIMARY_MANIFEST}" "${holdout_manifest}" <<'PY'
import csv
import sys

def targets(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return sorted({row["target"] for row in csv.DictReader(handle, delimiter="\t")})

print("yes" if targets(sys.argv[1]) == targets(sys.argv[2]) else "no")
PY
)"
  if [[ "${same_as_primary}" == "yes" ]]; then
    ln -sfn "$(realpath "${RBP_FEATURES}")" "${holdout_features}"
    cp "${RBP_FEATURES}.json" "${holdout_features}.json"
    ln -sfn "$(realpath "${RBP_CACHE}")" "${holdout_cache}"
  elif [[ ! -s "${holdout_features}" ]]; then
    "${PYTHON_BIN}" -B scripts/build_rbp_positional_features.py \
      --dataset "${DATASET}" \
      --manifest "${holdout_manifest}" \
      --out "${holdout_features}" \
      >> "${LOG_DIR}/rbp_train_only.log" 2>&1
  fi
  if [[ "${same_as_primary}" != "yes" ]]; then
    "${PYTHON_BIN}" -B scripts/convert_rbp_positional_tsv_to_memmap.py \
      --source "${holdout_features}" \
      --output "${holdout_cache}" \
      >> "${LOG_DIR}/rbp_train_only.log" 2>&1
  fi
done

printf '%s\n' "${RBP_CACHE}" > "${STATE}/rbp_primary_path.txt"
date -Iseconds > "${STATE}/rbp.ready"
printf '%s preparation_complete\n' "$(date -Iseconds)" >> "${LOG_DIR}/submission_prepare.log"
