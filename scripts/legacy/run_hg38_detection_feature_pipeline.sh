#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${HOME}/summer/m1A_m6A}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
RNAFOLD_BIN="${RNAFOLD_BIN:-${HOME}/anaconda3/envs/RNAfold/bin/RNAfold}"
OUT="rmbase12_work/detection_strict_20260814"
DATA="${OUT}/hg38_modified_vs_unannotated_strict_splits.tsv"
SPLIT="split_detection_cluster201"
FEATURES="${OUT}/features"

cd "${ROOT}"
mkdir -p "${FEATURES}" "${OUT}/logs" "${OUT}/state"

while [[ ! -s "${OUT}/state/data_pipeline.complete" ]]; do
  printf '%s waiting_for_detection_data_features\n' "$(date -Iseconds)"
  sleep 60
done

RNAFOLD_OUT="${FEATURES}/rnafold_41nt_modified_vs_unannotated.tsv"
if [[ ! -s "${OUT}/state/rnafold.ready" ]]; then
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -B scripts/run_rmbase11_rnafold.py \
    --dataset "${DATA}" \
    --out "${RNAFOLD_OUT}" \
    --rnafold-bin "${RNAFOLD_BIN}" \
    --batch-size 20000 \
    --existing-rnafold rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv \
    2>&1 | tee "${OUT}/logs/10_rnafold_detection.log"
  "${PYTHON_BIN}" - "${DATA}" "${RNAFOLD_OUT}" <<'PY'
import csv
import sys
from pathlib import Path

def count(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle, delimiter="\t"))

dataset_rows = count(sys.argv[1])
fold_rows = count(sys.argv[2])
if dataset_rows != fold_rows:
    raise SystemExit(f"RNAfold coverage mismatch: dataset={dataset_rows} fold={fold_rows}")
print(f"rnafold_coverage=PASS rows={fold_rows}")
PY
  date -Iseconds > "${OUT}/state/rnafold.ready"
fi

FULL_MANIFEST="rmbase11_full_work/rbp_encode_eclip_all/encode_eclip_file_manifest.tsv"
TOP30="${FEATURES}/encode_eclip_detection_train_top30_manifest.tsv"
COVERAGE="${FEATURES}/encode_eclip_detection_train_target_overlap.tsv"
if [[ ! -s "${OUT}/state/rbp_manifest.ready" ]]; then
  "${PYTHON_BIN}" -B scripts/select_rbp_manifest_by_site_overlap.py \
    --dataset "${DATA}" \
    --manifest "${FULL_MANIFEST}" \
    --out-manifest "${TOP30}" \
    --out-coverage "${COVERAGE}" \
    --top-n 30 \
    --window 200 \
    --min-overlap-sites 1 \
    --split-column "${SPLIT}" \
    --split-value train 2>&1 | tee "${OUT}/logs/11_select_detection_train_rbp.log"
  test "$(awk -F '\t' 'NR > 1 { seen[$1] = 1 } END { print length(seen) }' "${TOP30}")" -eq 30
  date -Iseconds > "${OUT}/state/rbp_manifest.ready"
fi

RBP_TSV="${FEATURES}/modified_vs_unannotated_top30_rbp_positional.tsv"
if [[ ! -s "${OUT}/state/rbp_tsv.ready" ]]; then
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -B scripts/build_rbp_positional_features.py \
    --dataset "${DATA}" \
    --manifest "${TOP30}" \
    --out "${RBP_TSV}" 2>&1 | tee "${OUT}/logs/12_build_detection_rbp.log"
  date -Iseconds > "${OUT}/state/rbp_tsv.ready"
fi

RBP_MEMMAP="${FEATURES}/modified_vs_unannotated_top30_rbp_positional.memmap"
if [[ ! -s "${OUT}/state/rbp_memmap.ready" ]]; then
  "${PYTHON_BIN}" -B scripts/convert_rbp_positional_tsv_to_memmap.py \
    --source "${RBP_TSV}" \
    --output "${RBP_MEMMAP}" 2>&1 | tee "${OUT}/logs/13_convert_detection_rbp_memmap.log"
  test -s "${RBP_MEMMAP}/complete"
  date -Iseconds > "${OUT}/state/rbp_memmap.ready"
fi

date -Iseconds > "${OUT}/state/feature_pipeline.complete"
printf '%s detection_feature_pipeline_complete\n' "$(date -Iseconds)"
