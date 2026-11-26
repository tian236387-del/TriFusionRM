#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${HOME}/summer/m1A_m6A}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
MMSEQS_BIN="${MMSEQS_BIN:-${HOME}/anaconda3/bin/mmseqs}"
THREADS="${THREADS:-32}"

cd "${ROOT}"

POS="rmbase12_work/submission_strict/multirm11_hg38_conflict_clean_strict_splits.tsv"
KNOWN="rmbase12_work/submission_strict/multirm12_hg38_coordinate_multilabel_41_101_201nt.tsv"
REF="rmbase12_work/reference/gencode_v49"
GENOME="${REF}/GRCh38.primary_assembly.genome.fa.gz"
GTF="${REF}/gencode.v49.annotation.gtf.gz"
OUT="rmbase12_work/detection_strict_20260814"
LOG="${OUT}/logs"
STATE="${OUT}/state"
mkdir -p "${OUT}" "${LOG}" "${STATE}"

while [[ ! -s "${REF}/download.ready" ]]; do
  printf '%s waiting_for_gencode_v49\n' "$(date -Iseconds)"
  sleep 60
done

gzip -t "${GENOME}"
gzip -t "${GTF}"
(cd "${REF}" && sha256sum -c SHA256SUMS)

NEG="${OUT}/hg38_matched_unannotated_candidates.tsv"
FAIL="${OUT}/unmatched_positive_candidates.tsv"
if [[ ! -s "${STATE}/negatives.ready" ]]; then
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -B scripts/build_hg38_matched_unannotated_negatives.py \
    --positives "${POS}" \
    --known-modifications "${KNOWN}" \
    --genome "${GENOME}" \
    --gtf "${GTF}" \
    --output "${NEG}" \
    --failures "${FAIL}" \
    --summary "${OUT}/negative_generation_summary.json" \
    --exclusion-radius 25 \
    --local-radius 20000 \
    --seed 42 2>&1 | tee "${LOG}/01_generate_negatives.log"
  date -Iseconds > "${STATE}/negatives.ready"
fi

if [[ ! -s "${STATE}/negative_audit.ready" ]]; then
  "${PYTHON_BIN}" -B scripts/audit_hg38_matched_unannotated_negatives.py \
    --positives "${POS}" \
    --known-modifications "${KNOWN}" \
    --negatives "${NEG}" \
    --output "${OUT}/negative_audit" \
    --exclusion-radius 25 \
    --minimum-match-rate 0.90 \
    --maximum-gc-smd 0.15 \
    --strict 2>&1 | tee "${LOG}/02_audit_negatives.log"
  date -Iseconds > "${STATE}/negative_audit.ready"
fi

if [[ ! -s "${STATE}/prepare.ready" ]]; then
  "${PYTHON_BIN}" -B scripts/build_hg38_detection_strict_splits.py prepare \
    --positives "${POS}" \
    --negatives "${NEG}" \
    --output-dir "${OUT}" 2>&1 | tee "${LOG}/03_prepare_detection.log"
  date -Iseconds > "${STATE}/prepare.ready"
fi

MM_PREFIX="${OUT}/detection201_c080"
MM_CLUSTER="${MM_PREFIX}_cluster.tsv"
if [[ ! -s "${STATE}/mmseqs.ready" ]]; then
  "${MMSEQS_BIN}" easy-linclust \
    "${OUT}/detection_unique_201nt.fa" \
    "${MM_PREFIX}" \
    "${OUT}/mmseqs_tmp" \
    --min-seq-id 0.80 \
    -c 0.80 \
    --cov-mode 0 \
    --cluster-mode 0 \
    --threads "${THREADS}" 2>&1 | tee "${LOG}/04_mmseqs_detection.log"
  test -s "${MM_CLUSTER}"
  date -Iseconds > "${STATE}/mmseqs.ready"
fi

if [[ ! -s "${STATE}/finalize.ready" ]]; then
  "${PYTHON_BIN}" -B scripts/build_hg38_detection_strict_splits.py finalize \
    --positives "${POS}" \
    --output-dir "${OUT}" \
    --mmseqs-cluster-tsv "${MM_CLUSTER}" \
    --seed 42 2>&1 | tee "${LOG}/05_finalize_detection.log"
  "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
path = Path("rmbase12_work/detection_strict_20260814/detection_split_audit.json")
audit = json.loads(path.read_text())
if audit.get("overall_status") != "PASS":
    raise SystemExit(f"Detection split audit failed: {audit}")
PY
  date -Iseconds > "${STATE}/finalize.ready"
fi

sha256sum \
  "${POS}" "${KNOWN}" "${NEG}" \
  "${OUT}/hg38_modified_vs_unannotated_strict_splits.tsv" \
  "${OUT}/multirm11_detection_aligned_positive_strict_splits.tsv" \
  > "${OUT}/DATA_SHA256SUMS"
date -Iseconds > "${STATE}/data_pipeline.complete"
printf '%s detection_data_pipeline_complete\n' "$(date -Iseconds)"
