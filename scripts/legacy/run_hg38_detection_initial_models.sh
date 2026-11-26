#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${HOME}/summer/m1A_m6A}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
OUT="rmbase12_work/detection_strict_20260814"
DATA="${OUT}/hg38_modified_vs_unannotated_strict_splits.tsv"
SPLIT="split_detection_cluster201"
SEED="${SEED:-42}"

cd "${ROOT}"
mkdir -p "${OUT}/logs" "${OUT}/state" "${OUT}/models"

while [[ ! -s "${OUT}/state/data_pipeline.complete" ]]; do
  printf '%s waiting_for_detection_data\n' "$(date -Iseconds)"
  sleep 60
done

"${PYTHON_BIN}" - <<'PY'
import csv
from collections import Counter
from pathlib import Path

path = Path("rmbase12_work/detection_strict_20260814/detection_split_support.tsv")
rows = list(csv.DictReader(path.open(), delimiter="\t"))
failures = []
for row in rows:
    if row["split_column"] != "split_detection_cluster201":
        continue
    for split in ("train", "valid", "test"):
        if int(row[split]) <= 0:
            failures.append((row["detection_class"], row["original_mod_type"], split))
if failures:
    raise SystemExit(f"Detection support audit failed: {failures}")
print(f"support_audit=PASS strata={len(rows)}")
PY

KMER_OUT="${OUT}/models/hash_kmer_seed${SEED}"
if [[ ! -s "${KMER_OUT}/audited/detection_metrics.json" ]]; then
  mkdir -p "${KMER_OUT}"
  "${PYTHON_BIN}" -B scripts/train_hg38_detection_hash_kmer.py \
    --dataset "${DATA}" \
    --output-dir "${KMER_OUT}" \
    --split-column "${SPLIT}" \
    --sequence-column sequence_201nt \
    --epochs 3 \
    --chunk-size 65536 \
    --seed "${SEED}" 2>&1 | tee "${OUT}/logs/06_hash_kmer_seed${SEED}.log"
  "${PYTHON_BIN}" -B scripts/evaluate_hg38_detection_predictions.py \
    --dataset "${DATA}" \
    --valid-predictions "${KMER_OUT}/valid_predictions.tsv" \
    --test-predictions "${KMER_OUT}/test_predictions.tsv" \
    --output-dir "${KMER_OUT}/audited" \
    --split-column "${SPLIT}" 2>&1 | tee "${OUT}/logs/07_hash_kmer_audit_seed${SEED}.log"
fi

SEQ_OUT="${OUT}/models/structdualrm_sequence_seed${SEED}"
if [[ ! -s "${SEQ_OUT}/audited/detection_metrics.json" ]]; then
  mkdir -p "${SEQ_OUT}"
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -B \
    scripts/train_motif_struct_fusion_v4.py \
    --dataset "${DATA}" \
    --class-names unmodified,modified \
    --rnafold "" \
    --rbp-features "" \
    --rbp-positional-features "" \
    --split "${SPLIT}" \
    --out-dir "${SEQ_OUT}" \
    --input-length 41 \
    --sequence-column sequence_41nt \
    --epochs 30 \
    --batch-size 256 \
    --grad-accum-steps 4 \
    --d-model 128 \
    --gat-layers 2 \
    --dropout 0.4 \
    --seq-encoder multi_length \
    --multi-length-columns sequence_41nt,sequence_101nt,sequence_201nt \
    --multi-length-lengths 41,101,201 \
    --multi-length-base-encoder motif \
    --afamily-classes "" \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --class-weight-power 0.0 \
    --loss ce \
    --aux-seq-weight 0.1 \
    --aux-struct-weight 0.0 \
    --length-aux-weight 0.05 \
    --gate-l1-weight 0.0 \
    --selection-metric macro_AUPR_ovr \
    --selection-focus-class modified \
    --selection-focus-metric AUPR_ovr \
    --selection-focus-weight 1.0 \
    --early-stopping-patience 5 \
    --early-stopping-min-delta 1e-4 \
    --disable-structure \
    --disable-position \
    --seed "${SEED}" 2>&1 | tee "${OUT}/logs/08_structdualrm_sequence_seed${SEED}.log"
  "${PYTHON_BIN}" -B scripts/evaluate_hg38_detection_predictions.py \
    --dataset "${DATA}" \
    --valid-predictions "${SEQ_OUT}/valid_predictions.tsv" \
    --test-predictions "${SEQ_OUT}/test_predictions.tsv" \
    --output-dir "${SEQ_OUT}/audited" \
    --split-column "${SPLIT}" 2>&1 | tee "${OUT}/logs/09_structdualrm_sequence_audit_seed${SEED}.log"
fi

date -Iseconds > "${OUT}/state/initial_models_seed${SEED}.complete"
printf '%s initial_detection_models_complete seed=%s\n' "$(date -Iseconds)" "${SEED}"
