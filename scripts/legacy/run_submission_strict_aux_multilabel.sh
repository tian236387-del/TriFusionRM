#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-${HOME}/summer/m1A_m6A}"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
WORK="${WORK:-rmbase12_work/submission_strict}"
RUN_ROOT="${RUN_ROOT:-rmbase12_work/submission_strict_runs_20260806}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
STATUS_LOG="${STATUS_LOG:-${RUN_ROOT}/job_status.tsv}"
DATASET="${DATASET:-${WORK}/multirm12_hg38_coordinate_multilabel_strict_splits.tsv}"
SEEDS=(42 17 73 101 202)

mkdir -p "${LOG_DIR}" "${WORK}/state"

for seed in "${SEEDS[@]}"; do
  job="multilabel_seed${seed}"
  out_dir="${RUN_ROOT}/multilabel/seed${seed}"
  mkdir -p "${out_dir}"
  if [[ -s "${out_dir}/metrics.json" && -s "${out_dir}/valid_predictions.tsv" && -s "${out_dir}/test_predictions.tsv" ]]; then
    continue
  fi

  # The auxiliary worker only handles tail jobs that the primary queue will not
  # reach until much later. The lock also prevents accidental duplicate starts.
  exec 9>"${out_dir}/auxiliary.lock"
  if ! flock -n 9; then
    continue
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${job}" auxiliary_started 0 "${out_dir}" >> "${STATUS_LOG}"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -B \
    scripts/train_structdualrm_multilabel_relational.py \
    --dataset "${DATASET}" \
    --rnafold rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv \
    --class-names Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m6Am,m7G,Y,AtoI \
    --target-field label_set \
    --fold-site-id-column source_site_id \
    --sequence-column sequence_41nt \
    --split split_cluster201 \
    --input-length 41 \
    --epochs 50 \
    --batch-size 1024 \
    --d-model 96 \
    --layers 2 \
    --heads 4 \
    --label-heads 4 \
    --dropout 0.2 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --pos-weight-cap 100 \
    --negative-weight 0.2 \
    --gamma-pos 0 \
    --gamma-neg 2 \
    --center-loss bce \
    --lambda-window 1 \
    --lambda-center 0 \
    --lambda-relation 0.1 \
    --lambda-cardinality 0.05 \
    --selection-metric macro_AUPR \
    --early-stopping-patience 8 \
    --early-stopping-min-delta 1e-4 \
    --seed "${seed}" \
    --out-dir "${out_dir}" \
    > "${LOG_DIR}/${job}.log" 2>&1
  code=$?
  set -e

  if [[ "${code}" -ne 0 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${job}" auxiliary_failed "${code}" "${out_dir}" >> "${STATUS_LOG}"
    printf '%s\n' "${job}" > "${WORK}/state/aux_multilabel.failed"
    exit "${code}"
  fi
  if [[ ! -s "${out_dir}/metrics.json" || ! -s "${out_dir}/valid_predictions.tsv" || ! -s "${out_dir}/test_predictions.tsv" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${job}" auxiliary_incomplete 3 "${out_dir}" >> "${STATUS_LOG}"
    printf '%s\n' "${job}" > "${WORK}/state/aux_multilabel.failed"
    exit 3
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${job}" auxiliary_completed 0 "${out_dir}" >> "${STATUS_LOG}"
done

date -Iseconds > "${WORK}/state/aux_multilabel.ready"
