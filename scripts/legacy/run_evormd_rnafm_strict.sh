#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
CONTEXT_MODE="${CONTEXT_MODE:-full}"
SHUFFLE_CONTEXT="${SHUFFLE_CONTEXT:-0}"
if [[ "${CONTEXT_MODE}" == "sequence_only" ]]; then
  RUN_LABEL="sequence_only"
elif [[ "${SHUFFLE_CONTEXT}" == "1" ]]; then
  RUN_LABEL="${CONTEXT_MODE}_metadata_shuffled"
elif [[ "${CONTEXT_MODE}" == "safe" ]]; then
  RUN_LABEL="safe_context"
else
  RUN_LABEL="full_metadata_enabled"
fi
OUT_DIR="${OUT_DIR:-rmbase12_work/official_reproduction_benchmark/evormd_rnafm_official_core_${RUN_LABEL}_cdhit80_seed42}"
PYTHON_BIN="${PYTHON_BIN:-.venvs/evormd_rnafm/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WEIGHT="${WEIGHT:-${HOME}/.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth}"
EXPECTED_SHA256="5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing EvoRMD environment: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${DATASET}" ]]; then
  echo "Missing strict-split dataset: ${DATASET}" >&2
  exit 2
fi
if [[ ! -f "${WEIGHT}" ]]; then
  echo "Missing RNA-FM weight: ${WEIGHT}" >&2
  exit 2
fi
ACTUAL_SHA256="$(sha256sum "${WEIGHT}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  echo "RNA-FM weight checksum mismatch: ${ACTUAL_SHA256}" >&2
  exit 2
fi
if [[ -e "${OUT_DIR}/checkpoint.pt" || -e "${OUT_DIR}/model.pt" ]]; then
  echo "Refusing to overwrite an existing run: ${OUT_DIR}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
SHUFFLE_ARGS=()
if [[ "${SHUFFLE_CONTEXT}" == "1" ]]; then
  SHUFFLE_ARGS=(--shuffle-context)
fi
CUDA_VISIBLE_DEVICES="${CUDA_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -B \
  scripts/train_rmbase11_official_reproduction_benchmark.py \
  --model evormd_rnafm_official_core \
  --dataset "${DATASET}" \
  --class-names "Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m6Am,m7G,Y,AtoI" \
  --split "${SPLIT}" \
  --out-dir "${OUT_DIR}" \
  --input-length 41 \
  --context-mode "${CONTEXT_MODE}" \
  "${SHUFFLE_ARGS[@]}" \
  --meta-dim 160 \
  --mlp-depth 1 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr 1e-3 \
  --lr-decay 1.0 \
  --weight-decay 0 \
  --class-weight-power 0 \
  --downsample-train-classes "m6A,m5C" \
  --downsample-alpha 0.6 \
  --selection-metric macro_AUPR_ovr \
  --seed "${SEED}"

echo "Done: ${OUT_DIR}"
