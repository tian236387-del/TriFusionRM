#!/usr/bin/env bash
set -euo pipefail

# Main-model experiment with class-specific 41/101/201 nt sequence-length fusion.
#
# Default variant:
#   multi_length_shared
#       one shared motif CNN+BiGRU backbone is applied to 41/101/201 nt windows;
#       length-specific heads and class-specific length gates fuse evidence.
#
# Example on server:
#   env -u LD_PRELOAD CUDA_ID=0 nohup bash scripts/run_multirm12_multi_length_fusion.sh > rmbase12_work/multirm12_multi_length_fusion.log 2>&1 &

if [[ -z "${PYTHON_BIN:-}" && -x "${HOME}/anaconda3/envs/rmbaseline/bin/python" ]]; then
  PYTHON_BIN="${HOME}/anaconda3/envs/rmbaseline/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

CUDA_ID="${CUDA_ID:-0}"
CLASS_NAMES="${CLASS_NAMES:-Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m6Am,m7G,Y,AtoI}"
SOURCE_DATASET="${SOURCE_DATASET:-rmbase12_work/datasets/rmbase_multirm12_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
DATASET="${DATASET:-rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv}"
DATASET_SUMMARY="${DATASET_SUMMARY:-rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.summary.json}"
HG38_FASTA="${HG38_FASTA:-${HOME}/summer/dataset/reference/hg38/hg38.fa}"
RNAFOLD="${RNAFOLD:-rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
OUT_ROOT="${OUT_ROOT:-rmbase12_work/models/multirm12_multi_length_fusion_cdhit80}"
METRICS_OUT="${METRICS_OUT:-rmbase12_work/metrics/multirm12_multi_length_fusion_cdhit80}"
SINGLE_LENGTH_REFERENCE="${SINGLE_LENGTH_REFERENCE:-rmbase12_work/metrics/multirm12_sequence_length_ablation_cdhit80/multirm12_sequence_length_per_class.tsv}"
BASELINE_REFERENCE="${BASELINE_REFERENCE:-rmbase12_work/metrics/multirm12_cdhit80_baseline_vs_structdualrm/cdhit80_baseline_vs_structdualrm_per_class.tsv}"

RBP_ROOT="${RBP_ROOT:-rmbase12_work/rbp_encode_eclip_all}"
RBP_MANIFEST="${RBP_MANIFEST:-rmbase11_full_work/rbp_encode_eclip_all/encode_eclip_top30_manifest.tsv}"
RBP_POSITIONAL="${RBP_POSITIONAL:-${RBP_ROOT}/rmbase_multirm12_encode_eclip_top30_rbp_positional_features.tsv}"
USE_RBP_POSITIONAL="${USE_RBP_POSITIONAL:-auto}"
REQUIRE_RBP_POSITIONAL="${REQUIRE_RBP_POSITIONAL:-0}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
D_MODEL="${D_MODEL:-128}"
DROPOUT="${DROPOUT:-0.4}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-0.5}"
LOSS="${LOSS:-ce}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
AUX_SEQ_WEIGHT="${AUX_SEQ_WEIGHT:-0.1}"
AUX_STRUCT_WEIGHT="${AUX_STRUCT_WEIGHT:-0.05}"
LENGTH_AUX_WEIGHT="${LENGTH_AUX_WEIGHT:-0.05}"
LENGTH_GATE_ENTROPY_WEIGHT="${LENGTH_GATE_ENTROPY_WEIGHT:-0.0}"
GATE_L1="${GATE_L1:-0.001}"
STRUCTURE_SCALE="${STRUCTURE_SCALE:-0.5}"
GATE_INIT_LOGIT="${GATE_INIT_LOGIT:--2.0}"
RBP_POS_SCALE="${RBP_POS_SCALE:-0.10}"
RBP_POS_GATE_L1="${RBP_POS_GATE_L1:-0.0}"
RBP_POS_GATE_INIT_LOGIT="${RBP_POS_GATE_INIT_LOGIT:--1.5}"
MODALITY_DROPOUT_STRUCT="${MODALITY_DROPOUT_STRUCT:-0.0}"
MODALITY_DROPOUT_RBP="${MODALITY_DROPOUT_RBP:-0.0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
SEED="${SEED:-42}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-0}"
RUN_INDEPENDENT="${RUN_INDEPENDENT:-0}"

for required in "${SOURCE_DATASET}" "${HG38_FASTA}" "${RNAFOLD}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_ROOT}" "${METRICS_OUT}" "${RBP_ROOT}"

if [[ ! -s "${DATASET}" ]]; then
  echo "[1/4] Extract common 101/201 nt windows while preserving the 41 nt strict split"
  "${PYTHON_BIN}" -B scripts/extract_rmbase_windows_from_fasta.py \
    --dataset "${SOURCE_DATASET}" \
    --fasta "${HG38_FASTA}" \
    --out "${DATASET}" \
    --summary-out "${DATASET_SUMMARY}" \
    --lengths 101,201 \
    --validate-existing-41 \
    --drop-mismatched-41 \
    --drop-invalid
else
  echo "[1/4] Reuse common long-window dataset: ${DATASET}"
fi

RBP_ARGS=()
if [[ "${USE_RBP_POSITIONAL}" == "1" || "${USE_RBP_POSITIONAL}" == "auto" ]]; then
  if [[ ! -s "${RBP_POSITIONAL}" && -s "${RBP_MANIFEST}" ]]; then
    echo "[2/4] Build RBP positional features for the common 12-class dataset"
    "${PYTHON_BIN}" -B scripts/build_rbp_positional_features.py \
      --dataset "${DATASET}" \
      --manifest "${RBP_MANIFEST}" \
      --out "${RBP_POSITIONAL}"
  fi
  if [[ -s "${RBP_POSITIONAL}" ]]; then
    RBP_ARGS=(--use-rbp-positional --rbp-positional-features "${RBP_POSITIONAL}")
    echo "[2/4] Use RBP positional features: ${RBP_POSITIONAL}"
  elif [[ "${REQUIRE_RBP_POSITIONAL}" == "1" || "${USE_RBP_POSITIONAL}" == "1" ]]; then
    echo "Missing required RBP positional feature file: ${RBP_POSITIONAL}" >&2
    exit 1
  else
    echo "[2/4] RBP positional feature file not found; continue without RBP branch"
  fi
else
  echo "[2/4] RBP positional branch disabled by USE_RBP_POSITIONAL=${USE_RBP_POSITIONAL}"
fi

MAX_ROW_ARGS=()
if [[ "${TRAIN_MAX_ROWS}" != "0" ]]; then
  MAX_ROW_ARGS=(--max-rows "${TRAIN_MAX_ROWS}")
fi

COMMON_ARGS=(
  --dataset "${DATASET}"
  --rnafold "${RNAFOLD}"
  --split "${SPLIT}"
  --class-names "${CLASS_NAMES}"
  --input-length 41
  --sequence-column sequence_41nt
  --multi-length-columns sequence_41nt,sequence_101nt,sequence_201nt
  --multi-length-lengths 41,101,201
  --multi-length-base-encoder motif
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --d-model "${D_MODEL}"
  --dropout "${DROPOUT}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --class-weight-power "${CLASS_WEIGHT_POWER}"
  --loss "${LOSS}"
  --focal-gamma "${FOCAL_GAMMA}"
  --gat-layers 2
  --selection-metric macro_AUPR_ovr
  --disable-position
  --seq-encoder multi_length
  --aux-seq-weight "${AUX_SEQ_WEIGHT}"
  --aux-struct-weight "${AUX_STRUCT_WEIGHT}"
  --length-aux-weight "${LENGTH_AUX_WEIGHT}"
  --length-gate-entropy-weight "${LENGTH_GATE_ENTROPY_WEIGHT}"
  --gate-l1-weight "${GATE_L1}"
  --structure-scale "${STRUCTURE_SCALE}"
  --gate-init-logit "${GATE_INIT_LOGIT}"
  --rbp-pos-scale "${RBP_POS_SCALE}"
  --rbp-pos-gate-l1-weight "${RBP_POS_GATE_L1}"
  --rbp-pos-gate-init-logit "${RBP_POS_GATE_INIT_LOGIT}"
  --modality-dropout-struct "${MODALITY_DROPOUT_STRUCT}"
  --modality-dropout-rbp "${MODALITY_DROPOUT_RBP}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}"
  --seed "${SEED}"
  --resume
  "${RBP_ARGS[@]}"
  "${MAX_ROW_ARGS[@]}"
)

run_variant() {
  local name="$1"
  shift
  local out_dir="${OUT_ROOT}/${name}"
  if [[ -s "${out_dir}/metrics.json" ]]; then
    echo "Reuse completed ${name}: ${out_dir}/metrics.json"
  else
    echo "[3/4] Train ${name}"
    CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
      "${COMMON_ARGS[@]}" \
      --out-dir "${out_dir}" \
      "$@"
  fi
  echo "[4/4] Summarize ${name}"
  "${PYTHON_BIN}" -B scripts/summarize_multirm12_multi_length_fusion.py \
    --model-dir "${out_dir}" \
    --out-dir "${METRICS_OUT}/${name}" \
    --model-name "${name}" \
    --class-names "${CLASS_NAMES}" \
    --single-length-reference "${SINGLE_LENGTH_REFERENCE}" \
    --baseline-reference "${BASELINE_REFERENCE}"
}

run_variant multi_length_shared --multi-length-share-encoder

if [[ "${RUN_INDEPENDENT}" == "1" ]]; then
  run_variant multi_length_independent
fi

echo "Done. Summary files under: ${METRICS_OUT}"
