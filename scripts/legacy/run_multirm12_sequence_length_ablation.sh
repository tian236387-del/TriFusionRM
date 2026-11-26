#!/usr/bin/env bash
set -euo pipefail

# Compare 41/101/201 nt sequence windows on exactly the same sites and split.

if [[ -z "${PYTHON_BIN:-}" && -x "${HOME}/anaconda3/envs/rmbaseline/bin/python" ]]; then
  PYTHON_BIN="${HOME}/anaconda3/envs/rmbaseline/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

CUDA_ID="${CUDA_ID:-0}"
CLASS_NAMES="Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m6Am,m7G,Y,AtoI"
SOURCE_DATASET="${SOURCE_DATASET:-rmbase12_work/datasets/rmbase_multirm12_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
DATASET="${DATASET:-rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv}"
DATASET_SUMMARY="${DATASET_SUMMARY:-rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.summary.json}"
HG38_FASTA="${HG38_FASTA:-${HOME}/summer/dataset/reference/hg38/hg38.fa}"
RNAFOLD="${RNAFOLD:-rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
OUT_ROOT="${OUT_ROOT:-rmbase12_work/models/multirm12_sequence_length_ablation_cdhit80}"
METRICS_OUT="${METRICS_OUT:-rmbase12_work/metrics/multirm12_sequence_length_ablation_cdhit80}"

EPOCHS="${EPOCHS:-50}"
D_MODEL="${D_MODEL:-128}"
DROPOUT="${DROPOUT:-0.4}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-0.5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
SEED="${SEED:-42}"

for required in "${SOURCE_DATASET}" "${HG38_FASTA}" "${RNAFOLD}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_ROOT}" "${METRICS_OUT}"

if [[ ! -s "${DATASET}" ]]; then
  echo "[1/5] Extract common 101/201 nt windows while preserving the 41 nt strict split"
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
  echo "[1/5] Reuse common long-window dataset: ${DATASET}"
fi

COMMON_ARGS=(
  --dataset "${DATASET}"
  --rnafold "${RNAFOLD}"
  --split "${SPLIT}"
  --class-names "${CLASS_NAMES}"
  --epochs "${EPOCHS}"
  --d-model "${D_MODEL}"
  --dropout "${DROPOUT}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --class-weight-power "${CLASS_WEIGHT_POWER}"
  --gat-layers 2
  --seq-encoder motif
  --disable-position
  --disable-structure
  --aux-struct-weight 0
  --structure-scale 0
  --gate-l1-weight 0
  --selection-metric macro_AUPR_ovr
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --seed "${SEED}"
  --resume
)

run_length() {
  if [[ "$#" -ne 5 ]]; then
    echo "run_length requires: variant length sequence_column batch_size grad_accum_steps" >&2
    return 2
  fi
  local variant_name
  local input_length
  local sequence_column
  local batch_size
  local grad_accum_steps
  local variant_out
  variant_name="$1"
  input_length="$2"
  sequence_column="$3"
  batch_size="$4"
  grad_accum_steps="$5"
  variant_out="${OUT_ROOT}/${variant_name}"
  if [[ -s "${variant_out}/metrics.json" ]]; then
    echo "Reuse completed ${variant_name}: ${variant_out}/metrics.json"
    return 0
  fi
  echo "Training ${variant_name}: length=${input_length}, batch=${batch_size}, accum=${grad_accum_steps}"
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    "${COMMON_ARGS[@]}" \
    --input-length "${input_length}" \
    --sequence-column "${sequence_column}" \
    --batch-size "${batch_size}" \
    --grad-accum-steps "${grad_accum_steps}" \
    --out-dir "${variant_out}"
}

echo "[2/5] Train 41 nt control"
run_length length_41 41 sequence_41nt 1024 1

echo "[3/5] Train 101 nt model"
run_length length_101 101 sequence_101nt 512 2

echo "[4/5] Train 201 nt model"
run_length length_201 201 sequence_201nt 256 4

echo "[5/5] Summarize per-class sequence-length effects"
"${PYTHON_BIN}" -B scripts/summarize_multirm12_sequence_length_ablation.py \
  --root "${OUT_ROOT}" \
  --out-dir "${METRICS_OUT}" \
  --class-names "${CLASS_NAMES}"

echo "Done: ${METRICS_OUT}/multirm12_sequence_length_macro.tsv"
echo "Done: ${METRICS_OUT}/multirm12_sequence_length_per_class.tsv"
