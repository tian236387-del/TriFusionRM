#!/usr/bin/env bash
set -euo pipefail

# Quantitative RNAfold ablation for the MultiRM-aligned 12-class dataset.
# All variants share one CD-HIT80 strict split and differ only in structure use.
# Safe to rerun: RNAfold and model training use resume/checkpoint behavior.

if [[ -z "${PYTHON_BIN:-}" && -x "${HOME}/anaconda3/envs/rmbaseline/bin/python" ]]; then
  PYTHON_BIN="${HOME}/anaconda3/envs/rmbaseline/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
if [[ -z "${RNAFOLD_BIN:-}" && -x "${HOME}/anaconda3/envs/RNAfold/bin/RNAfold" ]]; then
  RNAFOLD_BIN="${HOME}/anaconda3/envs/RNAfold/bin/RNAfold"
else
  RNAFOLD_BIN="${RNAFOLD_BIN:-RNAfold}"
fi
CUDA_ID="${CUDA_ID:-0}"
CLASS_NAMES="Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m6Am,m7G,Y,AtoI"
RAW_DATASET="${RAW_DATASET:-rmbase12_work/datasets/rmbase_multirm12_hg38_41nt_all_sites.tsv}"
DATASET="${DATASET:-rmbase12_work/datasets/rmbase_multirm12_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
RNAFOLD="${RNAFOLD:-rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv}"
EXISTING_RNAFOLD="${EXISTING_RNAFOLD:-rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv}"
CDHIT_FASTA="${CDHIT_FASTA:-rmbase12_work/cluster/cdhit80/rmbase_multirm12_41nt_unique.fa}"
CDHIT_PREFIX="${CDHIT_PREFIX:-rmbase12_work/cluster/cdhit80/rmbase_multirm12_cdhit80}"
CDHIT_SUMMARY="${CDHIT_SUMMARY:-rmbase12_work/datasets/rmbase_multirm12_cdhit80_strict_splits_summary.json}"
CDHIT_THREADS="${CDHIT_THREADS:-32}"
OUT_ROOT="${OUT_ROOT:-rmbase12_work/models/multirm12_rnafold_ablation_cdhit80}"
METRICS_OUT="${METRICS_OUT:-rmbase12_work/metrics/multirm12_rnafold_ablation_cdhit80}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
D_MODEL="${D_MODEL:-128}"
DROPOUT="${DROPOUT:-0.4}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-0.5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
SEED="${SEED:-42}"

for required in "${RAW_DATASET}" "${EXISTING_RNAFOLD}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_ROOT}" "${METRICS_OUT}"

if [[ ! -s "${DATASET}" ]]; then
  echo "[1/5] Build one CD-HIT-est 80% cluster split shared by all variants"
  "${PYTHON_BIN}" -B scripts/create_rmbase11_cdhit_cluster_split.py \
    --dataset "${RAW_DATASET}" \
    --out "${DATASET}" \
    --summary "${CDHIT_SUMMARY}" \
    --fasta "${CDHIT_FASTA}" \
    --cdhit-output-prefix "${CDHIT_PREFIX}" \
    --class-names "${CLASS_NAMES}" \
    --identity 0.8 \
    --coverage-shorter 0.8 \
    --coverage-longer 0.8 \
    --threads "${CDHIT_THREADS}" \
    --deduplicate-identical-sequences
else
  echo "[1/5] Reuse existing strict split: ${DATASET}"
fi

echo "[2/5] Incremental RNAfold: reuse 10 common classes and fold only m6Am/AtoI"
"${PYTHON_BIN}" -B scripts/run_rmbase11_rnafold.py \
  --dataset "${DATASET}" \
  --existing-rnafold "${EXISTING_RNAFOLD}" \
  --out "${RNAFOLD}" \
  --rnafold-bin "${RNAFOLD_BIN}" \
  --batch-size 5000 \
  --resume

COMMON_ARGS=(
  --dataset "${DATASET}"
  --rnafold "${RNAFOLD}"
  --split "${SPLIT}"
  --class-names "${CLASS_NAMES}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --d-model "${D_MODEL}"
  --dropout "${DROPOUT}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --class-weight-power "${CLASS_WEIGHT_POWER}"
  --gat-layers 2
  --seq-encoder motif
  --disable-position
  --selection-metric macro_AUPR_ovr
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --seed "${SEED}"
  --resume
)

run_variant() {
  if [[ "$#" -lt 1 ]]; then
    echo "run_variant requires a variant name" >&2
    return 2
  fi
  local variant_name
  local variant_out
  variant_name="$1"
  variant_out="${OUT_ROOT}/${variant_name}"
  shift
  if [[ -s "${variant_out}/metrics.json" ]]; then
    echo "Reuse completed ${variant_name}: ${variant_out}/metrics.json"
    return 0
  fi
  echo "Training ${variant_name}"
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    "${COMMON_ARGS[@]}" \
    --out-dir "${variant_out}" \
    "$@"
}

echo "[3/5] Train sequence-only control"
run_variant sequence_only \
  --disable-structure \
  --aux-struct-weight 0 \
  --structure-scale 0 \
  --gate-l1-weight 0

echo "[4/5] Train real and shuffled RNAfold variants"
run_variant rnafold_mfe
run_variant rnafold_mfe_shuffled --shuffle-structure

echo "[5/5] Summarize quantitative structure gain"
"${PYTHON_BIN}" -B scripts/summarize_multirm12_rnafold_ablation.py \
  --root "${OUT_ROOT}" \
  --out-dir "${METRICS_OUT}" \
  --class-names "${CLASS_NAMES}"

echo "Done: ${METRICS_OUT}/multirm12_rnafold_ablation_macro.tsv"
echo "Done: ${METRICS_OUT}/multirm12_rnafold_ablation_per_class.tsv"
echo "Done: ${METRICS_OUT}/multirm12_rnafold_structure_gain.tsv"
