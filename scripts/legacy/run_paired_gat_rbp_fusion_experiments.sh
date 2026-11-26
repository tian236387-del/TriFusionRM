#!/usr/bin/env bash
set -euo pipefail

# Run the proposed position-aware sequence/structure/RBP fusion model and
# ablations on the strict RMBase11 MMseqs80 split.  Intended for the Linux GPU
# server from the project root, e.g.:
#
#   cd ~/summer/m1A_m6A
#   conda activate RNAfold
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_paired_gat_rbp_fusion_experiments.sh

DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_mmseqs80_strict_splits.tsv}"
RNAFOLD="${RNAFOLD:-rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv}"
RBP="${RBP:-rmbase11_full_work/rbp/rmbase11_rbp_features.tsv}"
OUT_ROOT="${OUT_ROOT:-rmbase11_full_work/models/paired_gat_rbp_fusion_strict_experiments}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
D_MODEL="${D_MODEL:-128}"
SEQ_LAYERS="${SEQ_LAYERS:-2}"
SEQ_HEADS="${SEQ_HEADS:-4}"
GAT_LAYERS="${GAT_LAYERS:-2}"
LOSS="${LOSS:-ce}"

mkdir -p "${OUT_ROOT}"

run_model() {
  local name="$1"
  shift
  echo "===== Running ${name} ====="
  python -B scripts/train_paired_gat_rbp_fusion.py \
    --dataset "${DATASET}" \
    --rnafold "${RNAFOLD}" \
    --rbp-features "${RBP}" \
    --split split_mmseqs80_cluster \
    --out-dir "${OUT_ROOT}/${name}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --d-model "${D_MODEL}" \
    --seq-layers "${SEQ_LAYERS}" \
    --seq-heads "${SEQ_HEADS}" \
    --gat-layers "${GAT_LAYERS}" \
    --loss "${LOSS}" \
    --selection-metric macro_AUPR_ovr \
    --resume \
    "$@"
}

run_model full
run_model sequence_only --disable-structure --disable-rbp
run_model sequence_structure --disable-rbp
run_model sequence_rbp --disable-structure
run_model no_position --disable-position
run_model no_gate --disable-gate
run_model shuffled_structure --shuffle-structure
run_model shuffled_rbp --shuffle-rbp

python -B scripts/summarize_paired_gat_rbp_fusion_experiments.py \
  --proposed-root "${OUT_ROOT}" \
  --baseline-root rmbase11_full_work/official_reproduction_benchmark/strict_mmseqs80_split \
  --out-dir rmbase11_full_work/metrics/paired_gat_rbp_fusion_strict

echo "Done. Summary: rmbase11_full_work/metrics/paired_gat_rbp_fusion_strict"
