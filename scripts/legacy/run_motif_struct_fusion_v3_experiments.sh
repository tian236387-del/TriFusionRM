#!/usr/bin/env bash
set -euo pipefail

# V3: motif-centric sequence encoder + controlled RNAfold structure residual.

DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_mmseqs80_strict_splits.tsv}"
RNAFOLD="${RNAFOLD:-rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv}"
RBP="${RBP:-rmbase11_full_work/rbp/rmbase11_rbp_features.tsv}"
OUT_ROOT="${OUT_ROOT:-rmbase11_full_work/models/motif_struct_fusion_v3}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
D_MODEL="${D_MODEL:-128}"

mkdir -p "${OUT_ROOT}"

run_model() {
  local name="$1"
  shift
  echo "===== Running V3 ${name} ====="
  python -B scripts/train_motif_struct_fusion_v3.py \
    --dataset "${DATASET}" \
    --rnafold "${RNAFOLD}" \
    --rbp-features "${RBP}" \
    --split split_mmseqs80_cluster \
    --out-dir "${OUT_ROOT}/${name}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --d-model "${D_MODEL}" \
    --gat-layers 2 \
    --selection-metric macro_AUPR_ovr \
    --resume \
    "$@"
}

run_model full
run_model sequence_only --disable-structure --aux-struct-weight 0 --structure-scale 0
run_model no_position --disable-position
run_model shuffled_structure --shuffle-structure
run_model with_rbp --use-rbp

python -B scripts/summarize_paired_gat_rbp_fusion_experiments.py \
  --proposed-root "${OUT_ROOT}" \
  --baseline-root rmbase11_full_work/official_reproduction_benchmark/strict_mmseqs80_split \
  --out-dir rmbase11_full_work/metrics/motif_struct_fusion_v3_strict

echo "Done. Summary: rmbase11_full_work/metrics/motif_struct_fusion_v3_strict"
