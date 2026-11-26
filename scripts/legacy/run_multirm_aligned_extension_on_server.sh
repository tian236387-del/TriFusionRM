#!/usr/bin/env bash
set -euo pipefail

# Supplementary MultiRM-aligned RMBase experiment pipeline.
#
# Run from the project root on the GPU server:
#
#   bash scripts/run_multirm_aligned_extension_on_server.sh multirm12
#
# Optional second argument controls CUDA device, e.g.:
#
#   bash scripts/run_multirm_aligned_extension_on_server.sh extended13 0

PRESET="${1:-multirm12}"
CUDA_ID="${2:-0}"

if [[ "${PRESET}" == "multirm12" ]]; then
  CLASS_NAMES="Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m6Am,m7G,Y,AtoI"
elif [[ "${PRESET}" == "extended13" ]]; then
  CLASS_NAMES="Am,Cm,Gm,Um,D,Y,m1A,m5C,m5U,m6A,m6Am,m7G,AtoI"
else
  echo "Unknown preset: ${PRESET}. Use multirm12 or extended13." >&2
  exit 2
fi

DATASET="rmbase12_work/datasets/rmbase_${PRESET}_hg38_41nt_all_sites.tsv"
CDHIT_DATASET="rmbase12_work/datasets/rmbase_${PRESET}_hg38_41nt_all_sites_cdhit80_strict_splits.tsv"
CDHIT_SUMMARY="rmbase12_work/datasets/rmbase_${PRESET}_cdhit80_strict_splits_summary.json"
FASTA="rmbase12_work/cluster/cdhit80/rmbase_${PRESET}_hg38_41nt_all_sites.fa"
CDHIT_PREFIX="rmbase12_work/cluster/cdhit80/rmbase_${PRESET}_cdhit80"
RNAFOLD_OUT="rmbase12_work/structure/rnafold_41nt_${PRESET}_all_sites.tsv"
BASELINE_ROOT="rmbase12_work/official_reproduction_benchmark/${PRESET}_cdhit80_strict_split"
PROPOSED_ROOT="rmbase12_work/models/motif_struct_fusion_v4_${PRESET}_cdhit80"
METRICS_ROOT="rmbase12_work/metrics/${PRESET}_cdhit80_baseline_vs_structdualrm"

echo "[1/6] Build ${PRESET} dataset, downloading only missing RMBase source files"
python -B scripts/build_rmbase_multirm_aligned_dataset.py \
  --preset "${PRESET}" \
  --source-dirs rmbase11_work/extracted rmbase12_work/extracted \
  --out-dir rmbase12_work \
  --download-missing \
  --insecure-download

echo "[2/6] Create CD-HIT80 strict split"
python -B scripts/create_rmbase11_cdhit_cluster_split.py \
  --dataset "${DATASET}" \
  --out "${CDHIT_DATASET}" \
  --summary "${CDHIT_SUMMARY}" \
  --fasta "${FASTA}" \
  --cdhit-output-prefix "${CDHIT_PREFIX}" \
  --cdhit-est "$(command -v cd-hit-est)" \
  --identity 0.8 \
  --word-size 5 \
  --coverage-shorter 0.8 \
  --coverage-longer 0.8 \
  --threads 32 \
  --memory-mb 0 \
  --deduplicate-identical-sequences \
  --class-names "${CLASS_NAMES}"

echo "[3/6] Run RNAfold incrementally, reusing existing RMBase11 folds"
python -B scripts/run_rmbase11_rnafold.py \
  --dataset "${CDHIT_DATASET}" \
  --existing-rnafold rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv \
  --out "${RNAFOLD_OUT}" \
  --rnafold-bin RNAfold \
  --batch-size 5000 \
  --resume

echo "[4/6] Train three architecture-adapted baselines"
for model in multirm_official transrnam_official evormd_adapted; do
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" python -B scripts/train_rmbase11_official_reproduction_benchmark.py \
    --model "${model}" \
    --dataset "${CDHIT_DATASET}" \
    --split split_cdhit80_cluster \
    --class-names "${CLASS_NAMES}" \
    --out-dir "${BASELINE_ROOT}/${model}" \
    --epochs 50 \
    --batch-size 2048 \
    --resume
done

echo "[5/6] Train StructDualRM no-position main variant on the same extended split"
CUDA_VISIBLE_DEVICES="${CUDA_ID}" python -B scripts/train_motif_struct_fusion_v4.py \
  --dataset "${CDHIT_DATASET}" \
  --rnafold "${RNAFOLD_OUT}" \
  --split split_cdhit80_cluster \
  --class-names "${CLASS_NAMES}" \
  --out-dir "${PROPOSED_ROOT}/no_position" \
  --disable-position \
  --epochs 50 \
  --batch-size 1024 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.0001 \
  --resume

echo "[6/6] Summarize metrics"
python -B scripts/summarize_cdhit80_baseline_vs_structdualrm.py \
  --baseline-root "${BASELINE_ROOT}" \
  --proposed-dir "${PROPOSED_ROOT}/no_position" \
  --out-dir "${METRICS_ROOT}" \
  --class-names "${CLASS_NAMES}"

echo "Done. Summary:"
echo "  ${METRICS_ROOT}/cdhit80_baseline_vs_structdualrm_macro.tsv"
echo "  ${METRICS_ROOT}/cdhit80_baseline_vs_structdualrm_per_class.tsv"
