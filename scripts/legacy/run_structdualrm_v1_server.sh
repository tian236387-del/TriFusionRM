#!/usr/bin/env bash
set -euo pipefail

# Reproducible server entry point for the isolated StructDualRM V1 dataset.
ROOT="${1:-/home/nefu1020220084/summer/StructDualRM}"
PROJECT="${2:-/home/nefu1020220084/summer/m1A_m6A}"
PYTHON="${PYTHON:-python3}"
MMSEQS="${MMSEQS:-${PROJECT}/tools/mmseqs2/mmseqs/bin/mmseqs}"
RNAFOLD="${RNAFOLD:-RNAfold}"
THREADS="${THREADS:-32}"
SEED="${SEED:-42}"

mkdir -p "${ROOT}/data" "${ROOT}/structure" "${ROOT}/cluster" "${ROOT}/logs" "${ROOT}/models" "${ROOT}/results"
exec > >(tee -a "${ROOT}/logs/build_v1.log") 2>&1

SOURCE="${PROJECT}/rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv"
GENOME="${PROJECT}/rmbase12_work/reference/gencode_v49/GRCh38.primary_assembly.genome.fa.gz"
GTF="${PROJECT}/rmbase12_work/reference/gencode_v49/gencode.v49.annotation.gtf.gz"
POS="${ROOT}/data/positive_clean.tsv"
KNOWN="${ROOT}/data/known_all_target_coordinates.tsv"
NEG="${ROOT}/data/matched_unannotated_negatives.tsv"
COMBINED="${ROOT}/data/dataset_13class_unclustered.tsv"
SPLIT="${ROOT}/data/dataset_13class_cluster201_splits.tsv"
CLUSTER_PREFIX="${ROOT}/cluster/mmseqs201"

echo "[1/7] Canonicalize the 12 positive classes"
"${PYTHON}" "${PROJECT}/scripts/prepare_structdualrm_v1.py" canonicalize \
  --source "${SOURCE}" --out-dir "${ROOT}/data"

echo "[2/7] Build one matched unannotated candidate per positive"
"${PYTHON}" "${PROJECT}/scripts/build_hg38_matched_unannotated_negatives.py" \
  --positives "${POS}" --known-modifications "${KNOWN}" \
  --genome "${GENOME}" --gtf "${GTF}" \
  --output "${NEG}" \
  --failures "${ROOT}/data/negative_failures.tsv" \
  --summary "${ROOT}/data/negative_summary.json" \
  --exclusion-radius 25 --local-radius 20000 --seed "${SEED}"

echo "[3/7] Combine positives and unmodified candidates into 13 classes"
"${PYTHON}" "${PROJECT}/scripts/prepare_structdualrm_v1.py" combine \
  --positives "${POS}" --negatives "${NEG}" --out-dir "${ROOT}/data"

echo "[4/7] RNAfold the 41-nt windows (resume capable)"
"${PYTHON}" "${PROJECT}/scripts/reuse_rnafold_by_sequence.py" \
  --dataset "${COMBINED}" \
  --out "${ROOT}/structure/rnafold_41nt_13class.tsv" \
  --rnafold-bin "${RNAFOLD}" --batch-size 2000 \
  --existing-rnafold "${PROJECT}/rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv"

echo "[5/7] Cluster all 201-nt sequences at 80% identity/80% coverage"
"${MMSEQS}" easy-linclust "${ROOT}/data/sequence_201nt_unique.fa" \
  "${CLUSTER_PREFIX}" "${ROOT}/cluster/tmp" \
  --dbtype 2 --min-seq-id 0.8 -c 0.8 --cov-mode 0 --cluster-mode 2 \
  --similarity-type 2 --threads "${THREADS}" --remove-tmp-files 1

echo "[6/7] Join sequence clusters and matched pairs before splitting"
"${PYTHON}" "${PROJECT}/scripts/prepare_structdualrm_v1.py" split \
  --dataset "${COMBINED}" \
  --site-to-record "${ROOT}/data/site_to_sequence_record.tsv" \
  --cluster-tsv "${CLUSTER_PREFIX}_cluster.tsv" \
  --out-dir "${ROOT}/data" --seed "${SEED}"

echo "[7/7] Run leakage and label audits"
"${PYTHON}" "${PROJECT}/scripts/prepare_structdualrm_v1.py" audit \
  --dataset "${SPLIT}" --out-dir "${ROOT}/results"

echo "BUILD COMPLETE: ${ROOT}"
