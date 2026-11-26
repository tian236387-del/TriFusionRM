#!/usr/bin/env bash
set -euo pipefail

TAG="${TAG:-20260726}"
CUDA_ID="${CUDA_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-500}"

factor_root="rmbase12_work/models/final_modality_factorial_ablation_seed42_${TAG}"
factor_summary="rmbase12_work/metrics/final_modality_factorial_ablation_seed42_${TAG}"
multiseed_root="rmbase12_work/models/final_main_multiseed_${TAG}"
multiseed_summary="rmbase12_work/metrics/final_main_multiseed_${TAG}"

CUDA_ID="${CUDA_ID}" \
PYTHON_BIN="${PYTHON_BIN}" \
OUT_ROOT="${factor_root}" \
SUMMARY_DIR="${factor_summary}" \
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS}" \
bash scripts/run_final_modality_factorial_ablation.sh

CUDA_ID="${CUDA_ID}" \
PYTHON_BIN="${PYTHON_BIN}" \
OUT_ROOT="${multiseed_root}" \
SUMMARY_DIR="${multiseed_summary}" \
bash scripts/run_final_main_multiseed.sh

echo "Factorial results: ${factor_summary}"
echo "Multi-seed results: ${multiseed_summary}"
