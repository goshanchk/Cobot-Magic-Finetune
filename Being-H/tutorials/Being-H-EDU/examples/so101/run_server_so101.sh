#!/usr/bin/env bash
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
set -euo pipefail

# Being-H SO101 inference server.
# Override MODEL_PATH before launching:
#   MODEL_PATH=/path/to/checkpoint bash examples/so101/run_server_so101.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SERVER_PORT="${SERVER_PORT:-8080}"
MODEL_PATH="${MODEL_PATH:-/path/to/your/so101-checkpoint}"
DATA_CONFIG_NAME="${DATA_CONFIG_NAME:-so101}"
DATASET_NAME="${DATASET_NAME:-so101_posttrain}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-so101}"

STATS_SELECTION_MODE="${STATS_SELECTION_MODE:-auto}"
METADATA_VARIANT="${METADATA_VARIANT:-}"

NUM_INFERENCE_TIMESTEPS="${NUM_INFERENCE_TIMESTEPS:-4}"
USE_MPG="${USE_MPG:-True}"
MPG_LAMBDA="${MPG_LAMBDA:-0.1}"
MPG_NUM_PROJECTIONS="${MPG_NUM_PROJECTIONS:-32}"
MPG_REFINEMENT_ITERS="${MPG_REFINEMENT_ITERS:-1}"
ENABLE_RTC="${ENABLE_RTC:-False}"

echo "=========================================="
echo "Starting SO101 inference server"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Port: ${SERVER_PORT}"
echo "Data config: ${DATA_CONFIG_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Embodiment: ${EMBODIMENT_TAG}"
echo "RTC enabled: ${ENABLE_RTC}"
echo "MPG enabled: ${USE_MPG}"
echo "=========================================="

if [[ "${MODEL_PATH}" == /path/to/* || ! -e "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is not set to an existing checkpoint: ${MODEL_PATH}" >&2
  echo "Launch with: MODEL_PATH=/path/to/checkpoint bash examples/so101/run_server_so101.sh" >&2
  exit 2
fi

CMD=(
  python -m BeingH.inference.run_server_vla
  --model-path "${MODEL_PATH}"
  --port "${SERVER_PORT}"
  --data-config-name "${DATA_CONFIG_NAME}"
  --dataset-name "${DATASET_NAME}"
  --embodiment-tag "${EMBODIMENT_TAG}"
  --prompt-template long
  --max-view-num -1
  --no-use-fixed-view
  --stats-selection-mode "${STATS_SELECTION_MODE}"
)

if [[ -n "${METADATA_VARIANT}" ]]; then
  CMD+=(--metadata-variant "${METADATA_VARIANT}")
fi

if [[ -n "${NUM_INFERENCE_TIMESTEPS}" ]]; then
  CMD+=(--num-inference-timesteps "${NUM_INFERENCE_TIMESTEPS}")
fi

if [[ "${USE_MPG}" == "True" ]]; then
  CMD+=(
    --use-mpg "${USE_MPG}"
    --mpg-lambda "${MPG_LAMBDA}"
    --mpg-num-projections "${MPG_NUM_PROJECTIONS}"
    --mpg-refinement-iters "${MPG_REFINEMENT_ITERS}"
  )
fi

if [[ "${ENABLE_RTC}" == "True" ]]; then
  CMD+=(--enable-rtc)
else
  CMD+=(--no-enable-rtc)
fi

printf 'Executing command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
