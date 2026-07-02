#!/usr/bin/env bash
# Cobot Magic post-training for Being-H05. Set NUM_GPUS/CUDA_VISIBLE_DEVICES for the node.

set -euo pipefail

export PYTHONPATH=${PYTHONPATH:-.}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=true

# ============ Model and Data Paths ============
# Override these from the shell if your checkpoints live elsewhere.
PRETRAIN_MODEL=${PRETRAIN_MODEL:-${PWD}/checkpoint_models/being/InternVL3_5-2B}
EXPERT_MODEL=${EXPERT_MODEL:-${PWD}/checkpoint_models/being/Qwen3-0.6B}
RESUME_PATH=${RESUME_PATH:-${PWD}/checkpoint_models/being/Being-H05-2B}
DATASET_DIR=${DATASET_DIR:-/home/dual4090/workspace/apanasevich/cobot_magic_sber}
DATASET_CONFIG_FILE=${DATASET_CONFIG_FILE:-configs/posttrain/cobot_magic/cobot_magic_sber.yaml}

# Keep dataset_info.py in sync with DATASET_DIR before launching if you change it.

# ============ Training Configuration ============
NUM_GPUS=${NUM_GPUS:-2}
MAX_STEPS=${MAX_STEPS:-50000}
SAVE_STEPS=${SAVE_STEPS:-5000}
SAVE_STEPS_START=${SAVE_STEPS_START:-5000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
GRAD_ACCUMULATION_STEPS=${GRAD_ACCUMULATION_STEPS:-1}

# ============ Weighted Action Loss ============
# Unified gripper-like dims: right q6 -> 56, left q6 -> 63.
ACTION_LOSS_GRIPPER_WEIGHT=${ACTION_LOSS_GRIPPER_WEIGHT:-4.0}
ACTION_LOSS_LATE_CHUNK_START=${ACTION_LOSS_LATE_CHUNK_START:-10}
ACTION_LOSS_LATE_CHUNK_WEIGHT=${ACTION_LOSS_LATE_CHUNK_WEIGHT:-2.0}

# ============ Data Loading ============
NUM_WORKERS=${NUM_WORKERS:-4}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}

# ============ Sequence Configuration ============
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-8704}
EXPECTED_NUM_TOKENS=${EXPECTED_NUM_TOKENS:-8192}
PREFER_BUFFER_BEFORE=${PREFER_BUFFER_BEFORE:-4096}
MAX_BUFFER_SIZE=${MAX_BUFFER_SIZE:-4}
ATTN_MODE=${ATTN_MODE:-causal}

# ============ Image Configuration ============
FORCE_IMAGE_SIZE=${FORCE_IMAGE_SIZE:-224}
MAX_VIEW_NUM=${MAX_VIEW_NUM:--1}
USE_FIXED_VIEW=${USE_FIXED_VIEW:-False}
DOWN_SAMPLE_RATIO=${DOWN_SAMPLE_RATIO:-0.5}

# ============ Action Configuration ============
# Dataset action[t] is absolute next joint state. --is_relative True converts
# target joints to target_delta = q_target - q_current in the dataloader.
ACTION_CHUNK_LENGTH=${ACTION_CHUNK_LENGTH:-24}
IS_RELATIVE=${IS_RELATIVE:-True}

# ============ Freezing Configuration ============
# Existing Being-H flags are enough for the requested training scope:
# - freeze_mllm=True freezes VLM/backbone layers while leaving action modules trainable.
# - freeze_vit_mlp=False keeps the vision connector/projector trainable.
FREEZE_MLLM=${FREEZE_MLLM:-True}
FREEZE_VIT_MLP=${FREEZE_VIT_MLP:-False}
FREEZE_LLM=${FREEZE_LLM:-False}
FREEZE_VIT=${FREEZE_VIT:-False}

# ============ MPG / RTC ============
USE_MPG=${USE_MPG:-True}
MPG_LAMBDA=${MPG_LAMBDA:-0.1}
MPG_NUM_PROJECTIONS=${MPG_NUM_PROJECTIONS:-32}
MPG_REFINEMENT_ITERS=${MPG_REFINEMENT_ITERS:-1}
MPG_GATE_TEMPERATURE=${MPG_GATE_TEMPERATURE:-1.0}
MPG_USE_STOP_GRADIENT=${MPG_USE_STOP_GRADIENT:-True}

USE_TRAINING_TIME_RTC=${USE_TRAINING_TIME_RTC:-False}
SIMULATED_DELAY=${SIMULATED_DELAY:-0}
RTC_DELAY_EXP_WEIGHT=${RTC_DELAY_EXP_WEIGHT:-True}
USE_INFERENCE_PREFIX_OVERWRITE=${USE_INFERENCE_PREFIX_OVERWRITE:-True}

# ============ Output Configuration ============
RUN_NAME=${RUN_NAME:-cobot_magic_sber_beingh05_2gpu_action_stack_chunk${ACTION_CHUNK_LENGTH}}
OUTPUT_DIR=${OUTPUT_DIR:-${PWD}/logs/outputs/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-${PWD}/logs/tensorboard}
LOG_FILE=${LOG_FILE:-${PWD}/logs/stdout/${RUN_NAME}.log}
RUN_IN_TMUX=${RUN_IN_TMUX:-True}
TMUX_SESSION=${TMUX_SESSION:-beingh_cobot_magic_sber}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "$(dirname "${LOG_FILE}")"

export PRETRAIN_MODEL EXPERT_MODEL RESUME_PATH DATASET_DIR DATASET_CONFIG_FILE
export NUM_GPUS MAX_STEPS SAVE_STEPS SAVE_STEPS_START LEARNING_RATE WEIGHT_DECAY WARMUP_RATIO GRAD_ACCUMULATION_STEPS
export ACTION_LOSS_GRIPPER_WEIGHT ACTION_LOSS_LATE_CHUNK_START ACTION_LOSS_LATE_CHUNK_WEIGHT
export NUM_WORKERS PREFETCH_FACTOR MAX_NUM_TOKENS EXPECTED_NUM_TOKENS PREFER_BUFFER_BEFORE MAX_BUFFER_SIZE ATTN_MODE
export FORCE_IMAGE_SIZE MAX_VIEW_NUM USE_FIXED_VIEW DOWN_SAMPLE_RATIO ACTION_CHUNK_LENGTH IS_RELATIVE
export FREEZE_MLLM FREEZE_VIT_MLP FREEZE_LLM FREEZE_VIT
export USE_MPG MPG_LAMBDA MPG_NUM_PROJECTIONS MPG_REFINEMENT_ITERS MPG_GATE_TEMPERATURE MPG_USE_STOP_GRADIENT
export USE_TRAINING_TIME_RTC SIMULATED_DELAY RTC_DELAY_EXP_WEIGHT USE_INFERENCE_PREFIX_OVERWRITE
export RUN_NAME OUTPUT_DIR LOG_DIR LOG_FILE CUDA_VISIBLE_DEVICES MASTER_PORT

if [ "${RUN_IN_TMUX}" = "True" ] && [ -z "${TMUX:-}" ]; then
  if command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
      echo "tmux session already exists: ${TMUX_SESSION}"
      echo "Attach: tmux attach -t ${TMUX_SESSION}"
      echo "Log: ${LOG_FILE}"
      exit 0
    fi
    tmux_env=(
      RUN_IN_TMUX=False
      PRETRAIN_MODEL="${PRETRAIN_MODEL}"
      EXPERT_MODEL="${EXPERT_MODEL}"
      RESUME_PATH="${RESUME_PATH}"
      DATASET_DIR="${DATASET_DIR}"
      DATASET_CONFIG_FILE="${DATASET_CONFIG_FILE}"
      NUM_GPUS="${NUM_GPUS}"
      MAX_STEPS="${MAX_STEPS}"
      SAVE_STEPS="${SAVE_STEPS}"
      SAVE_STEPS_START="${SAVE_STEPS_START}"
      LEARNING_RATE="${LEARNING_RATE}"
      WEIGHT_DECAY="${WEIGHT_DECAY}"
      WARMUP_RATIO="${WARMUP_RATIO}"
      GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS}"
      ACTION_LOSS_GRIPPER_WEIGHT="${ACTION_LOSS_GRIPPER_WEIGHT}"
      ACTION_LOSS_LATE_CHUNK_START="${ACTION_LOSS_LATE_CHUNK_START}"
      ACTION_LOSS_LATE_CHUNK_WEIGHT="${ACTION_LOSS_LATE_CHUNK_WEIGHT}"
      NUM_WORKERS="${NUM_WORKERS}"
      PREFETCH_FACTOR="${PREFETCH_FACTOR}"
      MAX_NUM_TOKENS="${MAX_NUM_TOKENS}"
      EXPECTED_NUM_TOKENS="${EXPECTED_NUM_TOKENS}"
      PREFER_BUFFER_BEFORE="${PREFER_BUFFER_BEFORE}"
      MAX_BUFFER_SIZE="${MAX_BUFFER_SIZE}"
      ATTN_MODE="${ATTN_MODE}"
      FORCE_IMAGE_SIZE="${FORCE_IMAGE_SIZE}"
      MAX_VIEW_NUM="${MAX_VIEW_NUM}"
      USE_FIXED_VIEW="${USE_FIXED_VIEW}"
      DOWN_SAMPLE_RATIO="${DOWN_SAMPLE_RATIO}"
      ACTION_CHUNK_LENGTH="${ACTION_CHUNK_LENGTH}"
      IS_RELATIVE="${IS_RELATIVE}"
      FREEZE_MLLM="${FREEZE_MLLM}"
      FREEZE_VIT_MLP="${FREEZE_VIT_MLP}"
      FREEZE_LLM="${FREEZE_LLM}"
      FREEZE_VIT="${FREEZE_VIT}"
      USE_MPG="${USE_MPG}"
      MPG_LAMBDA="${MPG_LAMBDA}"
      MPG_NUM_PROJECTIONS="${MPG_NUM_PROJECTIONS}"
      MPG_REFINEMENT_ITERS="${MPG_REFINEMENT_ITERS}"
      MPG_GATE_TEMPERATURE="${MPG_GATE_TEMPERATURE}"
      MPG_USE_STOP_GRADIENT="${MPG_USE_STOP_GRADIENT}"
      USE_TRAINING_TIME_RTC="${USE_TRAINING_TIME_RTC}"
      SIMULATED_DELAY="${SIMULATED_DELAY}"
      RTC_DELAY_EXP_WEIGHT="${RTC_DELAY_EXP_WEIGHT}"
      USE_INFERENCE_PREFIX_OVERWRITE="${USE_INFERENCE_PREFIX_OVERWRITE}"
      RUN_NAME="${RUN_NAME}"
      OUTPUT_DIR="${OUTPUT_DIR}"
      LOG_DIR="${LOG_DIR}"
      LOG_FILE="${LOG_FILE}"
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
      MASTER_PORT="${MASTER_PORT:-29106}"
    )
    tmux_env_cmd=()
    for item in "${tmux_env[@]}"; do
      tmux_env_cmd+=("$(printf '%q' "$item")")
    done
    tmux new -d -s "${TMUX_SESSION}" "cd $(printf '%q' "${PWD}") && env ${tmux_env_cmd[*]} bash $(printf '%q' "$0")"
    echo "Started tmux session: ${TMUX_SESSION}"
    echo "Attach: tmux attach -t ${TMUX_SESSION}"
    echo "Log: ${LOG_FILE}"
    exit 0
  else
    echo "WARNING: tmux is not installed; running in foreground." >&2
  fi
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "Dataset not found: ${DATASET_DIR}" >&2
  exit 1
fi

for pair in \
  "PRETRAIN_MODEL:${PRETRAIN_MODEL}" \
  "EXPERT_MODEL:${EXPERT_MODEL}" \
  "RESUME_PATH:${RESUME_PATH}" \
  "DATASET_CONFIG_FILE:${DATASET_CONFIG_FILE}"; do
  label=${pair%%:*}
  path=${pair#*:}
  if [ ! -e "${path}" ]; then
    echo "WARNING: ${label} does not exist yet: ${path}" >&2
  fi
done

# Save launch context for reproducibility.
cp "$0" "${OUTPUT_DIR}/"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=${MASTER_PORT:-29106} \
  BeingH/train/train.py \
  --mllm_path "${PRETRAIN_MODEL}" \
  --expert_path "${EXPERT_MODEL}" \
  --resume_from "${RESUME_PATH}" \
  --resume_model_only True \
  --layer_module Qwen3MoTDecoderLayer \
  --use_expert True \
  --use_flow_matching True \
  --llm_qk_norm True \
  --freeze_mllm "${FREEZE_MLLM}" \
  --freeze_llm "${FREEZE_LLM}" \
  --freeze_vit "${FREEZE_VIT}" \
  --freeze_vit_mlp "${FREEZE_VIT_MLP}" \
  --action_chunk_length "${ACTION_CHUNK_LENGTH}" \
  --is_relative "${IS_RELATIVE}" \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --max_num_tokens_per_sample "${MAX_NUM_TOKENS}" \
  --expected_num_tokens "${EXPECTED_NUM_TOKENS}" \
  --prefer_buffer_before "${PREFER_BUFFER_BEFORE}" \
  --max_buffer_size "${MAX_BUFFER_SIZE}" \
  --attn_mode "${ATTN_MODE}" \
  --max_view_num "${MAX_VIEW_NUM}" \
  --use_fixed_view "${USE_FIXED_VIEW}" \
  --force_image_size "${FORCE_IMAGE_SIZE}" \
  --down_sample_ratio "${DOWN_SAMPLE_RATIO}" \
  --dataset_config_file "${DATASET_CONFIG_FILE}" \
  --save_merged_metadata True \
  --conv_style being_h0 \
  --vision_select_layer -1 \
  --prompt_template long \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOG_DIR}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --max_steps "${MAX_STEPS}" \
  --save_model_only False \
  --save_steps "${SAVE_STEPS}" \
  --save_steps_start "${SAVE_STEPS_START}" \
  --logging_steps 10 \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler cosine \
  --grad_checkpoint False \
  --gradient_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
  --action_loss_gripper_weight "${ACTION_LOSS_GRIPPER_WEIGHT}" \
  --action_loss_late_chunk_start "${ACTION_LOSS_LATE_CHUNK_START}" \
  --action_loss_late_chunk_weight "${ACTION_LOSS_LATE_CHUNK_WEIGHT}" \
  --use_mpg "${USE_MPG}" \
  --mpg_lambda "${MPG_LAMBDA}" \
  --mpg_num_projections "${MPG_NUM_PROJECTIONS}" \
  --mpg_refinement_iters "${MPG_REFINEMENT_ITERS}" \
  --mpg_gate_temperature "${MPG_GATE_TEMPERATURE}" \
  --mpg_use_stop_gradient "${MPG_USE_STOP_GRADIENT}" \
  --use_training_time_rtc "${USE_TRAINING_TIME_RTC}" \
  --simulated_delay "${SIMULATED_DELAY}" \
  --rtc_delay_exp_weight "${RTC_DELAY_EXP_WEIGHT}" \
  --use_inference_prefix_overwrite "${USE_INFERENCE_PREFIX_OVERWRITE}" \
  2>&1 | tee "${LOG_FILE}"
