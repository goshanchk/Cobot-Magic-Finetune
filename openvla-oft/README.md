# Cobot Magic Fine-Tuning

This repository contains the Cobot Magic integration for OpenVLA-OFT.
The recommended training path is the direct `lerobot` PyTorch dataloader. The original upstream README is kept as `README_openvla.md`.

## Code Map

```text
experiments/robot/openvla_utils.py                         # OpenVLA helper utilities, TF imported lazily only when needed
prismatic/vla/constants.py                                 # ACTION_DIM=14, PROPRIO_DIM=14, ACTION_CHUNK=24
prismatic/vla/datasets/datasets.py                         # LeRobotDataset, LeRobotBatchTransform, RLDS wrappers
prismatic/vla/datasets/__init__.py                         # dataset exports
prismatic/vla/datasets/rlds/oxe/*.py                       # optional RLDS/OXE registration
prismatic/models/action_heads.py                           # L1 action head forward + configurable size
vla-scripts/finetune.py                                    # dataset_format, FSDP, logging, freeze_vla, head size args
vla-scripts/cobot_openvlaoft_zmq.py                        # ZeroMQ inference server for the ALOHA client protocol
```

Important launch args:

```text
--dataset_format lerobot|rlds
--distributed_backend ddp|fsdp
--freeze_vla True|False
--num_images_in_input 1|2|3
--use_film True|False
--use_proprio True|False
--action_head_hidden_dim N
--action_head_num_blocks N
--logger tensorboard|wandb|none
--dataloader_num_workers N
--lerobot_use_precomputed_stats True|False
--lerobot_sample_by_episode True|False
--lerobot_episode_cache_size N
```

## Dataset

Expected source dataset root:

```text
/path/to/cobot_magic_sber
```

Direct LeRobot mapping:

```text
observation.images.camera_2 -> primary image
observation.images.camera_1 -> left wrist image
observation.images.camera_0 -> right wrist image
observation.state           -> raw state, shape [26]; loader keeps first 14 joint dims
action                      -> raw action, shape [26]; loader keeps first 14 joint dims
task_index                  -> language_instruction via meta/tasks.jsonl
```

Only the joint-space part is trained: `left_q0..left_q6 + right_q0..right_q6`.
The raw FK EEF xyz/rpy coordinates stay in the source dataset but are dropped by the loader, because current ALOHA control supports joint commands only.

Cobot constants:

```text
ACTION_DIM = 14
PROPRIO_DIM = 14
NUM_ACTIONS_CHUNK = 24
ACTION_PROPRIO_NORMALIZATION_TYPE = bounds
```

Camera selection:

```text
--num_images_in_input 1   primary only
--num_images_in_input 2   primary + left wrist
--num_images_in_input 3   primary + left wrist + right wrist
```

The direct LeRobot backend uses this split:

```text
train: 7041 episodes
val:   300 episodes
split_seed: 42
```

Performance notes for the direct LeRobot backend:

```text
lerobot_use_precomputed_stats=True  uses meta/stats.json sliced to 14D joints; avoids scanning all parquet at startup
lerobot_sample_by_episode=True      shuffles episodes first, then timesteps; reuses decoded video cache
dataloader_num_workers>0            lets CPU workers prepare video batches while GPU is training
```

## Environment

Create the `finetune_env` micromamba environment from the repo metadata, then install the repo in editable mode. Run from the repo root:

```bash
cd /path/to/openvla-oft
export REPO_ROOT=$PWD
export DATASET_DIR=/path/to/cobot_magic_sber

micromamba create -n finetune_env python=3.10 -y
micromamba run -n finetune_env pip install -U pip setuptools wheel
micromamba run -n finetune_env pip install -e .

mkdir -p logs/stdout logs/runs
```

Quick environment check:

```bash
micromamba run -n finetune_env python - <<'PY'
import torch
import tensorboard
import prismatic
print('torch', torch.__version__)
print('tensorboard', tensorboard.__version__)
print('openvla-oft import ok')
PY
```

## Smoke: Frozen VLA, Action Head Only

```bash
tmux new -d -s openvla_frozen_smoke \
  "cd ${REPO_ROOT} && \
   CUDA_VISIBLE_DEVICES=0,1 \
   micromamba run -n finetune_env torchrun \
   --standalone --nnodes 1 --nproc-per-node 2 \
   vla-scripts/finetune.py \
   --vla_path openvla/openvla-7b \
   --dataset_format lerobot \
   --data_root_dir ${DATASET_DIR} \
   --dataset_name cobot_magic_sber \
   --run_root_dir ${REPO_ROOT}/logs/runs \
   --distributed_backend fsdp \
   --freeze_vla True \
   --use_l1_regression True \
   --use_diffusion False \
   --use_film True \
   --num_images_in_input 3 \
   --use_proprio True \
   --batch_size 1 \
   --grad_accumulation_steps 1 \
   --learning_rate 5e-4 \
   --num_steps_before_decay 80 \
   --max_steps 1 \
   --use_val_set False \
   --save_freq 1000 \
   --image_aug False \
   --lora_rank 8 \
   --action_head_hidden_dim 512 \
   --action_head_num_blocks 1 \
   --logger tensorboard \
   --log_freq 1 \
   --shuffle_buffer_size 256 \
   --run_id_note frozen-smoke--3cam--film--chunk24--h512 \
   2>&1 | tee logs/stdout/openvla_frozen_smoke.log"
```

Watch:

```bash
tail -f ${REPO_ROOT}/logs/stdout/openvla_frozen_smoke.log
```

## Smoke: Unfrozen Lightweight Training

```bash
tmux new -d -s openvla_unfrozen_smoke \
  "cd ${REPO_ROOT} && \
   CUDA_VISIBLE_DEVICES=0,1 \
   COBOT_MAGIC_NUM_ACTIONS_CHUNK=1 \
   micromamba run -n finetune_env torchrun \
   --standalone --nnodes 1 --nproc-per-node 2 \
   vla-scripts/finetune.py \
   --vla_path openvla/openvla-7b \
   --dataset_format lerobot \
   --data_root_dir ${DATASET_DIR} \
   --dataset_name cobot_magic_sber \
   --run_root_dir ${REPO_ROOT}/logs/runs \
   --distributed_backend fsdp \
   --freeze_vla False \
   --use_l1_regression True \
   --use_diffusion False \
   --use_film False \
   --num_images_in_input 1 \
   --use_proprio False \
   --batch_size 1 \
   --grad_accumulation_steps 1 \
   --learning_rate 5e-4 \
   --num_steps_before_decay 80 \
   --max_steps 1 \
   --use_val_set False \
   --save_freq 1000 \
   --image_aug False \
   --lora_rank 8 \
   --action_head_hidden_dim 512 \
   --action_head_num_blocks 1 \
   --logger none \
   --log_freq 1 \
   --shuffle_buffer_size 256 \
   --run_id_note unfrozen-smoke--1cam--chunk1--lora8--h512 \
   2>&1 | tee logs/stdout/openvla_unfrozen_smoke.log"
```

Watch:

```bash
tail -f ${REPO_ROOT}/logs/stdout/openvla_unfrozen_smoke.log
```

## Full Training

For n-GPU node, expose n GPUs and use `--nproc-per-node n`, n is the number of GPUs.

```bash
tmux new -d -s openvla_full_train \
  "cd ${REPO_ROOT} && \
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   NCCL_DEBUG=INFO \
   TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
   CUDA_DEVICE_MAX_CONNECTIONS=1 \
   micromamba run -n finetune_env torchrun \
   --standalone --nnodes 1 --nproc-per-node 8 \
   vla-scripts/finetune.py \
   --vla_path openvla/openvla-7b \
   --dataset_format lerobot \
   --data_root_dir ${DATASET_DIR} \
   --dataset_name cobot_magic_sber \
   --run_root_dir ${REPO_ROOT}/logs/runs \
   --distributed_backend fsdp \
   --freeze_vla False \
   --use_l1_regression True \
   --use_diffusion False \
   --use_film True \
   --num_images_in_input 3 \
   --use_proprio True \
   --batch_size 1 \
   --grad_accumulation_steps 8 \
   --learning_rate 5e-4 \
   --num_steps_before_decay 10000 \
   --max_steps 20000 \
   --use_val_set True \
   --val_freq 1000 \
   --val_time_limit 180 \
   --save_freq 2000 \
   --image_aug True \
   --lora_rank 8 \
   --action_head_hidden_dim 2048 \
   --action_head_num_blocks 1 \
   --logger tensorboard \
   --log_freq 10 \
   --shuffle_buffer_size 100000 \
   --dataloader_num_workers 2 \
   --dataloader_prefetch_factor 2 \
   --lerobot_episode_cache_size 4 \
   --lerobot_use_precomputed_stats True \
   --lerobot_sample_by_episode True \
   --run_id_note full--3cam--film--chunk24--joints14--lora8--h2048--fastloader \
   2>&1 | tee logs/stdout/openvla_full_train.log"
```

## Logs

Stdout logs:

```bash
tail -f ${REPO_ROOT}/logs/stdout/openvla_full_train.log
```

TensorBoard:

```bash
cd ${REPO_ROOT}
micromamba run -n finetune_env tensorboard \
  --logdir ${REPO_ROOT}/logs/runs \
  --host 127.0.0.1 \
  --port 6006
```

Open through localhost:

```text
http://localhost:6006
```

Tmux checking:

```bash
tmux ls
tmux attach -t openvla_full_train
# detach without stopping: Ctrl-b, then d
```

## Inference: ZeroMQ Server

The robot client uses a ZeroMQ `REQ` socket and expects a server-side `REP` socket. The OpenVLA-OFT adapter listens on `0.0.0.0:5055`, receives 3 JPEG-base64 cameras plus a 14D joint proprio vector, and returns absolute joint actions with shape `[num_actions, 14]`.

Checkpoint layout expected by the loader:

```text
/path/to/openvla_20000_chkpt/
  lora_adapter/
  action_head--20000_checkpoint.pt
  proprio_projector--20000_checkpoint.pt
  vision_backbone--20000_checkpoint.pt
  dataset_statistics.json
  config.json
  model-*.safetensors
```

Run from the OpenVLA-OFT repo root:

```bash
cd /path/to/openvla-oft
export OPENVLA_CHECKPOINT=/path/to/openvla_20000_chkpt

CUDA_VISIBLE_DEVICES=0 \
micromamba run -n finetune_env python vla-scripts/cobot_openvlaoft_zmq.py \
  --pretrained_checkpoint ${OPENVLA_CHECKPOINT} \
  --unnorm_key cobot_magic_sber \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --lora_rank 8 \
  --host 0.0.0.0 \
  --port 5055
```

Use `--use_film False` only for checkpoints trained without FiLM. Use `--use_relative_actions True` only if the checkpoint returns delta actions; the Cobot Magic joint-only checkpoints are trained to return absolute joint targets.

