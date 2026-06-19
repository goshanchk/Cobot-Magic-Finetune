# Cobot Magic Fine-Tuning with Isaac-GR00T

This folder contains the Isaac-GR00T N1.7 integration for the Cobot Magic LeRobot dataset. The upstream NVIDIA README is kept as `README_groot.md`.

## Code Map

```text
examples/CobotMagic/cobot_magic_config.py        # Cobot Magic modality config: 3 cameras, 14D joint state/action
examples/CobotMagic/cobot_groot_zmq.py           # ZeroMQ inference server for the ALOHA client protocol
gr00t/experiment/launch_finetune.py             # CLI entry point for fine-tuning
gr00t/experiment/experiment.py                  # Trainer setup, DeepSpeed/DDP, TensorBoard/W&B logging
gr00t/configs/finetune_config.py                # fine-tune CLI config
gr00t/configs/training/training_config.py       # training/logging config
logs/stdout/                                    # stdout logs from tmux/manual runs
logs/outputs/                                   # checkpoints, configs, TensorBoard files
```

## Dataset

Expected dataset root:

```text
/path/to/cobot_magic_sber
```

The current Cobot Magic config uses:

```text
video.cam_high
video.cam_left_wrist
video.cam_right_wrist
state.all_arms      # 14D joints
action.all_arms     # 24-step chunk of 14D relative joint delta targets during training
```

The raw dataset also contains `left_eef` and `right_eef` FK xyz/rpy groups, but they are not used for training because current ALOHA control supports joint commands only. Training uses 24-step relative joint deltas computed from absolute dataset actions; the ZMQ server converts predicted deltas back to absolute joint targets for the robot client.

Training commands below build the same train/validation split as OpenVLA: they start from `meta/validation_episodes.json` and, if needed, sample extra hold-out episodes up to `300` with `split_seed=42`. Training excludes those hold-out episodes; offline validation evaluates the saved checkpoint on the same split.

## Environment

GR00T uses `uv`. Create and verify the environment from this folder:

```bash
cd /path/to/Isaac-GR00T
uv sync
uv run python -c "import gr00t, torch; print('GR00T ok', torch.__version__)"
```

## HuggingFace Access

Before training, request access to the gated Cosmos backbone in the browser:

```text
https://huggingface.co/nvidia/Cosmos-Reason2-2B
```

Login on the server with a HuggingFace read token:

```bash
cd /path/to/Isaac-GR00T
.venv/bin/hf auth login
```

## Smoke: Lightweight Training, Frozen Diffusion

It freezes the LLM, visual encoder, and diffusion model, and trains the projector/action adapter part.

```bash
cd /path/to/Isaac-GR00T
export DATASET_DIR=/path/to/cobot_magic_sber
mkdir -p logs/stdout logs/outputs

tmux new -d -s groot_cobot_projector \
  "cd $PWD && \
   CUDA_VISIBLE_DEVICES=0,1 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   .venv/bin/torchrun --nproc_per_node=2 --master_port=29501 \
   gr00t/experiment/launch_finetune.py \
   --base_model_path nvidia/GR00T-N1.7-3B \
   --dataset_path ${DATASET_DIR} \
   --embodiment_tag NEW_EMBODIMENT \
   --num_gpus 2 \
   --output_dir $PWD/logs/outputs/cobot_magic_2gpu_projector_test \
   --save_steps 10 \
   --save_total_limit 5 \
   --max_steps 10 \
   --warmup_ratio 0.05 \
   --weight_decay 1e-5 \
   --learning_rate 1e-4 \
   --use_tensorboard \
   --global_batch_size 2 \
   --random_rotation_angle 5 \
   --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
   --dataloader_num_workers 1 \
   --shard_size 1024 \
   --num_shards_per_epoch 100000 \
   --episode_sampling_rate 0.1 \
   --modality_config_path examples/CobotMagic/cobot_magic_config.py \
   --exclude-validation-episodes \
   --validation-split-path meta/validation_episodes.json \
   --validation-episodes-target 300 \
   --split-seed 42 \
   --no-tune-diffusion-model \
   2>&1 | tee logs/stdout/cobot_magic_2gpu_projector_test.log"
```

## Training with Diffusion Unfrozen

```bash
cd /path/to/Isaac-GR00T
export DATASET_DIR=/path/to/cobot_magic_sber
mkdir -p logs/stdout logs/outputs

tmux new -d -s groot_cobot_full \
  "cd $PWD && \
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   .venv/bin/torchrun --nproc_per_node=8 --master_port=29500 \
   gr00t/experiment/launch_finetune.py \
   --base_model_path nvidia/GR00T-N1.7-3B \
   --dataset_path ${DATASET_DIR} \
   --embodiment_tag NEW_EMBODIMENT \
   --num_gpus 8 \
   --output_dir $PWD/logs/outputs/cobot_magic_full \
   --save_steps 1000 \
   --save_total_limit 5 \
   --max_steps 35000 \
   --warmup_ratio 0.05 \
   --weight_decay 1e-5 \
   --learning_rate 5e-5 \
   --use_tensorboard \
   --global_batch_size 32 \
   --random_rotation_angle 5 \
   --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
   --dataloader_num_workers 4 \
   --shard_size 1024 \
   --num_shards_per_epoch 100000 \
   --episode_sampling_rate 0.1 \
   --modality_config_path examples/CobotMagic/cobot_magic_config.py \
   --exclude-validation-episodes \
   --validation-split-path meta/validation_episodes.json \
   --validation-episodes-target 300 \
   --split-seed 42 \
   --action-loss-gripper-weight 4.0 \
   --action-loss-late-chunk-start 10 \
   --action-loss-late-chunk-weight 2.0 \
   2>&1 | tee logs/stdout/cobot_magic_full.log"
```

## Offline Validation

The GR00T sharded training dataset does not run validation inside the training loop. In the Cobot Magic launch commands above, `--exclude-validation-episodes` removes the same 300 hold-out episodes used by OpenVLA from training. After a checkpoint is saved, use the offline validation script with the same split settings; it runs forward passes without optimizer steps and reports mean validation loss.

Quick validation on a subset:

```bash
cd /path/to/Isaac-GR00T
export DATASET_DIR=/path/to/cobot_magic_sber

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python examples/CobotMagic/eval_cobot_magic.py \
  --checkpoint_path logs/outputs/cobot_magic_full/checkpoint-30000 \
  --dataset_path ${DATASET_DIR} \
  --embodiment_tag NEW_EMBODIMENT \
  --modality_config_path examples/CobotMagic/cobot_magic_config.py \
  --validation_episodes_target 300 \
  --seed 42 \
  --output_dir logs/eval/cobot_magic_full_checkpoint_30000 \
  --device cuda:0 \
  --dtype bf16 \
  --batch_size 1 \
  --max_episodes 30 \
  --max_steps_per_episode 16
```

Full validation over all 300 validation episodes and all valid timesteps can be launched by disabling the limits:

```bash
.venv/bin/python examples/CobotMagic/eval_cobot_magic.py \
  --checkpoint_path logs/outputs/cobot_magic_full/checkpoint-30000 \
  --dataset_path ${DATASET_DIR} \
  --embodiment_tag NEW_EMBODIMENT \
  --modality_config_path examples/CobotMagic/cobot_magic_config.py \
  --validation_episodes_target 300 \
  --seed 42 \
  --output_dir logs/eval/cobot_magic_full_checkpoint_30000_full \
  --device cuda:0 \
  --dtype bf16 \
  --batch_size 1 \
  --max_episodes 0 \
  --max_steps_per_episode 0
```

Validation metrics are saved to:

```text
logs/eval/<run_name>/metrics.json
```

## Logs

Stdout:

```bash
cd /path/to/Isaac-GR00T
export RUN_NAME=cobot_magic_full
tail -f logs/stdout/${RUN_NAME}.log
```

TensorBoard:

```bash
cd /path/to/Isaac-GR00T
export RUN_NAME=cobot_magic_full
.venv/bin/tensorboard --logdir logs/outputs/${RUN_NAME}/tensorboard --host 0.0.0.0 --port 6006
```

Tmux:

```bash
tmux ls
tmux attach -t groot_cobot_projector
# detach without stopping: Ctrl-b, then d
```

## Inference: ZeroMQ Server

The robot client uses a ZeroMQ `REQ` socket and expects a server-side `REP` socket. The GR00T adapter listens on `0.0.0.0:5055`, receives 3 JPEG-base64 cameras plus a 14D joint proprio vector, and returns absolute joint actions with shape `[num_actions, 14]`.

Checkpoint layout expected by `Gr00tPolicy`:

```text
/path/to/gr00t_checkpoint-30000/
  config.json
  embodiment_id.json
  processor_config.json
  statistics.json
  model-*.safetensors
  model.safetensors.index.json
```

Run from the Isaac-GR00T repo root:

```bash
cd /path/to/Isaac-GR00T
export GROOT_CHECKPOINT=/path/to/gr00t_checkpoint-30000

CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python examples/CobotMagic/cobot_groot_zmq.py \
  --model_path ${GROOT_CHECKPOINT} \
  --embodiment_tag new_embodiment \
  --modality_config_path examples/CobotMagic/cobot_magic_config.py \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5055
```

Default Cobot Magic GR00T checkpoints are trained to predict relative joint deltas, and this server converts them to absolute joint targets before replying. Use `--absolute_actions` only for old checkpoints that already output absolute targets.
