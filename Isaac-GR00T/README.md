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

The training command excludes a reproducible hold-out set built from
`meta/validation_episodes.json`, extended to 300 episodes with seed 42 when
needed.

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

## Training

This configuration fine-tunes the diffusion/flow-matching action decoder and
multimodal projector while keeping the LLM and visual encoder frozen. Rank-8
LoRA adapters are applied to the VLM attention projections.

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
   --use_lora \
   --lora_rank 8 \
   --lora_alpha 16 \
   --lora_dropout 0.05 \
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

GR00T LoRA runs are saved as full Hugging Face checkpoints: the base weights,
action head, and injected `lora_A`/`lora_B` tensors are stored together in the
model safetensors files. The inference policy detects these tensor keys,
reconstructs the adapters, loads the checkpoint, and calls
`merge_and_unload()` before the first inference request. Plain checkpoints and
already-merged checkpoints are loaded directly without applying LoRA again.
