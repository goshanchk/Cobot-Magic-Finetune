# Cobot Magic Fine-Tuning with LeRobot SmolVLA

This folder contains the LeRobot SmolVLA integration for the Cobot Magic LeRobot dataset. The upstream LeRobot README is kept as `README_lerobot.md`.

## Code Map

```text
src/lerobot/configs/default.py                         # DatasetConfig, adds dataset.joint_only_dim
src/lerobot/datasets/factory.py                        # JointOnlyDataset wrapper for 14D joints-only training
src/lerobot/policies/smolvla/                          # SmolVLA policy/config/model code
src/lerobot/scripts/lerobot_train.py                   # LeRobot training entry point
src/lerobot/scripts/inference/cobot_smolvla_zmq.py     # ZeroMQ inference server for the ALOHA client protocol
logs_smolvla/stdout/                                   # stdout logs from tmux/manual runs
logs_smolvla/outputs/                                  # checkpoints and training artifacts
```

## Dataset

Original dataset root:

```text
/path/to/cobot_magic_sber
```

The raw dataset stores 26D state/action vectors:

```text
first 14 dims: left arm 7 joints + right arm 7 joints
last 12 dims:  left/right FK EEF xyz/rpy auxiliary coordinates
```

SmolVLA training uses only the 14D joint space:

```bash
--dataset.joint_only_dim=14
```

This slices both `observation.state` and `action` to the first 14 dimensions at dataloader time and exposes 14D metadata/statistics to the policy.

Camera mapping follows the client protocol:

```text
images_camera_0 / observation.images.camera_0 -> right wrist
images_camera_1 / observation.images.camera_1 -> left wrist
images_camera_2 / observation.images.camera_2 -> front/high camera
```

## Environment

LeRobot requires Python 3.12. Install both SmolVLA and dataset extras: training needs `datasets`, `jsonlines`, `pandas`, and `torchcodec` in addition to the policy code.

```bash
cd /path/to/lerobot

# If the local SOCKS proxy is broken and the server has direct internet, disable it for this shell.
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy

UV_HTTP_TIMEOUT=1200 uv sync
UV_HTTP_TIMEOUT=1200 uv pip install -e ".[dataset,smolvla]"
```

Tmux installation:

```bash
cd /path/to/lerobot
mkdir -p logs_smolvla/stdout

tmux new -d -s smolvla_install \
  "cd $PWD && \
   unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy && \
   UV_HTTP_TIMEOUT=1200 uv sync 2>&1 | tee logs_smolvla/stdout/install.log && \
   UV_HTTP_TIMEOUT=1200 uv pip install -e '.[dataset,smolvla]' 2>&1 | tee -a logs_smolvla/stdout/install.log"
```

Watch:

```bash
tail -f logs_smolvla/stdout/install.log
```

## Dataset Conversion

Current LeRobot expects dataset format `v3.0`. The original Cobot Magic dataset is `v2.1`, so keep it untouched for OpenVLA/GR00T and create a separate converted copy for SmolVLA.

```bash
cp -a /path/to/cobot_magic_sber /path/to/cobot_magic_sber_v3_0

cd /path/to/lerobot
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy

.venv/bin/python -m lerobot.scripts.convert_dataset_v21_to_v30 \
  --repo-id=cobot_magic_sber \
  --root=/path/to/cobot_magic_sber_v3_0 \
  --push-to-hub=false

cat /path/to/cobot_magic_sber_v3_0/meta/info.json | grep codebase_version
```

Training commands below should use:

```bash
export DATASET_DIR=/home/dual4090/workspace/apanasevich/cobot_magic_sber_v3_0
```

SmolVLA base expects camera feature names `observation.images.camera1/2/3`, while the Cobot dataset uses `observation.images.camera_0/1/2`. Use `--rename_map` in training; it only renames feature keys and keeps the same camera order:

```text
camera_0 -> camera1  # right wrist
camera_1 -> camera2  # left wrist
camera_2 -> camera3  # front/high
```

## Smoke Training

Use this before a long run. It checks dataset loading, joint-only slicing, model creation, forward/backward, logging, and checkpoint writing.

```bash
cd /path/to/lerobot
export DATASET_DIR=/home/dual4090/workspace/apanasevich/cobot_magic_sber_v3_0
mkdir -p logs_smolvla/stdout logs_smolvla/outputs

tmux new -d -s smolvla_cobot_smoke \
  "cd $PWD && \
   unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy && \
   CUDA_VISIBLE_DEVICES=0 \
   .venv/bin/python -m lerobot.scripts.lerobot_train \
   --policy.path=lerobot/smolvla_base \
   --policy.push_to_hub=false \
   --seed=42 \
   --dataset.repo_id=cobot_magic_sber \
   --dataset.root=${DATASET_DIR} \
   --dataset.joint_only_dim=14 \
   --dataset.relative_joint_actions=true \
   --policy.relative_joint_actions=true \
   --dataset.image_transforms.enable=true \
   --dataset.image_transforms.max_num_transforms=3 \
   --dataset.image_transforms.random_order=true \
   --policy.chunk_size=24 \
   --policy.n_action_steps=24 \
   --policy.state_noise_std=0.05 \
   --policy.state_dropout_prob=0.10 \
   --policy.action_loss_gripper_weight=4.0 \
   --policy.action_loss_late_chunk_start=10 \
   --policy.action_loss_late_chunk_weight=2.0 \
   --rename_map='{\"observation.images.camera_0\":\"observation.images.camera1\",\"observation.images.camera_1\":\"observation.images.camera2\",\"observation.images.camera_2\":\"observation.images.camera3\"}' \
   --batch_size=2 \
   --steps=10 \
   --save_freq=10 \
   --log_freq=1 \
   --num_workers=2 \
   --prefetch_factor=2 \
   --output_dir=logs_smolvla/outputs/cobot_magic_smolvla_smoke \
   --job_name=cobot_magic_smolvla_smoke \
   --policy.device=cuda \
   --wandb.enable=false \
   2>&1 | tee logs_smolvla/stdout/cobot_magic_smolvla_smoke.log"
```

Watch:

```bash
tail -f logs_smolvla/stdout/cobot_magic_smolvla_smoke.log
```

## Full Training

Recommended Cobot Magic run: pretrained `lerobot/smolvla_base`, 14D joint-only state/action, relative joint-delta targets, 24-step action chunks, action-expert-only fine-tuning, 50k steps.

For object/color selection, keep the VLM and vision encoder frozen. This reduces catastrophic forgetting of the pretrained visual-language grounding; only the SmolVLA action expert plus state/action/time projection layers are trained.

```bash
cd /path/to/lerobot
export DATASET_DIR=/path/to/cobot_magic_sber_v3_0
mkdir -p logs_smolvla/stdout logs_smolvla/outputs

tmux new -d -s smolvla_cobot_expert \
  "cd $PWD && \
   unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy && \
   CUDA_VISIBLE_DEVICES=0,1 \
   .venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 \
   -m lerobot.scripts.lerobot_train \
   --policy.path=lerobot/smolvla_base \
   --policy.push_to_hub=false \
   --seed=42 \
   --dataset.repo_id=cobot_magic_sber \
   --dataset.root=${DATASET_DIR} \
   --dataset.joint_only_dim=14 \
   --dataset.relative_joint_actions=true \
   --policy.relative_joint_actions=true \
   --dataset.image_transforms.enable=true \
   --dataset.image_transforms.max_num_transforms=3 \
   --dataset.image_transforms.random_order=true \
   --policy.chunk_size=24 \
   --policy.n_action_steps=24 \
   --policy.state_noise_std=0.05 \
   --policy.state_dropout_prob=0.10 \
   --policy.action_loss_gripper_weight=4.0 \
   --policy.action_loss_late_chunk_start=10 \
   --policy.action_loss_late_chunk_weight=2.0 \
   --rename_map='{\"observation.images.camera_0\":\"observation.images.camera1\",\"observation.images.camera_1\":\"observation.images.camera2\",\"observation.images.camera_2\":\"observation.images.camera3\"}' \
   --batch_size=8 \
   --steps=50000 \
   --save_freq=5000 \
   --log_freq=100 \
   --num_workers=2 \
   --prefetch_factor=2 \
   --output_dir=logs_smolvla/outputs/cobot_magic_smolvla_expert \
   --job_name=cobot_magic_smolvla_expert \
   --policy.device=cuda \
   --policy.freeze_vision_encoder=true \
   --policy.train_expert_only=true \
   --policy.train_state_proj=true \
   --policy.load_vlm_weights=true \
   --policy.optimizer_lr=5e-5 \
   --policy.scheduler_warmup_steps=1000 \
   --policy.scheduler_decay_steps=50000 \
   --wandb.enable=false \
   2>&1 | tee logs_smolvla/stdout/cobot_magic_smolvla_expert.log"
```

Watch:

```bash
tail -f logs_smolvla/stdout/cobot_magic_smolvla_expert.log
```

## Inference: ZeroMQ Server

The robot client uses a ZeroMQ `REQ` socket and expects a server-side `REP` socket. The server receives three JPEG-base64 cameras plus a 14D joint proprio vector and always returns absolute joint targets with shape `[num_actions, 14]`.

```bash
cd /path/to/lerobot
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy

CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python src/lerobot/scripts/inference/cobot_smolvla_zmq.py \
  --checkpoint_path logs_smolvla/outputs/cobot_magic_smolvla_expert/checkpoints/050000 \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5055 \
  --max_actions 10
```

PEFT/LoRA checkpoints contain `adapter_model.safetensors` instead of
`model.safetensors`. The inference server detects this layout automatically,
loads the base checkpoint from `adapter_config.json`, and calls
`merge_and_unload()` before serving requests. If the base path saved in the
adapter config is unavailable or you want to use a local copy, pass it explicitly:

```bash
.venv/bin/python src/lerobot/scripts/inference/cobot_smolvla_zmq.py \
  --checkpoint_path /path/to/checkpoints/smolvla_lora/050000 \
  --base_model_path /path/to/smolvla_base \
  --device cuda:0 \
  --port 5055
```

## Tmux

```bash
tmux ls
tmux attach -t smolvla_cobot_expert
# detach without stopping: Ctrl-b, then d
```
