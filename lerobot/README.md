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
export DATASET_DIR=/path/to/cobot_magic_sber_v3_0
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
export DATASET_DIR=/path/to/cobot_magic_sber_v3_0
mkdir -p logs_smolvla/stdout logs_smolvla/outputs

tmux new -d -s smolvla_cobot_smoke \
  "cd $PWD && \
   unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy && \
   CUDA_VISIBLE_DEVICES=0 \
   .venv/bin/python -m lerobot.scripts.lerobot_train \
   --policy.path=lerobot/smolvla_base \
   --policy.push_to_hub=false \
   --dataset.repo_id=cobot_magic_sber \
   --dataset.root=${DATASET_DIR} \
   --dataset.joint_only_dim=14 \
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

Recommended Cobot Magic run: pretrained `lerobot/smolvla_base`, 14D joint-only state/action, full unfrozen fine-tuning, 50k steps.

```bash
cd /path/to/lerobot
export DATASET_DIR=/path/to/cobot_magic_sber_v3_0
mkdir -p logs_smolvla/stdout logs_smolvla/outputs

tmux new -d -s smolvla_cobot_full_state14_50k \
  "cd $PWD && \
   unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy && \
   CUDA_VISIBLE_DEVICES=0,1 \
   .venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 \
   -m lerobot.scripts.lerobot_train \
   --policy.path=lerobot/smolvla_base \
   --policy.push_to_hub=false \
   --dataset.repo_id=cobot_magic_sber \
   --dataset.root=${DATASET_DIR} \
   --dataset.joint_only_dim=14 \
   --rename_map='{"observation.images.camera_0":"observation.images.camera1","observation.images.camera_1":"observation.images.camera2","observation.images.camera_2":"observation.images.camera3"}' \
   --batch_size=8 \
   --steps=50000 \
   --save_freq=5000 \
   --log_freq=100 \
   --num_workers=4 \
   --prefetch_factor=2 \
   --output_dir=logs_smolvla/outputs/cobot_magic_smolvla_2gpu_full_unfrozen_state14_50k \
   --job_name=cobot_magic_smolvla_2gpu_full_unfrozen_state14_50k \
   --policy.device=cuda \
   --policy.freeze_vision_encoder=false \
   --policy.train_expert_only=false \
   --policy.load_vlm_weights=true \
   --policy.optimizer_lr=5e-5 \
   --policy.scheduler_warmup_steps=1000 \
   --policy.scheduler_decay_steps=50000 \
   --wandb.enable=false \
   2>&1 | tee logs_smolvla/stdout/cobot_magic_smolvla_2gpu_full_unfrozen_state14_50k.log"
```

Watch:

```bash
tail -f logs_smolvla/stdout/cobot_magic_smolvla_2gpu_full_unfrozen_state14_50k.log
```

## Inference: ZeroMQ Server

The robot client uses a ZeroMQ `REQ` socket and expects a server-side `REP` socket. The SmolVLA adapter listens on `0.0.0.0:5055`, receives 3 JPEG-base64 cameras plus a 14D joint proprio vector, and returns absolute joint actions with shape `[num_actions, 14]`.

Checkpoint layout expected by LeRobot:

```text
logs_smolvla/outputs/cobot_magic_smolvla_2gpu_50k/checkpoints/050000/
  pretrained_model/
    config.json
    model.safetensors
    policy_preprocessor.json
    policy_postprocessor.json
    train_config.json
  training_state/
```

Run from the LeRobot repo root:

```bash
cd /path/to/lerobot
export SMOLVLA_CHECKPOINT=/path/to/checkpoints/050000

CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python src/lerobot/scripts/inference/cobot_smolvla_zmq.py \
  --checkpoint_path ${SMOLVLA_CHECKPOINT} \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5055
```

You may pass either the checkpoint directory or its `pretrained_model` subdirectory:

```text
/path/to/checkpoints/050000
/path/to/checkpoints/050000/pretrained_model
```

Use `--relative_actions` only if a checkpoint returns delta actions. The Cobot Magic joint-only training setup is intended to learn absolute joint targets.

## Tmux

```bash
tmux ls
tmux attach -t smolvla_cobot_2gpu_50k
# detach without stopping: Ctrl-b, then d
```
