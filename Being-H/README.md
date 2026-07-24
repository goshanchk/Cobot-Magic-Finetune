# Being-H Cobot Magic Fine-Tuning

This folder contains the Being-H0.5 integration for the Cobot Magic LeRobot-style dataset. The upstream Being-H README is kept as `README_beign.md`.

## Code Map

```text
Being-H05/configs/data_config.py                         # CobotMagicSberDataConfig: 3 cameras, 14D bimanual joints
Being-H05/configs/dataset_info.py                        # cobot_magic_sber_posttrain registry entry
Being-H05/configs/posttrain/cobot_magic/cobot_magic_sber.yaml
Being-H05/BeingH/dataset/datasets/vla_dataset.py         # relative joint-delta conversion for joint modalities
Being-H05/logs/stdout/                                   # stdout logs from local runs
Being-H05/logs/outputs/                                  # checkpoints and training artifacts
```

## Dataset

Expected local dataset root:

```text
/path/to/cobot_magic_sber
```

The raw dataset stores 26D state/action vectors:

```text
0..6    left arm joints, including q6/gripper-like dimension
7..13   right arm joints, including q6/gripper-like dimension
14..25  FK-derived EEF xyz/rpy auxiliary coordinates
```

Being-H05 training uses only the first 14 joint dimensions. The Cobot mapping into Being-H's 200D unified space is:

```text
raw left_q0..left_q6   -> state/action.left_arm_joint_position -> unified dims 57..63
raw right_q0..right_q6 -> state/action.arm_joint_position      -> unified dims 50..56
```

Camera mapping:

```text
observation.images.camera_0 -> video.cam_right_wrist
observation.images.camera_1 -> video.cam_left_wrist
observation.images.camera_2 -> video.cam_high
```

The dataset action is absolute next joint state: `action[t] = observation.state[t+1]`. The launch script uses `--is_relative True`; for joint modalities the dataloader converts targets to:

```text
action_delta = target_joint - current_joint
```

## Weighted Action Loss

The Cobot script enables extra weighting for gripper-like and late-chunk action dimensions:

```text
ACTION_LOSS_GRIPPER_WEIGHT=4.0
ACTION_LOSS_LATE_CHUNK_START=10
ACTION_LOSS_LATE_CHUNK_WEIGHT=2.0
```

For the Being-H unified action vector, the Cobot gripper-like dimensions are:

```text
right q6 -> unified action dim 56
left q6  -> unified action dim 63
```

The late-chunk weight applies to action timesteps `>= 10` inside the 24-step chunk.

At real-robot inference, `BeingHPolicy.get_action()` converts Cobot joint deltas back to absolute joint targets:

```text
absolute_target = current_qpos + predicted_delta
```

## Environment

Being-H05 uses Python 3.10. From `Being-H/Being-H05`:

```bash
cd /path/to/cobot_magic_finetune/Being-H/Being-H05

micromamba create -n beingh python=3.10 -y
micromamba activate beingh

pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt --no-deps
pip install av

export TMPDIR=$PWD/.tmp
export PIP_CACHE_DIR=$PWD/.pip_cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

pip install flash-attn==2.8.3.post1 \
    --no-build-isolation \
    --no-cache-dir

mkdir -p logs/stdout logs/outputs logs/tensorboard
```

## Checkpoints

The training script expects local checkpoint directories. Download the models from Hugging Face first, then point the variables at those local directories.

Example:

```bash
mkdir -p /path/to/cobot_magic_finetune/Being-H/Being-H05/checkpoint_models/beingh
cd /path/to/cobot_magic_finetune/Being-H/Being-H05/checkpoint_models/beingh

hf auth login
hf download OpenGVLab/InternVL3_5-2B --local-dir InternVL3_5-2B
hf download Qwen/Qwen3-0.6B --local-dir Qwen3-0.6B
hf download BeingBeyond/Being-H05-2B --local-dir Being-H05-2B
```

Then export local paths before launch:

```bash
export PRETRAIN_MODEL=/path/to/cobot_magic_finetune/Being-H/Being-H05/checkpoint_models/beingh/InternVL3_5-2B
export EXPERT_MODEL=/path/to/cobot_magic_finetune/Being-H/Being-H05/checkpoint_models/beingh/Qwen3-0.6B
export RESUME_PATH=/path/to/cobot_magic_finetune/Being-H/Being-H05/checkpoint_models/beingh/Being-H05-2B
```

Export the dataset path before launch. Keep the `cobot_magic_sber_posttrain`
entry in `Being-H05/configs/dataset_info.py` consistent with the same location.

## Full Training

This uses the existing Being-H freeze flags:

```text
--freeze_mllm True       freeze VLM/backbone layers
--freeze_vit_mlp False   keep the vision connector/projector trainable
--use_expert True        train/use the action expert path
--use_flow_matching True train the continuous flow/action head
--use_mpg True           train MPG adaptation/projection modules
USE_LORA=True            train rank-8 adapters while base VLM weights stay frozen
```

Launch:

```bash
cd /path/to/cobot_magic_finetune/Being-H/Being-H05
micromamba activate beingh

export PRETRAIN_MODEL=/path/to/cobot_magic_finetune/Being-H/Being-H05/checkpoint_models/beingh/InternVL3_5-2B
export EXPERT_MODEL=/path/to/cobot_magic_finetune/Being-H/Being-H05/checkpoint_models/beingh/Qwen3-0.6B
export RESUME_PATH=/path/to/cobot_magic_finetune/Being-H/Being-H05/checkpoint_models/beingh/Being-H05-2B
export DATASET_DIR=/path/to/cobot_magic_sber

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_GPUS=8 \
RUN_NAME=cobot_magic_sber_beingh05_8h100 \
TMUX_SESSION=beingh_cobot_magic_sber_8h100 \
USE_LORA=True \
LORA_RANK=8 \
LORA_ALPHA=16 \
LORA_DROPOUT=0.05 \
bash scripts/train/train_cobot_magic_sber_2gpu.sh
```

This starts tmux session `beingh_cobot_magic_sber_8h100`. Attach with:

```bash
tmux attach -t beingh_cobot_magic_sber_8h100
```

## Inference

```bash
cd /path/to/cobot_magic_finetune/Being-H/Being-H05
micromamba activate beingh

export MODEL_PATH=/path/to/cobot_magic_finetune/Being-H/Being-H05/logs/outputs/cobot_magic_sber_beingh05_8h100/checkpoint-final

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
python scripts/inference/cobot_beingh_zmq.py \
  --model_path ${MODEL_PATH} \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5055
```

 For extra safety during first robot tests, add a joint-step guard:

```bash
python scripts/inference/cobot_beingh_zmq.py \
  --model_path ${MODEL_PATH} \
  --device cuda:0 \
  --port 5055 \
  --max_abs_step_delta 0.5
```

Being-H LoRA runs are also saved as full checkpoints rather than adapter-only
directories. At inference, the loader checks the state-dict keys. Checkpoints
with `lora_A`/`lora_B` tensors are reconstructed and merged before the first
forward pass; plain or already-merged checkpoints bypass the LoRA path.
