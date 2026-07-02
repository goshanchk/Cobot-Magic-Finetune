# SO101 Quickstart

This guide runs the shortest current SO101 path: install, prepare the local dataset path, train, and start the server.

## 1. Install

```bash
cd tutorials/Being-H-EDU
bash scripts/setup_beingh_env.sh .venv
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
```

## 2. Prepare Paths

Prepare these local paths:

```text
/path/to/InternVL3_5-2B
/path/to/Qwen3-0.6B
/path/to/being-h05-checkpoint
$SO101_DATASET
```

The SO101 dataset is published at [BeingBeyond/Being-H-EDU_SO101](https://huggingface.co/datasets/BeingBeyond/Being-H-EDU_SO101). Follow [so101_data_processing.md](so101_data_processing.md) to download it, select the `pick_cube_plate_trimmed` dataset root, and run the one-time action delta conversion.

## 3. Configure Dataset

Create a local config from `configs/posttrain/so101/so101_example.yaml`:

```bash
python -c 'from pathlib import Path; import os; src=Path("configs/posttrain/so101/so101_example.yaml"); dst=Path("configs/posttrain/so101/so101_local.yaml"); dst.write_text(src.read_text().replace("/path/to/datasets/Being-H-EDU_SO101/pick_cube_plate_trimmed", os.environ["SO101_DATASET"]))'
```

It should contain:

```yaml
so101_posttrain:
  dataset_names:
  - so101.pick_cube_plate
  dataset_path_overrides:
    so101.pick_cube_plate: /local/path/to/Being-H-EDU_SO101/pick_cube_plate_trimmed
```

## 4. Train

```bash
PRETRAIN_MODEL=/path/to/InternVL3_5-2B \
EXPERT_MODEL=/path/to/Qwen3-0.6B \
RESUME_PATH=/path/to/being-h05-checkpoint \
bash examples/so101/train_so101_beingh.sh \
  --dataset-config configs/posttrain/so101/so101_local.yaml \
  --gpus 4 \
  --steps 20000
```

Outputs are written under `outputs/` by default.

## 5. Start Server

```bash
MODEL_PATH=/path/to/so101-checkpoint \
bash examples/so101/run_server_so101.sh
```
