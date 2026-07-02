# SO101 Data Processing

Being-H-EDU is an educational tutorial workspace. This page documents the current SO101 data path and how to connect the Hugging Face dataset to the training config.

## Public Dataset

The SO101 dataset is published at [BeingBeyond/Being-H-EDU_SO101](https://huggingface.co/datasets/BeingBeyond/Being-H-EDU_SO101). It contains LeRobot-format datasets for the `pick_cube_plate` manipulation task.

The task instruction is:

```text
Pick the cube into the plate.
```

The dataset can be used with Being-H-EDU post-training, and it is also a standard LeRobot-style robot dataset that can be loaded by other imitation-learning or robot-policy pipelines that support the same schema.

## Dataset Variants

The Hugging Face dataset repository contains these variants:

| Directory | LeRobot version | Episodes | Frames | Intended use |
| --- | ---: | ---: | ---: | --- |
| `pick_cube_plate` | v2.1 | 210 | 51,987 | Raw merged dataset before quality filtering. Use for auditing or custom filtering. |
| `pick_cube_plate_filtered` | v2.1 | 189 | 45,001 | Quality-filtered dataset with short or likely incomplete episodes removed. |
| `pick_cube_plate_trimmed` | v2.1 | 189 | 35,006 | Recommended v2.1 Being-H-EDU training set. It keeps the filtered episodes and trims static or uninformative frames. |
| `pick_cube_plate_v3.0` | v3.0 | 210 | 51,987 | Same raw task data packaged in the LeRobot v3.0 layout for newer LeRobot tooling. |

For Being-H-EDU post-training, start with `pick_cube_plate_trimmed` unless you specifically need raw trajectories or want to reproduce preprocessing.

## Download Location

Download the dataset to a writable local directory:

```bash
cd tutorials/Being-H-EDU

export DATA_ROOT=${DATA_ROOT:-$HOME/datasets}
mkdir -p "$DATA_ROOT"

huggingface-cli download BeingBeyond/Being-H-EDU_SO101 \
  --repo-type dataset \
  --local-dir "$DATA_ROOT/Being-H-EDU_SO101"
```

Point training to the recommended v2.1 dataset variant:

```bash
export SO101_DATASET="$DATA_ROOT/Being-H-EDU_SO101/pick_cube_plate_trimmed"
```

Validate that the selected path is a LeRobot dataset root:

```bash
test -f "$SO101_DATASET/meta/info.json"
test -d "$SO101_DATASET/data"
test -d "$SO101_DATASET/videos"
```

## Expected Layout

The dataset root used by `SO101_DATASET` should look like:

```text
$SO101_DATASET/
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl
    stats.json
  data/
    chunk-000/
      episode_000000.parquet
      ...
  videos/
    chunk-000/
      observation.images.external/
        episode_000000.mp4
        ...
      observation.images.wrist/
        episode_000000.mp4
        ...
```

The recommended `pick_cube_plate_trimmed` variant has these properties:

- LeRobot `codebase_version`: `v2.1`
- `robot_type`: `so_follower`
- task: `Pick the cube into the plate.`
- episodes: `189`
- frames: `35006`
- fps: `30`
- cameras: `observation.images.external` at `480x640`, `observation.images.wrist` at `240x320`

## Expected Columns

The training config expects these LeRobot columns:

- `observation.state`: six SO101 joint/gripper positions.
- `action`: six dimensions. The first five joint dimensions must be deltas relative to `observation.state`; the final gripper dimension stays absolute.
- `observation.images.external`: external camera video.
- `observation.images.wrist`: wrist camera video.
- `task_index`: task/instruction index used by the LeRobot metadata.

## Convert Absolute Actions To Delta

The public SO101 dataset stores actions as standard robot action targets. `configs/data_config.py` expects the first five action dimensions to be joint deltas relative to `observation.state`, so run this conversion once on your writable local copy:

```bash
python examples/so101/convert_so101_actions_to_delta.py "$SO101_DATASET"
```

The converter updates parquet files in place, recomputes `meta/stats.json` and `meta/episodes_stats.jsonl`, and writes:

```text
$SO101_DATASET/meta/action_delta_conversion.json
$SO101_DATASET/meta/stats.before_action_delta.json
$SO101_DATASET/meta/episodes_stats.before_action_delta.jsonl
```

If the conversion marker already exists, the script refuses to run again. Use `--force` only when you intentionally want to redo the conversion from a fresh absolute-action copy.

## Configure Training

Create a local training YAML that points `so101.pick_cube_plate` to the converted dataset path:

```bash
python -c 'from pathlib import Path; import os; src=Path("configs/posttrain/so101/so101_example.yaml"); dst=Path("configs/posttrain/so101/so101_local.yaml"); dst.write_text(src.read_text().replace("/path/to/datasets/Being-H-EDU_SO101/pick_cube_plate_trimmed", os.environ["SO101_DATASET"]))'
```

The resulting config should contain:

```yaml
so101_posttrain:
  dataset_names:
  - so101.pick_cube_plate

  dataset_path_overrides:
    so101.pick_cube_plate: /local/path/to/Being-H-EDU_SO101/pick_cube_plate_trimmed

  data_config_names:
  - "so101"

  embodiment_tags:
  - "so101"
```

Keep the default key `so101.pick_cube_plate` unless you are registering multiple SO101 datasets. It matches `configs/dataset_info.py`, the Hugging Face dataset README, and the inference metadata variant used by this tutorial.

## Launch Training

Use the local YAML with the existing launcher:

```bash
PRETRAIN_MODEL=/path/to/InternVL3_5-2B \
EXPERT_MODEL=/path/to/Qwen3-0.6B \
RESUME_PATH=/path/to/being-h05-checkpoint \
bash examples/so101/train_so101_beingh.sh \
  --dataset-config configs/posttrain/so101/so101_local.yaml \
  --gpus 4 \
  --steps 20000
```

See [training.md](training.md) for other launcher options.

## Optional Processing For Your Own SO101 Data

If you collect or convert your own SO101 data, the expected flow is:

```text
raw or converted LeRobot sessions
  -> convert to LeRobot v2.1 if needed
  -> merge sessions
  -> filter short or invalid episodes
  -> trim static frames
  -> convert absolute actions to deltas
  -> train with configs/posttrain/so101/so101_example.yaml or so101_local.yaml
```

This repository keeps only the SO101-specific helper scripts:

```text
tools/so101_processing/
  PIPELINE.md
  merge_datasets.py
  trim_static.py
  analyze_stillness.py
```

Clone external tools such as LeRobot or any4lerobot separately when raw data still needs format conversion or filtering.

### Merge Multiple Sessions

```bash
python tools/so101_processing/merge_datasets.py \
  --srcs /path/to/session_001 /path/to/session_002 /path/to/session_003 \
  --dst /path/to/so101_merged
```

### Trim Static Frames

Analyze first:

```bash
python tools/so101_processing/analyze_stillness.py \
  --root /path/to/so101_merged \
  --threshold 0.5
```

Then trim:

```bash
python tools/so101_processing/trim_static.py \
  --src /path/to/so101_merged \
  --dst /path/to/so101_trimmed \
  --threshold 0.5 \
  --buffer 3
```

`--threshold` is measured from action changes between frames. Start with `0.5` and inspect the resulting episodes before training.

## Validation Checklist

Before launching training, check:

1. `meta/info.json` reports LeRobot `codebase_version` `v2.1`.
2. `meta/tasks.jsonl` contains the expected instruction.
3. `meta/action_delta_conversion.json` exists after the local action delta conversion.
4. `data/chunk-000/*.parquet` exists and includes `action`, `observation.state`, `timestamp`, `frame_index`, `episode_index`, `index`, and `task_index`.
5. `videos/chunk-000/observation.images.external/` and `videos/chunk-000/observation.images.wrist/` contain matching episode videos.
