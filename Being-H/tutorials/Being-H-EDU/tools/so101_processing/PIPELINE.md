# SO101 Data Processing Pipeline

This folder contains small utilities for preparing SO101 LeRobot datasets before post-training.

## Typical Flow

```text
raw or converted LeRobot sessions
  -> merge sessions
  -> inspect short or static episodes
  -> trim static frames
  -> convert absolute actions to deltas
  -> train with configs/posttrain/so101/so101_example.yaml
```

## Merge Sessions

```bash
python tools/so101_processing/merge_datasets.py \
  --srcs /path/to/session_001 /path/to/session_002 \
  --dst /path/to/so101_merged
```

## Analyze Static Frames

```bash
python tools/so101_processing/analyze_stillness.py \
  --root /path/to/so101_merged \
  --threshold 0.5
```

## Trim Static Frames

```bash
python tools/so101_processing/trim_static.py \
  --src /path/to/so101_merged \
  --dst /path/to/so101_trimmed \
  --threshold 0.5 \
  --buffer 3
```

## Convert Absolute Actions To Deltas

```bash
python examples/so101/convert_so101_actions_to_delta.py /path/to/so101_trimmed
```

## Notes

- `--threshold` is action-delta magnitude per frame.
- `--buffer` keeps a few frames before and after motion starts or stops.
- The training config expects the first five SO101 action dimensions to be deltas and the gripper action to stay absolute.
- This repository does not vendor LeRobot or any4lerobot. Use external conversion tools separately when raw data is not yet in LeRobot format.
