# Data Configuration

SO101 data is configured in two places:

- `configs/dataset_info.py`: dataset keys and default paths.
- `configs/data_config.py`: modality definitions and action/state mapping.

## Dataset Registry

The default registry is:

```python
DATASET_INFO = {
    "so101_posttrain": {
        "so101.pick_cube_plate": {
            "dataset_path": "/path/to/datasets/Being-H-EDU_SO101/pick_cube_plate_trimmed",
            "embodiment": "SO101",
            "embodiment_tag": "so101",
            "subtask": "so101.pick_cube_plate",
        },
    },
}
```

For public use, prefer YAML overrides instead of editing Python:

```yaml
so101_posttrain:
  dataset_names:
  - so101.pick_cube_plate
  dataset_path_overrides:
    so101.pick_cube_plate: /path/to/datasets/Being-H-EDU_SO101/pick_cube_plate_trimmed
```

## SO101 Data Config

`SO101DataConfig` maps:

- `observation.images.external` to `video.front_view`
- `observation.images.wrist` to `video.wrist_view`
- `observation.state[0:6]` to `state.joint_positions`
- `action[0:5]` to `action.joint_positions`
- `action[5:6]` to `action.gripper`

The first five action dimensions are expected to be joint deltas. The gripper dimension remains absolute.

## Multiple Tasks

The current SO101 example uses the public dataset key `so101.pick_cube_plate`. Users with multiple tasks can add task-specific keys:

```python
DATASET_INFO["so101_posttrain"]["so101.my_task"] = {
    "dataset_path": "/path/to/datasets/so101/my_task",
    "embodiment": "SO101",
    "embodiment_tag": "so101",
    "subtask": "so101.my_task",
}
```

Then use the same key in `dataset_names` and `dataset_path_overrides`.
