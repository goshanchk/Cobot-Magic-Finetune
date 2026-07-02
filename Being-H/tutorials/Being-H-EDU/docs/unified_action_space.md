# Unified Action Space For SO101

Being-H uses a 200-dimensional unified action/state space internally. Being-H-EDU only exposes the SO101 mapping needed by this tutorial.

## SO101 Mapping

| Modality | Source | Unified Range | Meaning |
| --- | --- | --- | --- |
| `state.joint_positions` | `observation.state[0:6]` | `6:12` | six absolute SO101 joint/gripper positions |
| `action.joint_positions` | `action[0:5]` | `6:11` | five delta joint actions |
| `action.gripper` | `action[5:6]` | `11:12` | absolute gripper action |

The mapping is defined in `configs/data_config.py`.

## Action Convention

Before training:

- Convert the first five SO101 action dimensions to deltas.
- Keep the final gripper dimension absolute.
- Keep observation state as absolute joint/gripper positions.

Use `examples/so101/convert_so101_actions_to_delta.py` if your LeRobot dataset stores absolute joint actions.
