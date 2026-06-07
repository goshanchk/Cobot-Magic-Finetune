from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


cobot_magic_config = {
    # Keys must match meta/modality.json under "video".
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
    ),
    # 26D state split in meta/modality.json:
    # all_arms=14D, left_eef=6D xyz+rpy, right_eef=6D xyz+rpy.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["all_arms", "left_eef", "right_eef"],
    ),
    # Start with ABSOLUTE/NON_EEF for all groups because the dataset stores
    # action[t] = observation.state[t+1], and EEF rotations are euler_xyz rather
    # than GR00T's XYZ_ROT6D / XYZ_ROTVEC EEF formats.
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["all_arms", "left_eef", "right_eef"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(cobot_magic_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
