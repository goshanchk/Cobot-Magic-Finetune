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
    # Use joint-space only. The dataset also exposes FK EEF xyz/rpy groups,
    # but current ALOHA control consumes only the 14 bimanual joint values.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["all_arms"],
    ),
    # action.all_arms stores relative joint deltas during training; inference converts them back to absolute targets.
    "action": ModalityConfig(
        delta_indices=list(range(24)),
        modality_keys=["all_arms"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
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
