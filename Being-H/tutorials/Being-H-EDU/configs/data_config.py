# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import random
from abc import ABC, abstractmethod
from typing import Dict, List

from pydantic import BaseModel, Field

from BeingH.dataset.transform.base import ComposedModalityTransform, ModalityTransform
from BeingH.dataset.transform.state_action import StateActionToTensor, StateActionTransform
from BeingH.utils.schema import RotationType


class ModalityConfig(BaseModel):
    """Sampling configuration for one modality group."""

    delta_indices: list[int]
    modality_keys: list[str]


class ModalityDef(BaseModel):
    source_column: str = Field(..., description="Original column name in the Parquet file")
    start: int = Field(..., description="Start dimension index in the column")
    end: int = Field(..., description="End dimension index in the column, exclusive")
    absolute: bool = True
    rotation_type: RotationType | None = Field(None, description="Rotation representation type, if applicable")
    continuous: bool = True


class BaseDataConfig(ABC):
    def __init__(
        self,
        embodiment_tag,
        use_fixed_view,
        max_view_num,
        obs_indices=None,
        action_indices=None,
    ):
        self.embodiment_tag = embodiment_tag
        self.use_fixed_view = use_fixed_view
        self.max_view_num = max_view_num
        self.obs_indices = obs_indices or [0]
        self.action_indices = action_indices or list(range(16))

    @abstractmethod
    def define_modalities(self) -> Dict[str, ModalityDef]:
        pass

    @abstractmethod
    def get_transforms(self) -> ModalityTransform:
        pass

    def get_sampling_indices(self) -> Dict[str, List[int]]:
        sampling_map = {}
        for key in self.VIDEO_KEYS + self.STATE_KEYS:
            sampling_map[key] = self.obs_indices
        for key in self.ACTION_KEYS:
            sampling_map[key] = self.action_indices
        return sampling_map

    def add_video_modality(self, modalities):
        if self.use_fixed_view:
            video_keys = [next(iter(self.VIDEO_SOURCE_COLUMNS))]
        elif self.max_view_num == -1:
            video_keys = list(self.VIDEO_SOURCE_COLUMNS.keys())
        else:
            max_view_num = min(self.max_view_num, len(self.VIDEO_SOURCE_COLUMNS))
            video_keys = random.sample(list(self.VIDEO_SOURCE_COLUMNS.keys()), max_view_num)

        for video_key in video_keys:
            modalities[video_key] = ModalityDef(
                source_column=self.VIDEO_SOURCE_COLUMNS[video_key],
                start=0,
                end=0,
            )
        return modalities


class SO101DataConfig(BaseDataConfig):
    """
    SO101 robot data config for 6-DOF joint-position control.

    Expected LeRobot columns:
      observation.state: [shoulder_pan, shoulder_lift, elbow_flex,
                          wrist_flex, wrist_roll, gripper]
      action:            first 5 dims as joint deltas + final absolute gripper
      observation.images.external: external camera
      observation.images.wrist:    wrist camera
    """

    VIDEO_KEYS = ["video.front_view", "video.wrist_view"]
    VIDEO_SOURCE_COLUMNS = {
        "video.front_view": "observation.images.external",
        "video.wrist_view": "observation.images.wrist",
    }

    STATE_KEYS = ["state.joint_positions"]
    ACTION_KEYS = ["action.joint_positions", "action.gripper"]
    LANGUAGE_KEYS = ["language.instruction"]

    # Map SO101 joints into the right-arm slots in the unified action space.
    UNIFIED_MAPPING = {
        "state.joint_positions": (6, 12),
        "action.joint_positions": (6, 11),
        "action.gripper": (11, 12),
    }

    state_normalization_modes = {
        "state.joint_positions": "min_max",
    }
    action_normalization_modes = {
        "action.joint_positions": "min_max",
        "action.gripper": "min_max",
    }

    def get_feature_meta(self):
        return {
            "state.joint_positions": (
                "6-d absolute joint positions (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)",
                6,
            ),
            "action.joint_positions": (
                "5-d delta joint positions (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)",
                5,
            ),
            "action.gripper": ("1-d absolute gripper position", 1),
        }

    def define_modalities(self) -> Dict[str, ModalityDef]:
        modalities = {
            "language.instruction": ModalityDef(
                source_column="task_index",
                start=0,
                end=0,
            ),
            "state.joint_positions": ModalityDef(
                source_column="observation.state",
                start=0,
                end=6,
                absolute=True,
            ),
            "action.joint_positions": ModalityDef(
                source_column="action",
                start=0,
                end=5,
                absolute=False,
            ),
            "action.gripper": ModalityDef(
                source_column="action",
                start=5,
                end=6,
                absolute=True,
            ),
        }
        return self.add_video_modality(modalities)

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                normalization_modes=self.state_normalization_modes,
            ),
            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                normalization_modes=self.action_normalization_modes,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


DATA_CONFIG_MAP = {
    "so101": SO101DataConfig,
}
