#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import logging
from pprint import pformat

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_PREFIX, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .streaming_dataset import StreamingLeRobotDataset



class _JointOnlyMeta:
    def __init__(self, meta, joint_dim: int):
        self._meta = meta
        self._features = copy.deepcopy(meta.features)
        self._stats = copy.deepcopy(meta.stats)
        self._joint_dim = joint_dim
        self._slice_feature(ACTION)
        self._slice_feature("observation.state")
        self._slice_stats(ACTION)
        self._slice_stats("observation.state")

    def _slice_feature(self, key: str) -> None:
        if key not in self._features:
            return
        feature = self._features[key]
        shape = list(feature.get("shape", []))
        if len(shape) == 1 and shape[0] > self._joint_dim:
            feature["shape"] = [self._joint_dim]
        names = feature.get("names")
        if isinstance(names, list) and len(names) > self._joint_dim:
            feature["names"] = names[: self._joint_dim]

    def _slice_stats(self, key: str) -> None:
        if key not in self._stats:
            return
        for stat_name, value in list(self._stats[key].items()):
            if torch.is_tensor(value) and value.ndim >= 1 and value.shape[-1] > self._joint_dim:
                self._stats[key][stat_name] = value[..., : self._joint_dim]
            elif hasattr(value, "shape") and len(value.shape) >= 1 and value.shape[-1] > self._joint_dim:
                self._stats[key][stat_name] = value[..., : self._joint_dim]
            elif isinstance(value, list) and len(value) > self._joint_dim:
                self._stats[key][stat_name] = value[: self._joint_dim]

    @property
    def features(self):
        return self._features

    @property
    def stats(self):
        return self._stats

    def __getattr__(self, name):
        return getattr(self._meta, name)


class JointOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, joint_dim: int):
        self.dataset = dataset
        self.joint_dim = joint_dim
        self.meta = _JointOnlyMeta(dataset.meta, joint_dim)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        for key in (ACTION, "observation.state"):
            if key in item:
                item[key] = item[key][..., : self.joint_dim]
        return item

    def __getattr__(self, name):
        return getattr(self.dataset, name)

def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    if cfg.dataset.joint_only_dim is not None:
        dataset = JointOnlyDataset(dataset, cfg.dataset.joint_only_dim)

    return dataset
