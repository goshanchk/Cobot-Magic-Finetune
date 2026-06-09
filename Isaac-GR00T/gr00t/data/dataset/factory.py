# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.experiment.dist_utils import barrier


def _read_validation_episode_ids(path: Path) -> set[int]:
    with path.open("r") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return {int(x["episode_index"] if isinstance(x, dict) else x) for x in payload}
    if isinstance(payload, dict):
        for key in ("validation_episodes", "val", "episodes", "episode_indices"):
            if key in payload:
                return {int(x["episode_index"] if isinstance(x, dict) else x) for x in payload[key]}
    raise ValueError(f"Unsupported validation split format in {path}")


def _load_all_episode_ids(dataset_path: Path) -> list[int]:
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    with episodes_path.open("r") as f:
        return [int(json.loads(line)["episode_index"]) for line in f if line.strip()]


def _load_validation_episode_ids(
    dataset_path: str | Path,
    split_path: str,
    target_count: int = 0,
    split_seed: int = 42,
) -> set[int]:
    dataset_path = Path(dataset_path)
    path = Path(split_path)
    if not path.is_absolute():
        path = dataset_path / path
    if not path.exists():
        raise FileNotFoundError(f"Validation split file not found: {path}")

    val_ids = _read_validation_episode_ids(path)
    if target_count > 0 and len(val_ids) < target_count:
        all_ids = _load_all_episode_ids(dataset_path)
        train_pool = [idx for idx in all_ids if idx not in val_ids]
        needed = min(target_count - len(val_ids), len(train_pool))
        val_ids.update(random.Random(split_seed).sample(train_pool, needed))
    return val_ids


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        assert self.config.training.eval_strategy == "no", (
            "Sharded dataset does not support evaluation sets"
        )

        all_datasets = []
        all_weights = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        generate_stats(dataset_path)
                        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                else:
                    generate_stats(dataset_path)
                    generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                barrier()
                excluded_episode_indices = set()
                if self.config.data.exclude_validation_episodes:
                    excluded_episode_indices = _load_validation_episode_ids(
                        dataset_path,
                        self.config.data.validation_split_path,
                        self.config.data.validation_episodes_target,
                        self.config.data.split_seed,
                    )

                dataset = ShardedSingleStepDataset(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    video_backend=self.config.data.video_backend,
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                    excluded_episode_indices=excluded_episode_indices,
                )
                datasets.append(dataset)
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        return (
            ShardedMixtureDataset(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=True,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            ),
            None,
        )
