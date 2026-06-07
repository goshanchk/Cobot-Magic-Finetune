"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from collections import OrderedDict
from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Type

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            wrist_keys = sorted(k for k in rlds_batch["observation"].keys() if "wrist" in k)
            for k in wrist_keys:
                img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                pixel_values_wrist = self.image_transform(img_wrist)
                all_wrist_pixels.append(pixel_values_wrist)
            if all_wrist_pixels:
                return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict


CAMERA_KEY_MAP = {
    "primary": "observation.images.camera_2",
    "left_wrist": "observation.images.camera_1",
    "right_wrist": "observation.images.camera_0",
}


def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _episode_parquet_path(dataset_dir: Path, info: dict, episode_index: int) -> Path:
    chunk = _episode_chunk(episode_index, info["chunks_size"])
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    return dataset_dir / rel


def _episode_video_path(dataset_dir: Path, info: dict, episode_index: int, video_key: str) -> Path:
    chunk = _episode_chunk(episode_index, info["chunks_size"])
    rel = info["video_path"].format(episode_chunk=chunk, episode_index=episode_index, video_key=video_key)
    return dataset_dir / rel


def _read_video_frames(path: Path) -> np.ndarray:
    frames = [np.asarray(frame, dtype=np.uint8) for frame in iio.imiter(path)]
    if not frames:
        raise ValueError(f"Video contains no frames: {path}")
    return np.stack(frames, axis=0)


def _bounds_normalize(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    denom = np.maximum(high - low, 1e-6)
    return (2.0 * (values - low) / denom - 1.0).astype(np.float32)


def _stats_dict(values: np.ndarray) -> Dict[str, Any]:
    return {
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "max": values.max(0).tolist(),
        "min": values.min(0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _make_lerobot_splits(
    dataset_dir: Path,
    train: bool,
    val_episodes_target: int = 300,
    split_seed: int = 42,
) -> List[int]:
    episodes = _load_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    validation = set(_load_json(dataset_dir / "meta" / "validation_episodes.json"))
    kept = [int(row["episode_index"]) for row in episodes]
    train_indices = [idx for idx in kept if idx not in validation]
    val_indices = [idx for idx in kept if idx in validation]

    if val_episodes_target > 0 and len(val_indices) < val_episodes_target:
        needed = min(val_episodes_target - len(val_indices), len(train_indices))
        extra_val = set(random.Random(split_seed).sample(train_indices, needed))
        val_indices = sorted(val_indices + list(extra_val))
        train_indices = [idx for idx in train_indices if idx not in extra_val]

    return train_indices if train else val_indices


@dataclass
class LeRobotBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        current_action = sample["actions"][0]
        actions = sample["actions"]
        img = Image.fromarray(sample["image"])
        lang = sample["language_instruction"].lower()

        prompt_builder = self.prompt_builder_fn("openvla")
        future_actions_string = "".join(self.action_tokenizer(actions[1:]))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": sample["dataset_name"],
            "actions": actions,
        }

        if self.use_wrist_image:
            wrist_pixels = []
            for key in ("left_wrist_image", "right_wrist_image"):
                if key in sample:
                    wrist_pixels.append(self.image_transform(Image.fromarray(sample[key])))
            if wrist_pixels:
                return_dict["pixel_values_wrist"] = torch.cat(wrist_pixels, dim=0)
        if self.use_proprio:
            return_dict["proprio"] = sample["proprio"]

        return return_dict


class LeRobotDataset(IterableDataset):
    """Direct PyTorch loader for the local Cobot Magic Sber LeRobot-like dataset."""

    def __init__(
        self,
        data_root_dir: Path,
        dataset_name: str,
        batch_transform: LeRobotBatchTransform,
        train: bool = True,
        num_images_in_input: int = 1,
        val_episodes_target: int = 300,
        split_seed: int = 42,
        episode_cache_size: int = 2,
        normalization_statistics: Dict[str, Any] | None = None,
    ) -> None:
        if dataset_name != "cobot_magic_sber":
            raise ValueError(f"LeRobotDataset currently supports only cobot_magic_sber, got {dataset_name}")
        if not 1 <= num_images_in_input <= 3:
            raise ValueError(f"num_images_in_input must be 1..3 for {dataset_name}, got {num_images_in_input}")

        self.data_root_dir = Path(data_root_dir)
        self.dataset_name = dataset_name
        self.batch_transform = batch_transform
        self.train = train
        self.num_images_in_input = num_images_in_input
        self.info = _load_json(self.data_root_dir / "meta" / "info.json")
        self.tasks = {row["task_index"]: row["task"] for row in _load_jsonl(self.data_root_dir / "meta" / "tasks.jsonl")}
        self.episode_indices = _make_lerobot_splits(self.data_root_dir, train, val_episodes_target, split_seed)
        self._episode_cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()
        self.episode_cache_size = episode_cache_size

        self.transition_indices: List[Tuple[int, int]] = []
        actions_for_stats, proprios_for_stats = [], []
        for episode_index in self.episode_indices:
            table = pq.read_table(_episode_parquet_path(self.data_root_dir, self.info, episode_index))
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                raise ValueError(f"Expected action shape [T, {ACTION_DIM}], got {actions.shape} for episode {episode_index}")
            if states.ndim != 2 or states.shape[1] != PROPRIO_DIM:
                raise ValueError(f"Expected state shape [T, {PROPRIO_DIM}], got {states.shape} for episode {episode_index}")
            actions_for_stats.append(actions)
            proprios_for_stats.append(states)
            for t in range(actions.shape[0]):
                self.transition_indices.append((episode_index, t))

        all_actions = np.concatenate(actions_for_stats, axis=0)
        all_proprios = np.concatenate(proprios_for_stats, axis=0)
        if normalization_statistics is None:
            self.dataset_statistics = {
                dataset_name: {
                    "action": _stats_dict(all_actions),
                    "proprio": _stats_dict(all_proprios),
                    "num_transitions": len(self.transition_indices),
                    "num_trajectories": len(self.episode_indices),
                }
            }
        else:
            self.dataset_statistics = normalization_statistics
        stats = self.dataset_statistics[dataset_name]
        self.action_low = np.asarray(stats["action"]["min"], dtype=np.float32)
        self.action_high = np.asarray(stats["action"]["max"], dtype=np.float32)
        self.proprio_low = np.asarray(stats["proprio"]["min"], dtype=np.float32)
        self.proprio_high = np.asarray(stats["proprio"]["max"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.transition_indices)

    def _load_episode(self, episode_index: int) -> Dict[str, Any]:
        if episode_index in self._episode_cache:
            self._episode_cache.move_to_end(episode_index)
            return self._episode_cache[episode_index]

        table = pq.read_table(_episode_parquet_path(self.data_root_dir, self.info, episode_index))
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        episode = {"state": states, "action": actions, "task_index": task_indices}

        camera_views = ("primary", "left_wrist", "right_wrist")[: self.num_images_in_input]
        for view in camera_views:
            frames = _read_video_frames(_episode_video_path(self.data_root_dir, self.info, episode_index, CAMERA_KEY_MAP[view]))
            if frames.shape[0] != states.shape[0]:
                raise ValueError(
                    f"Frame count mismatch for episode {episode_index}, {view}: "
                    f"video={frames.shape[0]}, parquet={states.shape[0]}"
                )
            episode[view] = frames

        self._episode_cache[episode_index] = episode
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return episode

    def _make_sample(self, episode_index: int, t: int) -> Dict[str, Any]:
        episode = self._load_episode(episode_index)
        actions = episode["action"]
        states = episode["state"]
        end = min(t + NUM_ACTIONS_CHUNK, actions.shape[0])
        action_chunk = actions[t:end]
        if action_chunk.shape[0] < NUM_ACTIONS_CHUNK:
            pad = np.repeat(action_chunk[-1:], NUM_ACTIONS_CHUNK - action_chunk.shape[0], axis=0)
            action_chunk = np.concatenate([action_chunk, pad], axis=0)
        action_chunk = _bounds_normalize(action_chunk, self.action_low, self.action_high)
        proprio = _bounds_normalize(states[t], self.proprio_low, self.proprio_high)
        task_index = int(episode["task_index"][t])

        sample = {
            "dataset_name": self.dataset_name,
            "language_instruction": self.tasks[task_index],
            "image": episode["primary"][t],
            "actions": action_chunk,
            "proprio": proprio,
        }
        if self.num_images_in_input >= 2:
            sample["left_wrist_image"] = episode["left_wrist"][t]
        if self.num_images_in_input >= 3:
            sample["right_wrist_image"] = episode["right_wrist"][t]
        return sample

    def __iter__(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank, world_size = 0, 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            rank = rank * worker_info.num_workers + worker_info.id
            world_size *= worker_info.num_workers

        epoch = 0
        while True:
            indices = list(range(len(self.transition_indices)))
            if self.train:
                rng = random.Random(17 + epoch)
                rng.shuffle(indices)
            indices = indices[rank::world_size]
            for idx in indices:
                episode_index, t = self.transition_indices[idx]
                yield self.batch_transform(self._make_sample(episode_index, t))
            if not self.train:
                break
            epoch += 1


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        num_images_in_input: int = 1,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights

        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix or "cobot_magic" in self.data_mix:
            cobot_camera_views = ("primary", "left_wrist", "right_wrist")
            if not 1 <= num_images_in_input <= len(cobot_camera_views):
                raise ValueError(
                    f"num_images_in_input must be 1..{len(cobot_camera_views)} for {self.data_mix}, "
                    f"got {num_images_in_input}"
                )
            load_camera_views = cobot_camera_views[:num_images_in_input]
        else:
            default_camera_views = ("primary", "wrist")
            if not 1 <= num_images_in_input <= len(default_camera_views):
                raise ValueError(
                    f"num_images_in_input must be 1..{len(default_camera_views)} for {self.data_mix}, "
                    f"got {num_images_in_input}"
                )
            load_camera_views = default_camera_views[:num_images_in_input]

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        from prismatic.vla.datasets.rlds import make_interleaved_dataset

        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        from prismatic.vla.datasets.rlds import make_single_dataset

        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
