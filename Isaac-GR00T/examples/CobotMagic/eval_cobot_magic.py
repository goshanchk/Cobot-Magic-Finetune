#!/usr/bin/env python
"""Offline validation for Cobot Magic fine-tuned GR00T checkpoints.

This intentionally stays outside the training loop because GR00T's sharded
training dataset currently disables eval sets. The script loads a checkpoint,
reads validation episode ids from the LeRobot dataset metadata, and computes the
same model loss used during training without optimizer steps.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import get_default_config
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.model import MODEL_REGISTRY


def load_modality_config(modality_config_path: str) -> None:
    path = Path(modality_config_path)
    if not path.exists() or path.suffix != ".py":
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")
    sys.path.append(str(path.parent))
    importlib.import_module(path.stem)
    print(f"Loaded modality config: {path}")


def _read_validation_episode_ids(path: Path) -> list[int]:
    with path.open("r") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [int(x["episode_index"] if isinstance(x, dict) else x) for x in payload]
    if isinstance(payload, dict):
        for key in ("validation_episodes", "val", "episodes", "episode_indices"):
            if key in payload:
                return [int(x["episode_index"] if isinstance(x, dict) else x) for x in payload[key]]
    raise ValueError(f"Unsupported validation split format in {path}")


def _load_all_episode_ids(dataset_path: Path) -> list[int]:
    with (dataset_path / "meta" / "episodes.jsonl").open("r") as f:
        return [int(json.loads(line)["episode_index"]) for line in f if line.strip()]


def load_validation_episode_ids(
    dataset_path: Path,
    split_file: str,
    target_count: int = 0,
    split_seed: int = 42,
) -> list[int]:
    path = Path(split_file)
    if not path.is_absolute():
        path = dataset_path / split_file
    val_ids = set(_read_validation_episode_ids(path))
    if target_count > 0 and len(val_ids) < target_count:
        all_ids = _load_all_episode_ids(dataset_path)
        train_pool = [idx for idx in all_ids if idx not in val_ids]
        needed = min(target_count - len(val_ids), len(train_pool))
        val_ids.update(random.Random(split_seed).sample(train_pool, needed))
    return sorted(val_ids)


def to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(to_device(v, device) for v in batch)
    return batch


def choose_steps(length: int, action_horizon: int, max_steps: int | None, seed: int) -> list[int]:
    effective_length = max(0, length - action_horizon + 1)
    if effective_length <= 0:
        return []
    steps = list(range(effective_length))
    if max_steps is not None and max_steps > 0 and len(steps) > max_steps:
        rng = random.Random(seed)
        steps = sorted(rng.sample(steps, max_steps))
    return steps


def build_config(args: argparse.Namespace, embodiment_tag: str):
    dataset_paths = [str(args.dataset_path)]
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.training.start_from_checkpoint = str(args.checkpoint_path)
    config.training.output_dir = str(args.output_dir)
    config.training.num_gpus = 1
    config.training.use_wandb = False
    config.training.use_tensorboard = False
    config.training.skip_weight_loading = False
    config.training.transformers_local_files_only = args.local_files_only
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = False
    config.model.tune_diffusion_model = False
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.data.shard_size = args.shard_size
    config.data.episode_sampling_rate = 1.0
    config.data.num_shards_per_epoch = 1
    config.data.video_backend = args.video_backend
    config.data.seed = args.seed
    config.validate()
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, type=Path)
    parser.add_argument("--dataset_path", required=True, type=Path)
    parser.add_argument("--embodiment_tag", default="NEW_EMBODIMENT")
    parser.add_argument("--modality_config_path", default="examples/CobotMagic/cobot_magic_config.py")
    parser.add_argument("--validation_split", default="meta/validation_episodes.json")
    parser.add_argument("--validation_episodes_target", type=int, default=0)
    parser.add_argument("--output_dir", default="logs/eval/cobot_magic_eval", type=Path)
    parser.add_argument("--output_json", default=None, type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_episodes", type=int, default=30)
    parser.add_argument("--max_steps_per_episode", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video_backend", default="torchcodec")
    parser.add_argument("--shard_size", type=int, default=1024)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.dataset_path = args.dataset_path.resolve()
    args.checkpoint_path = args.checkpoint_path.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.output_dir / "metrics.json")

    embodiment = EmbodimentTag.resolve(args.embodiment_tag)
    embodiment_tag = embodiment.value
    load_modality_config(args.modality_config_path)
    config = build_config(args, embodiment_tag)

    save_cfg_dir = args.output_dir / "eval_cfg"
    save_cfg_dir.mkdir(parents=True, exist_ok=True)
    pipeline_cls = MODEL_REGISTRY.get(type(config.model))
    pipeline = pipeline_cls(config, save_cfg_dir)
    pipeline.setup()
    model = pipeline.return_model()
    collator = pipeline.return_collator()
    processor = pipeline.return_processor()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device(args.device)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    modality_configs = config.data.modality_configs[embodiment_tag]
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        video_backend=config.data.video_backend,
    )
    episode_to_loader_idx = {
        int(meta["episode_index"]): idx for idx, meta in enumerate(loader.episodes_metadata)
    }
    val_episode_ids = load_validation_episode_ids(
        args.dataset_path, args.validation_split, args.validation_episodes_target, args.seed
    )
    val_episode_ids = [ep for ep in val_episode_ids if ep in episode_to_loader_idx]
    if args.max_episodes and args.max_episodes > 0:
        val_episode_ids = val_episode_ids[: args.max_episodes]
    if not val_episode_ids:
        raise ValueError("No validation episodes selected")

    action_delta_indices = modality_configs["action"].delta_indices
    action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1

    total_loss = 0.0
    total_batches = 0
    total_samples = 0
    per_episode = []
    pending = []
    start = time.time()

    def flush_batch(samples: list[dict]) -> float:
        batch = collator(samples)
        batch = to_device(batch, device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                outputs = model(batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return float(loss.detach().cpu().item())

    progress = tqdm(val_episode_ids, desc="Evaluating episodes")
    for ep_id in progress:
        loader_idx = episode_to_loader_idx[ep_id]
        episode_df = loader[loader_idx]
        steps = choose_steps(
            len(episode_df),
            action_horizon,
            args.max_steps_per_episode,
            args.seed + ep_id,
        )
        ep_losses = []
        for step in steps:
            vla_step = extract_step_data(
                episode_df,
                step,
                modality_configs,
                embodiment,
                allow_padding=config.data.allow_padding,
            )
            pending.append(processor([{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]))
            if len(pending) >= args.batch_size:
                loss = flush_batch(pending)
                total_loss += loss
                total_batches += 1
                total_samples += len(pending)
                ep_losses.append(loss)
                pending = []
                progress.set_postfix(loss=f"{loss:.4f}", avg=f"{total_loss / total_batches:.4f}")
        per_episode.append(
            {
                "episode_index": ep_id,
                "num_steps": len(steps),
                "mean_loss": float(np.mean(ep_losses)) if ep_losses else None,
            }
        )

    if pending:
        loss = flush_batch(pending)
        total_loss += loss
        total_batches += 1
        total_samples += len(pending)

    metrics = {
        "checkpoint_path": str(args.checkpoint_path),
        "dataset_path": str(args.dataset_path),
        "validation_split": str(args.validation_split),
        "validation_episodes_target": args.validation_episodes_target,
        "num_episodes": len(val_episode_ids),
        "num_samples": total_samples,
        "num_batches": total_batches,
        "mean_loss": total_loss / max(total_batches, 1),
        "batch_size": args.batch_size,
        "max_steps_per_episode": args.max_steps_per_episode,
        "action_horizon": action_horizon,
        "elapsed_sec": time.time() - start,
        "per_episode": per_episode,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: v for k, v in metrics.items() if k != "per_episode"}, indent=2))
    print(f"Saved metrics to {output_json}")


if __name__ == "__main__":
    main()
