#!/usr/bin/env python
"""ZeroMQ JSON server adapter for Cobot Magic SmolVLA inference.

Protocol:
- REP socket bound to tcp://0.0.0.0:<port>
- receives a JSON string with base64 JPEG images, instruction, and proprio
- returns {"actions": [[...14 floats...], ...]} with absolute joint targets
  New Cobot Magic SmolVLA checkpoints predict relative deltas internally; this server converts them to absolute.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
from pathlib import Path
import traceback
from typing import Any

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
import zmq

from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


CAMERA_FIELD_TO_DATASET_KEY = {
    "images_camera_0": "observation.images.camera_0",  # right wrist
    "images_camera_1": "observation.images.camera_1",  # left wrist
    "images_camera_2": "observation.images.camera_2",  # front/high
}

FULL_MODEL_FILENAME = "model.safetensors"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
ADAPTER_MODEL_FILENAME = "adapter_model.safetensors"


def decode_jpeg_b64(value: str) -> np.ndarray:
    raw = base64.b64decode(value)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(image, dtype=np.uint8)


def image_hwc_to_chw_float(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image shape [H, W, 3], got {image.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).to(torch.float32)
    return tensor / 255.0


def latest_decoded_image(request: dict[str, Any], field: str) -> torch.Tensor:
    encoded = request[field]
    if not encoded:
        raise ValueError(f"Request field `{field}` must contain at least one image")
    return image_hwc_to_chw_float(decode_jpeg_b64(encoded[-1]))


def resolve_pretrained_model_dir(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()
    if (path / "pretrained_model").is_dir():
        return path / "pretrained_model"
    return path


def load_smolvla_policy(
    model_dir: Path,
    *,
    device: str,
    strict: bool,
    local_files_only: bool,
    base_model_path: str | None = None,
) -> SmolVLAPolicy:
    """Load either a full SmolVLA checkpoint or a PEFT adapter checkpoint.

    LeRobot saves PEFT checkpoints without ``model.safetensors``. For inference,
    reconstruct the base policy with the fine-tuned policy config, load the
    adapter, and merge its LoRA weights into the base model in memory.
    """
    full_model_path = model_dir / FULL_MODEL_FILENAME
    adapter_config_path = model_dir / ADAPTER_CONFIG_FILENAME
    adapter_model_path = model_dir / ADAPTER_MODEL_FILENAME

    if full_model_path.is_file():
        policy_config = PreTrainedConfig.from_pretrained(model_dir, local_files_only=True)
        policy_config.device = device
        return SmolVLAPolicy.from_pretrained(
            model_dir,
            config=policy_config,
            local_files_only=local_files_only,
            strict=strict,
        )

    adapter_files = (adapter_config_path, adapter_model_path)
    if not all(path.is_file() for path in adapter_files):
        missing = [path.name for path in adapter_files if not path.is_file()]
        raise FileNotFoundError(
            f"Checkpoint {model_dir} contains neither {FULL_MODEL_FILENAME} nor a complete PEFT adapter. "
            f"Missing adapter files: {', '.join(missing)}"
        )

    try:
        from peft import PeftConfig, PeftModel
    except ImportError as exc:
        raise ImportError(
            "This is a LoRA/PEFT checkpoint, but `peft` is not installed. "
            "Install the SmolVLA extra with `uv pip install -e '.[smolvla]'`."
        ) from exc

    # The checkpoint's config contains the trained 14D state/action schema and
    # must override the schema stored in the generic SmolVLA base checkpoint.
    policy_config = PreTrainedConfig.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    if policy_config.type != "smolvla":
        raise ValueError(f"Expected a SmolVLA checkpoint, got policy type {policy_config.type!r}")
    policy_config.device = device

    peft_config = PeftConfig.from_pretrained(str(model_dir), local_files_only=True)
    resolved_base_model = base_model_path or peft_config.base_model_name_or_path
    if not resolved_base_model:
        raise ValueError(
            "adapter_config.json does not contain `base_model_name_or_path`. "
            "Pass the base checkpoint explicitly with --base_model_path."
        )

    print(f"Loading LoRA base model: {resolved_base_model}")
    base_policy = SmolVLAPolicy.from_pretrained(
        resolved_base_model,
        config=policy_config,
        local_files_only=local_files_only,
        strict=strict,
    )
    print(f"Loading LoRA adapter: {model_dir}")
    peft_policy = PeftModel.from_pretrained(
        base_policy,
        str(model_dir),
        config=peft_config,
        is_trainable=False,
    )
    print("Merging LoRA adapter into SmolVLA base model")
    merged_policy = peft_policy.merge_and_unload(safe_merge=True)
    merged_policy.to(device)
    merged_policy.eval()
    return merged_policy


def _camera_tensor_for_feature(feature_key: str, images_by_field: dict[str, torch.Tensor]) -> torch.Tensor:
    lower = feature_key.lower()
    if "camera_0" in lower or "camera1" in lower or "right" in lower:
        return images_by_field["images_camera_0"]
    if "camera_1" in lower or "camera2" in lower or "left" in lower:
        return images_by_field["images_camera_1"]
    if "camera_2" in lower or "camera3" in lower or "front" in lower or "high" in lower or "base" in lower:
        return images_by_field["images_camera_2"]
    raise KeyError(
        f"Cannot map policy image feature `{feature_key}` to client cameras. "
        f"Expected camera_0/camera1, camera_1/camera2, camera_2/camera3, or right/left/front/high/base."
    )


def _as_actions_array(actions: Any) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 in actions [B, N, D], got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected actions shape [N, D] or [1, N, D], got {arr.shape}")
    if arr.shape[1] != 14:
        raise ValueError(f"Expected action dim 14, got {arr.shape[1]}")
    return arr


class CobotSmolVLAZMQServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_dir = resolve_pretrained_model_dir(args.checkpoint_path)
        self.policy = load_smolvla_policy(
            self.model_dir,
            device=args.device,
            strict=args.strict,
            local_files_only=args.local_files_only,
            base_model_path=args.base_model_path,
        )
        self.policy.eval()
        self.policy.reset()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(self.model_dir),
            preprocessor_overrides={"device_processor": {"device": args.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.image_feature_keys = list(self.policy.config.image_features.keys())
        if not self.image_feature_keys:
            raise ValueError("Loaded SmolVLA checkpoint has no image features in config")
        checkpoint_relative_actions = getattr(self.policy.config, "relative_joint_actions", None)
        if args.relative_actions is None:
            if checkpoint_relative_actions is None:
                raise ValueError(
                    "Checkpoint does not record whether actions are relative or absolute. "
                    "Pass exactly one of --relative_actions or --absolute_actions after verifying its training config."
                )
            self.relative_actions = checkpoint_relative_actions
        else:
            self.relative_actions = args.relative_actions
            if (
                checkpoint_relative_actions is not None
                and checkpoint_relative_actions != self.relative_actions
            ):
                raise ValueError(
                    "CLI action mode disagrees with checkpoint config: "
                    f"checkpoint relative_joint_actions={checkpoint_relative_actions}, "
                    f"CLI relative_actions={self.relative_actions}."
                )
        print(f"Loaded SmolVLA checkpoint: {self.model_dir}")
        print(f"Image features: {self.image_feature_keys}")
        print(f"Action feature: {self.policy.config.action_feature}")
        print(f"Action output mode: {'relative deltas' if self.relative_actions else 'absolute targets'}")
        self._validate_processor_statistics()

    def _validate_processor_statistics(self) -> None:
        pre_config_path = self.model_dir / "policy_preprocessor.json"
        post_config_path = self.model_dir / "policy_postprocessor.json"
        if not pre_config_path.is_file() or not post_config_path.is_file():
            raise FileNotFoundError("Checkpoint is missing saved normalization processor configs")

        with pre_config_path.open("r") as f:
            pre_config = json.load(f)
        with post_config_path.open("r") as f:
            post_config = json.load(f)

        def find_step(config: dict[str, Any], registry_name: str) -> dict[str, Any]:
            matches = [step for step in config.get("steps", []) if step.get("registry_name") == registry_name]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one {registry_name!r} step in saved processor, found {len(matches)}"
                )
            return matches[0]

        pre_step = find_step(pre_config, "normalizer_processor")
        post_step = find_step(post_config, "unnormalizer_processor")
        pre_norm_map = pre_step["config"].get("norm_map", {})
        post_norm_map = post_step["config"].get("norm_map", {})
        if pre_norm_map.get("ACTION") != post_norm_map.get("ACTION"):
            raise ValueError(
                "Training normalizer and inference unnormalizer use different ACTION normalization modes"
            )

        policy_norm_map = self.policy.config.normalization_mapping
        for feature_type in ("STATE", "ACTION"):
            policy_mode = policy_norm_map.get(feature_type)
            policy_mode = getattr(policy_mode, "value", policy_mode)
            processor_mode = pre_norm_map.get(feature_type)
            if policy_mode != processor_mode:
                raise ValueError(
                    f"Policy config {feature_type} normalization {policy_mode!r} does not match "
                    f"saved processor mode {processor_mode!r}"
                )

        pre_state_path = self.model_dir / pre_step["state_file"]
        post_state_path = self.model_dir / post_step["state_file"]
        if not pre_state_path.is_file() or not post_state_path.is_file():
            raise FileNotFoundError("Checkpoint is missing saved normalization tensors")
        pre_state = load_file(pre_state_path)
        post_state = load_file(post_state_path)

        for key in ("action.mean", "action.std", "action.min", "action.max"):
            if key not in pre_state or key not in post_state:
                raise KeyError(f"Checkpoint normalization tensors are missing {key!r}")
            if not torch.equal(pre_state[key], post_state[key]):
                raise ValueError(f"Training and inference action statistics differ for {key}")
            if pre_state[key].shape != (14,) or not torch.isfinite(pre_state[key]).all():
                raise ValueError(f"Invalid saved action statistic {key}: shape={tuple(pre_state[key].shape)}")

        for key in (
            "observation.state.mean",
            "observation.state.std",
            "observation.state.min",
            "observation.state.max",
        ):
            if key not in pre_state:
                raise KeyError(f"Checkpoint normalization tensors are missing {key!r}")
            if pre_state[key].shape != (14,) or not torch.isfinite(pre_state[key]).all():
                raise ValueError(f"Invalid saved state statistic {key}: shape={tuple(pre_state[key].shape)}")
        digest = hashlib.sha256()
        for feature in ("action", "observation.state"):
            for stat_name in ("mean", "std", "min", "max"):
                key = f"{feature}.{stat_name}"
                if key not in pre_state:
                    raise KeyError(f"Checkpoint normalization tensors are missing {key!r}")
                values = pre_state[key].to(dtype=torch.float32).contiguous().cpu()
                digest.update(f"{key}\0".encode())
                digest.update(values.numpy().tobytes())
        actual_hash = digest.hexdigest()
        expected_hash = getattr(self.policy.config, "normalization_statistics_sha256", None)
        expected_representation = getattr(
            self.policy.config, "normalization_statistics_representation", None
        )
        required_representation = (
            "relative_chunk_action_and_absolute_state"
            if self.relative_actions
            else "absolute_action_and_absolute_state"
        )
        if expected_representation != required_representation:
            if not self.args.allow_unverified_normalization:
                raise ValueError(
                    "Checkpoint normalization representation is missing or inconsistent: "
                    f"expected {required_representation!r}, got {expected_representation!r}"
                )
            logging.warning("Allowing unverified normalization representation for legacy checkpoint")
        if expected_hash is None:
            if not self.args.allow_unverified_normalization:
                raise ValueError(
                    "Checkpoint has no normalization statistics fingerprint; use a new checkpoint or pass "
                    "--allow_unverified_normalization only after manual verification"
                )
            logging.warning("Allowing legacy checkpoint without normalization statistics fingerprint")
        elif actual_hash != expected_hash:
            raise ValueError(
                f"Saved processor statistics fingerprint {actual_hash} does not match model config {expected_hash}"
            )

        if torch.any(pre_state["action.std"] <= 0) or torch.any(pre_state["observation.state.std"] <= 0):
            raise ValueError("Checkpoint normalization contains non-positive standard deviations")

    def _request_to_batch(self, request: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        if request.get("type") != "inference":
            raise ValueError(f"Unsupported request type: {request.get('type')!r}")

        images_by_field = {
            field: latest_decoded_image(request, field) for field in CAMERA_FIELD_TO_DATASET_KEY
        }
        proprio = np.asarray(request["proprio"], dtype=np.float32)
        if proprio.ndim == 1:
            proprio = proprio[None, :]
        if proprio.ndim != 2 or proprio.shape[1] != 14:
            raise ValueError(f"Expected proprio shape [T, 14], got {proprio.shape}")
        if not np.isfinite(proprio).all():
            raise ValueError("Proprio contains NaN or Inf values")

        batch: dict[str, Any] = {
            OBS_STATE: torch.from_numpy(proprio[-1].copy()).to(torch.float32),
            "task": str(request["instruction"]),
        }
        for feature_key in self.image_feature_keys:
            batch[feature_key] = _camera_tensor_for_feature(feature_key, images_by_field)
        return batch, proprio

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        batch, proprio = self._request_to_batch(request)
        batch = self.preprocessor(batch)
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch)
            actions = self.postprocessor(actions)
        actions = _as_actions_array(actions)
        if self.relative_actions:
            actions = proprio[-1][None, :] + actions
        if not np.isfinite(actions).all():
            raise ValueError("Model actions contain NaN or Inf values")
        if self.args.max_actions > 0:
            actions = actions[: self.args.max_actions]
        return {"actions": actions.astype(float).tolist()}

    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.RCVTIMEO = self.args.recv_timeout_ms
        socket.SNDTIMEO = self.args.send_timeout_ms
        socket.linger = 0
        endpoint = f"tcp://{self.args.host}:{self.args.port}"
        socket.bind(endpoint)
        print(f"Cobot SmolVLA ZMQ server listening on {endpoint}")
        while True:
            try:
                request = json.loads(socket.recv_string())
                reply = self.infer(request)
            except zmq.Again:
                continue
            except KeyboardInterrupt:
                break
            except Exception as exc:  # noqa: BLE001
                logging.error(traceback.format_exc())
                reply = {"error": str(exc)}
            socket.send_string(json.dumps(reply))
        socket.close()
        context.term()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, help="Path to checkpoint or checkpoint/pretrained_model directory.")
    parser.add_argument(
        "--base_model_path",
        default=None,
        help=(
            "Optional local/Hugging Face base SmolVLA checkpoint for a LoRA checkpoint. "
            "By default it is read from adapter_config.json."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--max_actions", type=int, default=0, help="If >0, truncate returned action chunk to this many actions.")
    action_mode = parser.add_mutually_exclusive_group()
    action_mode.add_argument(
        "--relative_actions",
        action="store_true",
        dest="relative_actions",
        help="Predictions are relative joint deltas; add current proprio once.",
    )
    action_mode.add_argument(
        "--absolute_actions",
        action="store_false",
        dest="relative_actions",
        help="Predictions are already absolute joint targets.",
    )
    parser.add_argument(
        "--allow_unverified_normalization",
        action="store_true",
        help="Allow legacy checkpoints without a saved normalization fingerprint after manual verification.",
    )
    parser.add_argument("--local_files_only", action="store_true", help="Do not download missing files from HuggingFace Hub.")
    parser.add_argument("--strict", action="store_true", help="Use strict safetensors loading.")
    parser.add_argument("--recv_timeout_ms", type=int, default=10000)
    parser.add_argument("--send_timeout_ms", type=int, default=10000)
    parser.set_defaults(relative_actions=None)
    return parser.parse_args()


def main() -> None:
    CobotSmolVLAZMQServer(parse_args()).run()


if __name__ == "__main__":
    main()
