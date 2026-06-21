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
import io
import json
import logging
from pathlib import Path
import traceback
from typing import Any

import numpy as np
from PIL import Image
import torch
import zmq

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


CAMERA_FIELD_TO_DATASET_KEY = {
    "images_camera_0": "observation.images.camera_0",  # right wrist
    "images_camera_1": "observation.images.camera_1",  # left wrist
    "images_camera_2": "observation.images.camera_2",  # front/high
}


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
        self.policy = SmolVLAPolicy.from_pretrained(
            self.model_dir,
            device=args.device,
            local_files_only=args.local_files_only,
            strict=args.strict,
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
        print(f"Loaded SmolVLA checkpoint: {self.model_dir}")
        print(f"Image features: {self.image_feature_keys}")
        print(f"Action feature: {self.policy.config.action_feature}")

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
        if self.args.relative_actions:
            actions = proprio[-1][None, :] + actions
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--max_actions", type=int, default=0, help="If >0, truncate returned action chunk to this many actions.")
    parser.add_argument("--absolute_actions", action="store_true", help="Use only for old checkpoints that already output absolute joint targets.")
    parser.add_argument("--local_files_only", action="store_true", help="Do not download missing files from HuggingFace Hub.")
    parser.add_argument("--strict", action="store_true", help="Use strict safetensors loading.")
    parser.add_argument("--recv_timeout_ms", type=int, default=10000)
    parser.add_argument("--send_timeout_ms", type=int, default=10000)
    args = parser.parse_args()
    args.relative_actions = not args.absolute_actions
    return args


def main() -> None:
    CobotSmolVLAZMQServer(parse_args()).run()


if __name__ == "__main__":
    main()
