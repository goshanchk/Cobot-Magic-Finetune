"""ZeroMQ JSON server adapter for Cobot Magic GR00T inference.

Protocol:
- REP socket bound to tcp://0.0.0.0:<port>
- receives a JSON string with base64 JPEG images, instruction, and proprio
- returns {"actions": [[...14 floats...], ...]} with absolute joint targets
"""

from __future__ import annotations

import argparse
import base64
import importlib
import io
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import zmq
from PIL import Image

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


def decode_jpeg_b64(value: str) -> np.ndarray:
    raw = base64.b64decode(value)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(image, dtype=np.uint8)


def load_modality_config(path: str | None) -> None:
    if not path:
        return
    config_path = Path(path)
    if not config_path.exists() or config_path.suffix != ".py":
        raise FileNotFoundError(f"Expected .py modality config, got: {path}")
    sys.path.append(str(config_path.parent))
    importlib.import_module(config_path.stem)
    print(f"Loaded modality config: {config_path}")


def _stack_sequence(images: list[np.ndarray], name: str) -> np.ndarray:
    if not images:
        raise ValueError(f"Request field `{name}` must contain at least one image")
    return np.stack(images, axis=0).astype(np.uint8)[None, ...]  # [B=1, T, H, W, C]


def _as_actions_array(action: Any, key: str = "all_arms") -> np.ndarray:
    if isinstance(action, dict):
        if key in action:
            action = action[key]
        elif f"action.{key}" in action:
            action = action[f"action.{key}"]
        else:
            raise KeyError(f"Cannot find action key `{key}` in action dict keys={list(action.keys())}")
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected actions shape [N, D] or [1, N, D], got {arr.shape}")
    if arr.shape[1] != 14:
        raise ValueError(f"Expected action dim 14, got {arr.shape[1]}")
    return arr


class CobotGR00TZMQServer:
    def __init__(self, args: argparse.Namespace) -> None:
        load_modality_config(args.modality_config_path)
        self.args = args
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
            model_path=args.model_path,
            device=args.device,
            strict=not args.no_strict,
        )
        self.language_key = self.policy.language_key

    def _request_to_observation(self, request: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        if request.get("type") != "inference":
            raise ValueError(f"Unsupported request type: {request.get('type')!r}")

        cam0 = [decode_jpeg_b64(x) for x in request["images_camera_0"]]
        cam1 = [decode_jpeg_b64(x) for x in request["images_camera_1"]]
        cam2 = [decode_jpeg_b64(x) for x in request["images_camera_2"]]
        instruction = str(request["instruction"])
        proprio = np.asarray(request["proprio"], dtype=np.float32)
        if proprio.ndim == 1:
            proprio = proprio[None, :]
        if proprio.ndim != 2 or proprio.shape[1] != 14:
            raise ValueError(f"Expected proprio shape [T, 14], got {proprio.shape}")

        observation = {
            "video": {
                # Dataset mapping: camera_2=front/high, camera_1=left wrist, camera_0=right wrist.
                "cam_high": _stack_sequence(cam2, "images_camera_2"),
                "cam_left_wrist": _stack_sequence(cam1, "images_camera_1"),
                "cam_right_wrist": _stack_sequence(cam0, "images_camera_0"),
            },
            "state": {
                "all_arms": proprio[None, ...],  # [B=1, T, 14]
            },
            "language": {
                self.language_key: [[instruction]],
            },
        }
        return observation, proprio

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        observation, proprio = self._request_to_observation(request)
        action_chunk, _ = self.policy.get_action(observation)
        actions = _as_actions_array(action_chunk, key="all_arms")
        if self.args.relative_actions:
            actions = proprio[-1][None, :] + actions
        return {"actions": actions.astype(float).tolist()}

    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        endpoint = f"tcp://{self.args.host}:{self.args.port}"
        socket.bind(endpoint)
        print(f"Cobot GR00T ZMQ server listening on {endpoint}")
        while True:
            try:
                request = json.loads(socket.recv_string())
                reply = self.infer(request)
            except Exception as exc:  # noqa: BLE001
                logging.error(traceback.format_exc())
                reply = {"error": str(exc)}
            socket.send_string(json.dumps(reply))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--embodiment_tag", default="NEW_EMBODIMENT")
    parser.add_argument("--modality_config_path", default="examples/CobotMagic/cobot_magic_config.py")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--relative_actions", action="store_true", help="Convert model deltas to absolute joint targets before replying.")
    parser.add_argument("--no_strict", action="store_true", help="Disable GR00T policy strict input/output validation.")
    return parser.parse_args()


def main() -> None:
    CobotGR00TZMQServer(parse_args()).run()


if __name__ == "__main__":
    main()
