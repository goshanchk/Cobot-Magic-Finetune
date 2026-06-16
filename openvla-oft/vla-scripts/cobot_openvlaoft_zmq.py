"""ZeroMQ JSON server adapter for Cobot Magic OpenVLA-OFT inference.

Protocol:
- REP socket bound to tcp://0.0.0.0:<port>
- receives a JSON string with base64 JPEG images, instruction, and proprio
- returns {"actions": [[...14 floats...], ...]} with absolute joint targets
"""

from __future__ import annotations

import base64
import io
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import draccus
import numpy as np
import zmq
from PIL import Image

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM


def decode_jpeg_b64(value: str) -> np.ndarray:
    raw = base64.b64decode(value)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(image, dtype=np.uint8)


def _latest_image(images: list[np.ndarray], name: str) -> np.ndarray:
    if not images:
        raise ValueError(f"Request field `{name}` must contain at least one image")
    return images[-1]


def _as_actions_array(actions: Any) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected actions shape [N, D], got {arr.shape}")
    if arr.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action dim {ACTION_DIM}, got {arr.shape[1]}")
    return arr


class CobotOpenVLAOFTZMQServer:
    def __init__(self, cfg: "OpenVLAOFTZMQConfig") -> None:
        self.cfg = cfg
        self.vla = get_vla(cfg)
        self.processor = get_processor(cfg)

        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)

        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)

        if cfg.unnorm_key:
            assert cfg.unnorm_key in self.vla.norm_stats, (
                f"Action un-norm key {cfg.unnorm_key} not found in VLA norm_stats. "
                f"Available keys: {list(self.vla.norm_stats.keys())}"
            )

    def _request_to_observation(self, request: dict[str, Any]) -> tuple[dict[str, Any], str, np.ndarray]:
        if request.get("type") != "inference":
            raise ValueError(f"Unsupported request type: {request.get('type')!r}")

        cam0 = [decode_jpeg_b64(x) for x in request["images_camera_0"]]
        cam1 = [decode_jpeg_b64(x) for x in request["images_camera_1"]]
        cam2 = [decode_jpeg_b64(x) for x in request["images_camera_2"]]
        instruction = str(request["instruction"])
        proprio = np.asarray(request["proprio"], dtype=np.float32)
        if proprio.ndim == 1:
            proprio = proprio[None, :]
        if proprio.ndim != 2 or proprio.shape[1] != PROPRIO_DIM:
            raise ValueError(f"Expected proprio shape [T, {PROPRIO_DIM}], got {proprio.shape}")

        # OpenVLA-OFT expects a primary full image plus optional wrist images.
        # Dataset mapping: camera_2=front/high, camera_1=left wrist, camera_0=right wrist.
        observation = {
            "full_image": _latest_image(cam2, "images_camera_2"),
            "left_wrist_image": _latest_image(cam1, "images_camera_1"),
            "right_wrist_image": _latest_image(cam0, "images_camera_0"),
            "state": proprio[-1],
        }
        return observation, instruction, proprio

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        observation, instruction, proprio = self._request_to_observation(request)
        actions = get_vla_action(
            self.cfg,
            self.vla,
            self.processor,
            observation,
            instruction,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )
        actions = _as_actions_array(actions)
        if self.cfg.use_relative_actions:
            actions = proprio[-1][None, :] + actions
        return {"actions": actions.astype(float).tolist()}

    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        endpoint = f"tcp://{self.cfg.host}:{self.cfg.port}"
        socket.bind(endpoint)
        print(f"Cobot OpenVLA-OFT ZMQ server listening on {endpoint}")
        while True:
            try:
                request = json.loads(socket.recv_string())
                reply = self.infer(request)
            except Exception as exc:  # noqa: BLE001
                logging.error(traceback.format_exc())
                reply = {"error": str(exc)}
            socket.send_string(json.dumps(reply))


@dataclass
class OpenVLAOFTZMQConfig:
    host: str = "0.0.0.0"
    port: int = 5055

    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 3
    use_proprio: bool = True
    center_crop: bool = True
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    use_relative_actions: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    seed: int = 7


@draccus.wrap()
def main(cfg: OpenVLAOFTZMQConfig) -> None:
    CobotOpenVLAOFTZMQServer(cfg).run()


if __name__ == "__main__":
    main()
