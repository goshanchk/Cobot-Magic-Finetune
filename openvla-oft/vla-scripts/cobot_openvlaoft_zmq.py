"""ZeroMQ JSON server adapter for Cobot Magic OpenVLA-OFT inference.

Protocol:
- REP socket bound to tcp://0.0.0.0:<port>
- receives a JSON string with base64 JPEG images, instruction, and proprio
- returns {"actions": [[...14 floats...], ...]} with absolute joint targets
"""

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
            action_stats = self.vla.norm_stats[cfg.unnorm_key]["action"]
            proprio_stats = self.vla.norm_stats[cfg.unnorm_key]["proprio"]
            if len(action_stats["min"]) != ACTION_DIM or len(proprio_stats["min"]) != PROPRIO_DIM:
                raise ValueError("Checkpoint statistics do not match the configured Cobot Magic dimensions")
            print(f"Action mode: {'relative -> absolute' if cfg.use_relative_actions else 'absolute'}")
            print(f"Normalization key: {cfg.unnorm_key}; action/proprio dims: {ACTION_DIM}/{PROPRIO_DIM}")

    def _warn_proprio_ood(self, proprio: np.ndarray) -> None:
        if not self.cfg.unnorm_key:
            return
        stats = self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]
        low = np.asarray(stats["min"], dtype=np.float32)
        high = np.asarray(stats["max"], dtype=np.float32)
        span = np.maximum(high - low, 1e-6)
        outside = np.flatnonzero((proprio < low - 0.05 * span) | (proprio > high + 0.05 * span))
        if outside.size:
            logging.warning(
                "Proprio OOD dimensions %s; current=%s, train_min=%s, train_max=%s",
                outside.tolist(), proprio[outside].tolist(), low[outside].tolist(), high[outside].tolist(),
            )

    # def _validate_targets(self, actions: np.ndarray, proprio: np.ndarray) -> None:
    #     if not np.isfinite(proprio).all():
    #         raise ValueError("Received non-finite proprio; refusing to move the robot")
    #     if not np.isfinite(actions).all():
    #         raise ValueError("Model produced non-finite actions; refusing to move the robot")

    #     target_delta = actions - proprio[-1][None, :]
    #     arm_indices = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    #     gripper_indices = np.asarray([6, 13])
    #     max_arm_delta = float(np.abs(target_delta[:, arm_indices]).max())
    #     max_gripper_delta = float(np.abs(target_delta[:, gripper_indices]).max())
    #     if max_arm_delta > self.cfg.max_arm_target_delta:
    #         raise ValueError(
    #             f"Unsafe arm target rejected: delta {max_arm_delta:.4f} exceeds "
    #             f"{self.cfg.max_arm_target_delta:.4f}. Check action mode, joint order, and normalization."
    #         )
    #     if max_gripper_delta > self.cfg.max_gripper_target_delta:
    #         raise ValueError(
    #             f"Unsafe gripper target rejected: delta {max_gripper_delta:.4f} exceeds "
    #             f"{self.cfg.max_gripper_target_delta:.4f}. Check action mode and normalization."
    #         )

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
        self._warn_proprio_ood(proprio[-1])
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
            # New Cobot Magic checkpoints are trained to predict relative joint deltas.
            # The robot client expects absolute joint targets, so convert here.
            actions = proprio[-1][None, :] + actions
        # self._validate_targets(actions, proprio)
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
    base_model_path: Union[str, Path] = "openvla/openvla-7b"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 3
    use_proprio: bool = True
    action_head_hidden_dim: int | None = None
    action_head_num_blocks: int = 1
    center_crop: bool = True
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    use_relative_actions: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    max_arm_target_delta: float = 0.75
    max_gripper_target_delta: float = 3.0
    seed: int = 7


@draccus.wrap()
def main(cfg: OpenVLAOFTZMQConfig) -> None:
    CobotOpenVLAOFTZMQServer(cfg).run()


if __name__ == "__main__":
    main()
