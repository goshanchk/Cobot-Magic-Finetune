"""ZeroMQ JSON server adapter for Cobot Magic Being-H inference.

Protocol:
- REP socket bound to tcp://<host>:<port>
- receives a JSON string with base64 JPEG images, instruction, and proprio
- returns {"actions": [[...14 floats...], ...]} with absolute joint targets

Request format matches the existing Cobot Magic GR00T/OpenVLA adapters:
{
    "type": "inference",
    "images_camera_0": ["<base64-jpeg>", ...],  # right wrist, latest frame is used
    "images_camera_1": ["<base64-jpeg>", ...],  # left wrist, latest frame is used
    "images_camera_2": ["<base64-jpeg>", ...],  # high/front, latest frame is used
    "instruction": "...",
    "proprio": [[left7..., right7...], ...]      # latest row is current qpos
}
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import traceback
from typing import Any

import numpy as np
import zmq
from PIL import Image

from BeingH.inference.beingh_policy import BeingHPolicy
from BeingH.utils.constants import INSTRUCTION_TEMPLATE


PROPRIO_DIM = 14
ARM_DIM = 7


def decode_jpeg_b64(value: str) -> np.ndarray:
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a non-empty base64 JPEG string")
    raw = base64.b64decode(value)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(image, dtype=np.uint8)


def _required_sequence(request: dict[str, Any], name: str) -> list[Any]:
    if name not in request:
        raise ValueError(f"Request is missing required field `{name}`")
    value = request[name]
    if not isinstance(value, list):
        raise ValueError(f"Request field `{name}` must be a list, got {type(value).__name__}")
    if len(value) == 0:
        raise ValueError(f"Request field `{name}` must contain at least one item")
    return value


def _latest_image(values: list[str], name: str) -> np.ndarray:
    return decode_jpeg_b64(_required_sequence({name: values}, name)[-1])


def _as_chunk(values: Any, name: str, dim: int = ARM_DIM) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"Expected {name} shape [N, {dim}], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


class CobotBeingHZMQServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.policy = BeingHPolicy(
            model_path=args.model_path,
            data_config_name=args.data_config_name,
            dataset_name=args.dataset_name,
            embodiment_tag=args.embodiment_tag,
            instruction_template=INSTRUCTION_TEMPLATE,
            max_view_num=args.max_view_num,
            use_fixed_view=args.use_fixed_view,
            enable_rtc=args.enable_rtc,
            metadata_variant=args.metadata_variant,
            stats_selection_mode=args.stats_selection_mode,
            device=args.device,
        )

    def _request_to_observation(self, request: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        if request.get("type") != "inference":
            raise ValueError(f"Unsupported request type: {request.get('type')!r}")

        cam0 = _latest_image(_required_sequence(request, "images_camera_0"), "images_camera_0")
        cam1 = _latest_image(_required_sequence(request, "images_camera_1"), "images_camera_1")
        cam2 = _latest_image(_required_sequence(request, "images_camera_2"), "images_camera_2")

        instruction = str(request.get("instruction", ""))
        if not instruction.strip():
            raise ValueError("Request field `instruction` must be a non-empty string")

        if "proprio" not in request:
            raise ValueError("Request is missing required field `proprio`")
        proprio = np.asarray(request["proprio"], dtype=np.float32)
        if proprio.ndim == 1:
            proprio = proprio[None, :]
        if proprio.ndim != 2 or proprio.shape[1] != PROPRIO_DIM:
            raise ValueError(f"Expected proprio shape [T, {PROPRIO_DIM}], got {proprio.shape}")
        if not np.isfinite(proprio).all():
            raise ValueError("Received non-finite proprio; refusing to move the robot")

        current_qpos = proprio[-1]
        left_qpos = current_qpos[:ARM_DIM]
        right_qpos = current_qpos[ARM_DIM:PROPRIO_DIM]

        observation = {
            # Dataset mapping: camera_2=front/high, camera_1=left wrist, camera_0=right wrist.
            "video.cam_high": cam2,
            "video.cam_left_wrist": cam1,
            "video.cam_right_wrist": cam0,
            # Being-H Cobot config uses right arm as arm_joint_position and left arm separately.
            "state.arm_joint_position": right_qpos,
            "state.left_arm_joint_position": left_qpos,
            "language.instruction": instruction,
        }
        return observation, current_qpos

    def _validate_targets(self, actions: np.ndarray, current_qpos: np.ndarray) -> None:
        if actions.ndim != 2 or actions.shape[1] != PROPRIO_DIM:
            raise ValueError(f"Expected output actions shape [N, {PROPRIO_DIM}], got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Model produced non-finite actions; refusing to move the robot")
        if self.args.max_abs_step_delta > 0:
            deltas = np.abs(actions - current_qpos[None, :])
            max_delta = float(deltas.max())
            if max_delta > self.args.max_abs_step_delta:
                raise ValueError(
                    f"Model output exceeds --max_abs_step_delta: {max_delta:.4f} > "
                    f"{self.args.max_abs_step_delta:.4f}; refusing to move the robot"
                )

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        observation, current_qpos = self._request_to_observation(request)
        result = self.policy.get_action(observation)

        right_actions = _as_chunk(result["action.arm_joint_position"], "action.arm_joint_position")
        left_actions = _as_chunk(result["action.left_arm_joint_position"], "action.left_arm_joint_position")
        num_steps = min(len(left_actions), len(right_actions), self.args.num_actions)
        if num_steps <= 0:
            raise ValueError("Policy returned an empty action chunk")

        # Robot client protocol expects raw Cobot order: [left7, right7].
        actions = np.concatenate([left_actions[:num_steps], right_actions[:num_steps]], axis=1)
        self._validate_targets(actions, current_qpos)

        reply: dict[str, Any] = {"actions": actions.astype(float).tolist()}
        if self.args.return_debug:
            left_delta = result.get("action_delta.left_arm_joint_position")
            right_delta = result.get("action_delta.arm_joint_position")
            if left_delta is not None and right_delta is not None:
                reply["action_deltas"] = np.concatenate(
                    [_as_chunk(left_delta, "action_delta.left_arm_joint_position")[:num_steps],
                     _as_chunk(right_delta, "action_delta.arm_joint_position")[:num_steps]],
                    axis=1,
                ).astype(float).tolist()
        return reply

    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        endpoint = f"tcp://{self.args.host}:{self.args.port}"
        socket.bind(endpoint)
        print(f"Cobot Being-H ZMQ server listening on {endpoint}", flush=True)
        while True:
            try:
                request = json.loads(socket.recv_string())
                reply = self.infer(request)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                logging.error("Inference request failed:\n%s", tb)
                reply = {"error": str(exc), "error_traceback": tb}
            socket.send_string(json.dumps(reply))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_config_name", default="cobot_magic_sber")
    parser.add_argument("--dataset_name", default="cobot_magic_sber_posttrain")
    parser.add_argument("--embodiment_tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--num_actions", type=int, default=24)
    parser.add_argument("--max_view_num", type=int, default=-1)
    parser.add_argument("--use_fixed_view", action="store_true")
    parser.add_argument("--enable_rtc", action="store_true")
    parser.add_argument("--metadata_variant", default=None)
    parser.add_argument("--stats_selection_mode", default="auto", choices=("auto", "task", "embodiment"))
    parser.add_argument(
        "--max_abs_step_delta",
        type=float,
        default=0.0,
        help="Optional safety guard in joint units. 0 disables the guard.",
    )
    parser.add_argument("--return_debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    CobotBeingHZMQServer(parse_args()).run()


if __name__ == "__main__":
    main()
