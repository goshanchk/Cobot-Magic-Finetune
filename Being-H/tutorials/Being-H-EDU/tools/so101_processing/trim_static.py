#!/usr/bin/env python3
"""
裁剪 LeRobot dataset 开头和末尾的静止段。

用法:
    python trim_dataset.py \
        --src /path/to/src_dataset \
        --dst /path/to/dst_dataset \
        --threshold 0.5 \
        --buffer 3
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def find_trim_range(actions: np.ndarray, threshold: float, buffer: int):
    """
    返回 (start_idx, end_idx)，即裁剪后保留的帧范围 [start_idx, end_idx)。
    buffer: 在第一个/最后一个有效帧前后各保留的缓冲帧数。
    """
    deltas = np.linalg.norm(np.diff(actions, axis=0), axis=1)  # shape: (T-1,)
    T = len(actions)

    # 找第一个 delta > threshold 的位置
    start_idx = 0
    for i, d in enumerate(deltas):
        if d >= threshold:
            start_idx = max(0, i - buffer)
            break

    # 找最后一个 delta > threshold 的位置
    end_idx = T
    for i, d in enumerate(reversed(deltas)):
        if d >= threshold:
            end_idx = min(T, T - 1 - i + 1 + buffer)
            break

    # 保证至少保留 10 帧
    if end_idx - start_idx < 10:
        start_idx = 0
        end_idx = T

    return start_idx, end_idx


def compute_stats(values: np.ndarray):
    """计算 mean/std/min/max/count，兼容 1D 和 2D。"""
    if values.ndim == 1:
        values = values[:, None]
    return {
        "mean": values.mean(axis=0).tolist(),
        "std":  values.std(axis=0).tolist(),
        "min":  values.min(axis=0).tolist(),
        "max":  values.max(axis=0).tolist(),
        "count": [len(values)],
    }


def trim_video(src_video: Path, dst_video: Path, start_frame: int, end_frame: int, fps: int):
    """用 ffmpeg 按帧范围裁剪视频。"""
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame) / fps
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.6f}",
        "-i", str(src_video),
        "-t", f"{duration_sec:.6f}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-an",
        str(dst_video),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src_video}:\n{result.stderr.decode()}")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="源 dataset 路径")
    parser.add_argument("--dst", required=True, help="输出 dataset 路径")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="判断静止的 action delta 阈值（默认 0.5）")
    parser.add_argument("--buffer", type=int, default=3,
                        help="有效动作前后保留的缓冲帧数（默认 3）")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    # 读取 info.json
    with open(src / "meta" / "info.json") as f:
        info = json.load(f)

    fps = info["fps"]
    total_episodes = info["total_episodes"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    print(f"源数据集: {src}")
    print(f"目标路径: {dst}")
    print(f"Episodes: {total_episodes}, FPS: {fps}")
    print(f"视频流: {video_keys}")
    print(f"threshold={args.threshold}, buffer={args.buffer}\n")

    # 创建目标目录结构
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    for vk in video_keys:
        (dst / "videos" / "chunk-000" / vk).mkdir(parents=True, exist_ok=True)

    # 复制不变的 meta 文件
    shutil.copy(src / "meta" / "tasks.jsonl", dst / "meta" / "tasks.jsonl")

    episodes_out = []
    episodes_stats_out = []
    global_index = 0
    total_frames_new = 0

    # 用于最终重算全局 stats
    all_actions = []
    all_states = []
    all_timestamps = []

    for ep_idx in range(total_episodes):
        src_parquet = src / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(src_parquet)

        actions = np.array(df["action"].tolist())
        start_idx, end_idx = find_trim_range(actions, args.threshold, args.buffer)
        trimmed_len = end_idx - start_idx

        print(f"EP{ep_idx:03d}: {len(df)} -> {trimmed_len} 帧  "
              f"(裁头 {start_idx}, 裁尾 {len(df)-end_idx})")

        # ── 裁剪 parquet ──
        df_trim = df.iloc[start_idx:end_idx].copy()
        df_trim["frame_index"] = list(range(trimmed_len))
        df_trim["index"] = list(range(global_index, global_index + trimmed_len))
        # 重置 timestamp 从 0 开始
        df_trim["timestamp"] = [i / fps for i in range(trimmed_len)]

        dst_parquet = dst / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        df_trim.to_parquet(dst_parquet, index=False)

        # ── 裁剪视频 ──
        for vk in video_keys:
            src_video = src / "videos" / "chunk-000" / vk / f"episode_{ep_idx:06d}.mp4"
            dst_video = dst / "videos" / "chunk-000" / vk / f"episode_{ep_idx:06d}.mp4"
            trim_video(src_video, dst_video, start_idx, end_idx, fps)

        # ── 收集统计数据 ──
        actions_trim = np.array(df_trim["action"].tolist())
        states_trim  = np.array(df_trim["observation.state"].tolist())
        timestamps_trim = df_trim["timestamp"].values

        all_actions.append(actions_trim)
        all_states.append(states_trim)
        all_timestamps.append(timestamps_trim)

        # per-episode stats
        ep_stats = {
            "episode_index": ep_idx,
            "stats": {
                "action":            compute_stats(actions_trim),
                "observation.state": compute_stats(states_trim),
                "timestamp":         compute_stats(timestamps_trim),
                "frame_index":       compute_stats(np.arange(trimmed_len, dtype=np.float64)),
                "episode_index":     compute_stats(np.full(trimmed_len, ep_idx, dtype=np.float64)),
                "index":             compute_stats(np.arange(global_index, global_index + trimmed_len, dtype=np.float64)),
            }
        }
        episodes_stats_out.append(ep_stats)

        # episodes.jsonl entry
        tasks = df_trim["task_index"].iloc[0]
        # 读取原始 task 名称
        with open(src / "meta" / "episodes.jsonl") as f:
            src_episodes = [json.loads(l) for l in f]
        ep_tasks = src_episodes[ep_idx]["tasks"]
        episodes_out.append({
            "episode_index": ep_idx,
            "tasks": ep_tasks,
            "length": trimmed_len,
        })

        global_index += trimmed_len
        total_frames_new += trimmed_len

    # ── 写 episodes.jsonl ──
    with open(dst / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes_out:
            f.write(json.dumps(ep) + "\n")

    # ── 写 episodes_stats.jsonl ──
    with open(dst / "meta" / "episodes_stats.jsonl", "w") as f:
        for es in episodes_stats_out:
            f.write(json.dumps(es) + "\n")

    # ── 重算全局 stats.json ──
    all_actions_np    = np.concatenate(all_actions, axis=0)
    all_states_np     = np.concatenate(all_states, axis=0)
    all_timestamps_np = np.concatenate(all_timestamps, axis=0)

    global_stats = {
        "action":            compute_stats(all_actions_np),
        "observation.state": compute_stats(all_states_np),
        "timestamp":         compute_stats(all_timestamps_np),
        "frame_index":       compute_stats(np.arange(total_frames_new, dtype=np.float64)),
        "episode_index":     compute_stats(np.array([ep["episode_index"] for ep in episodes_out
                                                      for _ in range(ep["length"])], dtype=np.float64)),
        "index":             compute_stats(np.arange(total_frames_new, dtype=np.float64)),
    }
    with open(dst / "meta" / "stats.json", "w") as f:
        json.dump(global_stats, f, indent=4)

    # ── 更新 info.json ──
    info["total_frames"] = total_frames_new
    info["total_videos"] = total_episodes * len(video_keys)
    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"\n完成！总帧数: {total_frames_new} (原 {info['total_frames']} -> 裁剪后 {total_frames_new})")
    print(f"节省帧数: {33913 - total_frames_new} ({(33913 - total_frames_new)/33913*100:.1f}%)")
    print(f"输出路径: {dst}")


if __name__ == "__main__":
    main()
