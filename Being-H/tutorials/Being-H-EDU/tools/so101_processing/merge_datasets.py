#!/usr/bin/env python3
"""
merge_datasets.py

将多个 LeRobot v2.1 数据集合并为一个，重新连续编号。

用法:
    python merge_datasets.py \
        --srcs /path/to/ds_001 /path/to/ds_002 /path/to/ds_003 \
        --dst  /path/to/ds_merged
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def load_info(root):
    with open(os.path.join(root, "meta", "info.json")) as f:
        return json.load(f)


# ── Stats helpers ─────────────────────────────────────────────────────────────

def compute_stats(arr: np.ndarray):
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "mean":  arr.mean(axis=0).tolist(),
        "std":   arr.std(axis=0).tolist(),
        "min":   arr.min(axis=0).tolist(),
        "max":   arr.max(axis=0).tolist(),
        "count": [len(arr)],
    }

def merge_global_stats(all_ep_stats):
    if not all_ep_stats:
        return {}
    keys = list(all_ep_stats[0].keys())
    merged = {}
    for key in keys:
        per = [s[key] for s in all_ep_stats if key in s]
        if not per:
            continue
        counts = [s["count"][0] for s in per]
        total = sum(counts)
        merged_mean = np.sum([np.array(s["mean"]) * c / total for s, c in zip(per, counts)], axis=0)
        merged_std  = np.sqrt(np.sum([np.array(s["std"])**2 * c / total for s, c in zip(per, counts)], axis=0))
        merged_min  = np.minimum.reduce([np.array(s["min"]) for s in per])
        merged_max  = np.maximum.reduce([np.array(s["max"]) for s in per])
        merged[key] = {
            "mean":  merged_mean.tolist(),
            "std":   merged_std.tolist(),
            "min":   merged_min.tolist(),
            "max":   merged_max.tolist(),
            "count": [total],
        }
    return merged

def compute_episode_stats(df, scalar_features):
    stats = {}
    for key in scalar_features:
        if key not in df.columns:
            continue
        first = df[key].iloc[0]
        if isinstance(first, (list, np.ndarray)):
            arr = np.array(df[key].tolist(), dtype=np.float64)
        elif isinstance(first, (int, float, np.integer, np.floating)):
            arr = df[key].values.astype(np.float64).reshape(-1, 1)
        else:
            continue
        stats[key] = compute_stats(arr)
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def merge_datasets(src_roots: list, dst_root: str):
    dst = Path(dst_root)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    # 以第一个数据集的 info 为基准
    base_info = load_info(src_roots[0])
    fps = base_info["fps"]
    chunks_size = base_info.get("chunks_size", 1000)
    features = base_info["features"]
    scalar_features = [k for k, v in features.items() if v.get("dtype") not in ("video", "image")]
    video_keys      = [k for k, v in features.items() if v.get("dtype") == "video"]

    for vk in video_keys:
        (dst / "videos" / "chunk-000" / vk).mkdir(parents=True, exist_ok=True)
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # 收集所有 tasks，去重合并
    all_tasks = {}
    for src in src_roots:
        tasks_path = os.path.join(src, "meta", "tasks.jsonl")
        if os.path.exists(tasks_path):
            for t in load_jsonl(tasks_path):
                lang = t.get("task", t.get("language_instruction", ""))
                if lang not in all_tasks.values():
                    idx = len(all_tasks)
                    all_tasks[idx] = lang

    new_ep_idx = 0
    global_frame_offset = 0
    new_episodes = []
    all_ep_stats = []

    for src_root in src_roots:
        src = Path(src_root)
        info = load_info(src_root)
        src_chunks_size = info.get("chunks_size", 1000)
        episodes = load_jsonl(src / "meta" / "episodes.jsonl")

        # 构建 task 名称 -> 新 task_index 的映射
        src_tasks = {}
        tasks_path = src / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            for t in load_jsonl(tasks_path):
                lang = t.get("task", t.get("language_instruction", ""))
                src_tasks[t["task_index"]] = lang

        print(f"\n[{src_root}]  {len(episodes)} episodes")

        for ep in sorted(episodes, key=lambda e: e["episode_index"]):
            old_idx = ep["episode_index"]
            old_chunk = old_idx // src_chunks_size
            old_parquet = src / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_idx:06d}.parquet"

            df = pd.read_parquet(old_parquet)
            length = len(df)

            # 重新编号
            df["episode_index"] = new_ep_idx
            df["frame_index"]   = list(range(length))
            df["index"]         = list(range(global_frame_offset, global_frame_offset + length))
            df["timestamp"]     = [i / fps for i in range(length)]

            # 更新 task_index
            if "task_index" in df.columns and src_tasks:
                def remap_task(old_ti):
                    lang = src_tasks.get(int(old_ti), "")
                    for new_ti, new_lang in all_tasks.items():
                        if new_lang == lang:
                            return new_ti
                    return 0
                df["task_index"] = df["task_index"].apply(remap_task)

            new_chunk = new_ep_idx // chunks_size
            dst_parquet = dst / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_ep_idx:06d}.parquet"
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst_parquet, index=False)

            # 复制视频
            for vk in video_keys:
                src_vid = src / "videos" / f"chunk-{old_chunk:03d}" / vk / f"episode_{old_idx:06d}.mp4"
                dst_vid = dst / "videos" / f"chunk-{new_chunk:03d}" / vk / f"episode_{new_ep_idx:06d}.mp4"
                dst_vid.parent.mkdir(parents=True, exist_ok=True)
                if src_vid.exists():
                    shutil.copy2(src_vid, dst_vid)
                else:
                    print(f"  WARNING: video not found: {src_vid}")

            ep_stats = compute_episode_stats(df, scalar_features)
            all_ep_stats.append(ep_stats)

            # 获取 task 名称列表
            ep_task_indices = df["task_index"].unique().tolist() if "task_index" in df.columns else []
            ep_tasks = [all_tasks[ti] for ti in ep_task_indices if ti in all_tasks]
            if not ep_tasks:
                ep_tasks = ep.get("tasks", [])

            new_episodes.append({
                "episode_index": new_ep_idx,
                "tasks": ep_tasks,
                "length": length,
            })

            print(f"  ep {old_idx} -> ep {new_ep_idx}  ({length} frames)")
            new_ep_idx += 1
            global_frame_offset += length

    total_episodes = new_ep_idx
    total_frames   = global_frame_offset

    # 写 meta
    save_jsonl(new_episodes, dst / "meta" / "episodes.jsonl")

    ep_stats_records = [
        {"episode_index": ep["episode_index"], "stats": st}
        for ep, st in zip(new_episodes, all_ep_stats)
    ]
    save_jsonl(ep_stats_records, dst / "meta" / "episodes_stats.jsonl")

    tasks_out = [{"task_index": ti, "task": lang} for ti, lang in all_tasks.items()]
    save_jsonl(tasks_out, dst / "meta" / "tasks.jsonl")

    global_stats = merge_global_stats(all_ep_stats)
    with open(dst / "meta" / "stats.json", "w") as f:
        json.dump(global_stats, f, indent=4)

    new_info = dict(base_info)
    new_info["total_episodes"] = total_episodes
    new_info["total_frames"]   = total_frames
    new_info["total_tasks"]    = len(all_tasks)
    new_info["total_chunks"]   = math.ceil(total_episodes / chunks_size)
    new_info["total_videos"]   = total_episodes * len(video_keys)
    new_info["splits"]         = {"train": f"0:{total_episodes}"}
    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)

    print(f"\n完成: {total_episodes} episodes, {total_frames} frames -> {dst_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcs", nargs="+", required=True, help="源数据集路径列表")
    parser.add_argument("--dst",  required=True, help="输出数据集路径")
    args = parser.parse_args()
    merge_datasets(args.srcs, args.dst)

if __name__ == "__main__":
    main()
