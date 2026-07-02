#!/usr/bin/env python3
"""
analyze_stillness.py

分析 LeRobot v2.1 dataset 每个 episode 开头/末尾的静止段情况（只读，不修改数据）。

用法:
    python analyze_stillness.py --root /path/to/dataset
    python analyze_stillness.py --root /path/to/dataset --threshold 0.5 --top 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def analyze(root: str, threshold: float, top: int):
    root = Path(root)
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)

    all_start, all_end, all_total = [], [], []

    for ep_idx in range(total_episodes):
        chunk = ep_idx // chunks_size
        parquet = root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet)
        actions = np.array(df["action"].tolist())
        deltas = np.linalg.norm(np.diff(actions, axis=0), axis=1)
        total = len(df)

        ss = 0
        for d in deltas:
            if d < threshold: ss += 1
            else: break

        es = 0
        for d in reversed(deltas):
            if d < threshold: es += 1
            else: break

        all_start.append(ss)
        all_end.append(es)
        all_total.append(total)

    all_start = np.array(all_start)
    all_end   = np.array(all_end)
    all_total = np.array(all_total)

    print(f"数据集: {root}")
    print(f"Episodes: {total_episodes}  threshold={threshold}\n")

    print("=== 开头静止帧 ===")
    print(f"  均值:   {all_start.mean():.1f} 帧  ({all_start.mean()/all_total.mean()*100:.1f}%)")
    print(f"  中位数: {np.median(all_start):.0f} 帧")
    print(f"  最大:   {all_start.max()} 帧  (EP{all_start.argmax():03d})")
    print(f"  最小:   {all_start.min()} 帧")
    print(f"  >10帧:  {(all_start > 10).sum()}/{total_episodes}")
    print(f"  >30帧:  {(all_start > 30).sum()}/{total_episodes}")

    print("\n=== 末尾静止帧 ===")
    print(f"  均值:   {all_end.mean():.1f} 帧  ({all_end.mean()/all_total.mean()*100:.1f}%)")
    print(f"  中位数: {np.median(all_end):.0f} 帧")
    print(f"  最大:   {all_end.max()} 帧  (EP{all_end.argmax():03d})")
    print(f"  最小:   {all_end.min()} 帧")
    print(f"  >10帧:  {(all_end > 10).sum()}/{total_episodes}")
    print(f"  >30帧:  {(all_end > 30).sum()}/{total_episodes}")

    print(f"\n=== 静止最严重的 Top {top} episodes ===")
    worst = np.argsort(all_start + all_end)[::-1][:top]
    print(f"{'EP':>5} {'Total':>7} {'StartStill':>11} {'EndStill':>9} {'合计':>6}")
    print("-" * 45)
    for i in worst:
        print(f"  {i:03d}  {all_total[i]:>6}  {all_start[i]:>10}  {all_end[i]:>8}  {all_start[i]+all_end[i]:>5}")

    print("\n=== 分布直方图 (开头) ===")
    bins = [0, 5, 10, 20, 30, 50, 100, 999]
    counts, _ = np.histogram(all_start, bins=bins)
    for i in range(len(counts)):
        bar = "█" * min(counts[i], 50)
        print(f"  [{bins[i]:>3},{bins[i+1]:>3}): {counts[i]:>3}  {bar}")

    print("\n=== 分布直方图 (末尾) ===")
    counts, _ = np.histogram(all_end, bins=bins)
    for i in range(len(counts)):
        bar = "█" * min(counts[i], 50)
        print(f"  [{bins[i]:>3},{bins[i+1]:>3}): {counts[i]:>3}  {bar}")

    total_still = (all_start + all_end).sum()
    print(f"\n总静止帧: {total_still}  占总帧数 {total_still/all_total.sum()*100:.1f}%")
    print(f"建议: python trim_static.py --src {root} --dst <output> --threshold {threshold} --buffer 3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",      required=True, help="数据集路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="静止判断阈值（默认 0.5）")
    parser.add_argument("--top",       type=int,   default=10,  help="显示最严重的 N 个 episode")
    args = parser.parse_args()
    analyze(args.root, args.threshold, args.top)


if __name__ == "__main__":
    main()
