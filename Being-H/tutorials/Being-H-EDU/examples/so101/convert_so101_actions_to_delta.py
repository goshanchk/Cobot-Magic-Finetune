#!/usr/bin/env python
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
"""Convert SO101 LeRobot action joints from absolute positions to deltas."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


VECTOR_COLUMNS = {"action", "observation.state"}
SCALAR_COLUMNS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _stack_column(df: pd.DataFrame, column: str) -> np.ndarray:
    values = df[column].to_numpy()
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.stack(values)
    return values.reshape(-1, 1)


def _stats_for_array(array: np.ndarray, include_count: bool = False) -> dict[str, list[float]]:
    array = np.asarray(array)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    stats = {
        "mean": np.mean(array, axis=0).astype(float).tolist(),
        "std": np.std(array, axis=0).astype(float).tolist(),
        "min": np.min(array, axis=0).astype(float).tolist(),
        "max": np.max(array, axis=0).astype(float).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).astype(float).tolist(),
    }
    if include_count:
        stats.pop("q01")
        stats.pop("q99")
        stats["count"] = [int(array.shape[0])]
    return stats


def _episode_stats(df: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    stats = {}
    for column in VECTOR_COLUMNS | SCALAR_COLUMNS:
        if column in df.columns:
            stats[column] = _stats_for_array(_stack_column(df, column), include_count=True)
    return stats


def _global_stats(parquet_paths: list[Path]) -> dict[str, dict[str, list[float]]]:
    arrays_by_column: dict[str, list[np.ndarray]] = {}
    for path in parquet_paths:
        df = pd.read_parquet(path)
        for column in VECTOR_COLUMNS | SCALAR_COLUMNS:
            if column in df.columns:
                arrays_by_column.setdefault(column, []).append(_stack_column(df, column))

    return {
        column: _stats_for_array(np.concatenate(arrays, axis=0))
        for column, arrays in arrays_by_column.items()
    }


def _convert_file(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    action = np.stack(df["action"].to_numpy()).astype(np.float32, copy=True)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32, copy=False)

    before_first = action[0].copy()
    action[:, :5] = action[:, :5] - state[:, :5]
    after_first = action[0].copy()

    df["action"] = list(action)
    df.to_parquet(path, index=False)
    return len(df), before_first, after_first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--force", action="store_true", help="Run even if the conversion marker exists.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    meta_dir = dataset_dir / "meta"
    marker_path = meta_dir / "action_delta_conversion.json"
    if marker_path.exists() and not args.force:
        raise SystemExit(f"Refusing to run: conversion marker already exists at {marker_path}")

    parquet_paths = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No parquet files found under {dataset_dir / 'data'}")

    sample_df = pd.read_parquet(parquet_paths[0])
    sample_action = np.stack(sample_df["action"].to_numpy())
    sample_state = np.stack(sample_df["observation.state"].to_numpy())
    mean_abs_action = np.mean(np.abs(sample_action[:, :5]), axis=0)
    mean_abs_action_minus_state = np.mean(np.abs(sample_action[:, :5] - sample_state[:, :5]), axis=0)

    if float(np.mean(mean_abs_action)) < 5.0 and not args.force:
        raise SystemExit(
            "Refusing to run: action first five dims already look like deltas. "
            f"mean_abs_action={mean_abs_action.tolist()}"
        )

    stats_path = meta_dir / "stats.json"
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    if stats_path.exists():
        stats_path.replace(meta_dir / "stats.before_action_delta.json")
    if episodes_stats_path.exists():
        episodes_stats_path.replace(meta_dir / "episodes_stats.before_action_delta.jsonl")

    total_rows = 0
    first_before = None
    first_after = None
    for path in parquet_paths:
        rows, before, after = _convert_file(path)
        total_rows += rows
        if first_before is None:
            first_before = before
            first_after = after

    stats = _global_stats(parquet_paths)
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=4)

    with episodes_stats_path.open("w") as f:
        for path in parquet_paths:
            df = pd.read_parquet(path)
            episode_index = int(np.asarray(df["episode_index"].iloc[0]).reshape(-1)[0])
            f.write(json.dumps({"episode_index": episode_index, "stats": _episode_stats(df)}) + "\n")

    marker = {
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "num_parquet_files": len(parquet_paths),
        "total_rows": total_rows,
        "operation": "action[:, :5] = action[:, :5] - observation.state[:, :5]; action[:, 5] unchanged",
        "precheck_mean_abs_action_first5": mean_abs_action.astype(float).tolist(),
        "precheck_mean_abs_action_minus_state_first5": mean_abs_action_minus_state.astype(float).tolist(),
        "first_action_before": first_before.astype(float).tolist() if first_before is not None else None,
        "first_action_after": first_after.astype(float).tolist() if first_after is not None else None,
        "backups": {
            "stats": str(meta_dir / "stats.before_action_delta.json"),
            "episodes_stats": str(meta_dir / "episodes_stats.before_action_delta.jsonl"),
        },
    }
    with marker_path.open("w") as f:
        json.dump(marker, f, indent=4)

    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
