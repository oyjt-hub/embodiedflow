#!/usr/bin/env python3
"""Small, real-image OpenPI offline evaluator for EmbodiedFlow."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--episode-indices-file")
    return p.parse_args()


def letterbox(path: Path, frame_index: int) -> np.ndarray:
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        frame = None
        for i, item in enumerate(container.decode(video=0)):
            if i == frame_index:
                frame = item.to_ndarray(format="rgb24")
                break
        if frame is None:
            raise ValueError(f"frame {frame_index} not found in {path}")
    h, w = frame.shape[:2]
    scale = min(224 / h, 224 / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    y, x = (224 - nh) // 2, (224 - nw) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas.transpose(2, 0, 1)


def camera_alias(key: str) -> str:
    name = key.rsplit(".", 1)[-1]
    return {"front": "cam_high", "left": "cam_left_wrist", "right": "cam_right_wrist"}.get(name, name)


def main():
    args = parse_args()
    root = Path(args.data)
    info = json.loads((root / "meta" / "info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))
    camera_keys = [k for k in info.get("features", {}) if k.startswith("observation.images.")]
    if not camera_keys:
        raise ValueError("dataset has no observation.images.* camera streams")
    if args.episode_indices_file:
        episodes = json.loads(Path(args.episode_indices_file).read_text())
    else:
        episodes = list(range(int(info.get("total_episodes", 0))))
    if not episodes:
        raise ValueError("dataset has no episodes to evaluate")
    if len(episodes) > args.samples:
        positions = np.linspace(0, len(episodes) - 1, args.samples, dtype=int)
        episodes = [episodes[i] for i in positions]

    tasks = {}
    task_file = root / "meta" / "tasks.jsonl"
    if task_file.exists():
        for line in task_file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                tasks[int(row.get("task_index", len(tasks)))] = row.get("task", "")

    from openpi.policies import policy_config
    from openpi.training import config as config_lib

    cfg = config_lib.get_config(args.config)
    started = time.perf_counter()
    policy = policy_config.create_trained_policy(
        cfg, args.checkpoint, sample_kwargs={"num_steps": 10}
    )
    load_seconds = time.perf_counter() - started

    predictions, targets, per_episode = [], [], []
    latencies = []
    for episode in episodes:
        chunk = int(episode) // chunks_size
        filename = f"episode_{int(episode):06d}"
        parquet_path = root / "data" / f"chunk-{chunk:03d}" / f"{filename}.parquet"
        table = pq.read_table(parquet_path, columns=["observation.state", "action", "task_index"])
        if table.num_rows < 2:
            continue
        frame_index = 0
        row = table.slice(frame_index, 1).to_pylist()[0]
        images = {}
        for key in camera_keys:
            video = root / "videos" / f"chunk-{chunk:03d}" / key / f"{filename}.mp4"
            images[camera_alias(key)] = letterbox(video, frame_index)
        observation = {
            "images": images,
            "state": np.asarray(row["observation.state"], dtype=np.float32),
            "prompt": tasks.get(int(row.get("task_index", 0)), ""),
        }
        infer_started = time.perf_counter()
        result = policy.infer(observation)
        latencies.append((time.perf_counter() - infer_started) * 1000)
        predicted = np.asarray(result["actions"], dtype=np.float32)
        if predicted.ndim != 2:
            raise ValueError(f"unexpected action output shape: {predicted.shape}")
        target_rows = table.slice(frame_index, min(len(predicted), table.num_rows - frame_index)).to_pylist()
        target = np.asarray([r["action"] for r in target_rows], dtype=np.float32)
        count = min(len(predicted), len(target))
        predicted, target = predicted[:count], target[:count]
        if predicted.shape[1] != target.shape[1]:
            raise ValueError(f"action dimension mismatch: predicted={predicted.shape}, target={target.shape}")
        mae = float(np.abs(predicted - target).mean())
        rmse = float(np.sqrt(np.square(predicted - target).mean()))
        predictions.append(predicted)
        targets.append(target)
        per_episode.append({"episode_index": int(episode), "samples": count, "mae": mae, "rmse": rmse})

    if not predictions:
        raise ValueError("no evaluable episodes found")
    pred_all = np.concatenate(predictions, axis=0)
    target_all = np.concatenate(targets, axis=0)
    baseline_action = target_all.mean(axis=0, keepdims=True)
    mae = float(np.abs(pred_all - target_all).mean())
    rmse = float(np.sqrt(np.square(pred_all - target_all).mean()))
    baseline_mae = float(np.abs(np.broadcast_to(baseline_action, target_all.shape) - target_all).mean())
    report = {
        "model_family": "openpi",
        "config": args.config,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": str(root.resolve()),
        "episodes": [x["episode_index"] for x in per_episode],
        "samples": len(per_episode),
        "action_steps": int(len(pred_all)),
        "mae": mae,
        "rmse": rmse,
        "baseline_mean_action_mae": baseline_mae,
        "improvement_vs_baseline": 1.0 - mae / baseline_mae if baseline_mae else 0.0,
        "mean_inference_ms": float(np.mean(latencies)),
        "checkpoint_load_seconds": load_seconds,
        "per_episode": per_episode,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
