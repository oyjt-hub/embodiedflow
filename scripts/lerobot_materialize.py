#!/usr/bin/env python3
"""Materialize a LeRobot dataset version with optional crop and FPS resampling."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_hwc(value):
    image = _as_numpy(value)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            if image.size and float(np.nanmax(image)) <= 1.5:
                image = image * 255.0
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _nearest_indices(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    right = np.searchsorted(times, targets, side="left")
    right = np.clip(right, 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    use_left = np.abs(times[left] - targets) <= np.abs(times[right] - targets)
    return np.where(use_left, left, right)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--target-fps", type=int, required=True)
    parser.add_argument("--episodes-json", required=True)
    parser.add_argument("--crop-start-s", type=float, default=0.0)
    parser.add_argument("--crop-end-s", type=float)
    parser.add_argument("--source-version-id", type=int, required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve()
    stage_root = output_root.with_name(output_root.name + ".partial")
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selected = [int(x) for x in json.loads(args.episodes_json)]
    if not selected:
        raise ValueError("至少选择一个 Episode")
    if args.target_fps < 1 or args.target_fps > 120:
        raise ValueError("target_fps 必须在 1 到 120 之间")
    if args.crop_start_s < 0 or (args.crop_end_s is not None and args.crop_end_s <= args.crop_start_s):
        raise ValueError("裁剪时间范围无效")
    if output_root.exists() or stage_root.exists():
        raise FileExistsError(f"输出目录已存在：{output_root}")

    info = json.loads((source_root / "meta" / "info.json").read_text(encoding="utf-8"))
    source_fps = float(info.get("fps") or 0)
    if source_fps <= 0:
        raise ValueError("源数据版本缺少有效 fps")
    source = LeRobotDataset(
        repo_id=args.repo_id,
        root=source_root,
        episodes=selected,
        download_videos=False,
        video_backend="pyav",
    )
    available = [int(x) for x in source.episodes]
    if available != selected:
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"源数据中不存在 Episode：{missing}")

    output_features = {key: dict(value) for key, value in info.get("features", {}).items()}
    # This installed LeRobot validator compares ndarray.shape directly with
    # feature["shape"]. Tuples keep that check compatible with JSON list metadata.
    for feature in output_features.values():
        if feature.get("shape") is not None:
            feature["shape"] = tuple(feature["shape"])
    writer = LeRobotDataset.create(
        repo_id=f"embodiedflow/materialized_{args.source_version_id}_{os.getpid()}",
        root=stage_root,
        fps=args.target_fps,
        robot_type=info.get("robot_type"),
        features=output_features,
        use_videos=True,
        video_backend="pyav",
    )

    timestamps = np.asarray([float(x) for x in source.hf_dataset["timestamp"]], dtype=np.float64)
    episode_ids = np.asarray([int(x) for x in source.hf_dataset["episode_index"]], dtype=np.int64)
    ranges = source.episode_data_index
    per_episode = []
    for local_episode, source_episode in enumerate(available):
        begin = int(ranges["from"][local_episode])
        end = int(ranges["to"][local_episode])
        rows = np.arange(begin, end, dtype=np.int64)
        rows = rows[episode_ids[rows] == source_episode]
        if not len(rows):
            raise ValueError(f"Episode {source_episode} 没有数据帧")
        times = timestamps[rows]
        start_s = max(0.0, args.crop_start_s)
        episode_end_s = float(times[-1]) + 1.0 / source_fps
        end_s = min(episode_end_s, args.crop_end_s) if args.crop_end_s is not None else episode_end_s
        if end_s <= start_s:
            raise ValueError(f"Episode {source_episode} 在当前裁剪范围内没有帧")
        frame_count = max(1, int(np.floor((end_s - start_s) * args.target_fps + 1e-9)))
        target_times = start_s + np.arange(frame_count, dtype=np.float64) / args.target_fps
        source_positions = _nearest_indices(times, target_times)
        selected_rows = rows[source_positions]
        selected_times = times[source_positions]
        row_data = source.hf_dataset.select(selected_rows.tolist())
        video_keys = [
            key for key, feature in info.get("features", {}).items()
            if feature.get("dtype") == "video"
        ]
        task_name = None
        # Decode each camera in short sequential batches. Calling dataset[row]
        # for every frame reopens/seeks all three videos on every sample.
        for batch_start in range(0, len(selected_rows), 32):
            batch_end = min(batch_start + 32, len(selected_rows))
            batch_images = source._query_videos(
                {key: selected_times[batch_start:batch_end].tolist() for key in video_keys},
                source_episode,
            ) if video_keys else {}
            for key, frames in batch_images.items():
                if frames.ndim == 3:
                    batch_images[key] = frames.unsqueeze(0)
            for row_index in range(batch_start, batch_end):
                raw = row_data[row_index]
                task_index = int(raw["task_index"])
                task_name = str(source.meta.tasks[task_index])
                values = {"task": task_name}
                local_index = row_index - batch_start
                for feature_name, feature in info.get("features", {}).items():
                    if feature_name in {"timestamp", "frame_index", "episode_index", "index", "task_index"}:
                        continue
                    if feature.get("dtype") == "video":
                        value = batch_images[feature_name][local_index]
                        value = _image_hwc(value)
                    elif feature.get("dtype") == "image":
                        value = _image_hwc(raw[feature_name])
                    else:
                        value = _as_numpy(raw[feature_name]).astype(
                            np.dtype(feature["dtype"]), copy=False
                        )
                    values[feature_name] = value
                writer.add_frame(values)
        writer.save_episode()
        per_episode.append({
            "output_episode": len(per_episode),
            "source_episode": source_episode,
            "frames": len(selected_rows),
            "duration_s": round(len(selected_rows) / args.target_fps, 6),
            "task": task_name,
            "source_start_s": round(float(times[source_positions[0]]), 6),
            "source_end_s": round(float(times[source_positions[-1]]), 6),
        })
        print(f"[materialize] episode {source_episode} -> {len(per_episode)-1}: {len(selected_rows)} frames", flush=True)

    stage_report = {
        "source_version_id": args.source_version_id,
        "source_path": str(source_root),
        "source_fps": source_fps,
        "target_fps": args.target_fps,
        "crop_start_s": args.crop_start_s,
        "crop_end_s": args.crop_end_s,
        "episode_count": len(per_episode),
        "frame_count": sum(x["frames"] for x in per_episode),
        "episodes": per_episode,
    }
    (stage_root / "meta" / "source_episode_mapping.json").write_text(
        json.dumps(stage_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.move(str(stage_root), str(output_root))
    total_bytes = sum(p.stat().st_size for p in output_root.rglob("*") if p.is_file())
    stage_report.update({"output_path": str(output_root), "total_bytes": total_bytes})
    report_path.write_text(json.dumps(stage_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[materialize] complete episodes={len(per_episode)} frames={stage_report['frame_count']} bytes={total_bytes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
