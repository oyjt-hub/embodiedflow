"""生成一个含 12 类可控异常的 LeRobot 风格数据集（1 个干净 Episode + 12 个异常 Episode）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 30
LEN = 100
DIM = 6
ANOMALIES = [
    "clean",                 # 0
    "missing_timestamp",     # 1
    "time_reversal",         # 2
    "irregular_gap",         # 3
    "missing_frames",        # 4（gripper 少 10 帧）
    "video_corrupt",         # 5（全零帧）
    "camera_action_misalign",# 6（相机时钟 +0.5s）
    "joint_limit",           # 7
    "action_jump",           # 8
    "gripper_anomaly",       # 9
    "episode_incomplete",    # 10（声明 100 实际 30）
    "static_frames",         # 11（状态几乎不变）
    "trajectory_break",      # 12
]


def _frames(kind: str, n: int, h: int = 48, w: int = 64) -> np.ndarray:
    if kind == "video_corrupt":
        return np.zeros((n, h, w, 3), dtype=np.uint8)
    base = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        base[i, :, :, 0] = (i * 3) % 200 + 30
        base[i, :, :, 1] = 60
        base[i, :, :, 2] = 120
    return base


def build(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "videos" / "chunk-000" / "observation.images.top").mkdir(parents=True, exist_ok=True)

    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick up the red cube"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    episode_rows = []
    for idx, kind in enumerate(ANOMALIES):
        n = 30 if kind == "episode_incomplete" else LEN
        t = np.arange(n, dtype=np.float64) / FPS

        state = np.zeros((n, DIM), dtype=np.float64)
        for d in range(DIM):
            state[:, d] = 0.4 * np.sin(2 * np.pi * (0.5 + 0.1 * d) * t + d)
        action = np.roll(state, -1, axis=0) * 0.9
        gripper = (np.arange(n) // 10 % 2).astype(np.float64)

        if kind == "missing_timestamp":
            t[10] = np.nan
        elif kind == "time_reversal":
            t[10], t[11] = t[11], t[10]
        elif kind == "irregular_gap":
            t[50:] += 5.0
        elif kind == "missing_frames":
            n_cam = n - 10  # 视频少 10 帧，parquet 列长保持一致
        elif kind == "joint_limit":
            state[50, 0] = 5.0
        elif kind == "action_jump":
            action[50:, 0] += 50.0
        elif kind == "gripper_anomaly":
            gripper[20] = 5.0
        elif kind == "static_frames":
            state[:] = state[0]
        elif kind == "trajectory_break":
            state[50:, 0] += 500.0

        cam_off = 0.5 if kind == "camera_action_misalign" else 0.0
        if kind != "missing_frames":
            n_cam = n

        table = pa.table(
            {
                "timestamp": pa.array(t, pa.float64()),
                "frame_index": pa.array(np.arange(n), pa.int64()),
                "episode_index": pa.array(np.full(n, idx), pa.int64()),
                "index": pa.array(np.arange(n) + idx * LEN, pa.int64()),
                "task_index": pa.array(np.zeros(n, dtype=np.int64), pa.int64()),
                "observation.state": pa.array(state.tolist(), pa.list_(pa.float64(), DIM)),
                "action": pa.array(action.tolist(), pa.list_(pa.float64(), DIM)),
                "observation.gripper": pa.array(gripper.reshape(-1, 1).tolist(), pa.list_(pa.float64(), 1)),
            }
        )
        pq.write_table(table, root / "data" / "chunk-000" / f"episode_{idx:06d}.parquet")

        frames = _frames(kind, n_cam)
        import imageio.v2 as iio

        iio.mimsave(
            root / "videos" / "chunk-000" / "observation.images.top" / f"episode_{idx:06d}.mp4",
            frames, fps=FPS, quality=6,
        )

        row = {"episode_index": idx, "tasks": ["pick up the red cube"], "length": LEN}
        if cam_off:
            row["camera_time_offset"] = cam_off
        episode_rows.append(row)

    with (root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for row in episode_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    info = {
        "codebase_version": "v2.1",
        "robot_type": "piper",
        "fps": FPS,
        "total_episodes": len(ANOMALIES),
        "total_frames": LEN * len(ANOMALIES),
        "features": {
            "observation.state": {"dtype": "float64", "shape": [DIM]},
            "action": {"dtype": "float64", "shape": [DIM]},
            "observation.gripper": {"dtype": "float64", "shape": [1]},
            "observation.images.top": {"dtype": "video", "shape": [48, 64, 3]},
        },
        "anomaly_plan": ANOMALIES,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return root


if __name__ == "__main__":
    import sys

    out = build(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ef_fixture"))
    print(f"fixture written to {out}")
