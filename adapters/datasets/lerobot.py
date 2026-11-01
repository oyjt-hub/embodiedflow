"""LeRobot 数据集读取器（v2.1 布局：meta + data parquet + videos mp4）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMAGE_PREFIX = "observation.images."
# 关节顺序 / 单位 / 坐标系由本体规范提供；缺省时按维度占位并标记 unknown
DEFAULT_UNITS = {"observation.state": "rad", "action": "rad"}


@dataclass
class StreamData:
    name: str
    t: np.ndarray
    kind: str                       # continuous | discrete | image
    values: np.ndarray | None = None
    units: str = ""
    frame: str = ""
    dim: int = 0
    nominal_hz: float = 0.0
    video_path: str | None = None
    frames: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        """逻辑长度：数值流=行数；图像流=真实帧数（时间轴长度，不受解码上限影响）。"""
        if self.values is not None:
            return int(self.values.shape[0])
        if self.kind == "image":
            return int(len(self.t))
        if self.frames is not None:
            return int(self.frames.shape[0])
        return int(len(self.t))


@dataclass
class EpisodeData:
    index: int
    task: str
    declared_length: int | None
    fps: float
    streams: dict[str, StreamData]
    meta: dict = field(default_factory=dict)

    def dominant_length(self) -> int:
        lens = [s.length for s in self.streams.values() if s.kind != "image"]
        if not lens:
            lens = [s.length for s in self.streams.values()]
        if not lens:
            return 0
        vals, counts = np.unique(lens, return_counts=True)
        return int(vals[int(np.argmax(counts))])

    def get(self, name: str) -> StreamData | None:
        return self.streams.get(name)


class LeRobotAdapter:
    """只读适配器：绝不修改原始目录。"""

    def __init__(self, root: str | Path, decode_frames: int = 0) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {self.root}")
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"meta/info.json missing in {self.root}")
        self._info = json.loads(info_path.read_text(encoding="utf-8"))
        self.decode_frames = decode_frames

    # ---------------- meta ----------------
    def info(self) -> dict:
        return self._info

    @property
    def fps(self) -> float:
        return float(self._info.get("fps", 0) or 0)

    @property
    def total_episodes(self) -> int:
        return int(self._info.get("total_episodes", 0) or 0)

    def tasks(self) -> dict[int, str]:
        p = self.root / "meta" / "tasks.jsonl"
        out: dict[int, str] = {}
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    out[int(row["task_index"])] = str(row.get("task", ""))
        return out

    def episode_metas(self) -> list[dict]:
        p = self.root / "meta" / "episodes.jsonl"
        rows: list[dict] = []
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        if not rows:
            rows = [{"episode_index": i, "length": None} for i in range(self.total_episodes)]
        return rows

    # ---------------- data ----------------
    def _parquet_for(self, idx: int) -> Path | None:
        cands = sorted((self.root / "data").glob(f"chunk-*/episode_{idx:06d}.parquet"))
        if cands:
            return cands[0]
        cands = sorted((self.root / "data").glob(f"**/episode_{idx:06d}.parquet"))
        return cands[0] if cands else None

    def _video_for(self, idx: int, camera: str) -> Path | None:
        cam = f"{IMAGE_PREFIX}{camera}"
        cands = sorted((self.root / "videos").glob(f"chunk-*/{cam}/episode_{idx:06d}.mp4"))
        if cands:
            return cands[0]
        cands = sorted((self.root / "videos").glob(f"**/{cam}/episode_{idx:06d}.mp4"))
        return cands[0] if cands else None

    def read_episode(self, idx: int, task_map: dict[int, str] | None = None) -> EpisodeData:
        import pyarrow.parquet as pq

        metas = {int(m["episode_index"]): m for m in self.episode_metas()}
        meta = metas.get(idx, {})
        declared = meta.get("length")
        task_map = task_map if task_map is not None else self.tasks()

        streams: dict[str, StreamData] = {}
        pq_path = self._parquet_for(idx)
        rows = 0
        if pq_path is not None:
            table = pq.read_table(pq_path)
            data = {c: table.column(c).to_pylist() for c in table.column_names}
            rows = table.num_rows
            t = np.asarray(data.get("timestamp", []), dtype=np.float64)
            if t.size == 0:
                t = np.arange(rows, dtype=np.float64) / (self.fps or 1.0)
            for col, values in data.items():
                if col in {"timestamp", "frame_index", "episode_index", "index", "task_index"}:
                    continue
                arr = np.asarray(values)
                if arr.ndim == 1 and arr.dtype == object:
                    continue
                kind = "continuous"
                if "gripper" in col:
                    kind = "discrete"
                streams[col] = StreamData(
                    name=col,
                    t=t,
                    kind=kind,
                    values=arr,
                    units=DEFAULT_UNITS.get(col, ""),
                    dim=int(arr.shape[1]) if arr.ndim > 1 else 1,
                    nominal_hz=self.fps,
                )

        # 视频流（不解码时只登记路径与名义时间轴）
        for video_dir in sorted((self.root / "videos").glob("**/observation.images.*")):
            camera = video_dir.name[len(IMAGE_PREFIX):]
            vpath = self._video_for(idx, camera)
            if vpath is None:
                continue
            n = int(declared or rows or 0)
            try:  # 容器内真实帧数（只读元数据，不解码）
                import imageio_ffmpeg

                true_n, _secs = imageio_ffmpeg.count_frames_and_secs(str(vpath))
                if true_n and true_n > 0:
                    n = int(true_n)
            except Exception:  # noqa: BLE001
                pass
            offset = float(meta.get("camera_time_offset", 0.0) or 0.0)
            t = np.arange(n, dtype=np.float64) / (self.fps or 1.0) + offset
            sd = StreamData(
                name=f"{IMAGE_PREFIX}{camera}",
                t=t,
                kind="image",
                nominal_hz=self.fps,
                video_path=str(vpath),
            )
            if self.decode_frames:
                sd.frames = _decode_frames(vpath, self.decode_frames)
                decoded = int(sd.frames.shape[0]) if sd.frames is not None else 0
                sd.meta["decoded_frames"] = decoded
                sd.meta["decode_truncated"] = bool(decoded >= self.decode_frames and n > self.decode_frames)
            streams[sd.name] = sd

        task = ""
        if meta.get("tasks"):
            task = str(meta["tasks"][0])
        return EpisodeData(
            index=idx,
            task=task,
            declared_length=int(declared) if declared is not None else None,
            fps=self.fps,
            streams=streams,
            meta=meta,
        )


def _decode_frames(path: Path, limit: int) -> np.ndarray:
    """解码前 limit 帧为 (N,H,W,3) uint8；失败返回空数组。"""
    try:
        import imageio.v3 as iio

        frames = []
        for i, frame in enumerate(iio.imiter(str(path), plugin="pyav")):
            frames.append(np.asarray(frame)[..., :3])
            if i + 1 >= limit:
                break
        return np.stack(frames) if frames else np.zeros((0, 1, 1, 3), dtype=np.uint8)
    except Exception:  # noqa: BLE001
        try:
            import imageio.v2 as iio2

            rd = iio2.get_reader(str(path))
            frames = [np.asarray(f)[..., :3] for _, f in zip(range(limit), rd)]
            rd.close()
            return np.stack(frames) if frames else np.zeros((0, 1, 1, 3), dtype=np.uint8)
        except Exception:  # noqa: BLE001
            return np.zeros((0, 1, 1, 3), dtype=np.uint8)
