"""多频对齐、切分与归一化（归一化只拟合训练集）。"""
from __future__ import annotations

import numpy as np

from adapters.datasets.lerobot import EpisodeData


def align_to_camera_clock(
    ep: EpisodeData,
    target_hz: float | None = None,
    max_dt: float = 0.05,
    discrete_prefixes: tuple[str, ...] = ("gripper",),
) -> dict:
    """把 100 Hz 动作/状态对齐到相机时钟；离散量保持，连续量线性插值。"""
    hz = target_hz or ep.fps or 30.0
    cam = next((s for n, s in ep.streams.items() if s.kind == "image"), None)
    if cam is None or not len(cam.t):
        base_t = np.arange(ep.dominant_length(), dtype=np.float64) / hz
    else:
        base_t = np.asarray(cam.t, dtype=np.float64)

    out: dict[str, dict] = {}
    for name, s in ep.streams.items():
        if s.kind == "image" or s.values is None or not s.length:
            continue
        src_t = np.asarray(s.t, dtype=np.float64)
        ok = np.isfinite(src_t)
        if ok.sum() < 2:
            out[name] = {"t": base_t, "values": None, "mask": np.zeros(base_t.shape, bool),
                         "reason": "时间戳有效点不足"}
            continue
        src_t, vals = src_t[ok], np.asarray(s.values, dtype=np.float64)[ok]
        is_discrete = any(p in name for p in discrete_prefixes)
        if is_discrete:
            idx = np.clip(np.searchsorted(src_t, base_t, side="right") - 1, 0, len(src_t) - 1)
            aligned = vals[idx]
            residual = np.abs(base_t - src_t[idx])
        else:
            aligned = np.stack(
                [np.interp(base_t, src_t, vals[:, d]) for d in range(vals.shape[1])], axis=1
            ) if vals.ndim > 1 else np.interp(base_t, src_t, vals)
            nearest = np.clip(np.searchsorted(src_t, base_t), 0, len(src_t) - 1)
            residual = np.abs(base_t - src_t[nearest])
        out[name] = {
            "t": base_t,
            "values": aligned,
            "mask": residual <= max_dt,
            "max_residual_s": float(np.max(residual)) if residual.size else 0.0,
            "invalid_frames": int((residual > max_dt).sum()),
            "interpolation": "hold" if is_discrete else "linear",
            "target_hz": hz,
            "tolerance_s": max_dt,
        }
    return out


def split_episodes(
    indices: list[int],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, list[int]]:
    """按 Episode 切分，重跑结果必须完全一致（同一 seed）。"""
    total = sum(ratios)
    ratios = tuple(r / total for r in ratios)
    rng = np.random.default_rng(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train:n_train + n_val])
    test = sorted(shuffled[n_train + n_val:])
    return {"train": train, "val": val, "test": test}


def fit_norm_stats(
    episodes: list[EpisodeData],
    keys: tuple[str, ...] = ("observation.state", "action"),
) -> dict:
    """只用训练集拟合 mean/std。"""
    stats: dict[str, dict] = {}
    for key in keys:
        chunks = []
        for ep in episodes:
            s = ep.streams.get(key)
            if s is not None and s.values is not None and s.length:
                chunks.append(np.asarray(s.values, dtype=np.float64))
        if not chunks:
            continue
        arr = np.concatenate(chunks, axis=0)
        stats[key] = {
            "mean": np.mean(arr, axis=0).round(6).tolist(),
            "std": np.std(arr, axis=0).round(6).tolist(),
            "count": int(arr.shape[0]),
            "fitted_on": "train_only",
        }
    return stats
