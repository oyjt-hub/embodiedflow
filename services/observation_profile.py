"""训练与推理共用的观测画像（ObservationProfile）。

V08 缺口：掩码派生的空间关系此前只写进派生 Sidecar，训练和推理各自
不知道该怎么用它。这里把「哪些派生量进入观测、以什么顺序、怎么归一化」
固化成一份带版本号的画像，训练侧和推理侧都按同一份实现取值。

约定：
- profile 有唯一 name + version；version 变了就是不同的观测空间，
  训练产物不能跨版本直接推理。
- 每个字段声明 source（sidecar 里的键）与 scale（归一化除数）。
- 缺失字段一律填 0.0 并在 missing 里记录，不做静默插值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Field:
    name: str
    source: str
    scale: float = 1.0


@dataclass(frozen=True)
class ObservationProfile:
    name: str
    version: str
    fields: tuple[Field, ...] = field(default_factory=tuple)

    def keys(self) -> list[str]:
        return [f.name for f in self.fields]


# 只有图像统计量的最小画像
IMAGE_ONLY = ObservationProfile(
    name="image_only",
    version="1.0.0",
    fields=(),
)

# 图像 + 同帧掩码几何（V08 目标画像）
IMAGE_MASK_GEOMETRY = ObservationProfile(
    name="image_mask_geometry",
    version="1.0.0",
    fields=(
        Field("mask_present", "matched", 1.0),
        Field("mask_area_ratio", "area_ratio", 1.0),
        Field("center_dx_norm", "center_dx_norm", 1.0),
        Field("center_dy_norm", "center_dy_norm", 1.0),
        Field("center_distance_norm", "center_distance_norm", 1.0),
        Field("bbox_iou", "bbox_iou", 1.0),
        Field("left_right", "left_right", 1.0),
        Field("top_bottom", "top_bottom", 1.0),
        Field("time_delta_s", "time_delta_s", 0.1),
    ),
)

PROFILES: dict[str, ObservationProfile] = {
    p.name: p for p in (IMAGE_ONLY, IMAGE_MASK_GEOMETRY)
}

DEFAULT_PROFILE = IMAGE_MASK_GEOMETRY.name


def get_profile(name: str | None) -> ObservationProfile:
    key = name or DEFAULT_PROFILE
    if key not in PROFILES:
        raise KeyError(f"未知观测画像: {key}")
    return PROFILES[key]


def _coerce(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return {"left": -1.0, "right": 1.0, "above": 1.0,
                "below": -1.0, "same": 0.0, "unknown": 0.0}.get(value, 0.0)
    return 0.0


def vectorize(profile: ObservationProfile, sidecar: dict[str, Any]) -> dict[str, Any]:
    """把一份 sidecar 转成固定顺序的观测向量。

    返回 {profile, version, keys, values, missing}。missing 列出缺失字段，
    缺失位置的数值是 0.0 —— 调用方据此判断这条样本可用性。
    """
    values: list[float] = []
    missing: list[str] = []
    for f in profile.fields:
        raw = sidecar.get(f.source)
        if raw is None:
            missing.append(f.name)
            values.append(0.0)
            continue
        values.append(round(_coerce(raw) / (f.scale or 1.0), 6))
    return {
        "profile": profile.name,
        "version": profile.version,
        "keys": profile.keys(),
        "values": values,
        "missing": missing,
        "dim": len(values),
    }
