"""FLORA 自动标注适配器。

- 真实实现：拿到 NVIDIA FLORA 权重/服务后填入 `FloraAdapter`
- 现在提供 `MockFlora`：确定性假标签，用于先跑通平台管道，
  界面与数据库都标注 model_version="mock"，不冒充真实标注。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class FloraLabel:
    episode_index: int
    kind: str
    payload: dict
    confidence: float


class MockFlora:
    """按 Episode 索引 deterministically 生成标签，便于复现与测试。"""

    model_name = "flora"
    model_version = "mock"

    def annotate(self, episode_index: int, kind: str = "episode") -> FloraLabel:
        h = int(hashlib.sha256(f"{episode_index}:{kind}".encode()).hexdigest()[:8], 16)
        conf = round(0.55 + (h % 450) / 1000.0, 3)          # 0.55 ~ 0.999
        success = (h % 3) != 0                              # 约 2/3 判成功
        payload = {
            "success": success,
            "task_label": ["pick", "place", "open", "stack"][h % 4],
            "source": "mock",
        }
        if kind == "time-range":
            payload["range_s"] = [round((h % 30) / 10, 1), round(((h % 30) / 10) + 2.5, 1)]
        return FloraLabel(episode_index, kind, payload, conf)


class FloraAdapter:
    """真实 FLORA 接入点（待权重）。保持与 MockFlora 相同的 annotate 签名。"""

    model_name = "flora"

    def __init__(self, model_version: str, endpoint: str | None = None) -> None:
        self.model_version = model_version
        self.endpoint = endpoint

    def annotate(self, episode_index: int, kind: str = "episode") -> FloraLabel:
        raise NotImplementedError("FLORA 权重/服务未接入，请先使用 MockFlora")
