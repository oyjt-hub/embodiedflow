"""模型家族注册表：声明每个 VLA 家族怎么跑、权重与入口在哪、当前是否可用。

设计原则：**如实标注**。路径不存在就标 missing，只有评测配置就标 eval_only，
绝不让界面显示成"可直接训练"。
"""
from __future__ import annotations

from pathlib import Path

FAMILIES: dict[str, dict] = {
    "openpi": {
        "label": "OpenPI (π0.5)",
        "adapter": "adapters.training.openpi",
        "kind": "train+infer",
        "repo": "/home/oyjt/workspace/openpi_v2",
        "entry": "scripts/train.py",
        "weights": ["/home/oyjt/workspace/checkpoints/4items_pi05base_rtc_v2_all_lora"],
        "notes": "已跑通 1 步真实训练（debug_pi05）",
    },
    "bc": {
        "label": "行为克隆基线",
        "adapter": "adapters.training.local_bc",
        "kind": "train",
        "notes": "小显存/CPU 可跑的可复验基线",
    },
    "holobrain": {
        "label": "HoloBrain 0.5 (sim)",
        "adapter": "docker:holobrain_srv",
        "kind": "train+infer",
        "weights": ["/home/oyjt/models/HoloBrain_v0.5_sim"],
        "hf_repo": "HorizonRobotics/HoloBrain_v0.5_sim",
        "images": ["holobrain_srv:v0.5", "holobrain_srv:v0.6", "holobrain_srv:v0.7", "holobrain_srv:v0.8"],
        "train_entry": "python -u -m projects.holobrain_0_5.train --config /workspace/hb_pkg/configs/config_holobrain_0_5.py",
        "notes": "容器内训练；本机曾观测到 hb_e10/hb_e10b/hb_e10c 连续重启调参",
    },
    "lingbotvla": {
        "label": "LingBot-VLA",
        "adapter": "rlinf-eval",
        "kind": "eval",
        "config": "/home/oyjt/RLinf/evaluations/robotwin/robotwin_click_bell_lingbotvla_eval.yaml",
        "notes": "目前只有 RLinf 的 RoboTwin 评测配置；未发现独立权重目录",
    },
}


def _exists(p: str) -> bool:
    return Path(p).expanduser().exists()


def describe(name: str) -> dict:
    f = dict(FAMILIES[name])
    f["name"] = name
    f["weights_found"] = [w for w in f.get("weights", []) if _exists(w)]
    f["weights_missing"] = [w for w in f.get("weights", []) if not _exists(w)]
    if f.get("config"):
        f["config_found"] = _exists(f["config"])
    if f.get("repo"):
        f["repo_found"] = _exists(f["repo"])
    usable = bool(f["weights_found"]) or bool(f.get("repo_found")) or bool(f.get("config_found"))
    f["status"] = "available" if usable and f["kind"].startswith("train") else (
        "eval_only" if usable else "missing"
    )
    return f


def all_families() -> list[dict]:
    return [describe(n) for n in FAMILIES]
