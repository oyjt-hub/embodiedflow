"""OpenPI 训练适配器：把平台 JobSpec 翻译成受控的上游训练调用。

支持两种模式：
  - dry_run：只做物料校验（配置存在、数据/权重路径可达），不启动训练
  - run    ：用独立子进程调用锁定版本的官方训练入口

上游命令约定（openpi 用 tyro CLI）：
  python scripts/train.py <config_name> --exp-name X --num-train-steps N ...
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OpenPIProbe:
    ok: bool
    repo: str
    config: str
    command: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dataset_path: str | None = None
    weights_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "repo": self.repo,
            "config": self.config,
            "command": " ".join(shlex.quote(c) for c in self.command),
            "missing": self.missing,
            "warnings": self.warnings,
            "dataset_path": self.dataset_path,
            "weights_path": self.weights_path,
            "mode": "dry_run",
        }


class OpenPIAdapter:
    """默认 dry_run；只有显式 execute=True 才真的起训练。"""

    def __init__(self, repo: str | Path, python: str | Path) -> None:
        self.repo = Path(repo).expanduser()
        self.python = str(python)

    def prepare(self, spec: dict) -> OpenPIProbe:
        config = str(spec.get("config") or "")
        missing: list[str] = []
        warnings: list[str] = []

        train_py = self.repo / "scripts" / "train.py"
        if not train_py.is_file():
            missing.append(f"训练入口缺失: {train_py}")
        if not config:
            missing.append("缺少 spec.config（openpi 训练配置名）")

        cfg_file = self.repo / "src" / "openpi" / "training" / "config.py"
        if config and cfg_file.is_file():
            if config not in cfg_file.read_text(encoding="utf-8", errors="ignore"):
                missing.append(f"配置 {config} 未在 config.py 中注册")

        dataset = spec.get("dataset_path")
        if dataset:
            dataset = str(Path(dataset).expanduser())
            if not Path(dataset).is_dir():
                missing.append(f"数据集目录不存在: {dataset}")
        else:
            warnings.append("未指定 dataset_path，将依赖配置里写死的 repo_id（需要 HF 缓存可用）")

        weights = spec.get("weights_path")
        if weights:
            weights = str(Path(weights).expanduser())
            if not Path(weights).is_dir():
                missing.append(f"基础权重目录不存在: {weights}")
        else:
            warnings.append("未指定 weights_path，将使用配置中的 weight_loader")

        cmd = [
            self.python, "scripts/train.py", config,
            "--exp-name", str(spec.get("exp_name") or "embodiedflow-job"),
            "--num-train-steps", str(int(spec.get("num_train_steps", 1))),
            "--batch-size", str(int(spec.get("batch_size", 1))),
            "--overwrite",
        ]
        if spec.get("resume"):
            cmd.append("--resume")

        return OpenPIProbe(
            ok=not missing,
            repo=str(self.repo),
            config=config,
            command=cmd,
            missing=missing,
            warnings=warnings,
            dataset_path=dataset,
            weights_path=weights,
        )
