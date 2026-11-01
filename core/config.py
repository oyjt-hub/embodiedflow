from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default)).expanduser()


@dataclass(frozen=True)
class Settings:
    root: Path
    data_root: Path
    db_path: Path
    raw_dir: Path
    derived_dir: Path
    runs_dir: Path
    artifacts_dir: Path
    cache_dir: Path
    tmp_dir: Path
    host_label: str

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    def relative(self, p: Path) -> str:
        """制品在平台内的相对 URI（便于日后迁到对象存储）。"""
        try:
            return str(Path(p).resolve().relative_to(self.data_root.resolve()))
        except ValueError:
            return str(p)


def load_settings() -> Settings:
    home = Path.home()
    root = _path("EF_ROOT", str(home / "workspace" / "embodiedflow"))
    data = _path("EF_DATA", str(home / "workspace" / "embodiedflow_data"))
    return Settings(
        root=root,
        data_root=data,
        db_path=_path("EF_DB", str(data / "metadata" / "platform.db")),
        raw_dir=data / "raw",
        derived_dir=data / "derived",
        runs_dir=data / "runs",
        artifacts_dir=data / "artifacts",
        cache_dir=data / "cache",
        tmp_dir=data / "tmp",
        host_label=os.environ.get("EF_HOST", os.uname().nodename),
    )


settings = load_settings()


def ensure_dirs() -> list[Path]:
    made = []
    for d in (
        settings.data_root,
        settings.db_path.parent,
        settings.raw_dir,
        settings.derived_dir,
        settings.runs_dir,
        settings.artifacts_dir,
        settings.cache_dir,
        settings.tmp_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
        made.append(d)
    return made
