"""模型制品摘要与登记（API / Worker 共用）。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy import select

from persistence.models import Artifact, Model, ModelVersion

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv"}


def tree_digest(path: Path, max_file_bytes: int = 8 * 1024 * 1024) -> tuple[str, int, int]:
    """目录可复现摘要：小文件取内容哈希，大文件只记 size，避免扫几十 GB。"""
    h = hashlib.sha256()
    files = 0
    total = 0
    root = Path(path)
    # 单文件也要能摘要（例如 checkpoint.pt）：rglob 在文件上返回空集，
    # 会得到空串摘要 e3b0c442...，看起来像"内容为空"。
    if root.is_file():
        size = root.stat().st_size
        h.update(f"{root.name}\0{size}\0".encode())
        if size <= max_file_bytes:
            h.update(hashlib.sha256(root.read_bytes()).digest())
        else:
            with root.open("rb") as fh:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(hashlib.sha256(chunk).digest())
        return h.hexdigest(), 1, size
    for p in sorted(root.rglob("*")):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = p.relative_to(path).as_posix()
        size = p.stat().st_size
        files += 1
        total += size
        h.update(f"{rel}\0{size}\0".encode())
        if size <= max_file_bytes:
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest(), files, total


def ensure_model(session, name: str, family: str = "openpi") -> Model:
    m = session.scalars(select(Model).where(Model.name == name)).first()
    if m is None:
        m = Model(name=name, family=family)
        session.add(m)
        session.flush()
    return m


def register_version(
    session,
    model: Model,
    version: str,
    weights_path: Path,
    *,
    attempt_id: int | None = None,
    mlflow_run_id: str | None = None,
    source: str = "real",
) -> tuple[ModelVersion, bool]:
    """登记模型版本；同摘要自动复用（不重复建版本）。返回 (版本, 是否新建)。"""
    digest, files, total = tree_digest(weights_path)
    existing = session.scalars(
        select(ModelVersion).where(ModelVersion.model_id == model.id, ModelVersion.sha256 == digest)
    ).first()
    if existing is not None:
        return existing, False
    mv = ModelVersion(
        model_id=model.id,
        version=version,
        source=source,
        attempt_id=attempt_id,
        mlflow_run_id=mlflow_run_id,
        weights_uri=str(weights_path),
        sha256=digest,
    )
    session.add(mv)
    session.flush()
    session.add(
        Artifact(
            job_attempt_id=attempt_id,
            kind="checkpoint",
            uri=str(weights_path),
            bytes=total,
            sha256=digest,
        )
    )
    return mv, True


def find_latest_checkpoint(root: str | Path, max_depth: int = 6) -> Path | None:
    """在目录树里找"像 checkpoint 的目录"，取最近修改的一个。

    识别依据（任一即可）：包含 `_CHECKPOINT_METADATA` 文件、`params` 子目录、
    或 `train_state` 子目录 —— openpi/Orbax 与常见训练框架都落在这些形态上。
    """
    base = Path(root).expanduser()
    if not base.exists():
        return None
    best: tuple[float, Path] | None = None
    for d in [base] + [p for p in base.rglob("*") if p.is_dir()]:
        try:
            if len(d.relative_to(base).parts) > max_depth:
                continue
        except ValueError:
            continue
        looks_like = (
            (d / "_CHECKPOINT_METADATA").exists()
            or (d / "params").is_dir()
            or (d / "train_state").is_dir()
        )
        if not looks_like:
            continue
        mtime = max((f.stat().st_mtime for f in d.rglob("*") if f.is_file()), default=0.0)
        if best is None or mtime > best[0]:
            best = (mtime, d)
    return best[1] if best else None
