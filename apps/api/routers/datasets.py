from __future__ import annotations

import hashlib
import json
from pathlib import Path

import hashlib
import shutil
import tarfile
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select

from adapters.datasets.lerobot import LeRobotAdapter
from core.config import settings
from persistence.db import session_scope
from persistence.models import (
    Dataset,
    DatasetVersion,
    Episode,
    Job,
    ModelVersion,
    QualityReport,
)
from services import align, quality

router = APIRouter(prefix="/datasets", tags=["datasets"])


class RegisterIn(BaseModel):
    name: str
    path: str
    version: str = "v1"
    robot_spec: str | None = None
    owner: str = "oyjt"


class ValidateIn(BaseModel):
    version: str
    decode_frames: int = Field(default=40, ge=0, le=200)
    log_mlflow: bool = False
    max_episodes: int = 0


class ProcessIn(BaseModel):
    version: str
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 0


def _dataset(s, dataset_id: int) -> Dataset:
    d = s.get(Dataset, dataset_id)
    if d is None:
        raise HTTPException(404, "dataset not found")
    return d


def _version(s, dataset_id: int, version: str) -> DatasetVersion:
    v = s.scalars(
        select(DatasetVersion).where(
            DatasetVersion.dataset_id == dataset_id, DatasetVersion.version == version
        )
    ).first()
    if v is None:
        raise HTTPException(404, f"version {version} not found")
    return v


@router.post("/upload", status_code=201)
async def upload(
    file: UploadFile = File(...),
    name: str = Form(...),
    version: str = Form("v1"),
    robot_spec: str | None = Form(None),
):
    """上传 tar.gz 数据集。中断上传只会在 tmp 留下半成品，绝不进入可训练区。"""
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    bytes_written = 0

    fd, tmp_path = tempfile.mkstemp(dir=settings.tmp_dir, suffix=".tar.gz")
    try:
        with open(fd, "wb") as out:
            while True:
                chunk = await file.read(4 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                bytes_written += len(chunk)
        sha = digest.hexdigest()

        dest = settings.raw_dir / name / version
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        # 安全解包：禁止绝对路径与 ..
        with tarfile.open(tmp_path, "r:gz") as tf:
            for member in tf.getmembers():
                target = (dest / member.name).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise HTTPException(400, f"非法路径: {member.name}")
            tf.extractall(dest)

        # 允许 tar 里多包一层目录
        root = dest
        if not (root / "meta" / "info.json").exists():
            subs = [d for d in dest.iterdir() if d.is_dir()]
            if len(subs) == 1 and (subs[0] / "meta" / "info.json").exists():
                root = subs[0]

        adapter = LeRobotAdapter(root)
        metas = adapter.episode_metas()
        total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())

        with session_scope() as s:
            ds = s.scalars(select(Dataset).where(Dataset.name == name)).first()
            if ds is None:
                ds = Dataset(name=name, format="lerobot", robot_spec=robot_spec, owner="upload")
                s.add(ds)
                s.flush()
            ver = DatasetVersion(
                dataset_id=ds.id, version=version, status="ready",
                episode_count=len(metas), total_bytes=total,
                manifest_uri=str(root / "meta" / "info.json"), sha256=sha,
                episode_indices=json.dumps([m.episode_index for m in metas]),
            )
            s.add(ver)
            s.flush()
            return {"dataset_id": ds.id, "version": ver.version, "episodes": len(metas),
                    "bytes": total, "upload_sha256": sha, "path": str(root)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/register", status_code=201)
def register(payload: RegisterIn):
    root = Path(payload.path).expanduser()
    try:
        adapter = LeRobotAdapter(root)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc

    info_path = root / "meta" / "info.json"
    digest = hashlib.sha256(info_path.read_bytes()).hexdigest()
    metas = adapter.episode_metas()
    total_bytes = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())

    with session_scope() as s:
        ds = s.scalars(select(Dataset).where(Dataset.name == payload.name)).first()
        if ds is None:
            ds = Dataset(name=payload.name, format="lerobot", robot_spec=payload.robot_spec,
                         owner=payload.owner)
            s.add(ds)
            s.flush()

        existing = s.scalars(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == ds.id,
                DatasetVersion.sha256 == digest,
            )
        ).first()
        if existing is not None:
            return {
                "dataset_id": ds.id,
                "version": existing.version,
                "deduplicated": True,
                "message": "同一份 meta/info.json（相同 sha256）已登记，未重复创建版本",
            }

        ver = DatasetVersion(
            dataset_id=ds.id,
            version=payload.version,
            status="ready",
            episode_count=len(metas),
            episode_indices=json.dumps([m.episode_index for m in metas]),
            total_bytes=total_bytes,
            manifest_uri=str(root / "meta" / "info.json"),
            sha256=digest,
        )
        s.add(ver)
        s.flush()

        for meta in metas:
            s.add(
                Episode(
                    dataset_version_id=ver.id,
                    episode_index=int(meta["episode_index"]),
                    task=(meta.get("tasks") or [None])[0],
                    n_frames=meta.get("length"),
                    fps=adapter.fps,
                    duration_s=(meta["length"] / adapter.fps) if meta.get("length") and adapter.fps else None,
                    storage_prefix=str(root),
                )
            )
        return {
            "dataset_id": ds.id,
            "version": ver.version,
            "episodes": len(metas),
            "bytes": total_bytes,
            "sha256": digest,
            "fps": adapter.fps,
            "path": str(root),
        }


@router.get("/{dataset_id}")
def detail(dataset_id: int):
    with session_scope() as s:
        ds = _dataset(s, dataset_id)
        versions = list(
            s.scalars(
                select(DatasetVersion)
                .where(DatasetVersion.dataset_id == dataset_id)
                .order_by(DatasetVersion.id.desc())
            )
        )
        return {
            "id": ds.id,
            "name": ds.name,
            "format": ds.format,
            "robot_spec": ds.robot_spec,
            "owner": ds.owner,
            "versions": [
                {
                    "version": v.version,
                    "status": v.status,
                    "episodes": v.episode_count,
                    "bytes": v.total_bytes,
                    "sha256": v.sha256,
                    "path": v.manifest_uri,
                }
                for v in versions
            ],
        }


@router.post("/{dataset_id}/validate")
def validate(dataset_id: int, payload: ValidateIn):
    with session_scope() as s:
        ds = _dataset(s, dataset_id)
        ver = _version(s, dataset_id, payload.version)
        root = Path(ver.manifest_uri).parent.parent
        ds_name, ver_name = ds.name, ver.version
        ver_id = ver.id

    adapter = LeRobotAdapter(root, decode_frames=payload.decode_frames)
    report = quality.build_report(adapter, max_episodes=(payload.max_episodes or None))

    out_dir = settings.artifacts_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ds_name}_{ver_name}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with session_scope() as s:
        row = QualityReport(
            dataset_version_id=ver_id,
            rules_version=report["rules_version"],
            episodes=report["episodes"],
            finding_count=len(report["findings"]),
            summary={"metrics": report["metrics"], "rule_hits": report["rule_hits"],
                     "by_episode": report["by_episode"]},
            report_uri=settings.relative(out_path),
        )
        s.add(row)

    mlflow_run_id = None
    if payload.log_mlflow:
        from services import tracking

        if tracking.available():
            metrics = {
                k: (m["value"] if m["value"] is not None else 0.0)
                for k, m in report["metrics"].items()
            }
            mlflow_run_id = tracking.log_run(
                experiment="data-quality",
                run_name=f"{ds_name}-{ver_name}",
                params={
                    "dataset": ds_name,
                    "version": ver_name,
                    "rules_version": report["rules_version"],
                    "episodes": report["episodes"],
                    "findings": len(report["findings"]),
                },
                metrics=metrics,
                artifacts=[out_path],
                tags={"source": "embodiedflow.p0.2", "rules_version": report["rules_version"]},
            )
    return {
        "mlflow_run_id": mlflow_run_id,
        "dataset_id": dataset_id,
        "version": payload.version,
        "episodes": report["episodes"],
        "findings": len(report["findings"]),
        "rule_hits": report["rule_hits"],
        "metrics": report["metrics"],
        "by_episode": report["by_episode"],
        "report_uri": settings.relative(out_path),
    }


@router.get("/{dataset_id}/quality-report")
def quality_report(dataset_id: int, version: str):
    with session_scope() as s:
        _dataset(s, dataset_id)
        ver = _version(s, dataset_id, version)
        row = s.scalars(
            select(QualityReport)
            .where(QualityReport.dataset_version_id == ver.id)
            .order_by(QualityReport.id.desc())
        ).first()
        if row is None:
            raise HTTPException(404, "尚未生成质检报告，请先调用 /validate")
        path = settings.data_root / row.report_uri
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/{dataset_id}/process")
def process(dataset_id: int, payload: ProcessIn):
    with session_scope() as s:
        ds = _dataset(s, dataset_id)
        ver = _version(s, dataset_id, payload.version)
        root = Path(ver.manifest_uri).parent.parent
        ds_name, ver_name = ds.name, ver.version

    adapter = LeRobotAdapter(root)
    metas = adapter.episode_metas()
    indices = [int(m["episode_index"]) for m in metas]
    split = align.split_episodes(indices, payload.ratios, payload.seed)

    train_eps = [adapter.read_episode(i) for i in split["train"]]
    stats = align.fit_norm_stats(train_eps)

    align_probe = align.align_to_camera_clock(adapter.read_episode(indices[0]), max_dt=0.05)
    probe_summary = {
        name: {
            "interpolation": v.get("interpolation"),
            "target_hz": v.get("target_hz"),
            "tolerance_s": v.get("tolerance_s"),
            "max_residual_s": round(v.get("max_residual_s", 0.0), 6),
            "invalid_frames": v.get("invalid_frames"),
            "aligned_rows": int(len(v["t"])) if v.get("t") is not None else 0,
        }
        for name, v in align_probe.items()
    }

    manifest = {
        "dataset": ds_name,
        "version": ver_name,
        "seed": payload.seed,
        "ratios": payload.ratios,
        "split": split,
        "split_sizes": {k: len(v) for k, v in split.items()},
        "normalization": stats,
        "alignment_probe_episode": indices[0],
        "alignment": probe_summary,
    }
    out_dir = settings.artifacts_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ds_name}_{ver_name}.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["uri"] = settings.relative(out_path)
    return manifest


@router.delete("/{dataset_id}/versions/{version}")
def delete_version(dataset_id: int, version: str):
    with session_scope() as s:
        _dataset(s, dataset_id)
        ver = _version(s, dataset_id, version)
        refs = int(
            s.scalar(
                __import__("sqlalchemy").select(__import__("sqlalchemy").func.count())
                .select_from(Job)
                .where(Job.dataset_version_id == ver.id)
            ) or 0
        )
        # 只统计"从该数据版本训练出来"的模型版本（attempt → job → dataset_version）
        model_refs = int(
            s.scalar(
                __import__("sqlalchemy").select(__import__("sqlalchemy").func.count())
                .select_from(ModelVersion)
                .join(JobAttempt, ModelVersion.attempt_id == JobAttempt.id)
                .join(Job, JobAttempt.job_id == Job.id)
                .where(Job.dataset_version_id == ver.id)
            ) or 0
        )
        if refs or model_refs:
            raise HTTPException(
                409,
                f"版本被引用，拒绝删除：jobs={refs}, model_versions={model_refs}",
            )
        s.delete(ver)
    return {"deleted": True, "dataset_id": dataset_id, "version": version}
