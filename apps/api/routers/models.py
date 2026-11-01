from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from persistence.db import session_scope
from persistence.models import Job, JobAttempt, Model, ModelVersion
from services.modeling import tree_digest

router = APIRouter(prefix="/models", tags=["models"])

class ModelIn(BaseModel):
    name: str
    family: str = "openpi"


class VersionIn(BaseModel):
    version: str
    weights_path: str
    source: str = "real"
    attempt_id: int | None = None
    mlflow_run_id: str | None = None


@router.get("")
def list_models():
    with session_scope() as s:
        out = []
        for m in s.scalars(select(Model).order_by(Model.id)):
            versions = list(
                s.scalars(select(ModelVersion).where(ModelVersion.model_id == m.id).order_by(ModelVersion.id))
            )
            out.append(
                {
                    "id": m.id,
                    "name": m.name,
                    "family": m.family,
                    "versions": [
                        {
                            "version": v.version,
                            "source": v.source,
                            "sha256": (v.sha256 or "")[:16],
                            "weights_uri": v.weights_uri,
                            "attempt_id": v.attempt_id,
                            "mlflow_run_id": v.mlflow_run_id,
                        }
                        for v in versions
                    ],
                }
            )
        return out


@router.post("", status_code=201)
def create_model(payload: ModelIn):
    with session_scope() as s:
        m = s.scalars(select(Model).where(Model.name == payload.name)).first()
        if m is None:
            m = Model(name=payload.name, family=payload.family)
            s.add(m)
            s.flush()
        return {"id": m.id, "name": m.name, "family": m.family}


@router.post("/{model_id}/versions", status_code=201)
def create_version(model_id: int, payload: VersionIn):
    path = Path(payload.weights_path).expanduser()
    if not path.exists():
        raise HTTPException(400, f"权重路径不存在: {path}")
    digest, files, total = tree_digest(path)

    with session_scope() as s:
        m = s.get(Model, model_id)
        if m is None:
            raise HTTPException(404, "model not found")
        existing = s.scalars(
            select(ModelVersion).where(
                ModelVersion.model_id == model_id, ModelVersion.sha256 == digest
            )
        ).first()
        if existing is not None:
            return {
                "deduplicated": True,
                "model_id": model_id,
                "version": existing.version,
                "sha256": digest,
                "message": "同一份权重（相同摘要）已登记",
            }
        if payload.attempt_id is not None and s.get(JobAttempt, payload.attempt_id) is None:
            raise HTTPException(400, "attempt_id 不存在")

        mv = ModelVersion(
            model_id=model_id,
            version=payload.version,
            source=payload.source,
            attempt_id=payload.attempt_id,
            mlflow_run_id=payload.mlflow_run_id,
            weights_uri=str(path),
            sha256=digest,
        )
        s.add(mv)
        s.flush()
        return {
            "model_id": model_id,
            "version": mv.version,
            "source": mv.source,
            "sha256": digest,
            "files": files,
            "bytes": total,
            "attempt_id": payload.attempt_id,
            "mlflow_run_id": payload.mlflow_run_id,
        }


@router.get("/{model_id}/versions/{version}/lineage")
def lineage(model_id: int, version: str):
    with session_scope() as s:
        mv = s.scalars(
            select(ModelVersion).where(
                ModelVersion.model_id == model_id, ModelVersion.version == version
            )
        ).first()
        if mv is None:
            raise HTTPException(404, "model version not found")
        out = {
            "model_version": version,
            "sha256": mv.sha256,
            "weights_uri": mv.weights_uri,
            "mlflow_run_id": mv.mlflow_run_id,
            "attempt": None,
            "job": None,
            "dataset_version_id": None,
        }
        if mv.attempt_id:
            att = s.get(JobAttempt, mv.attempt_id)
            if att:
                out["attempt"] = {
                    "id": att.id,
                    "attempt_no": att.attempt_no,
                    "status": att.status,
                    "exit_code": att.exit_code,
                }
                job = s.get(Job, att.job_id)
                if job:
                    out["job"] = {"id": job.id, "kind": job.kind, "name": job.name, "status": job.status}
                    out["dataset_version_id"] = job.dataset_version_id
        return out
