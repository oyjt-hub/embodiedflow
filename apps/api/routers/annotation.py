from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from persistence.db import session_scope
from services import annotation as ann

router = APIRouter(prefix="/annotation", tags=["annotation"])


class RunIn(BaseModel):
    dataset_version_id: int
    episode_indices: list[int]
    kind: str = "episode"
    spec_version: str = "v1"
    camera: str = "front"
    profile: str = "yam"
    instruction: str = ""


class ReviewIn(BaseModel):
    accept: bool
    reviewer: str


@router.get("/versions")
def list_versions():
    """供界面选择数据版本（id + 名称 + 状态）。"""
    from sqlalchemy import select
    from persistence.models import Dataset, DatasetVersion

    with session_scope() as s:
        rows = list(s.scalars(select(DatasetVersion).order_by(DatasetVersion.id.desc()).limit(50)))
        out = []
        for v in rows:
            d = s.get(Dataset, v.dataset_id)
            out.append({"id": v.id, "label": f"{d.name if d else '?'} / {v.version}",
                        "version": v.version, "status": v.status, "episodes": v.episode_count})
        return out


@router.post("/run", status_code=201)
def run(payload: RunIn):
    from apps.api.routers.vision import FloraJobIn, create_flora_job
    if not payload.episode_indices:
        raise HTTPException(422, "select at least one Episode")
    if payload.kind != "episode":
        raise HTTPException(422, "FLORA supports episode temporal annotations")
    result = create_flora_job(FloraJobIn(
        dataset_version_id=payload.dataset_version_id,
        episode=payload.episode_indices[0], episodes=payload.episode_indices,
        camera=payload.camera, profile=payload.profile, instruction=payload.instruction,
    ))
    return {**result, "model": "NVIDIA FLORA", "execution": "gpu_worker",
            "result_url": f"/workbench/api/vision/flora/jobs/{result['job_id']}/result"}


@router.get("/sampling/{dataset_version_id}")
def sampling(dataset_version_id: int):
    with session_scope() as s:
        return ann.sampling_plan(s, dataset_version_id)


@router.post("/annotations/{annotation_id}/review")
def review(annotation_id: int, payload: ReviewIn):
    try:
        with session_scope() as s:
            return ann.review(s, annotation_id, payload.accept, payload.reviewer)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/stats/{dataset_version_id}")
def stats(dataset_version_id: int):
    with session_scope() as s:
        return ann.stats(s, dataset_version_id)


@router.post("/publish", status_code=201)
def publish(dataset_version_id: int, version: str):
    try:
        with session_scope() as s:
            d = ann.publish(s, dataset_version_id, version)
            return {"dataset_version_id": d.id, "version": d.version,
                    "status": d.status, "episodes": d.episode_count}
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
