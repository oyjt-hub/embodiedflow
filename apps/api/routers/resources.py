from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from apps.api.schemas import GPUOut, JobOut
from core.enums import JobStatus
from persistence.db import session_scope
from persistence.models import GPUResource, Job
from services import resources as ressvc

router = APIRouter(prefix="/resources", tags=["resources"])


@router.get("/gpus")
def gpus(refresh: bool = True):
    with session_scope() as s:
        if refresh:
            ressvc.sync_gpus(s)
        return ressvc.gpu_summary(s)


@router.get("/jobs", response_model=list[JobOut])
def running_jobs():
    with session_scope() as s:
        stmt = select(Job).where(Job.status == JobStatus.RUNNING.value).order_by(Job.id.desc())
        return [JobOut.model_validate(j) for j in s.scalars(stmt)]
