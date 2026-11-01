from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


class JobCreate(BaseModel):
    kind: str = Field(description="smoke|gpu_probe|pipeline|train|eval|sim")
    name: str = ""
    spec: dict = Field(default_factory=dict)
    requested_gpus: int = 0
    gpu_memory_mb: int = 0
    max_attempts: int = 2
    priority: int = 100
    created_by: str = "web"


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    kind: str
    name: str
    status: str
    priority: int
    requested_gpus: int
    created_by: str
    created_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    error: str | None = None


class AttemptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    job_id: int
    attempt_no: int
    status: str
    worker_id: str
    pid: int | None = None
    exit_code: int | None = None
    gpu_uuids: list | None = None
    log_uri: str | None = None
    started_at: dt.datetime
    finished_at: dt.datetime | None = None


class GPUOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    uuid: str
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization: int
    state: str
    host: str
    updated_at: dt.datetime


class LogOut(BaseModel):
    job_id: int
    attempt_id: int | None
    path: str
    lines: list[str]
