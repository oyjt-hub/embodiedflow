from __future__ import annotations

from enum import Enum


class JobKind(str, Enum):
    SMOKE = "smoke"
    GPU_PROBE = "gpu_probe"
    PIPELINE = "pipeline"
    TRAIN = "train"
    EVAL = "eval"
    SIM = "sim"


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = {
    JobStatus.COMPLETED.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELLED.value,
}


class AttemptStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GPUState(str, Enum):
    FREE = "free"          # 整卡空闲
    PARTIAL = "partial"    # 有外部占用，但剩余显存足够调度
    LEASED = "leased"      # 已被平台租给某个 attempt
    BUSY = "busy"          # 外部占满，剩余显存不足
    UNKNOWN = "unknown"


class SourceKind(str, Enum):
    REAL = "real"
    SIM = "sim"
    MOCK = "mock"
    DRY_RUN = "dry_run"
    PLANNED = "planned"
