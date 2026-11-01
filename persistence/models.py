from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class SchemaVersion(Base):
    __tablename__ = "schema_version"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    applied_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    note: Mapped[str] = mapped_column(String(200), default="")


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    role: Mapped[str] = mapped_column(String(16), default="engineer")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    format: Mapped[str] = mapped_column(String(32), default="lerobot")
    robot_spec: Mapped[str | None] = mapped_column(String(64))
    owner: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    version: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="draft")
    episode_count: Mapped[int] = mapped_column(Integer, default=0)
    total_bytes: Mapped[int] = mapped_column(Integer, default=0)
    manifest_uri: Mapped[str | None] = mapped_column(String(512))
    episode_indices: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    __table_args__ = (UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),)


class Episode(Base):
    __tablename__ = "episodes"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True
    )
    episode_index: Mapped[int] = mapped_column(Integer)
    task: Mapped[str | None] = mapped_column(Text)
    n_frames: Mapped[int | None] = mapped_column(Integer)
    duration_s: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)
    success: Mapped[bool | None] = mapped_column(Boolean)
    storage_prefix: Mapped[str | None] = mapped_column(String(512))
    quality_flags: Mapped[dict | None] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("dataset_version_id", "episode_index", name="uq_episode"),)


class PipelineTask(Base):
    __tablename__ = "pipeline_tasks"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id"))
    input_version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"))
    output_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_versions.id")
    )
    steps: Mapped[dict | None] = mapped_column(JSON)
    report_uri: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class Job(Base):
    """统一任务队列。计划中的 TrainingJob 即 kind='train' 的 Job。"""

    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    priority: Mapped[int] = mapped_column(Integer, default=100)
    spec: Mapped[dict] = mapped_column(JSON, default=dict)
    requested_gpus: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=2)
    dataset_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_versions.id")
    )
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))
    experiment_id: Mapped[int | None] = mapped_column(ForeignKey("experiments.id"))
    created_by: Mapped[str] = mapped_column(String(64), default="system")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    error: Mapped[str | None] = mapped_column(Text)


class JobAttempt(Base):
    __tablename__ = "job_attempts"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    attempt_no: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="running")
    worker_id: Mapped[str] = mapped_column(String(64), default="")
    pid: Mapped[int | None] = mapped_column(Integer)
    exit_code: Mapped[int | None] = mapped_column(Integer)
    gpu_uuids: Mapped[list | None] = mapped_column(JSON)
    log_uri: Mapped[str | None] = mapped_column(String(512))
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    heartbeat_at: Mapped[dt.datetime | None] = mapped_column(DateTime)


class GPUResource(Base):
    __tablename__ = "gpu_resources"
    uuid: Mapped[str] = mapped_column(String(64), primary_key=True)
    index: Mapped[int] = mapped_column(Integer, default=0)
    name: Mapped[str] = mapped_column(String(64), default="")
    memory_total_mb: Mapped[int] = mapped_column(Integer, default=0)
    memory_used_mb: Mapped[int] = mapped_column(Integer, default=0)
    utilization: Mapped[int] = mapped_column(Integer, default=0)
    state: Mapped[str] = mapped_column(String(16), default="unknown")
    host: Mapped[str] = mapped_column(String(64), default="")
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class GPULease(Base):
    __tablename__ = "gpu_leases"
    id: Mapped[int] = mapped_column(primary_key=True)
    gpu_uuid: Mapped[str] = mapped_column(ForeignKey("gpu_resources.uuid"), index=True)
    attempt_id: Mapped[int] = mapped_column(
        ForeignKey("job_attempts.id", ondelete="CASCADE"), index=True
    )
    acquired_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    released_at: Mapped[dt.datetime | None] = mapped_column(DateTime)


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    project: Mapped[str] = mapped_column(String(64), default="default")
    mlflow_experiment_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class Model(Base):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    family: Mapped[str] = mapped_column(String(64), default="openpi")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"))
    version: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(16), default="real")
    attempt_id: Mapped[int | None] = mapped_column(ForeignKey("job_attempts.id"))
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    weights_uri: Mapped[str | None] = mapped_column(String(512))
    sha256: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_model_version"),)


class Artifact(Base):
    __tablename__ = "artifacts"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_attempt_id: Mapped[int | None] = mapped_column(ForeignKey("job_attempts.id"))
    kind: Mapped[str] = mapped_column(String(32), default="file")
    uri: Mapped[str] = mapped_column(String(512))
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class Device(Base):
    __tablename__ = "devices"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    robot_spec: Mapped[str] = mapped_column(String(64), default="piper")
    source: Mapped[str] = mapped_column(String(16), default="mock")
    state: Mapped[str] = mapped_column(String(16), default="offline")
    dof: Mapped[int | None] = mapped_column(Integer)
    control_hz: Mapped[float | None] = mapped_column(Float)
    config: Mapped[dict | None] = mapped_column(JSON)
    last_seen_at: Mapped[dt.datetime | None] = mapped_column(DateTime)


class Deployment(Base):
    __tablename__ = "deployments"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    device_id: Mapped[int] = mapped_column(ForeignKey("devices.id"))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    safety: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    stopped_at: Mapped[dt.datetime | None] = mapped_column(DateTime)


class Benchmark(Base):
    __tablename__ = "benchmarks"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    task: Mapped[str] = mapped_column(String(128), default="")


class BenchmarkVersion(Base):
    __tablename__ = "benchmark_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    benchmark_id: Mapped[int] = mapped_column(ForeignKey("benchmarks.id", ondelete="CASCADE"))
    version: Mapped[str] = mapped_column(String(32))
    engine: Mapped[str] = mapped_column(String(32), default="mujoco")
    episodes: Mapped[int] = mapped_column(Integer, default=0)
    seed: Mapped[int] = mapped_column(Integer, default=0)
    spec: Mapped[dict | None] = mapped_column(JSON)


class Evaluation(Base):
    __tablename__ = "evaluations"
    id: Mapped[int] = mapped_column(primary_key=True)
    benchmark_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("benchmark_versions.id")
    )
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))
    source: Mapped[str] = mapped_column(String(16), default="sim")
    status: Mapped[str] = mapped_column(String(16), default="pending")
    metrics: Mapped[dict | None] = mapped_column(JSON)
    report_uri: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class QualityReport(Base):
    __tablename__ = "quality_reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True
    )
    rules_version: Mapped[str] = mapped_column(String(16), default="1.0")
    episodes: Mapped[int] = mapped_column(Integer, default=0)
    finding_count: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[dict | None] = mapped_column(JSON)
    report_uri: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class AnnotationRun(Base):
    __tablename__ = "annotation_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(64), default="flora")
    model_version: Mapped[str] = mapped_column(String(32), default="mock")
    spec_version: Mapped[str] = mapped_column(String(32), default="v1")
    kind: Mapped[str] = mapped_column(String(16), default="episode")
    status: Mapped[str] = mapped_column(String(16), default="pending")
    episode_count: Mapped[int] = mapped_column(Integer, default=0)
    annotation_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime)


class Annotation(Base):
    __tablename__ = "annotations"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("annotation_runs.id", ondelete="CASCADE"), index=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"), index=True)
    episode_index: Mapped[int] = mapped_column(Integer, default=0)
    kind: Mapped[str] = mapped_column(String(16), default="episode")
    payload: Mapped[dict | None] = mapped_column(JSON)
    confidence: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(32), default="mock")
    spec_version: Mapped[str] = mapped_column(String(32), default="v1")
    status: Mapped[str] = mapped_column(String(16), default="auto")   # auto|accepted|rejected
    reviewer: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class WorkbenchDeployment(Base):
    """工作台推理服务部署记录（与 Deployment 表分开：这里按模型目录部署，
    不绑定 device / model_version 外键，用于重启后恢复。"""
    __tablename__ = "workbench_deployments"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_dir: Mapped[str] = mapped_column(String(512))
    port: Mapped[int] = mapped_column(Integer, default=0)
    pid: Mapped[int] = mapped_column(Integer, default=0)
    endpoint: Mapped[str] = mapped_column(String(256), default="")
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    stopped_at: Mapped[dt.datetime | None] = mapped_column(DateTime)


class LogRecord(Base):
    """训练 / 评测 / 任务日志的索引（只记路径与摘要，内容留在磁盘）。"""
    __tablename__ = "logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)   # train|eval|task|platform|other
    source_path: Mapped[str] = mapped_column(String(768), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    line_count: Mapped[int] = mapped_column(Integer, default=0)
    job_id: Mapped[int | None] = mapped_column(Integer, index=True)
    head: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
