from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import func, select

from persistence.db import session_scope
from persistence.models import (
    Dataset,
    QualityReport,
    DatasetVersion,
    Device,
    Evaluation,
    Job,
    Model,
    ModelVersion,
)

router = APIRouter(tags=["catalog"])

# 路由归属：实体 CRUD 在各自路由（/models、/devices、/evaluations），
# 本文件只保留只读聚合视图（/datasets 列表、/summary）。
# 注意：FastAPI 先注册先匹配，重复路径会让后注册的路由永远不生效。


@router.get("/datasets")
def datasets():
    with session_scope() as s:
        rows = list(s.scalars(select(Dataset).order_by(Dataset.id)))
        out = []
        for d in rows:
            versions = s.scalar(
                select(func.count()).select_from(DatasetVersion).where(DatasetVersion.dataset_id == d.id)
            )
            latest = s.scalars(
                select(DatasetVersion)
                .where(DatasetVersion.dataset_id == d.id)
                .order_by(DatasetVersion.id.desc())
            ).first()
            out.append(
                {
                    "id": d.id,
                    "name": d.name,
                    "format": d.format,
                    "robot_spec": d.robot_spec,
                    "owner": d.owner,
                    "versions": int(versions or 0),
                    "latest_version": latest.version if latest else None,
                    "episodes": latest.episode_count if latest else 0,
                    "status": latest.status if latest else None,
                    "created_at": d.created_at,
                }
            )
        return out


@router.get("/summary")
def summary():
    with session_scope() as s:
        def count(model):
            return int(s.scalar(select(func.count()).select_from(model)) or 0)

        real_devices = list(s.scalars(select(Device).where(Device.source == "real")))
        online = [d for d in real_devices if d.state == "online"]
        return {
            "devices_real": len(real_devices),
            "devices_online": len(online),
            "devices_online_rate": (round(len(online) / len(real_devices), 6) if real_devices else None),
            "quality_reports": count(QualityReport),
            "evaluations": count(Evaluation),
            "datasets": count(Dataset),
            "dataset_versions": count(DatasetVersion),
            "jobs": count(Job),
            "jobs_running": int(
                s.scalar(select(func.count()).select_from(Job).where(Job.status == "running")) or 0
            ),
            "jobs_failed": int(
                s.scalar(select(func.count()).select_from(Job).where(Job.status == "failed")) or 0
            ),
            "models": count(Model),
            "devices": count(Device),
            "evaluations": count(Evaluation),
        }
