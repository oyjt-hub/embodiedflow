from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from persistence.db import session_scope
from persistence.models import Deployment, Device, utcnow

router = APIRouter(prefix="/devices", tags=["devices"])


class DeviceIn(BaseModel):
    name: str
    robot_spec: str = "piper"
    source: str = "mock"          # real | mock | sim
    dof: int | None = None
    control_hz: float | None = None
    config: dict | None = None


def _view(d: Device) -> dict:
    return {
        "id": d.id,
        "name": d.name,
        "robot_spec": d.robot_spec,
        "source": d.source,
        "state": d.state,
        "dof": d.dof,
        "control_hz": d.control_hz,
        "last_seen_at": d.last_seen_at,
        "config": d.config,
    }


@router.get("")
def list_devices():
    with session_scope() as s:
        rows = [_view(d) for d in s.scalars(select(Device).order_by(Device.id))]
        real = [d for d in rows if d["source"] == "real"]
        online = [d for d in real if d["state"] == "online"]
        return {
            "devices": rows,
            "total": len(rows),
            "mock": sum(1 for d in rows if d["source"] == "mock"),
            "online_rate": {
                "value": round(len(online) / len(real), 6) if real else None,
                "numerator": len(online),
                "denominator": len(real),
                "note": "仅统计 source=real，Mock 不计入分母",
            },
        }


@router.post("", status_code=201)
def create_device(payload: DeviceIn):
    with session_scope() as s:
        exists = s.scalars(select(Device).where(Device.name == payload.name)).first()
        if exists:
            raise HTTPException(409, "设备名已存在")
        d = Device(
            name=payload.name,
            robot_spec=payload.robot_spec,
            source=payload.source,
            state="offline",
            dof=payload.dof,
            control_hz=payload.control_hz,
            config=payload.config or {},
        )
        s.add(d)
        s.flush()
        return _view(d)


@router.get("/{device_id}")
def get_device(device_id: int):
    with session_scope() as s:
        d = s.get(Device, device_id)
        if d is None:
            raise HTTPException(404, "device not found")
        deps = int(
            s.scalar(select(func.count()).select_from(Deployment).where(Deployment.device_id == device_id)) or 0
        )
        out = _view(d)
        out["deployments"] = deps
        return out


@router.post("/{device_id}/probe")
def probe(device_id: int):
    """连接测试 / 心跳：更新 last_seen_at 与在线状态。Mock 设备如实标记。"""
    with session_scope() as s:
        d = s.get(Device, device_id)
        if d is None:
            raise HTTPException(404, "device not found")
        d.last_seen_at = utcnow()
        d.state = "online" if d.source == "real" else "mock-online"
        return {"id": d.id, "state": d.state, "last_seen_at": d.last_seen_at, "source": d.source}
