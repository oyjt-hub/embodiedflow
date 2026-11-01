from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import edge

router = APIRouter(prefix="/edge", tags=["edge"])


class ActionIn(BaseModel):
    actions: list[float]


class EstopIn(BaseModel):
    reason: str = "manual"


@router.post("/heartbeat")
def heartbeat():
    return edge.BRIDGE.heartbeat()


@router.post("/actions")
def actions(payload: ActionIn):
    ok, info = edge.BRIDGE.submit_actions(payload.actions)
    if not ok:
        raise HTTPException(409, info["error"])
    return info


@router.get("/status")
def status():
    return edge.BRIDGE.status()


@router.post("/estop")
def estop(payload: EstopIn):
    return edge.BRIDGE.estop(payload.reason)


@router.post("/reset")
def reset():
    return edge.BRIDGE.reset()
