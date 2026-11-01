from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from persistence.db import session_scope
from persistence.models import Model
from services import families

router = APIRouter(prefix="/families", tags=["families"])


@router.get("")
def list_families():
    return {"families": families.all_families()}


@router.post("/seed", status_code=201)
def seed():
    """把已声明的家族写进 models 表，使其出现在模型仓库里。"""
    created = []
    with session_scope() as s:
        for f in families.all_families():
            exists = s.scalars(select(Model).where(Model.name == f["name"])).first()
            if exists is None:
                s.add(Model(name=f["name"], family=f["name"]))
                created.append(f["name"])
        s.flush()
    return {"created": created, "total_families": len(families.FAMILIES)}


@router.get("/{name}")
def get_family(name: str):
    if name not in families.FAMILIES:
        raise HTTPException(404, f"未注册的家族: {name}")
    return families.describe(name)
