from __future__ import annotations

from fastapi import APIRouter

from persistence.db import session_scope
from services import assistant

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.get("/ask")
def ask(q: str = ""):
    with session_scope() as s:
        return assistant.ask(s, q)
