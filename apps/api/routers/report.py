from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from core.config import settings
from persistence.db import session_scope
from services import demo_report

router = APIRouter(prefix="/report", tags=["report"])


@router.get("/mvp")
def mvp():
    with session_scope() as s:
        rep = demo_report.build(s)
    out = settings.artifacts_dir / "reports" / "mvp_acceptance.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(demo_report.to_markdown(rep), encoding="utf-8")
    rep["markdown_uri"] = settings.relative(out)
    return rep


@router.get("/mvp.md", response_class=PlainTextResponse)
def mvp_md():
    with session_scope() as s:
        rep = demo_report.build(s)
    return demo_report.to_markdown(rep)
