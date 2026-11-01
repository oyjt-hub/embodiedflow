from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import select

from core.config import settings
from persistence.db import session_scope
from persistence.models import Evaluation
from services import evaluation as evs

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


class EvalIn(BaseModel):
    log_paths: list[str]
    label: str = ""
    source: str = "real"
    model_version_id: int | None = None


@router.post("", status_code=201)
def create(payload: EvalIn):
    reports = []
    for lp in payload.log_paths:
        p = Path(lp).expanduser()
        if not p.is_file():
            raise HTTPException(400, f"日志不存在: {p}")
        reports.append(evs.evaluate(evs.parse_log(p)))

    summary = evs.summarize(reports)
    out_dir = settings.artifacts_dir / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = payload.label or "eval"
    json_path = out_dir / f"{name}.json"
    csv_path = out_dir / f"{name}.csv"
    payload_full = {"label": name, "summary": summary, "episodes": reports}
    json_path.write_text(json.dumps(payload_full, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path.write_text(evs.to_csv(reports), encoding="utf-8")

    with session_scope() as s:
        row = Evaluation(
            model_version_id=payload.model_version_id,
            source=payload.source,
            status="completed",
            metrics=summary,
            report_uri=settings.relative(json_path),
        )
        s.add(row)
        s.flush()
        eval_id = row.id
    return {
        "id": eval_id,
        "label": name,
        "summary": summary,
        "episodes": len(reports),
        "report_uri": settings.relative(json_path),
        "csv_uri": settings.relative(csv_path),
    }


@router.get("")
def list_evaluations():
    with session_scope() as s:
        rows = list(s.scalars(select(Evaluation).order_by(Evaluation.id.desc())))
        return [
            {
                "id": r.id,
                "source": r.source,
                "status": r.status,
                "summary": r.metrics,
                "report_uri": r.report_uri,
                "created_at": r.created_at,
            }
            for r in rows
        ]


@router.get("/{eval_id}/report")
def report(eval_id: int):
    with session_scope() as s:
        row = s.get(Evaluation, eval_id)
        if row is None or not row.report_uri:
            raise HTTPException(404, "evaluation not found")
        path = settings.data_root / row.report_uri
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/{eval_id}/report.csv", response_class=PlainTextResponse)
def report_csv(eval_id: int):
    with session_scope() as s:
        row = s.get(Evaluation, eval_id)
        if row is None or not row.report_uri:
            raise HTTPException(404, "evaluation not found")
        path = Path(settings.data_root / row.report_uri).with_suffix(".csv")
    return Path(path).read_text(encoding="utf-8")
