"""只读研发助手：把平台的查询能力包成可引用的答案。

设计约束（对应计划 P1.4）：
- 只读：不提供启动训练/删除数据/部署等写操作
- 每个结论都带引用（数据集版本 ID / job ID / run ID / 模型版本），可追溯
- 写入类请求直接拒绝，并说明理由
"""
from __future__ import annotations

from sqlalchemy import func, select

from persistence.models import (
    Dataset,
    DatasetVersion,
    Evaluation,
    Job,
    JobAttempt,
    Model,
    ModelVersion,
    QualityReport,
)

WRITE_INTENTS = ("启动", "训练", "删除", "部署", "执行", "提交")


def _cite(kind: str, ref) -> dict:
    return {"kind": kind, "ref": ref}


def dataset_quality(session) -> dict:
    datasets = list(session.scalars(select(Dataset).order_by(Dataset.id)))
    rows = []
    for d in datasets:
        v = session.scalars(
            select(DatasetVersion).where(DatasetVersion.dataset_id == d.id).order_by(DatasetVersion.id.desc())
        ).first()
        if v is None:
            continue
        q = session.scalars(
            select(QualityReport).where(QualityReport.dataset_version_id == v.id).order_by(QualityReport.id.desc())
        ).first()
        rows.append({
            "dataset": d.name,
            "version": v.version,
            "episodes": v.episode_count,
            "findings": q.finding_count if q else None,
            "cited": [_cite("dataset_version", v.id)],
        })
    worst = [r for r in rows if (r["findings"] or 0) > 0]
    return {
        "answer": f"共 {len(rows)} 个数据版本有质检记录；{len(worst)} 个存在异常。",
        "details": rows,
        "citations": [c for r in rows for c in r["cited"]],
    }


def training_status(session) -> dict:
    jobs = list(session.scalars(select(Job).order_by(Job.id.desc()).limit(10)))
    out = []
    for j in jobs:
        att = session.scalars(
            select(JobAttempt).where(JobAttempt.job_id == j.id).order_by(JobAttempt.id.desc())
        ).first()
        out.append({
            "job_id": j.id, "kind": j.kind, "name": j.name, "status": j.status,
            "attempts": int(session.scalar(
                select(func.count()).select_from(JobAttempt).where(JobAttempt.job_id == j.id)) or 0),
            "mlflow_run_id": getattr(att, "mlflow_run_id", None) if att else None,
            "cited": [_cite("job", j.id)] + ([_cite("mlflow_run", att.mlflow_run_id)] if att and att.mlflow_run_id else []),
        })
    done = sum(1 for j in out if j["status"] == "completed")
    return {
        "answer": f"最近 {len(out)} 个任务中 {done} 个已完成。",
        "details": out,
        "citations": [c for j in out for c in j["cited"]],
    }


def model_lineage(session, model_id: int | None = None) -> dict:
    stmt = select(ModelVersion).order_by(ModelVersion.id.desc()).limit(10)
    if model_id:
        stmt = select(ModelVersion).where(ModelVersion.model_id == model_id).order_by(ModelVersion.id.desc())
    rows = []
    for mv in session.scalars(stmt):
        m = session.get(Model, mv.model_id)
        rows.append({
            "model": m.name if m else "?",
            "version": mv.version,
            "sha256": (mv.sha256 or "")[:16],
            "attempt_id": mv.attempt_id,
            "mlflow_run_id": mv.mlflow_run_id,
            "cited": [_cite("model_version", f"{mv.model_id}:{mv.version}")],
        })
    return {
        "answer": f"共 {len(rows)} 个模型版本可追溯。",
        "details": rows,
        "citations": [c for r in rows for c in r["cited"]],
    }


def evaluation_summary(session) -> dict:
    rows = list(session.scalars(select(Evaluation).order_by(Evaluation.id.desc()).limit(5)))
    out = []
    for e in rows:
        s = e.metrics or {}
        out.append({
            "evaluation_id": e.id,
            "success": s.get("success"),
            "coverage": s.get("outcome_coverage"),
            "cited": [_cite("evaluation", e.id)],
        })
    return {
        "answer": f"最近 {len(out)} 次评测记录。",
        "details": out,
        "citations": [c for r in out for c in r["cited"]],
    }


def ask(session, question: str) -> dict:
    q = question or ""
    if any(w in q for w in WRITE_INTENTS):
        return {
            "allowed": False,
            "answer": "本助手为只读：启动训练、删除数据、部署或控制设备需在后续开放并校验权限。",
            "citations": [],
        }
    if any(w in q for w in ("质检", "质量", "异常", "数据")):
        r = dataset_quality(session)
    elif any(w in q for w in ("模型", "权重", "血缘")):
        r = model_lineage(session)
    elif any(w in q for w in ("评测", "成功率", "指标")):
        r = evaluation_summary(session)
    else:
        r = training_status(session)
    r["allowed"] = True
    r["question"] = q
    return r
