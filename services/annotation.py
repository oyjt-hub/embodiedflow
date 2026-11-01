"""FLORA 自动标注：作业执行、抽检复核、派生版本发布。"""
from __future__ import annotations

from sqlalchemy import select

from adapters.annotation.flora import MockFlora
from persistence.models import Annotation, AnnotationRun, DatasetVersion, utcnow

LOW_CONF = 0.70
MID_CONF = 0.90


def run_annotation(session, dataset_version_id: int, episode_indices: list[int],
                   kind: str = "episode", spec_version: str = "v1",
                   adapter=None) -> AnnotationRun:
    ad = adapter or MockFlora()
    run = AnnotationRun(
        dataset_version_id=dataset_version_id, model_name=ad.model_name,
        model_version=ad.model_version, spec_version=spec_version, kind=kind,
        status="running", episode_count=len(episode_indices),
    )
    session.add(run)
    session.flush()
    n = 0
    for idx in episode_indices:
        lab = ad.annotate(idx, kind)
        session.add(Annotation(
            run_id=run.id, dataset_version_id=dataset_version_id,
            episode_index=lab.episode_index, kind=lab.kind, payload=lab.payload,
            confidence=lab.confidence, model_version=ad.model_version,
            spec_version=spec_version,
        ))
        n += 1
    run.annotation_count = n
    run.status = "completed"
    run.finished_at = utcnow()
    session.flush()
    return run


def review(session, annotation_id: int, accept: bool, reviewer: str) -> dict:
    a = session.get(Annotation, annotation_id)
    if a is None:
        raise ValueError("annotation not found")
    a.status = "accepted" if accept else "rejected"
    a.reviewer = reviewer
    return {"annotation_id": a.id, "status": a.status}


def sampling_plan(session, dataset_version_id: int) -> dict:
    """按置信度分层抽样：<0.7 全检、0.7~0.9 抽 20%、>=0.9 抽 5%。"""
    anns = list(session.scalars(
        select(Annotation).where(Annotation.dataset_version_id == dataset_version_id)
    ))
    low = [a for a in anns if (a.confidence or 0) < LOW_CONF]
    mid = [a for a in anns if LOW_CONF <= (a.confidence or 0) < MID_CONF]
    high = [a for a in anns if (a.confidence or 0) >= MID_CONF]
    selected = low + mid[: max(1, int(len(mid) * 0.2))] + high[: max(1, int(len(high) * 0.05))]
    return {
        "total": len(anns),
        "low": len(low), "mid": len(mid), "high": len(high),
        "review_selected": len(selected),
        "coverage": {
            "value": round(len(selected) / len(anns), 6) if anns else None,
            "numerator": len(selected), "denominator": len(anns),
        },
        "annotation_ids": [a.id for a in selected],
    }


def stats(session, dataset_version_id: int) -> dict:
    anns = list(session.scalars(
        select(Annotation).where(Annotation.dataset_version_id == dataset_version_id)
    ))
    accepted = sum(1 for a in anns if a.status == "accepted")
    rejected = sum(1 for a in anns if a.status == "rejected")
    reviewed = accepted + rejected
    return {
        "annotations": len(anns),
        "auto": sum(1 for a in anns if a.status == "auto"),
        "accepted": accepted, "rejected": rejected,
        "acceptance_rate": {
            "value": round(accepted / reviewed, 6) if reviewed else None,
            "numerator": accepted, "denominator": reviewed,
        },
    }


def publish(session, dataset_version_id: int, version: str) -> DatasetVersion:
    src = session.get(DatasetVersion, dataset_version_id)
    if src is None:
        raise ValueError("source version not found")
    anns = list(session.scalars(
        select(Annotation).where(
            Annotation.dataset_version_id == dataset_version_id,
            Annotation.status.in_(("auto", "accepted")),
        )
    ))
    eps = sorted({a.episode_index for a in anns})
    derived = DatasetVersion(
        dataset_id=src.dataset_id, version=version, status="derived",
        episode_count=len(eps), total_bytes=src.total_bytes, manifest_uri=src.manifest_uri,
    )
    session.add(derived)
    session.flush()
    return derived
