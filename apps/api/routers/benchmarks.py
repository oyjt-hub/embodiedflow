from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from core.config import settings
from persistence.db import session_scope
from persistence.models import Benchmark, BenchmarkVersion, Evaluation
from services import benchmark as bms
from services import benchmark_arm as bma

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


class RunIn(BaseModel):
    name: str = "cartpole-balance"
    version: str = "v1"
    seeds: list[int] = [0, 1, 2, 3, 4]
    task: str | None = None          # arm_reach | cartpole_balance；None 则沿用 spec.task
    policy: str | None = None
    spec: dict = {}


class CompareIn(BaseModel):
    seeds: list[int] = [0, 1, 2, 3, 4, 5, 6, 7]
    policies: list[str] = ["proportional", "random"]
    task: str = "arm_reach"          # arm_reach | cartpole_balance
    spec: dict = {}


@router.post("/compare", status_code=201)
def compare(payload: CompareIn):
    _, mod = _resolve(payload.task, payload.spec)
    r = mod.compare(payload.seeds, payload.policies, payload.spec,
                    replay_dir=str(settings.artifacts_dir / "benchmarks" / "replay"))
    out_dir = settings.artifacts_dir / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"compare-{payload.task}-v1.json"
    path.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    r["report_uri"] = settings.relative(path)
    return r


TASKS = {"cartpole_balance": bms, "arm_reach": bma}


def _resolve(task: str | None, spec: dict):
    name = task or (spec or {}).get("task") or "cartpole_balance"
    mod = TASKS.get(name)
    if mod is None:
        raise HTTPException(400, f"未知 task: {name}（可选 {sorted(TASKS)}）")
    return name, mod


@router.post("/run", status_code=201)
def run(payload: RunIn):
    task_name, mod = _resolve(payload.task, payload.spec)
    spec = {**(payload.spec or {}), "task": task_name}
    policy = payload.policy or ("proportional" if task_name == "arm_reach" else "passive")
    replay_dir = str(settings.artifacts_dir / "benchmarks" / "replay")
    result = mod.run_benchmark(payload.seeds, spec, policy, replay_dir=replay_dir)

    out_dir = settings.artifacts_dir / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{payload.name}-{payload.version}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    with session_scope() as s:
        b = s.scalars(select(Benchmark).where(Benchmark.name == payload.name)).first()
        if b is None:
            b = Benchmark(name=payload.name, task=result["task"])
            s.add(b)
            s.flush()
        bv = BenchmarkVersion(
            benchmark_id=b.id,
            version=payload.version,
            engine=result["engine"],
            episodes=len(payload.seeds),
            seed=payload.seeds[0] if payload.seeds else 0,
            spec=result["spec"],
        )
        s.add(bv)
        s.flush()
        ev = Evaluation(
            benchmark_version_id=bv.id,
            source="sim",
            status="completed",
            metrics=result["success"],
            report_uri=settings.relative(path),
        )
        s.add(ev)
        s.flush()
        return {
            "task": task_name,
            "policy": policy,
            "benchmark_id": b.id,
            "benchmark_version_id": bv.id,
            "evaluation_id": ev.id,
            "success": result["success"],
            "episodes": result["episodes"][:5],
            "report_uri": settings.relative(path),
        }
