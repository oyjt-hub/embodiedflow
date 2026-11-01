"""当日上线工作台：数据 → 训练 → 评测 → 部署。

页面只提交结构化参数，后端复用现有 Worker 与模型仓库。
"""
from __future__ import annotations

import json
from pathlib import Path

import json as _json
import os
import socket
import subprocess
import time
import urllib.request

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select

from core.config import settings
from persistence.db import session_scope
from persistence.models import Dataset, DatasetVersion, Job
from services import jobs as jobsvc

router = APIRouter(prefix="/workbench", tags=["workbench"])

TORCH_PY = "/home/oyjt/workspace/openpi_v2/.venv/bin/python"
TRAINER = str(settings.root / "scripts" / "bc_train.py")
ROOT_HTML = settings.root / "apps" / "web" / "workbench.html"
SERVER = str(settings.root / "scripts" / "bc_server.py")


class TrainIn(BaseModel):
    dataset_version_id: int
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 256
    device: str = "cpu"


def _dataset_path(version_id: int) -> tuple[Dataset, DatasetVersion, Path]:
    with session_scope() as s:
        v = s.get(DatasetVersion, version_id)
        if v is None:
            raise HTTPException(404, f"数据版本 {version_id} 不存在")
        d = s.get(Dataset, v.dataset_id)
        mf = v.manifest_uri or ""
        root = Path(mf).parent.parent if mf.endswith("meta/info.json") else Path(mf)
        return d, v, root


@router.get("/api/datasets/{version_id}")
def dataset_detail(version_id: int):
    """模块二：数据集详情（基本信息）。"""
    d, v, path = _dataset_path(version_id)
    info = {}
    mf = path / "meta" / "info.json"
    if mf.exists():
        info = json.loads(mf.read_text(encoding="utf-8"))
    feats = info.get("features", {})
    st = (feats.get("observation.state") or {}).get("shape") or [None]
    ac = (feats.get("action") or {}).get("shape") or [None]
    return {
        "id": v.id, "dataset": d.name if d else "?",
        "format": d.format if d else None,
        "version": v.version, "status": v.status,
        "episodes": v.episode_count, "bytes": v.total_bytes,
        "sha256": v.sha256, "path": str(path),
        "fps": info.get("fps"), "robot_type": info.get("robot_type"),
        "state_dim": st[0], "action_dim": ac[0],
        "has_manifest": mf.exists(),
    }


@router.get("/api/datasets/{version_id}/episodes")
def dataset_episodes(version_id: int, limit: int = 50):
    """模块二：Episode 浏览器。"""
    from sqlalchemy import select

    from persistence.models import Episode

    with session_scope() as s:
        rows = list(s.scalars(
            select(Episode).where(Episode.dataset_version_id == version_id)
            .order_by(Episode.episode_index).limit(limit)))
        return [{"index": e.episode_index, "frames": e.n_frames,
                 "duration_s": round(e.duration_s, 2) if e.duration_s else None,
                 "task": e.task, "success": e.success} for e in rows]


@router.get("/api/datasets/{version_id}/manifest")
def dataset_manifest(version_id: int):
    """模块二：Dataset Manifest。"""
    _d, _v, path = _dataset_path(version_id)
    mf = path / "meta" / "info.json"
    if not mf.exists():
        raise HTTPException(404, "该版本没有 meta/info.json")
    return json.loads(mf.read_text(encoding="utf-8"))


@router.get("/api/preview/{version_id}")
def preview(version_id: int, episode: int = 0, frames: int = 120):
    """返回某个 Episode 的前 N 帧状态/动作，供页面画曲线。"""
    import glob as _glob
    import pyarrow.parquet as _pq

    _d, _v, path = _dataset_path(version_id)
    files = sorted(_glob.glob(str(path / "data" / "**" / "*.parquet"), recursive=True)) or \
            sorted(_glob.glob(str(path / "**" / "*.parquet"), recursive=True))
    for f in files:
        t = _pq.read_table(f, columns=["episode_index", "observation.state", "action"]).to_pydict()
        idx = [i for i, e in enumerate(t["episode_index"]) if int(e) == episode]
        if not idx:
            continue
        sel = idx[:frames]
        st = [[round(float(x), 4) for x in t["observation.state"][i]] for i in sel]
        ac = [[round(float(x), 4) for x in t["action"][i]] for i in sel]
        return {"episode": episode, "frames": len(sel),
                "dims": len(st[0]) if st else 0, "state": st, "action": ac}
    raise HTTPException(404, f"未找到 Episode {episode}")


@router.get("/api/processing/options")
def processing_options():
    """模块三：可选的数据处理项（规则清单）。"""
    rules = [
        {"key": "timestamp", "label": "时间戳完整性"},
        {"key": "gap", "label": "采样间隔不规则"},
        {"key": "frames", "label": "帧数缺失"},
        {"key": "video", "label": "视频损坏"},
        {"key": "align", "label": "相机-动作对齐"},
        {"key": "joint_limit", "label": "关节越限"},
        {"key": "jump", "label": "动作跳变"},
        {"key": "gripper", "label": "夹爪异常"},
        {"key": "incomplete", "label": "Episode 不完整"},
        {"key": "static", "label": "静止帧"},
        {"key": "break", "label": "轨迹断裂"},
    ]
    return {"rules": rules, "rules_version": "1.0"}


@router.post("/api/processing/jobs", status_code=201)
def processing_job(payload: dict):
    """模块三：对某个数据版本跑质检，返回发现项统计。

    同步执行（单版本前 N 个 Episode，通常几秒内）；结果写入 quality_reports。
    """
    import json as _json

    from adapters.datasets.lerobot import LeRobotAdapter
    from services import quality as qsvc

    vid = int(payload.get("dataset_version_id") or 0)
    max_ep = int(payload.get("max_episodes") or 5)
    if not vid:
        raise HTTPException(400, "缺少 dataset_version_id")

    d, v, path = _dataset_path(vid)
    adapter = LeRobotAdapter(path)
    report = qsvc.build_report(adapter, max_episodes=max_ep)

    # build_report 返回 dict：{rules_version, episodes, findings, by_episode,
    #                          rule_hits, metrics, config}
    findings = report.get("findings") or []
    rule_hits = report.get("rule_hits") or {}
    counts = dict(rule_hits)

    with session_scope() as s:
        from persistence.models import QualityReport
        rec = QualityReport(
            dataset_version_id=vid,
            rules_version=str(report.get("rules_version", "1.0")),
            episodes=int(report.get("episodes") or max_ep),
            finding_count=len(findings),
            summary={"by_kind": counts, "metrics": report.get("metrics")},
        )
        s.add(rec)
        s.flush()
        rid = rec.id

    return {"report_id": rid, "dataset_version_id": vid,
            "episodes_checked": int(report.get("episodes") or max_ep),
            "findings": len(findings), "by_kind": counts,
            "metrics": report.get("metrics"),
            "sample": findings[:5]}


@router.get("/api/experiments")
def experiments(limit: int = 20):
    """模块五：实验列表。

    数据源为平台自有表（job_attempts.mlflow_run_id），不依赖 MLflow 在线；
    若 MLflow 可达，附加实时指标，不可达则跳过（不报错）。
    """
    import urllib.request
    from sqlalchemy import desc, select

    from persistence.models import Job, JobAttempt, ModelVersion

    rows = []
    with session_scope() as s:
        attempts = list(s.scalars(
            select(JobAttempt).where(JobAttempt.mlflow_run_id.isnot(None))
            .order_by(desc(JobAttempt.id)).limit(limit)))
        for a in attempts:
            j = s.get(Job, a.job_id)
            mv = s.scalars(select(ModelVersion)
                           .where(ModelVersion.attempt_id == a.id)).first()
            spec = (j.spec or {}) if j else {}
            rows.append({
                "attempt_id": a.id, "job_id": a.job_id,
                "kind": j.kind if j else None, "name": j.name if j else None,
                "status": a.status, "run_id": a.mlflow_run_id,
                "started_at": str(a.started_at)[:19] if a.started_at else None,
                "model_version": mv.version if mv else None,
                "sha256": (mv.sha256 or "")[:12] if mv else None,
                "params": {k: spec.get(k) for k in
                           ("epochs", "lr", "device", "config", "exp_name") if k in spec},
            })

    # 可选：从 MLflow 拉指标
    mlflow_ok = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:15000/health", timeout=2) as r:
            mlflow_ok = r.status == 200
    except Exception:
        pass

    if mlflow_ok:
        for row in rows[:5]:
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:15000/api/2.0/mlflow/runs/get?run_id={row['run_id']}")
                with urllib.request.urlopen(req, timeout=3) as r:
                    d = _json.loads(r.read())
                m = d.get("run", {}).get("data", {}).get("metrics", [])
                row["metrics"] = {x["key"]: round(float(x["value"]), 6) for x in m[:6]}
            except Exception:
                row["metrics"] = None

    return {"mlflow_available": mlflow_ok, "count": len(rows), "runs": rows}


@router.get("/api/gpu")
def gpu_cluster():
    """模块六：GPU 算力管理（实时卡态 + 活跃租约 + 调度记录）。"""
    from sqlalchemy import desc, select

    from persistence.models import GPULease, GPUResource, Job, JobAttempt

    with session_scope() as s:
        cards = list(s.scalars(select(GPUResource).order_by(GPUResource.index)))
        leases = list(s.scalars(select(GPULease).where(GPULease.released_at.is_(None))))
        lease_map = {}
        for l in leases:
            a = s.get(JobAttempt, l.attempt_id)
            j = s.get(Job, a.job_id) if a else None
            lease_map[l.gpu_uuid] = {
                "attempt_id": l.attempt_id,
                "job_id": j.id if j else None,
                "job_name": j.name if j else None,
                "since": str(l.acquired_at)[:19],
            }
        recent = []
        for j in s.scalars(select(Job).where(Job.requested_gpus > 0)
                           .order_by(desc(Job.id)).limit(10)):
            recent.append({"id": j.id, "name": j.name, "status": j.status,
                           "gpus": j.requested_gpus,
                           "created_at": str(j.created_at)[:19]})

    out = []
    for c in cards:
        out.append({
            "index": c.index, "name": c.name,
            "memory_used_mb": c.memory_used_mb, "memory_total_mb": c.memory_total_mb,
            "utilization": c.utilization, "state": c.state,
            "lease": lease_map.get(c.uuid),
        })
    return {
        "cards": out,
        "total": len(out),
        "leased": len(lease_map),
        "free_mb_total": sum(c["memory_total_mb"] - c["memory_used_mb"] for c in out),
        "recent_jobs": recent,
    }


@router.get("/api/registry")
def registry(limit: int = 30):
    """模块八：模型仓库（版本 + 血缘 + 权重路径）。"""
    from sqlalchemy import desc, select

    from persistence.models import (Dataset, DatasetVersion, Job, JobAttempt,
                                    Model, ModelVersion)

    out = []
    with session_scope() as s:
        models = list(s.scalars(select(Model).order_by(Model.name)))
        for m in models:
            vers = list(s.scalars(
                select(ModelVersion).where(ModelVersion.model_id == m.id)
                .order_by(desc(ModelVersion.id)).limit(limit)))
            vlist = []
            for v in vers:
                lineage = None
                if v.attempt_id:
                    a = s.get(JobAttempt, v.attempt_id)
                    j = s.get(Job, a.job_id) if a else None
                    dv = (s.get(DatasetVersion, j.dataset_version_id)
                          if j and j.dataset_version_id else None)
                    ds = s.get(Dataset, dv.dataset_id) if dv else None
                    lineage = {
                        "job_id": j.id if j else None,
                        "job_name": j.name if j else None,
                        "dataset_version": f"{ds.name}/{dv.version}" if ds and dv else None,
                    }
                weights = v.weights_uri or ""
                vlist.append({
                    "version": v.version, "sha256": (v.sha256 or "")[:16],
                    "weights_uri": weights,
                    "weights_exists": os.path.exists(weights) if weights else None,
                    "source": v.source, "mlflow_run_id": (v.mlflow_run_id or "")[:16],
                    "lineage": lineage,
                })
            out.append({"model": m.name, "family": m.family, "versions": vlist})
    return {"count": len(out), "models": out}


@router.get("/api/robots")
def robots():
    """模块十：机器人设备管理（设备台账 + 执行契约状态 + 安全约束）。"""
    from sqlalchemy import select

    from persistence.models import Device
    from services import edge

    with session_scope() as s:
        rows = list(s.scalars(select(Device).order_by(Device.id)))
        devices = [{
            "id": d.id, "name": d.name, "robot_spec": d.robot_spec,
            "source": d.source, "state": d.state, "dof": d.dof,
            "last_seen": str(d.last_seen)[:19] if getattr(d, "last_seen", None) else None,
        } for d in rows]

    st = edge.BRIDGE.status()
    return {
        "devices": devices,
        "count": len(devices),
        "adapter": {
            "online": st["online"], "estopped": st["estopped"],
            "watchdog_ok": st["watchdog_ok"], "action_count": st["action_count"],
        },
        "safety": {
            "dof": st["dof"], "joint_limit": st["joint_limit"],
            "max_step_delta": st["max_step_delta"],
            "watchdog_timeout_s": st["watchdog_timeout_s"],
            "recent_rejected": st.get("rejected") or [],
        },
    }


@router.get("/api/logs")
def logs(kind: str = "", limit: int = 30):
    """日志索引：按类型统计 + 最近记录。"""
    from sqlalchemy import desc, func, select

    from persistence.models import LogRecord

    with session_scope() as s:
        stats = {k: n for k, n in s.execute(
            select(LogRecord.kind, func.count()).group_by(LogRecord.kind)).all()}
        stmt = select(LogRecord).order_by(desc(LogRecord.id)).limit(limit)
        if kind:
            stmt = (select(LogRecord).where(LogRecord.kind == kind)
                    .order_by(desc(LogRecord.id)).limit(limit))
        rows = list(s.scalars(stmt))
        items = [{"id": r.id, "kind": r.kind, "path": r.source_path,
                  "size": r.size_bytes, "lines": r.line_count,
                  "job_id": r.job_id} for r in rows]
        total = sum(stats.values())
    return {"total": total, "by_kind": stats, "items": items}


def _server_users() -> dict[str, dict]:
    """从 /etc/passwd 读取真实存在的普通用户（uid >= 1000）。"""
    users: dict[str, dict] = {}
    try:
        with open("/etc/passwd", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split(":")
                if len(parts) < 7:
                    continue
                name, _pw, uid, gid, _gecos, home, shell = parts[:7]
                try:
                    uid_i = int(uid)
                except ValueError:
                    continue
                if uid_i < 1000 or uid_i >= 65534:
                    continue
                if shell.endswith(("nologin", "false")):
                    continue
                users[name] = {"name": name, "uid": uid_i, "home": home}
    except OSError:
        pass
    return users


@router.get("/api/users")
def list_users():
    """可供登录的服务器用户名（不返回任何敏感信息）。"""
    return sorted(_server_users().values(), key=lambda u: u["uid"])


class LoginIn(BaseModel):
    username: str


@router.post("/api/login")
def login(payload: LoginIn):
    """输入服务器上存在的用户名即可登录（无密码，内网工具）。"""
    name = (payload.username or "").strip()
    if not name:
        raise HTTPException(400, "请输入用户名")
    users = _server_users()
    if name not in users:
        raise HTTPException(404, f"用户 {name} 不存在于本服务器")
    return {"ok": True, "username": name, "uid": users[name]["uid"],
            "home": users[name]["home"]}


@router.get("/api/running")
def running_jobs():
    """当前运行中的作业（总览页专用，只返回必要字段）。"""
    from sqlalchemy import desc, select

    from persistence.models import Job

    with session_scope() as s:
        rows = list(s.scalars(
            select(Job).where(Job.status == "running")
            .order_by(desc(Job.id)).limit(20)))
        return [{"id": j.id, "kind": j.kind, "name": j.name,
                 "gpus": j.requested_gpus,
                 "started_at": str(j.started_at)[:19] if j.started_at else None}
                for j in rows]


CLEAN_OPS = {
    "drop_any_finding": "剔除有任何质检异常的 Episode",
    "drop_static": "剔除静止帧过多的 Episode",
    "drop_incomplete": "剔除不完整的 Episode",
    "drop_joint_violation": "剔除关节越限的 Episode",
    "drop_action_jump": "剔除动作跳变的 Episode",
}

_OP_TO_RULE = {
    "drop_static": "static_frames",
    "drop_incomplete": "episode_incomplete",
    "drop_joint_violation": "joint_limit",
    "drop_action_jump": "action_jump",
}


TS_RULES = ("missing_timestamp", "irregular_gap", "time_reversal")


@router.get("/api/processing/timestamps/{version_id}")
def check_timestamps(version_id: int, max_episodes: int = 20):
    """时间戳检查：空值、间隔不齐、时间倒流，逐 Episode 汇总。"""
    from adapters.datasets.lerobot import LeRobotAdapter
    from services import quality as qsvc

    _d, v, path = _dataset_path(version_id)
    if not path.exists():
        raise HTTPException(400, f"数据目录不存在：{path}")
    report = qsvc.build_report(LeRobotAdapter(path), max_episodes=max_episodes)
    findings = [f for f in (report.get("findings") or [])
                if f.get("rule") in TS_RULES]

    per_ep: dict[int, dict] = {}
    for f in findings:
        ep = f.get("episode")
        if ep is None:
            continue
        row = per_ep.setdefault(int(ep), {"episode": int(ep), "missing_timestamp": 0,
                                          "irregular_gap": 0, "time_reversal": 0})
        row[f["rule"]] += int(f.get("count") or 1)

    bad = sorted(per_ep.values(), key=lambda r: -(r["missing_timestamp"] +
                   r["irregular_gap"] + r["time_reversal"]))
    return {
        "dataset_version_id": v.id,
        "episodes_checked": report.get("episodes"),
        "total_findings": len(findings),
        "bad_episodes": len(per_ep),
        "episodes": bad[:50],
        "metrics": {k: report.get("metrics", {}).get(k) for k in
                    ("timestamp_completeness",)},
    }


class AlignIn(BaseModel):
    dataset_version_id: int
    target_fps: float = 30.0
    max_episodes: int = 20


@router.post("/api/processing/align")
def align_plan(payload: AlignIn):
    """对齐计划：算出每个 Episode 按目标帧率重采样后的帧数。

    注意：本接口只出计划，不重写数据（重写需要生成新 parquet，另行实施）。
    """
    import glob as _glob

    import pyarrow.parquet as _pq

    _d, v, path = _dataset_path(payload.dataset_version_id)
    if not path.exists():
        raise HTTPException(400, f"数据目录不存在：{path}")
    if payload.target_fps <= 0:
        raise HTTPException(400, "target_fps 必须为正数")

    files = sorted(_glob.glob(str(path / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        files = sorted(_glob.glob(str(path / "**" / "*.parquet"), recursive=True))

    src_fps = None
    import json as _json
    mf = path / "meta" / "info.json"
    if mf.exists():
        try:
            src_fps = float(_json.loads(mf.read_text(encoding="utf-8")).get("fps") or 0) or None
        except Exception:
            src_fps = None

    counts: dict[int, int] = {}
    for f in files:
        tbl = _pq.read_table(f, columns=["episode_index"]).to_pydict()
        for e in tbl["episode_index"]:
            counts[int(e)] = counts.get(int(e), 0) + 1

    plan = []
    for ep, n in sorted(counts.items())[: payload.max_episodes]:
        dur = n / src_fps if src_fps else None
        target_n = int(round(dur * payload.target_fps)) if dur else None
        plan.append({"episode": ep, "frames": n, "duration_s": round(dur, 3) if dur else None,
                     "target_frames": target_n})
    return {"dataset_version_id": v.id, "source_fps": src_fps,
            "target_fps": payload.target_fps, "episodes": plan,
            "note": "仅计划，不重写数据"}


@router.get("/api/processing/clean-options")
def clean_options():
    return [{"key": k, "label": v} for k, v in CLEAN_OPS.items()]


class CleanIn(BaseModel):
    dataset_version_id: int
    ops: list[str] = ["drop_any_finding"]
    max_episodes: int = 200
    apply: bool = False


@router.post("/api/processing/clean")
def clean_dataset(payload: CleanIn):
    """按选定的清洗方式生成计划；apply=true 时落一个派生数据版本。"""
    from adapters.datasets.lerobot import LeRobotAdapter
    from services import quality as qsvc

    if not payload.ops:
        raise HTTPException(400, "至少选择一种清洗方式")
    unknown = [o for o in payload.ops if o not in CLEAN_OPS]
    if unknown:
        raise HTTPException(400, f"未知清洗方式: {unknown}")

    d, v, path = _dataset_path(payload.dataset_version_id)
    if not path.exists():
        raise HTTPException(400, f"数据目录不存在：{path}")

    report = qsvc.build_report(LeRobotAdapter(path), max_episodes=payload.max_episodes)
    findings = report.get("findings") or []

    bad_by_ep: dict[int, set[str]] = {}
    for f in findings:
        ep = f.get("episode")
        rule = f.get("rule")
        if ep is None:
            continue
        bad_by_ep.setdefault(int(ep), set()).add(rule or "")

    # 该版本声明包含的 Episode
    declared = []
    if v.episode_indices:
        try:
            declared = json.loads(v.episode_indices)
        except Exception:
            declared = []
    if not declared:
        declared = list(range(v.episode_count or 0))

    drop: set[int] = set()
    for ep, rules in bad_by_ep.items():
        if "drop_any_finding" in payload.ops and rules:
            drop.add(ep)
            continue
        for op in payload.ops:
            rule = _OP_TO_RULE.get(op)
            if rule and rule in rules:
                drop.add(ep)
                break

    keep = [e for e in declared if e not in drop]
    result = {
        "dataset_version_id": v.id,
        "checked_episodes": report.get("episodes"),
        "findings": len(findings),
        "declared_episodes": len(declared),
        "dropped_episodes": sorted(drop),
        "kept_episodes": len(keep),
        "apply": payload.apply,
        "new_version_id": None,
    }

    if payload.apply and keep:
        from persistence.models import DatasetVersion, utcnow

        with session_scope() as s:
            newv = DatasetVersion(
                dataset_id=v.dataset_id,
                version=f"{v.version}-clean{int(utcnow().timestamp())}",
                status="derived",
                episode_count=len(keep),
                total_bytes=v.total_bytes,
                manifest_uri=v.manifest_uri,
                episode_indices=json.dumps(keep),
            )
            s.add(newv)
            s.flush()
            result["new_version_id"] = newv.id
            result["new_version"] = newv.version
    return result


@router.get("/api/overview")
def overview():
    """模块一：系统总览。数据/训练/GPU/模型四类统计 + 最近活动。"""
    from sqlalchemy import desc, func, select
    from persistence.models import (Dataset, DatasetVersion, Episode, GPUResource,
                                    Job, Model, ModelVersion)

    with session_scope() as s:
        n_ds = s.scalar(select(func.count()).select_from(Dataset)) or 0
        n_ver = s.scalar(select(func.count()).select_from(DatasetVersion)) or 0
        n_ep = s.scalar(select(func.count()).select_from(Episode)) or 0
        total_bytes = s.scalar(select(func.coalesce(func.sum(DatasetVersion.total_bytes), 0))) or 0

        n_job = s.scalar(select(func.count()).select_from(Job)) or 0
        n_running = s.scalar(select(func.count()).select_from(Job)
                             .where(Job.status == "running")) or 0
        n_queued = s.scalar(select(func.count()).select_from(Job)
                            .where(Job.status == "queued")) or 0
        n_failed = s.scalar(select(func.count()).select_from(Job)
                            .where(Job.status == "failed")) or 0
        n_done = s.scalar(select(func.count()).select_from(Job)
                          .where(Job.status == "completed")) or 0

        gpus = list(s.scalars(select(GPUResource).order_by(GPUResource.uuid)))
        gpu_rows = [{"index": g.index, "name": g.name,
                     "memory_used_mb": g.memory_used_mb,
                     "memory_total_mb": g.memory_total_mb,
                     "utilization": g.utilization, "state": g.state} for g in gpus]

        n_model = s.scalar(select(func.count()).select_from(Model)) or 0
        n_mver = s.scalar(select(func.count()).select_from(ModelVersion)) or 0

        recent = []
        for j in s.scalars(select(Job).order_by(desc(Job.id)).limit(8)):
            recent.append({"id": j.id, "kind": j.kind, "name": j.name,
                           "status": j.status, "created_at": str(j.created_at)[:19]})

    return {
        "data": {"datasets": n_ds, "versions": n_ver, "episodes": n_ep,
                 "total_bytes": int(total_bytes)},
        "training": {"jobs": n_job, "running": n_running, "queued": n_queued,
                     "failed": n_failed, "completed": n_done},
        "gpu": {"count": len(gpu_rows),
                "free_mb_total": sum(g["memory_total_mb"] - g["memory_used_mb"] for g in gpu_rows),
                "cards": gpu_rows},
        "models": {"models": n_model, "versions": n_mver},
        "recent": recent,
    }


@router.get("/api/datasets")
def list_datasets(limit: int = 50, q: str = ""):
    """数据版本列表。q 在 SQL 里过滤（不能先 limit 再过滤，否则只能搜到最近 N 条）。"""
    with session_scope() as s:
        stmt = select(DatasetVersion, Dataset).join(
            Dataset, Dataset.id == DatasetVersion.dataset_id)
        if q:
            stmt = stmt.where(Dataset.name.ilike(f"%{q}%"))
        stmt = stmt.order_by(DatasetVersion.id.desc()).limit(limit)
        rows = [(v, d) for v, d in s.execute(stmt).all()]
        out = []
        for v, d in rows:
            out.append({"id": v.id, "label": f"{d.name if d else '?'} / {v.version}",
                        "status": v.status, "episodes": v.episode_count})
        return out


@router.post("/api/training/jobs", status_code=201)
def create_training(payload: TrainIn):
    d, v, path = _dataset_path(payload.dataset_version_id)
    if not path.exists():
        raise HTTPException(400, f"数据目录不存在：{path}")

    out_dir = settings.data_root / "workbench" / f"run_{payload.dataset_version_id}_{payload.epochs}e"
    cmd = [TORCH_PY, "-u", TRAINER,
           "--data", str(path), "--out", str(out_dir),
           "--epochs", str(payload.epochs), "--lr", str(payload.lr),
           "--batch", str(payload.batch_size), "--device", payload.device]
    spec = {
        "command": cmd,
        "exp_dir": str(out_dir),
        "metrics_path": str(out_dir / "metrics.json"),
        "checkpoint_path": str(out_dir / "checkpoint.pt"),
        "model_name": f"bc-{d.name}-v{v.version}",
        "mlflow": True,
        "mlflow_experiment": "workbench-bc",
    }
    with session_scope() as s:
        job = jobsvc.create_job(
            s, kind="train", name=f"bc-{d.name}-{payload.epochs}e",
            spec=spec, requested_gpus=0, max_attempts=1,
            created_by="workbench",
            dataset_version_id=payload.dataset_version_id,
        )
        return {"job_id": job.id, "status": job.status, "out_dir": str(out_dir)}


class EvalIn(BaseModel):
    model_dir: str
    dataset_version_id: int


class OpenPIIn(BaseModel):
    config: str = "debug_pi05"
    exp_name: str = "wb-pi05"
    gpus: int = 1


OPENPI_ROOT = Path("/home/oyjt/workspace/openpi_v2")


@router.get("/api/openpi/configs")
def openpi_configs():
    """已知可用的 OpenPI 训练配置。

    openpi 的配置注册表需要 import jax 才能读取，平台进程不装 jax，
    因此这里维护静态清单（来源于 openpi_v2 的 config 注册表）。
    """
    return {"root": str(OPENPI_ROOT),
            "exists": OPENPI_ROOT.exists(),
            "has_train": (OPENPI_ROOT / "scripts" / "train.py").exists(),
            "source": "static",
            "configs": ["debug_pi05", "pi05_aloha", "pi05_droid", "pi05_libero",
                        "pi0_aloha", "pi0_droid", "pi0_libero"]}


@router.post("/api/training/openpi", status_code=201)
def train_openpi(payload: OpenPIIn):
    train_py = OPENPI_ROOT / "scripts" / "train.py"
    if not train_py.exists():
        raise HTTPException(400, f"未找到 {train_py}")
    cmd = [TORCH_PY, "-u", str(train_py), payload.config,
           "--exp_name", payload.exp_name]
    spec = {
        "command": cmd,
        "cwd": str(OPENPI_ROOT),
        "exp_dir": str(Path("/home/oyjt/workspace/checkpoints") / payload.config / payload.exp_name),
        "mlflow": False,
        "env": {"OPENPI_DATA_HOME": "/home/oyjt/.cache/openpi"},
    }
    with session_scope() as s:
        job = jobsvc.create_job(
            s, kind="train_openpi", name=f"openpi-{payload.config}",
            spec=spec, requested_gpus=payload.gpus, max_attempts=1,
            created_by="workbench")
        return {"job_id": job.id, "config": payload.config,
                "exp_dir": spec["exp_dir"], "command": " ".join(cmd)}


@router.post("/api/evaluations", status_code=201)
def evaluate(payload: EvalIn):
    d, v, path = _dataset_path(payload.dataset_version_id)
    model_dir = Path(payload.model_dir)
    if not (model_dir / "checkpoint.pt").exists():
        raise HTTPException(400, f"未找到 checkpoint：{model_dir}")
    out = model_dir / "eval_report.json"
    cmd = [TORCH_PY, "-u", str(settings.root / "scripts" / "bc_eval.py"),
           "--data", str(path), "--model", str(model_dir), "--out", str(out)]
    spec = {"command": cmd, "exp_dir": str(model_dir), "metrics_path": str(out)}
    with session_scope() as s:
        job = jobsvc.create_job(
            s, kind="eval", name=f"eval-{d.name}", spec=spec,
            requested_gpus=0, max_attempts=1, created_by="workbench",
            dataset_version_id=payload.dataset_version_id)
        return {"job_id": job.id, "report_path": str(out)}


@router.get("/api/evaluations/{job_id}/report")
def eval_report(job_id: int):
    with session_scope() as s:
        j = s.get(Job, job_id)
        if j is None:
            raise HTTPException(404, "任务不存在")
        if j.status != "completed":
            raise HTTPException(409, f"任务状态 {j.status}，尚无报告")
        path = Path((j.spec or {}).get("metrics_path", ""))
    if not path.exists():
        raise HTTPException(404, f"报告文件不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


class DeployIn(BaseModel):
    model_dir: str


class PredictIn(BaseModel):
    state: list[float]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@router.post("/api/deployments", status_code=201)
def deploy(payload: DeployIn):
    from persistence.models import WorkbenchDeployment

    md = Path(payload.model_dir)
    if not (md / "checkpoint.pt").exists():
        raise HTTPException(400, f"未找到 checkpoint：{md}")

    port = _free_port()
    proc = subprocess.Popen(
        [TORCH_PY, "-u", SERVER, "--model", str(md), "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    ready = False
    for _ in range(60):
        try:
            with urllib.request.urlopen(url + "/health", timeout=1) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.3)
    if not ready:
        proc.terminate()
        raise HTTPException(500, "推理服务启动超时")

    with session_scope() as s:
        rec = WorkbenchDeployment(model_dir=str(md), port=port, pid=proc.pid,
                                  endpoint=f"{url}/predict", status="running")
        s.add(rec)
        s.flush()
        dep_id = rec.id
    return {"deployment_id": dep_id, "endpoint": f"{url}/predict",
            "predict_url": f"/workbench/api/deployments/{dep_id}/predict",
            "status": "running"}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


@router.post("/api/deployments/{dep_id}/predict")
def predict(dep_id: int, payload: PredictIn):
    from persistence.models import WorkbenchDeployment

    with session_scope() as s:
        rec = s.get(WorkbenchDeployment, dep_id)
        if rec is None:
            raise HTTPException(404, "部署不存在")
        if rec.status != "running":
            raise HTTPException(404, "部署不存在或未运行")
        base = rec.endpoint.rsplit("/predict", 1)[0]
        pid = rec.pid

    if not _alive(pid):
        raise HTTPException(503, "推理进程已退出（可能是服务重启），请重新部署")

    req = urllib.request.Request(
        base + "/predict",
        data=_json.dumps({"state": payload.state}).encode(),
        headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return _json.loads(r.read())
    except Exception as e:
        raise HTTPException(502, f"推理调用失败：{type(e).__name__}: {e}")


@router.post("/api/deployments/{dep_id}/stop")
def stop(dep_id: int):
    import signal as _sig

    from persistence.models import WorkbenchDeployment, utcnow

    with session_scope() as s:
        rec = s.get(WorkbenchDeployment, dep_id)
        if rec is None:
            raise HTTPException(404, "部署不存在")
        if rec.status == "running":
            try:
                os.killpg(os.getpgid(rec.pid), _sig.SIGTERM)
            except Exception:
                pass
            rec.status = "stopped"
            rec.stopped_at = utcnow()
        return {"deployment_id": dep_id, "status": rec.status}


def restore_deployments(max_restore: int = 3) -> dict:
    """API 启动时恢复上次处于 running 且进程已死的推理服务。

    防护：
    - 每次最多恢复 max_restore 个，避免异常时批量拉起
    - 只恢复目录与 checkpoint 都存在的记录
    - 已存活的不动（推理进程用独立会话，通常能熬过 API 重启）
    """
    from persistence.models import WorkbenchDeployment

    restored, skipped, failed = [], [], []
    with session_scope() as s:
        rows = list(s.query(WorkbenchDeployment)
                    .filter(WorkbenchDeployment.status == "running")
                    .order_by(WorkbenchDeployment.id.desc()).all())
    for rec in rows:
        if len(restored) >= max_restore:
            skipped.append({"id": rec.id, "reason": "超过恢复上限"})
            continue
        if _alive(rec.pid):
            skipped.append({"id": rec.id, "reason": "进程仍存活"})
            continue
        md = Path(rec.model_dir)
        if not (md / "checkpoint.pt").exists():
            failed.append({"id": rec.id, "reason": "checkpoint 缺失"})
            with session_scope() as s:
                r = s.get(WorkbenchDeployment, rec.id)
                if r:
                    r.status = "failed"
            continue
        try:
            port = _free_port()
            proc = subprocess.Popen(
                [TORCH_PY, "-u", SERVER, "--model", str(md), "--port", str(port)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            url = f"http://127.0.0.1:{port}"
            ok = False
            for _ in range(60):
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as r:
                        if r.status == 200:
                            ok = True
                            break
                except Exception:
                    time.sleep(0.3)
            if not ok:
                proc.terminate()
                failed.append({"id": rec.id, "reason": "启动超时"})
                continue
            with session_scope() as s:
                r = s.get(WorkbenchDeployment, rec.id)
                if r:
                    r.port, r.pid, r.endpoint = port, proc.pid, f"{url}/predict"
                    r.status = "running"
            restored.append({"id": rec.id, "port": port})
        except Exception as e:
            failed.append({"id": rec.id, "reason": f"{type(e).__name__}: {e}"})
    return {"restored": restored, "skipped": skipped, "failed": failed}


@router.get("/api/deployments")
def list_deployments():
    """列出部署；顺带把"进程已死但仍标 running"的记录纠正为 stopped。"""
    from persistence.models import WorkbenchDeployment, utcnow

    out = []
    with session_scope() as s:
        for rec in s.query(WorkbenchDeployment).order_by(WorkbenchDeployment.id.desc()).limit(20):
            if rec.status == "running" and not _alive(rec.pid):
                rec.status = "stopped"
                rec.stopped_at = utcnow()
            out.append({"id": rec.id, "status": rec.status,
                        "model_dir": rec.model_dir, "port": rec.port,
                        "endpoint": rec.endpoint})
    return out


@router.get("/api/jobs/{job_id}")
def job_status(job_id: int):
    with session_scope() as s:
        j = s.get(Job, job_id)
        if j is None:
            raise HTTPException(404, "任务不存在")
        tail = ""
        for lg in sorted((settings.data_root / "runs" / f"job_{j.id}").glob("attempt_*/worker.log")):
            tail = "\n".join(jobsvc.tail(lg, 30))
            break
        return {"id": j.id, "status": j.status, "name": j.name,
                "spec": j.spec, "log_tail": tail}


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def page():
    if ROOT_HTML.exists():
        return HTMLResponse(ROOT_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>workbench</h1><p>页面尚未生成</p>", status_code=200)
