from __future__ import annotations

import os
import subprocess

from sqlalchemy import select

from core.config import settings
from core.enums import GPUState
from persistence.models import GPULease, GPUResource, utcnow

QUERY_FIELDS = "index,uuid,name,memory.total,memory.used,utilization.gpu"
# 策略（可由环境变量覆盖）：只要"剩余显存 >= 需求 + 安全余量"就可调度，
# 不要求整卡空闲。整卡被外部占满时才判 busy。
ALLOW_PARTIAL = os.environ.get("EF_GPU_ALLOW_PARTIAL", "1") == "1"
MIN_FREE_MB = int(os.environ.get("EF_GPU_MIN_FREE_MB", "4096"))
SAFETY_MARGIN_MB = int(os.environ.get("EF_GPU_SAFETY_MARGIN_MB", "1024"))


def sample_gpus() -> list[dict]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    rows = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            rows.append(
                {
                    "index": int(float(parts[0])),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_total_mb": int(float(parts[3])),
                    "memory_used_mb": int(float(parts[4])),
                    "utilization": int(float(parts[5])),
                }
            )
        except ValueError:
            continue
    return rows


def sync_gpus(session) -> list[GPUResource]:
    now = utcnow()
    for row in sample_gpus():
        gpu = session.get(GPUResource, row["uuid"])
        if gpu is None:
            gpu = GPUResource(uuid=row["uuid"])
            session.add(gpu)
        gpu.index = row["index"]
        gpu.name = row["name"]
        gpu.memory_total_mb = row["memory_total_mb"]
        gpu.memory_used_mb = row["memory_used_mb"]
        gpu.utilization = row["utilization"]
        gpu.host = settings.host_label
        gpu.updated_at = now
    session.flush()
    _refresh_states(session)
    return list(session.scalars(select(GPUResource).order_by(GPUResource.index)))


def _free_mb(gpu: GPUResource) -> int:
    return max(0, int(gpu.memory_total_mb) - int(gpu.memory_used_mb))


def _refresh_states(session) -> None:
    leased = {
        lease.gpu_uuid
        for lease in session.scalars(select(GPULease).where(GPULease.released_at.is_(None)))
    }
    for gpu in session.scalars(select(GPUResource)):
        if gpu.uuid in leased:
            gpu.state = GPUState.LEASED.value
        elif _free_mb(gpu) <= 0 or (not ALLOW_PARTIAL and gpu.memory_used_mb > 0):
            gpu.state = GPUState.BUSY.value
        elif gpu.memory_used_mb > 1024:
            gpu.state = GPUState.PARTIAL.value
        else:
            gpu.state = GPUState.FREE.value


def schedulable_gpus(session, need_mb: int = 0) -> list[str]:
    """返回可调度的 GPU uuid：未被平台租用，且剩余显存 >= max(需求+余量, 最小门槛)。"""
    leased = {
        lease.gpu_uuid
        for lease in session.scalars(select(GPULease).where(GPULease.released_at.is_(None)))
    }
    required = max(int(need_mb) + SAFETY_MARGIN_MB, MIN_FREE_MB) if need_mb else MIN_FREE_MB
    gpus = sorted(session.scalars(select(GPUResource)), key=lambda g: g.index)
    out = []
    for gpu in gpus:
        if gpu.uuid in leased:
            continue
        if _free_mb(gpu) < required:
            continue
        out.append(gpu.uuid)
    # 优先整卡空闲的，再按剩余显存从多到少
    out.sort(key=lambda u: next((_free_mb(g) for g in gpus if g.uuid == u), 0), reverse=True)
    return out


def gpu_summary(session) -> list[dict]:
    rows = []
    for gpu in sorted(session.scalars(select(GPUResource)), key=lambda g: g.index):
        rows.append(
            {
                "index": gpu.index,
                "uuid": gpu.uuid,
                "name": gpu.name,
                "memory_total_mb": gpu.memory_total_mb,
                "memory_used_mb": gpu.memory_used_mb,
                "memory_free_mb": _free_mb(gpu),
                "utilization": gpu.utilization,
                "state": gpu.state,
            }
        )
    return rows


def total_gpus(session) -> int:
    return len(list(session.scalars(select(GPUResource))))


def _index_of(session, uuid: str) -> int | None:
    gpu = session.get(GPUResource, uuid)
    return gpu.index if gpu else None


def plan_allocation(session, job) -> list[str] | None:
    """为一个任务规划 GPU。

    返回 None 表示"资源暂时不够，应留队等待"；返回 [] 表示不需要 GPU。
    支持的 spec 字段：
      - gpu_memory_mb: 每卡需要的剩余显存
      - gpu_indices:   只允许用这些索引的卡（用于竞争/压测场景）
    """
    need_cards = int(job.requested_gpus or 0)
    if need_cards <= 0:
        return []
    spec = job.spec or {}
    need_mb = int(spec.get("gpu_memory_mb", 0) or 0)
    pinned = spec.get("gpu_indices")
    cands = schedulable_gpus(session, need_mb=need_mb)
    if pinned:
        allowed = {int(i) for i in pinned}
        cands = [u for u in cands if _index_of(session, u) in allowed]
    if len(cands) < need_cards:
        return None
    return cands[:need_cards]
