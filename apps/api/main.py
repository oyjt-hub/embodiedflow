from __future__ import annotations

from contextlib import asynccontextmanager

import hashlib
import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.routers import (
    annotation, assistant, benchmarks, catalog, datasets, devices, edge, evaluations,
    families, health, jobs, models, report, resources, vision,
)
from core.config import ensure_dirs
from persistence.db import session_scope
from persistence.migrate import upgrade
from persistence.models import User, UserSession
from services import resources as ressvc
from services.identity import reset_actor, set_actor


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    upgrade(verbose=True)
    with session_scope() as s:
        ressvc.sync_gpus(s)
    try:
        from apps.api.routers.workbench import restore_deployments
        info = restore_deployments()
        if info["restored"] or info["failed"]:
            print(f"[startup] 部署恢复: {info}")
    except Exception as e:
        print(f"[startup] 部署恢复跳过: {type(e).__name__}: {e}")
    yield


app = FastAPI(title="EmbodiedFlow API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _bind_session_actor(request: Request, call_next):
    raw_token = request.cookies.get("ef_session", "")
    actor = None
    if raw_token:
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        now = time.time()
        try:
            with session_scope() as session:
                user_session = session.get(UserSession, token_hash)
                if user_session and user_session.expires_at > now:
                    user = session.get(User, user_session.user_id)
                    actor = user.name if user else None
                    if now - user_session.last_seen_at > 300:
                        user_session.last_seen_at = now
        except Exception:
            actor = None
    context_token = set_actor(actor)
    try:
        return await call_next(request)
    finally:
        reset_actor(context_token)


# 可选鉴权：只有设置了 EF_API_TOKEN 才生效，默认关闭（不改变现有用法）。
# 令牌可放在 Authorization: Bearer <t> 或查询参数 ?token=<t>。
_API_TOKEN = os.environ.get("EF_API_TOKEN", "").strip()
_OPEN_PATHS = {"/", "/api/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def _require_token(request: Request, call_next):
    if not _API_TOKEN or request.method == "OPTIONS" or request.url.path in _OPEN_PATHS:
        return await call_next(request)
    auth = request.headers.get("authorization", "")
    supplied = auth[7:].strip() if auth.lower().startswith("bearer ") else request.query_params.get("token", "")
    if supplied != _API_TOKEN:
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)
app.include_router(health.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(resources.router, prefix="/api")
app.include_router(catalog.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(evaluations.router, prefix="/api")
app.include_router(report.router, prefix="/api")
app.include_router(assistant.router, prefix="/api")
app.include_router(benchmarks.router, prefix="/api")
app.include_router(families.router, prefix="/api")
app.include_router(annotation.router, prefix="/api")
app.include_router(edge.router, prefix="/api")
app.include_router(devices.router, prefix="/api")
app.include_router(vision.router)


@app.get("/")
def root() -> dict:
    return {"service": "EmbodiedFlow", "docs": "/docs", "health": "/api/health"}


# --- workbench（当日上线工作台）---
from apps.api.routers import workbench as _wb  # noqa: E402

app.include_router(_wb.router)
