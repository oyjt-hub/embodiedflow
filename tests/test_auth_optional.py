"""鉴权必须可选：未设置 token 时接口照常开放。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app


def test_open_when_no_token_configured():
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/api/summary").status_code == 200
