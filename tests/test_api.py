from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app


def test_health_and_job_creation():
    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["schema_version"] >= 1

        r = client.post(
            "/api/jobs",
            json={"kind": "smoke", "name": "api-test", "spec": {}, "requested_gpus": 0},
        )
        assert r.status_code == 202
        job = r.json()
        assert job["status"] == "queued"

        r = client.get(f"/api/jobs/{job['id']}")
        assert r.status_code == 200

        r = client.get("/api/jobs")
        assert r.status_code == 200
        assert any(j["id"] == job["id"] for j in r.json())

        r = client.get("/api/summary")
        assert r.status_code == 200
        assert r.json()["jobs"] >= 1
