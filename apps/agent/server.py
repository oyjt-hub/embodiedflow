"""Small EmbodiedFlow chat gateway using the official DeepSeek Harness SDK."""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from deepseek_harness import DeepSeekHarness

ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = Path(__file__).resolve().parent / "web"
DATA_ROOT = Path(os.environ.get("EMBODIEDFLOW_DATA_ROOT", "/home/oyjt/workspace/embodiedflow_data"))
DSH_HOME = Path(os.environ.get("EMBODIEDFLOW_DSH_HOME", DATA_ROOT / "agent" / "dsh_home"))
AGENT_WORKSPACE = Path(os.environ.get("EMBODIEDFLOW_AGENT_WORKSPACE", DATA_ROOT / "agent" / "workspace"))
MODEL = os.environ.get("DSH_MODEL", "deepseek-v4-flash")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "")
HOST = os.environ.get("EMBODIEDFLOW_AGENT_HOST", "0.0.0.0")
PORT = int(os.environ.get("EMBODIEDFLOW_AGENT_PORT", "18082"))
_harness: DeepSeekHarness | None = None
_run_lock = threading.Lock()


def get_harness() -> DeepSeekHarness:
    global _harness
    if _harness is None:
        if not API_KEY:
            raise RuntimeError("请在 embodiedflow_data/agent/agent.env 中配置 DEEPSEEK_API_KEY")
        DSH_HOME.mkdir(parents=True, exist_ok=True)
        AGENT_WORKSPACE.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, object] = {
            "provider": "deepseek-official",
            "model": MODEL,
            "cwd": str(AGENT_WORKSPACE.resolve()),
            "dsh_home": str(DSH_HOME.resolve()),
            "profile": "sdk-minimal",
            "api_key": API_KEY,
        }
        if BASE_URL:
            kwargs["base_url"] = BASE_URL
        _harness = DeepSeekHarness(**kwargs)
    return _harness


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, _format: str, *_args) -> None:
        return

    def _json(self, status: int, value: dict) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/health":
            self._json(200, {"ok": True, "provider_configured": bool(API_KEY), "model": MODEL})
            return
        if urlparse(self.path).path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self._json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64_000:
                raise ValueError("消息长度无效")
            payload = json.loads(self.rfile.read(length))
            message = str(payload.get("message", "")).strip()
            if not message:
                raise ValueError("请输入内容")
            if len(message) > 20_000:
                raise ValueError("单条消息不能超过 20000 个字符")
            session_id = str(payload.get("session_id") or uuid.uuid4())
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
                raise ValueError("session_id 格式无效")
            with _run_lock:
                result = get_harness().run(message, session_id=session_id)
            self._json(200, {
                "session_id": session_id,
                "response": result.final_response,
            })
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self._json(503, {"error": str(exc)})
        except Exception as exc:
            self._json(502, {"error": f"Agent 调用失败：{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    AGENT_WORKSPACE.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
