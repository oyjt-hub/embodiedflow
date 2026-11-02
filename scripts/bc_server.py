"""BC 推理服务：加载 checkpoint + norm.json，提供 POST /predict。

用标准库 http.server，不引入额外依赖；用带 torch 的解释器启动。
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch


class BC(torch.nn.Module):
    def __init__(self, i, o, h=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(i, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


MODEL = None
NORM = None
VERSION = ""


def load(model_dir: str):
    global MODEL, NORM, VERSION
    ck = torch.load(f"{model_dir}/checkpoint.pt", map_location="cpu")
    net = BC(ck["dim_in"], ck["dim_out"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    MODEL = net
    NORM = json.load(open(f"{model_dir}/norm.json"))
    VERSION = model_dir.rstrip("/").split("/")[-1]


def build_state(state, sidecar):
    prof_name = (NORM or {}).get("profile")
    if not prof_name:
        return list(state)
    from services.observation_profile import get_profile, vectorize
    return list(state) + list(vectorize(get_profile(prof_name), sidecar or {})["values"])


def predict(state: list[float], sidecar=None) -> dict:
    t0 = time.perf_counter()
    x = (np.asarray(build_state(state, sidecar), dtype=np.float32) - np.array(NORM["state_mean"], dtype=np.float32)) \
        / np.array(NORM["state_std"], dtype=np.float32)
    with torch.no_grad():
        y = MODEL(torch.tensor(x[None, :])).numpy()[0] * np.array(NORM["action_std"], dtype=np.float32) \
            + np.array(NORM["action_mean"], dtype=np.float32)
    return {"action": [round(float(v), 6) for v in y],
            "model_version": VERSION,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}


class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"ok": True, "model_version": VERSION})
        else:
            self._send({"detail": "not found"}, 404)

    def do_POST(self):
        if self.path != "/predict":
            self._send({"detail": "not found"}, 404)
            return
        try:
            n = int(self.headers.get("content-length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            st = data.get("state")
            if not isinstance(st, list):
                self._send({"detail": "state 必须是数组"}, 400)
                return
            self._send(predict(st, data.get('sidecar')))
        except Exception as e:
            self._send({"detail": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    load(a.model)
    print(f"[bc-server] model={VERSION} port={a.port}", flush=True)
    HTTPServer(("0.0.0.0", a.port), H).serve_forever()
