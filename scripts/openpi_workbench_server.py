#!/usr/bin/env python3
"""HTTP adapter around OpenPI policy.infer for EmbodiedFlow deployment."""
from __future__ import annotations

import argparse
import base64
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--num-steps", type=int, default=10)
    return p.parse_args()


def image_tensor(value) -> np.ndarray:
    if isinstance(value, str):
        raw = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        image = cv2.imdecode(np.frombuffer(base64.b64decode(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid base64 camera image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.asarray(value, dtype=np.uint8)
        if image.ndim != 3:
            raise ValueError("camera image must be HxWx3")
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
    h, w = image.shape[:2]
    scale = min(224 / h, 224 / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    y, x = (224 - nh) // 2, (224 - nw) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas.transpose(2, 0, 1)


def main():
    args = parse_args()
    from openpi.policies import policy_config
    from openpi.training import config as config_lib

    cfg = config_lib.get_config(args.config)
    policy = policy_config.create_trained_policy(
        cfg, args.checkpoint, sample_kwargs={"num_steps": args.num_steps}
    )
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "model_family": "openpi",
        "config": args.config,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "started_at": started_at,
        "input_images": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "input_state_dim": 14,
        "output": "actions [horizon, action_dim]",
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            print("[openpi-server] " + fmt % values, flush=True)

        def _json(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False, default=lambda x: np.asarray(x).tolist()).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path in ("/health", "/healthz"):
                return self._json(200, {"status": "ok", **metadata})
            if urlparse(self.path).path == "/metadata":
                return self._json(200, metadata)
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/predict":
                return self._json(404, {"error": "not found"})
            try:
                size = int(self.headers.get("content-length", "0"))
                request = json.loads(self.rfile.read(size))
                state = np.asarray(request.get("state"), dtype=np.float32)
                if state.shape != (14,):
                    raise ValueError("state must contain 14 values")
                supplied = request.get("images") or {}
                aliases = {
                    "cam_high": "cam_high", "front": "cam_high", "observation.images.front": "cam_high",
                    "cam_left_wrist": "cam_left_wrist", "left": "cam_left_wrist", "observation.images.left": "cam_left_wrist",
                    "cam_right_wrist": "cam_right_wrist", "right": "cam_right_wrist", "observation.images.right": "cam_right_wrist",
                }
                images = {}
                for key, value in supplied.items():
                    alias = aliases.get(key)
                    if alias:
                        images[alias] = image_tensor(value)
                missing = [k for k in metadata["input_images"] if k not in images]
                if missing:
                    raise ValueError("missing camera images: " + ", ".join(missing))
                observation = {"images": images, "state": state, "prompt": str(request.get("prompt") or "")}
                t0 = time.perf_counter()
                result = policy.infer(observation)
                latency_ms = (time.perf_counter() - t0) * 1000
                actions = np.asarray(result["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[-1] != 14 or not np.isfinite(actions).all():
                    raise ValueError(f"invalid action output shape {actions.shape}")
                return self._json(200, {
                    "actions": actions.tolist(),
                    "action": actions[0].tolist(),
                    "horizon": int(actions.shape[0]),
                    "latency_ms": latency_ms,
                    "model_version": Path(args.checkpoint).name,
                })
            except Exception as exc:
                return self._json(400, {"error": f"{type(exc).__name__}: {exc}"})

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"OPENPI_READY {json.dumps(metadata, ensure_ascii=False)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
