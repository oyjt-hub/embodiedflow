#!/usr/bin/env python3
"""HTTP inference adapter for a LeRobot ACT checkpoint."""
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
import torch


def image_tensor(value: str, height: int, width: int) -> torch.Tensor:
    raw = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    image = cv2.imdecode(np.frombuffer(base64.b64decode(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid base64 camera image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)


def camera_aliases(feature_key: str) -> set[str]:
    name = feature_key.removeprefix("observation.images.")
    aliases = {feature_key, name}
    lowered = name.lower()
    if any(token in lowered for token in ("front", "high", "head", "main")):
        aliases.update({"front", "cam_high", "observation.images.front"})
    if "left" in lowered:
        aliases.update({"left", "cam_left_wrist", "observation.images.left"})
    if "right" in lowered:
        aliases.update({"right", "cam_right_wrist", "observation.images.right"})
    return aliases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    from lerobot.common.policies.act.modeling_act import ACTPolicy

    model_dir = Path(args.model).resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    policy = ACTPolicy.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device).eval()
    features = config.get("input_features", {})
    image_keys = [key for key in features if key.startswith("observation.images.")]
    state_shape = features.get("observation.state", {}).get("shape", [])
    state_dim = int(state_shape[-1]) if state_shape else 0
    first_image = features.get(image_keys[0], {}) if image_keys else {}
    image_shape = first_image.get("shape", [3, 224, 224])
    height = int(image_shape[-2]) if len(image_shape) >= 3 else 224
    width = int(image_shape[-1]) if len(image_shape) >= 3 else 224
    metadata = {
        "status": "ok", "model_family": "act", "checkpoint": str(model_dir),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_images": image_keys, "input_state_dim": state_dim,
        "output": "action [action_dim]", "device": str(device),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            print("[act-server] " + fmt % values, flush=True)

        def _json(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/health", "/healthz"):
                return self._json(200, metadata)
            if path == "/metadata":
                return self._json(200, metadata)
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/predict":
                return self._json(404, {"error": "not found"})
            try:
                size = int(self.headers.get("content-length", "0"))
                request = json.loads(self.rfile.read(size))
                state = np.asarray(request.get("state", []), dtype=np.float32).reshape(-1)
                if state_dim and state.size != state_dim:
                    raise ValueError(f"state must contain {state_dim} values")
                supplied = request.get("images") or {}
                batch = {"observation.state": torch.as_tensor(state, device=device).unsqueeze(0)}
                for feature_key in image_keys:
                    supplied_value = next((supplied[key] for key in camera_aliases(feature_key)
                                           if key in supplied), None)
                    if supplied_value is None:
                        raise ValueError(f"missing camera image: {feature_key}")
                    batch[feature_key] = image_tensor(supplied_value, height, width).unsqueeze(0).to(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    policy.reset()
                    action = policy.select_action(batch).detach().float().cpu().numpy().reshape(-1)
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                return self._json(200, {"action": action.tolist(), "actions": [action.tolist()],
                                        "horizon": 1, "latency_ms": latency_ms,
                                        "model_version": model_dir.name})
            except Exception as exc:
                return self._json(400, {"error": f"{type(exc).__name__}: {exc}"})

    print(f"ACT_READY {json.dumps(metadata, ensure_ascii=False)}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
