#!/usr/bin/env python3
"""Run one LeRobot ACT training job and export compact metrics for EmbodiedFlow."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-train", required=True)
    ap.add_argument("--dataset-repo-id", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--episodes-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--metrics-path", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--learning-rate", type=float, required=True)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--n-action-steps", type=int, required=True)
    ap.add_argument("--num-workers", type=int, required=True)
    ap.add_argument("--pretrained-backbone", choices=("true", "false"), required=True)
    ap.add_argument("--use-amp", choices=("true", "false"), required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    episodes = json.loads(Path(args.episodes_file).read_text(encoding="utf-8"))
    if not episodes:
        raise SystemExit("empty training episode list")

    output_dir = Path(args.output_dir)
    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    log_freq = 1 if args.steps <= 20 else 10
    pretrained_weights = (
        "ResNet18_Weights.IMAGENET1K_V1"
        if args.pretrained_backbone == "true" else "null"
    )
    command = [
        args.lerobot_train,
        "--dataset.repo_id", args.dataset_repo_id,
        "--dataset.root", args.dataset_root,
        "--dataset.episodes", json.dumps([int(ep) for ep in episodes]),
        "--policy.type", "act",
        "--policy.device", "cuda",
        "--policy.chunk_size", str(args.chunk_size),
        "--policy.n_action_steps", str(args.n_action_steps),
        "--policy.optimizer_lr", str(args.learning_rate),
        "--policy.optimizer_lr_backbone", str(args.learning_rate),
        "--policy.use_amp", args.use_amp,
        "--policy.pretrained_backbone_weights", pretrained_weights,
        "--policy.push_to_hub", "false",
        "--output_dir", str(output_dir),
        "--job_name", output_dir.name,
        "--batch_size", str(args.batch_size),
        "--steps", str(args.steps),
        "--save_freq", str(args.steps),
        "--save_checkpoint", "true",
        "--num_workers", str(args.num_workers),
        "--log_freq", str(log_freq),
        "--wandb.enable", "false",
    ]
    print("ACT_COMMAND", json.dumps(command, ensure_ascii=False), flush=True)

    history: list[dict[str, float | int]] = []
    metric_pattern = re.compile(
        r"\bstep:\s*(\d+).*?\bloss:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    )
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        match = metric_pattern.search(line)
        if match:
            history.append({"step": int(match.group(1)), "value": float(match.group(2))})
    code = proc.wait()
    metrics: dict[str, object] = {
        "steps": args.steps if code == 0 else (history[-1]["step"] if history else 0),
        "batch_size": args.batch_size,
        "loss_history": [item["value"] for item in history],
        "loss_steps": [item["step"] for item in history],
    }
    if history:
        metrics["train_loss"] = history[-1]["value"]
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if code != 0:
        return code
    checkpoint = output_dir / "checkpoints" / "last" / "pretrained_model"
    if not (checkpoint / "config.json").is_file():
        print(f"ACT_CHECKPOINT_MISSING {checkpoint}", flush=True)
        return 2
    print(f"ACT_CHECKPOINT {checkpoint}", flush=True)
    print(f"ACT_METRICS {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
