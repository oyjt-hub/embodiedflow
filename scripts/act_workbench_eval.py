#!/usr/bin/env python3
"""Evaluate a LeRobot ACT checkpoint on selected held-out dataset frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--episodes-file")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.policies.act.modeling_act import ACTPolicy

    episodes = None
    if args.episodes_file:
        episodes = [int(item) for item in json.loads(Path(args.episodes_file).read_text(encoding="utf-8"))]
        if not episodes:
            raise ValueError("evaluation episode list is empty")
    dataset = LeRobotDataset(
        repo_id=args.repo_id, root=Path(args.dataset_root), episodes=episodes,
        download_videos=False, video_backend="pyav",
    )
    if len(dataset) == 0:
        raise ValueError("evaluation dataset is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ACTPolicy.from_pretrained(args.checkpoint).to(device)
    policy.eval()
    sample_count = min(args.samples, len(dataset))
    indices = np.unique(np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64)).tolist()
    errors: list[np.ndarray] = []
    episode_metrics: dict[int, list[np.ndarray]] = {}

    with torch.inference_mode():
        for index in indices:
            sample = dataset[index]
            batch = {
                key: value.unsqueeze(0).to(device)
                for key, value in sample.items()
                if torch.is_tensor(value)
            }
            if "observation.state" not in batch or "action" not in batch:
                raise ValueError("LeRobot sample must contain observation.state and action")
            policy.reset()
            predicted = policy.select_action(batch).detach().float().cpu().numpy().reshape(-1)
            target = sample["action"].detach().float().cpu().numpy()
            target = target.reshape(-1, target.shape[-1])[0] if target.ndim > 1 else target.reshape(-1)
            size = min(len(predicted), len(target))
            if size == 0:
                continue
            error = (predicted[:size] - target[:size]).astype(np.float64)
            errors.append(error)
            episode_id = int(sample.get("episode_index", -1))
            episode_metrics.setdefault(episode_id, []).append(error)

    if not errors:
        raise ValueError("no comparable ACT action samples were produced")
    values = np.concatenate(errors)
    per_episode = []
    for episode_id, episode_errors in sorted(episode_metrics.items()):
        joined = np.concatenate(episode_errors)
        per_episode.append({
            "episode_index": episode_id, "samples": len(episode_errors),
            "mae": round(float(np.abs(joined).mean()), 6),
            "rmse": round(float(np.sqrt(np.square(joined).mean())), 6),
        })
    report = {
        "model_family": "act", "model_dir": str(Path(args.checkpoint).resolve()),
        "samples": len(errors), "device": str(device),
        "mae": round(float(np.abs(values).mean()), 6),
        "rmse": round(float(np.sqrt(np.square(values).mean())), 6),
        "per_episode": per_episode,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ACT_EVAL {json.dumps(report, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
