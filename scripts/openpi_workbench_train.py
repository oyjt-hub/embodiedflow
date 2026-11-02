#!/usr/bin/env python3
"""Run an OpenPI preset with explicit EmbodiedFlow workbench overrides."""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
from collections.abc import Mapping
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--lora-rank", required=True, type=int)
    parser.add_argument("--freeze-strategy", required=True,
                        choices=("preset", "lora_only", "vision_lora", "full"))
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    import flax.nnx as nnx
    from openpi.shared import nnx_utils
    from openpi.training import config as config_lib

    cfg = config_lib.get_config(args.config)
    if not hasattr(cfg.data, "repo_id"):
        raise ValueError(f"{args.config} does not support a LeRobot dataset override")
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=args.dataset_repo_id))
    model = cfg.model
    if hasattr(model, "siglip_lora_rank"):
        model = dataclasses.replace(model, siglip_lora_rank=args.lora_rank)
    elif args.lora_rank:
        raise ValueError(f"{args.config} does not support configurable SigLIP LoRA rank")

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    if args.freeze_strategy == "preset":
        freeze_filter = cfg.freeze_filter
    elif args.freeze_strategy == "lora_only":
        freeze_filter = nnx.Not(lora_filter)
    elif args.freeze_strategy == "vision_lora":
        freeze_filter = nnx.All(
            nnx.Not(lora_filter),
            nnx.Not(nnx_utils.PathRegex("PaliGemma/img/.*")),
        )
    else:
        freeze_filter = nnx.Nothing

    schedule_updates: dict[str, Any] = {"peak_lr": args.learning_rate}
    if hasattr(cfg.lr_schedule, "decay_steps"):
        schedule_updates["decay_steps"] = args.steps
    if hasattr(cfg.lr_schedule, "warmup_steps"):
        schedule_updates["warmup_steps"] = min(
            int(cfg.lr_schedule.warmup_steps), max(0, args.steps // 10)
        )
    lr_schedule = dataclasses.replace(cfg.lr_schedule, **schedule_updates)
    return dataclasses.replace(
        cfg,
        exp_name=args.exp_name,
        checkpoint_base_dir="/home/oyjt/workspace/checkpoints",
        model=model,
        freeze_filter=freeze_filter,
        batch_size=args.batch_size,
        num_train_steps=args.steps,
        num_workers=args.num_workers,
        lr_schedule=lr_schedule,
        log_interval=max(1, min(int(cfg.log_interval), args.steps)),
        save_interval=max(1, min(int(cfg.save_interval), args.steps)),
        wandb_enabled=False,
        resume=False,
        overwrite=False,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    summary = {
        "config": args.config,
        "dataset_repo_id": args.dataset_repo_id,
        "exp_name": args.exp_name,
        "batch_size": cfg.batch_size,
        "steps": cfg.num_train_steps,
        "learning_rate": args.learning_rate,
        "siglip_lora_rank": args.lora_rank,
        "freeze_strategy": args.freeze_strategy,
        "num_workers": cfg.num_workers,
        "checkpoint_dir": str(cfg.checkpoint_dir),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    import wandb

    # This OpenPI checkout keeps the JAX trainer as scripts/train.py instead
    # of an importable openpi.training.train module.
    train_path = Path.cwd() / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_workbench_train_entry", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OpenPI trainer: {train_path}")
    train_lib = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = train_lib
    spec.loader.exec_module(train_lib)

    loss_history: list[float] = []
    logged_steps: list[int] = []
    original_log = wandb.log
    original_init = wandb.init

    def disabled_wandb_init(*_: Any, **__: Any) -> None:
        # OpenPI calls wandb.init(mode="disabled") when wandb_enabled=False.
        # The SDK replaces wandb.log with a no-op in that mode, which prevents
        # the local metric collector below from seeing training losses.
        return None

    def capture_log(data: dict[str, Any], step: int | None = None, *_: Any, **__: Any) -> None:
        value = None
        if isinstance(data, Mapping):
            for key in ("loss", "train_loss", "train/loss"):
                if key in data:
                    value = data[key]
                    break
        if value is None:
            return
        try:
            import numpy as np
            array = np.asarray(value).reshape(-1)
            if array.size != 1:
                return
            scalar = float(array[0])
            loss_history.append(scalar)
            logged_steps.append(int(step if step is not None else len(loss_history)))
            print(f"EmbodiedFlow metric step={logged_steps[-1]} train_loss={scalar:.8f}", flush=True)
        except (TypeError, ValueError):
            return

    wandb.init = disabled_wandb_init
    wandb.log = capture_log
    metrics_path = Path(args.metrics_path)
    try:
        train_lib.main(cfg)
    finally:
        wandb.log = original_log
        wandb.init = original_init
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            **summary,
            "logged_steps": logged_steps,
            "loss_history": loss_history,
            "final_train_loss": loss_history[-1] if loss_history else None,
            "min_train_loss": min(loss_history) if loss_history else None,
        }
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"EmbodiedFlow metrics saved: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
