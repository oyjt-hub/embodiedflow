"""Shared, versioned description of executable training backends."""
from __future__ import annotations

from pathlib import Path
from typing import Any

CAPABILITY_SCHEMA = "embodiedflow.training-capabilities"
CAPABILITY_VERSION = "1.0.0"


def _parameter(name: str, kind: str, default: Any, *, minimum=None, maximum=None,
               minimum_exclusive: bool = False, choices=None, unit=None) -> dict[str, Any]:
    item: dict[str, Any] = {"name": name, "type": kind, "default": default}
    if minimum is not None:
        item["minimum"] = minimum
    if maximum is not None:
        item["maximum"] = maximum
    if minimum is not None and minimum_exclusive:
        item["minimum_exclusive"] = True
    if choices is not None:
        item["choices"] = choices
    if unit:
        item["unit"] = unit
    return item


def training_capabilities(*, root: Path, torch_python: str, bc_trainer: Path,
                          bc_eval: Path, bc_server: Path, act_python: str,
                          act_cli: str, act_trainer: Path, act_eval: Path,
                          act_server: Path, openpi_root: Path,
                          openpi_trainer: Path, openpi_eval: Path,
                          openpi_server: Path, openpi_configs: list[str]) -> dict[str, Any]:
    """Build the capability contract from current environment and adapter files."""
    def backend(*, id: str, name: str, route: str, runtime: list[Path],
                train_parameters: list[dict[str, Any]],
                observation: dict[str, Any], gpu: dict[str, Any],
                configs: list[str] | None = None) -> dict[str, Any]:
        missing = [str(path) for path in runtime if not path.exists()]
        item: dict[str, Any] = {
            "id": id,
            "name": name,
            "available": not missing,
            "status": "ready" if not missing else "unavailable",
            "unavailable_requirements": missing,
            "train_route": route,
            "dataset_requirements": observation,
            "resource": gpu,
            "parameters": train_parameters,
            "evaluation": {"supported": not missing, "route": "/workbench/api/evaluations"},
            "deployment": {"supported": not missing, "route": "/workbench/api/deployments"},
        }
        if configs is not None:
            item["configs"] = list(configs)
        return item

    backends = [
        backend(
            id="bc", name="BC · 状态输入", route="/workbench/api/training/jobs",
            runtime=[Path(torch_python), bc_trainer, bc_eval, bc_server],
            observation={"required": ["observation.state", "action"], "images": "optional"},
            gpu={"required": False, "cpu_supported": True, "minimum_memory_mb": 0},
            train_parameters=[
                _parameter("epochs", "integer", 5, minimum=1, maximum=10000),
                _parameter("batch_size", "integer", 256, minimum=1, maximum=65536),
                _parameter("learning_rate", "number", 1e-3, minimum=0, maximum=1,
                           minimum_exclusive=True),
                _parameter("optimizer", "enum", "adamw", choices=["adam", "adamw"]),
                _parameter("weight_decay", "number", 1e-4, minimum=0, maximum=1),
                _parameter("holdout_episodes", "integer", 3, minimum=1, maximum=1000),
                _parameter("seed", "integer", 42, minimum=0, maximum=2147483647),
                _parameter("device", "enum", "cuda", choices=["cpu", "cuda"]),
                _parameter("gpu_memory_mb", "integer", 8192, minimum=1024, maximum=49140),
            ],
        ),
        backend(
            id="act", name="ACT · LeRobot", route="/workbench/api/training/act",
            runtime=[Path(act_python), Path(act_cli), act_trainer, act_eval, act_server],
            observation={"required": ["observation.images.*", "observation.state", "action"]},
            gpu={"required": True, "cpu_supported": False, "minimum_memory_mb": 5000},
            train_parameters=[
                _parameter("steps", "integer", 5000, minimum=1, maximum=500000),
                _parameter("batch_size", "integer", 4, minimum=1, maximum=128),
                _parameter("learning_rate", "number", 1e-5, minimum=0, maximum=1,
                           minimum_exclusive=True),
                _parameter("chunk_size", "integer", 50, minimum=1, maximum=200),
                _parameter("n_action_steps", "integer", 50, minimum=1, maximum=200),
                _parameter("use_amp", "boolean", False),
                _parameter("gpu_memory_mb", "integer", 20000, minimum=1024, maximum=49140),
                _parameter("gpu_index", "integer|null", None, minimum=0, maximum=7),
                _parameter("num_workers", "integer", 4, minimum=0, maximum=32),
                _parameter("pretrained_backbone", "boolean", True),
            ],
        ),
        backend(
            id="openpi", name="π0.5 · OpenPI", route="/workbench/api/training/openpi",
            runtime=[Path(torch_python), openpi_root / "scripts" / "train.py",
                     openpi_trainer, openpi_eval, openpi_server],
            observation={"required": ["observation.images.*", "observation.state", "action"],
                         "task_text": True},
            gpu={"required": True, "cpu_supported": False, "minimum_memory_mb": 18000},
            train_parameters=[
                _parameter("config", "enum", "4items_rtc_v2_lora", choices=openpi_configs),
                _parameter("steps", "integer", 5000, minimum=1, maximum=500000),
                _parameter("batch_size", "integer", 16, minimum=1, maximum=512),
                _parameter("learning_rate", "number", 1e-5, minimum=0, maximum=1,
                           minimum_exclusive=True),
                _parameter("lora_rank", "integer", 0, minimum=0, maximum=64),
                _parameter("freeze_strategy", "enum", "preset",
                           choices=["preset", "lora_only", "vision_lora", "full"]),
                _parameter("gpus", "integer", 1, minimum=1, maximum=8),
                _parameter("gpu_memory_mb", "integer", 22000, minimum=1024, maximum=49140),
                _parameter("num_workers", "integer", 8, minimum=0, maximum=64),
            ],
            configs=openpi_configs,
        ),
    ]
    return {
        "schema": CAPABILITY_SCHEMA,
        "version": CAPABILITY_VERSION,
        "backends": backends,
    }
