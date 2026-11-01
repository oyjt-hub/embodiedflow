"""训练作业完成后：写 MLflow + 自动登记模型版本。"""
from __future__ import annotations

import json
from pathlib import Path

from persistence.models import Job, JobAttempt
from services import modeling, tracking


def finalize(session, job: Job, attempt: JobAttempt, exit_code: int) -> dict:
    spec = job.spec or {}
    result: dict = {"mlflow_run_id": None, "model_version": None, "skipped": []}

    metrics_path = spec.get("metrics_path")
    ckpt_path = spec.get("checkpoint_path")
    if not ckpt_path and spec.get("exp_dir"):
        found = modeling.find_latest_checkpoint(spec["exp_dir"])
        if found is not None:
            ckpt_path = str(found)
            result["auto_discovered_checkpoint"] = str(found)
    want_mlflow = bool(spec.get("mlflow", False))

    metrics: dict = {}
    if metrics_path and Path(metrics_path).is_file():
        try:
            metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            result["skipped"].append(f"metrics 解析失败: {metrics_path}")

    if want_mlflow and exit_code == 0:
        if not tracking.available():
            result["skipped"].append("MLflow 不可用")
        else:
            artifacts = [Path(p) for p in (spec.get("artifact_paths") or []) if Path(p).exists()]
            if metrics_path and Path(metrics_path).is_file():
                artifacts.append(Path(metrics_path))
            run_id = tracking.log_run(
                experiment=spec.get("mlflow_experiment", "training"),
                run_name=f"{job.name}-job{job.id}-attempt{attempt.id}",
                params={
                    "job_id": job.id,
                    "attempt_id": attempt.id,
                    "dataset_version_id": job.dataset_version_id,
                    **{k: v for k, v in (spec.get("mlflow_params") or {}).items()},
                },
                metrics=metrics,
                artifacts=artifacts,
                tags={"source": "embodiedflow.worker", "kind": job.kind},
            )
            attempt.mlflow_run_id = run_id
            result["mlflow_run_id"] = run_id

    if ckpt_path and exit_code == 0 and Path(ckpt_path).exists():
        model = modeling.ensure_model(
            session, spec.get("model_name") or job.name, spec.get("model_family", "openpi")
        )
        version = spec.get("model_version") or f"job{job.id}-a{attempt.id}"
        mv, created = modeling.register_version(
            session,
            model,
            version,
            Path(ckpt_path),
            attempt_id=attempt.id,
            mlflow_run_id=attempt.mlflow_run_id,
            source=spec.get("source", "real"),
        )
        result["model_version"] = {
            "model": model.name,
            "version": mv.version,
            "sha256": (mv.sha256 or "")[:16],
            "created": created,
        }
    elif ckpt_path:
        result["skipped"].append(f"checkpoint 不存在或作业失败: {ckpt_path}")

    return result
