"""MLflow 追踪封装：Run 与平台 attempt 通过 run_id 关联。"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_URI = os.environ.get("EF_MLFLOW_URI", "http://127.0.0.1:15000")


def available(uri: str | None = None) -> bool:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=uri or DEFAULT_URI)
        client.search_experiments(max_results=1)
        return True
    except Exception:  # noqa: BLE001
        return False


def log_run(
    experiment: str,
    run_name: str,
    params: dict | None = None,
    metrics: dict | None = None,
    metric_history: dict | None = None,
    artifacts: list[Path] | None = None,
    tags: dict | None = None,
    uri: str | None = None,
) -> str:
    """写一个 MLflow run，返回 run_id。指标必须是标量。"""
    import mlflow

    mlflow.set_tracking_uri(uri or DEFAULT_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        for k, v in (params or {}).items():
            if v is not None:
                mlflow.log_param(k, v)
        history_keys = set((metric_history or {}).keys())
        for k, v in (metrics or {}).items():
            if k not in history_keys and isinstance(v, (int, float)) and v is not None:
                mlflow.log_metric(k, float(v))
        for key, values in (metric_history or {}).items():
            if not isinstance(values, (list, tuple)):
                continue
            for step, value in enumerate(values):
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value), step=step)
        for k, v in (tags or {}).items():
            mlflow.set_tag(k, str(v))
        for p in artifacts or []:
            p = Path(p)
            if p.is_file():
                mlflow.log_artifact(str(p))
            elif p.is_dir():
                mlflow.log_artifacts(str(p))
        return run.info.run_id
