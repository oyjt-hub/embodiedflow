#!/usr/bin/env python3
"""Execute a confirmed dataset-cleaning plan as a normal Worker job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from persistence.db import session_scope
from persistence.models import Job
from services.processing import execute_quality_clean


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    job_id = int(os.environ["EF_JOB_ID"])
    payload = json.loads(args.payload_json)
    payload["apply"] = True
    result = execute_quality_clean(**payload, job_id=job_id)
    output = Path(args.output)
    write_json(output, result)

    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is not None:
            spec = dict(job.spec or {})
            spec["report_path"] = str(output)
            if result.get("new_version_id"):
                spec["output_dataset_version_id"] = result["new_version_id"]
            job.spec = spec

    print(json.dumps({
        "job_id": job_id,
        "dataset_version_id": payload["dataset_version_id"],
        "new_version_id": result.get("new_version_id"),
        "dropped_episodes": len(result.get("dropped_episodes") or []),
        "kept_episodes": result.get("kept_episodes"),
        "report_path": str(output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
