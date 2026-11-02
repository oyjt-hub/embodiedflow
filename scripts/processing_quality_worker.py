#!/usr/bin/env python3
"""Run one dataset quality inspection and persist a compact report."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from adapters.datasets.lerobot import LeRobotAdapter
from persistence.db import session_scope
from persistence.models import QualityReport
from services import quality as quality_service


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version-id", type=int, required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--max-episodes", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_path}")
    raw = quality_service.build_report(
        LeRobotAdapter(dataset_path), max_episodes=args.max_episodes,
    )
    findings = raw.get("findings") or []
    by_kind = dict(raw.get("rule_hits") or {})
    output = Path(args.output)
    summary = {
        "dataset_version_id": args.dataset_version_id,
        "episodes_checked": int(raw.get("episodes") or 0),
        "findings": len(findings),
        "by_kind": by_kind,
        "metrics": raw.get("metrics") or {},
        "image_check": raw.get("image_check") or {},
        "sample": findings[:5],
    }

    with session_scope() as session:
        report = QualityReport(
            dataset_version_id=args.dataset_version_id,
            rules_version=str(raw.get("rules_version", "1.0")),
            episodes=summary["episodes_checked"],
            finding_count=summary["findings"],
            summary={"by_kind": by_kind, "metrics": summary["metrics"]},
            report_uri=str(output),
        )
        session.add(report)
        session.flush()
        summary["report_id"] = report.id
        write_json(output, summary)

    print(json.dumps({
        "dataset_version_id": args.dataset_version_id,
        "episodes_checked": summary["episodes_checked"],
        "findings": summary["findings"],
        "report_path": str(output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
