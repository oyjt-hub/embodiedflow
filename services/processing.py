from __future__ import annotations

import json
import uuid
from pathlib import Path

CLEAN_OPS = {
    "drop_any_finding": "剔除有任何质检异常的 Episode",
    "drop_static": "剔除静止帧过多的 Episode",
    "drop_incomplete": "剔除不完整的 Episode",
    "drop_joint_violation": "剔除关节越限的 Episode",
    "drop_action_jump": "剔除动作跳变的 Episode",
}

OP_TO_RULE = {
    "drop_static": "static_frames",
    "drop_incomplete": "episode_incomplete",
    "drop_joint_violation": "joint_limit",
    "drop_action_jump": "action_jump",
}


def execute_quality_clean(
    dataset_version_id: int,
    ops: list[str],
    max_episodes: int = 200,
    apply: bool = False,
    project_id: int | None = None,
    job_id: int | None = None,
) -> dict:
    """Inspect Episode quality and optionally publish a zero-copy filtered version."""
    from adapters.datasets.lerobot import LeRobotAdapter
    from persistence.db import session_scope
    from persistence.models import (
        DatasetVersion, PipelineTask, Project, utcnow,
    )
    from services import quality as quality_service

    if not ops:
        raise ValueError("至少选择一种清洗方式")
    unknown = [op for op in ops if op not in CLEAN_OPS]
    if unknown:
        raise ValueError(f"未知清洗方式: {unknown}")

    with session_scope() as session:
        version = session.get(DatasetVersion, dataset_version_id)
        if version is None:
            raise ValueError(f"数据版本 {dataset_version_id} 不存在")
        manifest = version.manifest_uri or ""
        path = Path(manifest).parent.parent if manifest.endswith("meta/info.json") else Path(manifest)
        version_data = {
            "id": version.id,
            "dataset_id": version.dataset_id,
            "version": version.version,
            "episode_count": version.episode_count,
            "total_bytes": version.total_bytes,
            "manifest_uri": version.manifest_uri,
            "episode_indices": version.episode_indices,
        }
        project = session.get(Project, project_id) if project_id else None
        if project_id and project is None:
            raise ValueError("项目不存在")
        project_name = project.name if project else None

    if not path.exists():
        raise ValueError(f"数据目录不存在：{path}")
    report = quality_service.build_report(
        LeRobotAdapter(path), max_episodes=max_episodes,
    )
    findings = report.get("findings") or []
    bad_by_episode: dict[int, set[str]] = {}
    for finding in findings:
        episode = finding.get("episode")
        if episode is not None:
            bad_by_episode.setdefault(int(episode), set()).add(finding.get("rule") or "")

    declared: list[int] = []
    if version_data["episode_indices"]:
        try:
            declared = json.loads(version_data["episode_indices"])
        except (TypeError, ValueError, json.JSONDecodeError):
            declared = []
    if not declared:
        declared = list(range(version_data["episode_count"] or 0))

    dropped: set[int] = set()
    for episode, rules in bad_by_episode.items():
        if "drop_any_finding" in ops and rules:
            dropped.add(episode)
            continue
        if any(OP_TO_RULE.get(op) in rules for op in ops):
            dropped.add(episode)
    kept = [episode for episode in declared if episode not in dropped]

    result = {
        "dataset_version_id": version_data["id"],
        "checked_episodes": report.get("episodes"),
        "findings": len(findings),
        "image_check": report.get("image_check"),
        "declared_episodes": len(declared),
        "dropped_episodes": sorted(dropped),
        "kept_episodes": len(kept),
        "apply": apply,
        "project_id": project_id,
        "project": project_name,
        "new_version_id": None,
    }

    if apply and kept and kept != declared:
        with session_scope() as session:
            stamp = int(utcnow().timestamp())
            new_version = DatasetVersion(
                dataset_id=version_data["dataset_id"],
                version=f"{version_data['version'][:14]}-c{stamp:x}{uuid.uuid4().hex[:5]}",
                status="derived",
                episode_count=len(kept),
                total_bytes=version_data["total_bytes"],
                manifest_uri=version_data["manifest_uri"],
                episode_indices=json.dumps(kept),
            )
            session.add(new_version)
            session.flush()
            steps = {
                "operator": "filter_episodes_by_quality",
                "ops": ops,
                "rules_version": report.get("rules_version", "1.0"),
                "max_episodes": max_episodes,
                "project_id": project_id,
                "project": project_name,
                "checked_episodes": int(report.get("episodes") or 0),
                "finding_count": len(findings),
                "dropped_episodes": sorted(dropped),
                "kept_episodes": len(kept),
                "metrics": report.get("metrics") or {},
            }
            session.add(PipelineTask(
                job_id=job_id,
                input_version_id=version_data["id"],
                output_version_id=new_version.id,
                project_id=project_id,
                steps=steps,
                status="completed",
            ))
            result.update({
                "new_version_id": new_version.id,
                "new_version": new_version.version,
                "pipeline": steps["operator"],
            })
    return result
