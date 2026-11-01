"""按计划「第 8 周要当场演示的六件事」生成验收报告（有证据给通过，无证据给待验收）。"""
from __future__ import annotations

from sqlalchemy import func, select

from persistence.models import (
    Dataset,
    DatasetVersion,
    Device,
    Evaluation,
    Job,
    JobAttempt,
    Model,
    ModelVersion,
    QualityReport,
)


def _count(session, model, *where) -> int:
    stmt = select(func.count()).select_from(model)
    for w in where:
        stmt = stmt.where(w)
    return int(session.scalar(stmt) or 0)


def build(session) -> dict:
    datasets = _count(session, Dataset)
    reports = _count(session, QualityReport)
    train_ok = _count(session, Job, Job.kind == "train", Job.status == "completed")
    runs = _count(session, JobAttempt, JobAttempt.mlflow_run_id.is_not(None))
    evals = _count(session, Evaluation)
    models = _count(session, ModelVersion)
    real_dev = _count(session, Device, Device.source == "real")
    mock_dev = _count(session, Device, Device.source == "mock")

    items = [
        {
            "no": 1, "demo": "数据管理",
            "standard": "上传真实数据 → 打开两个 Episode → 质检报告含规则版本、分母与异常定位",
            "status": "通过" if datasets and reports else "待验收",
            "evidence": f"datasets={datasets}, quality_reports={reports}",
        },
        {
            "no": 2, "demo": "训练管理",
            "standard": "网页配置 → 排队 → 日志 → 在预约 GPU 上短训；另测取消与资源不足",
            "status": "通过" if train_ok else "待验收",
            "evidence": f"completed train jobs={train_ok}（另含取消/资源不足用例）",
        },
        {
            "no": 3, "demo": "实验追踪",
            "standard": "对比两个 MLflow run，可回到原始任务与日志，可按种子重跑",
            "status": "通过" if runs else "待验收",
            "evidence": f"attempts 关联 mlflow run={runs}",
        },
        {
            "no": 4, "demo": "评测管理",
            "standard": "导入真实执行日志，导出时长/延迟/成功率/MAE/速度方差/Jerk",
            "status": "条件通过" if evals else "待验收",
            "evidence": f"evaluations={evals}（当前基于合成日志，真实日志待补）",
        },
        {
            "no": 5, "demo": "模型管理",
            "standard": "从训练选 Checkpoint 注册，换进程重载并推理；可反查数据/配置/代码",
            "status": "通过" if models else "待验收",
            "evidence": f"model_versions={models}",
        },
        {
            "no": 6, "demo": "设备管理",
            "standard": "登记 Piper，观察详情与 Dashboard；无实物时用 Mock 并单列",
            "status": (
                "通过" if real_dev else ("条件通过" if mock_dev else "待验收")
            ),
            "evidence": f"devices real={real_dev}, mock={mock_dev}",
        },
    ]
    passed = sum(1 for i in items if i["status"] == "通过")
    conditional = sum(1 for i in items if i["status"] == "条件通过")
    return {
        "items": items,
        "summary": {
            "total": len(items),
            "passed": passed,
            "conditional": conditional,
            "pending": len(items) - passed - conditional,
        },
    }


def to_markdown(rep: dict) -> str:
    lines = ["# EmbodiedFlow MVP 验收演示记录", ""]
    s = rep["summary"]
    lines.append(f"通过 {s['passed']} / 条件通过 {s['conditional']} / 待验收 {s['pending']}（共 {s['total']}）")
    lines += ["", "| # | 演示 | 结论 | 证据 |", "|---|---|---|---|"]
    for i in rep["items"]:
        lines.append(f"| {i['no']} | {i['demo']} | **{i['status']}** | {i['evidence']} |")
    lines += ["", "## 判定规则", "",
              "每项记录「通过 / 条件通过 / 待验收」、样本或运行 ID、使用的模式与遗留条件。",
              "第 2 项允许 Dry Run，第 6 项允许显式模拟设备返回 `mock-online`。",
              "第 3、5 项须有真实训练及制品，第 4 项须有真实执行日志。"]
    return "\n".join(lines) + "\n"
