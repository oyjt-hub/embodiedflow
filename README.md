# EmbodiedFlow · 单机研发平台（P0.1 骨架）

按 `EmbodiedFlow-系统建设与分阶段实施计划.md` 落地的第一阶段骨架。
当前实现的是 **P0.1**（工程骨架 + 持久任务队列 + 独立 Worker）与 **P0.3 的队列/GPU 租约底座**。

## 快速开始

```bash
cd ~/workspace/embodiedflow
bash ops/start_api.sh      # FastAPI  :8080
bash ops/start_worker.sh   # Worker   独立进程
bash ops/start_web.sh      # Streamlit :8501
bash ops/smoke.sh          # P0.1 验收：提交 → 执行 → 查日志
bash ops/status.sh         # 看进程与健康
bash ops/stop_all.sh
```

Web 通过 SSH 端口转发访问：`ssh -L 8501:127.0.0.1:8501 <USER>@<SERVER_HOST>`，然后浏览器开 `http://127.0.0.1:8501`。

API 端口默认 **18080**（8080 被别的服务占用）。验收记录见 `docs/STATUS_P0.1.md`。

## 已交付能力（可点、可验）

| 能力 | 落点 | 证据 |
|---|---|---|
| 项目目录与配置 | `core/config.py`、`configs/platform.yaml` | `EF_DATA` 下 7 个子目录自动创建 |
| 数据库与迁移 | `persistence/models.py`、`persistence/migrate.py` | `schema_version` 表，v1/v2，WAL |
| REST API | `apps/api/` | `/api/health`、`/api/jobs*`、`/api/resources/gpus`、`/api/summary`、`/docs` |
| 持久任务队列 | `services/jobs.py` | `UPDATE ... WHERE status='queued'` 抢占，attempt 编号 |
| 独立 Worker | `worker/main.py` | 子进程执行、日志落盘、心跳、取消（SIGTERM→SIGKILL） |
| GPU 登记与租约 | `services/resources.py` | `nvidia-smi` 采样，显存 >1 GiB 视为外部占用，按 attempt 租约 |
| 网页 UI | `apps/web/app.py` | 概览/任务/数据/算力/模型/设备/评测/Planned |
| 测试 | `tests/` | 生命周期、取消、无适配器显式失败、API 冒烟 |

**验收口径**（计划 P0.1）：网页提交 → 独立 Worker 领取执行 → 刷新后仍能查到结果。`ops/smoke.sh` 断言 `completed`。

**P0.2 已交付**：LeRobot 读取器、12 类质检规则（可定位 Episode 与时间点）、多频对齐、可复现切分、归一化只拟合训练集；GPU 调度改为「够显存即可用」。详见 `docs/STATUS_P0.1.md`、`docs/STATUS_P0.2.md`。

**P0.3 已交付**：资源感知抢占（争卡串行、留队等待）、卡位绑定 `gpu_indices`、失败重试预算 `max_attempts`、Worker 失联孤儿回收。详见 `docs/STATUS_P0.3.md`。

**P0.4 部分交付**：真实 LeRobot 数据验收（600 Episode）、MLflow 追踪、模型仓库与血缘、OpenPI 适配器 Dry Run。π0.5 实跑一步待续。详见 `docs/STATUS_P0.4.md`。

**P0.5 已交付**：离线评测器（时长/延迟 P50-P95/MAE/RMSE/速度·加速度方差/Jerk，缺数据给 N/A 不伪造）+ 覆盖率口径 + CSV/JSON 报告 + 4 个 API。详见 `docs/STATUS_P0.5.md`。

## 与计划的差异（如实记录）

| 计划要求 | 现状 | 处理 |
|---|---|---|
| Python 3.11 | ✅ 已达成 | 通过 `uv` 安装独立 CPython **3.11.14**，venv 无 pip 需用 `uv pip install --python .venv/bin/python` |
| 可写有配额的数据目录 | `/data` 根与 `/data/users` **不可写** | 暂用 `~/workspace/embodiedflow_data`（/home 余 82G）；待管理员分配后改 `EF_DATA` |
| 独占 GPU 时段 | 8 卡**全部被他人任务占用**（外部进程） | Worker 已实现"外部占用不调度"，实测会拒绝并记 `insufficient free GPUs` |
| MLflow | ✅ 已安装 3.17.2 | P0.4 接 Tracking；当前用 `Experiment` 表占位，`mlflow_run_id` 字段已留 |
| 计划中的 `TrainingJob` | 实现为统一 `Job(kind='train')` | 避免为每种任务复制队列表；`docs/ARCHITECTURE.md` 说明映射 |

## 目录

```
embodiedflow/
├── apps/api/         FastAPI 路由（health/jobs/resources/catalog）
├── apps/web/         Streamlit 页面
├── worker/           任务领取、子进程执行、心跳、取消
├── core/             配置、枚举
├── services/         任务与 GPU 服务
├── persistence/      SQLAlchemy 模型、迁移
├── ops/              启动/停止/状态/冒烟
├── configs/          平台配置模板
├── tests/            单元与接口测试
└── docs/             架构与接口说明
```

## 下一步（按计划顺序）

1. **P0.2** 数据层：LeRobot 读取器 + 上传会话 + 摘要校验 + 质检规则（12 类异常）
2. **P0.3 收尾**：两任务争卡/取消/Worker 重启的集成测试
3. **P0.4** MLflow 接入 + `OpenPIAdapter` + 模型登记（需一个可用的 GPU 窗口）

详见 `docs/STATUS_P0.5.md`。

- **P0.2b**：FLORA 自动标注管道（作业/记录/分层抽检/派生版本），模型待接 → `docs/STATUS_P0.2b.md`

- **P1.1**：MuJoCo 自动基准（6/8 可复现）→ `docs/STATUS_P1.1.md`
