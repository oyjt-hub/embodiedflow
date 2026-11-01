# P0.1 交付状态（2026-10-31）

计划文件：`~/workspace/EmbodiedFlow-系统建设与分阶段实施计划.md`（37667 B，已就位）
代码目录：`~/workspace/embodiedflow`
数据目录：`~/workspace/embodiedflow_data`

## 一、环境（实测）

| 项 | 结果 |
|---|---|
| Python | **3.11.14**（用 `uv` 安装的独立解释器，符合计划要求） |
| 依赖 | fastapi 0.162.0 / uvicorn 0.54.0 / sqlalchemy 2.1.6 / pydantic 2.14.2 / streamlit 1.66.1 / plotly 7.1.0 / **mlflow 3.17.2** |
| 数据库 | SQLite（WAL），`embodiedflow_data/metadata/platform.db`，schema **v2** |
| 端口 | API `127.0.0.1:18080`、Web `0.0.0.0:8501`（8080 被他人服务占用，已避开） |

## 二、P0.1 验收（计划要求：网页提交 → 独立 Worker 执行 → 刷新后仍可查）

```
[1] GET /api/health      -> {"ok":true, "python":"3.11.14", "schema_version":2}
[2] POST /api/jobs       -> 202, job_id=1, status=queued
[3] 轮询                 -> 01s queued → 02s running → 05s completed
[4] GET /api/jobs/1      -> {"status":"completed","started_at":...,"finished_at":...}
[5] GET /api/jobs/1/logs -> runs/job_1/attempt_1/worker.log
      embodiedflow worker ok
      python 3.11.14
      tick 0..4
      done
      # exit_code=0 cancelled=False
[6] 判定                 -> PASS
```

**结论：P0.1 通过。** 任务确实由独立 Worker 进程以子进程方式执行，日志落盘并可从 API 取回。

## 三、P0.3 底座（部分）

| 能力 | 实测证据 |
|---|---|
| GPU 登记 | `nvidia-smi` 采样写入 `gpu_resources`，8 卡全部登记（RTX 4090 49140 MiB） |
| 外部占用识别 | 8 卡 state 全部为 `external`（利用率 88–100%、显存 7.3–25.0 GiB 被他人任务占用） |
| 资源不足拒绝 | 提交 `requested_gpus=1` → `failed: insufficient free GPUs: need 1, have 0` |
| 只读探测任务 | `gpu_probe` 完成，日志含 8 卡完整 UUID/显存/利用率快照 |
| 任务态机 | `queued → running → completed/failed/cancelled`，`job_attempts` 独立编号 |
| 取消 | Worker 每秒查状态，命中即 SIGTERM 整个进程组，2s 后 SIGKILL |

## 四、测试

```
PYTHONPATH=. .venv/bin/python -m pytest tests -q
5 passed in 1.25s
```
覆盖：任务生命周期、取消后不可再领取、未实现 kind 显式失败、API 冒烟、GPU 采样结构。

## 五、与计划的偏差（如实记录）

| 计划 | 现状 | 说明 |
|---|---|---|
| 可写且有配额的项目数据目录 | **未落实** | `/data` 根与 `/data/users` 对 oyjt 不可写；暂用 `~/workspace/embodiedflow_data`（/home 余 ~82 G）。**这是首周 Day 1 需要管理员分配的事项** |
| 独占 GPU 时段 | **未落实** | 8 卡全部被他人的 holobrain/orchard 等任务占用。平台已能正确识别并拒绝调度，但真实训练（P0.4）要等窗口 |
| 计划中 `TrainingJob` 实体 | 实现为 `Job(kind='train')` | 统一队列，避免每类任务一套表；见 `docs/ARCHITECTURE.md` |

## 六、下一步

1. **P0.2**：LeRobot 读取器、上传会话、摘要校验、12 类质检规则
2. **P0.4 前置**：向管理员申请 1 张卡的时间窗 + 一个配额目录
3. 集成测试补齐：两任务争卡、Worker 重启恢复（读 `heartbeat_at` 判活）
