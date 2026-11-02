# 代码位置与说明

根目录：`/home/oyjt/workspace/embodiedflow`（软链到 `/mnt/20e266f0-…/oyjt/embodiedflow`）
数据目录：`/home/oyjt/workspace/embodiedflow_data`
Python：`.venv`（3.11.14）；训练/推理用 `/home/oyjt/workspace/openpi_v2/.venv`（带 torch）

## 一、目录职责

| 目录 | 作用 |
|---|---|
| `apps/api/main.py` | FastAPI 入口：lifespan（迁移/GPU 同步/部署恢复）、CORS、可选 token、挂载全部路由 |
| `apps/api/routers/` | 16 个路由模块，见下表 |
| `apps/web/workbench.html` | 工作台单页（HTML+CSS+JS 全内联，无构建、无 CDN） |
| `apps/web/app.py` | Streamlit 管理界面（10 页） |
| `services/` | 业务逻辑（任务队列、资源、质检、评测、模型、标注、边缘契约等） |
| `adapters/` | 外部形态适配：`datasets/lerobot.py`、`training/openpi.py`、`annotation/flora.py` |
| `persistence/` | `models.py`（SQLAlchemy 表）、`db.py`（连接与 pragma）、`migrate.py`（v1–v10 迁移） |
| `worker/main.py` | 独立 Worker：领任务、分配 GPU、起子进程、取消、失联恢复 |
| `scripts/` | 训练/评测/推理/导入/校验脚本（见下） |
| `ops/` | 启停脚本与自检 |
| `docs/` | 16 份状态与说明文档 |
| `tests/` | 75 项 pytest |

## 二、路由模块（`apps/api/routers/`）

| 文件 | 前缀 | 负责 |
|---|---|---|
| `workbench.py` | `/workbench` | **工作台主接口**：总览、数据集详情、预处理、训练、评测、部署、基准、实验、算力、模型、设备、日志 |
| `jobs.py` | `/api/jobs` | 任务提交/查询/取消/日志 |
| `datasets.py` | `/api/datasets` | 上传、登记、校验、质检、删除 |
| `resources.py` | `/api/resources` | GPU 台账与分配 |
| `models.py` | `/api/models` | 模型与版本 |
| `evaluations.py` | `/api/evaluations` | 评测记录 |
| `devices.py` | `/api/devices` | 设备台账 |
| `annotation.py` | `/api/annotation` | FLORA 标注作业 |
| `benchmarks.py` | `/api/benchmarks` | MuJoCo 基准对比 |
| `families.py` | `/api/families` | 模型家族注册表 |
| `edge.py` | `/api/edge` | 机器人执行契约（心跳/动作/急停） |
| `assistant.py` | `/api/assistant` | 只读问答 |
| `catalog.py` `report.py` `health.py` | `/api` | 聚合、演示报告、健康检查 |

## 三、脚本（`scripts/`）

| 脚本 | 用途 | 运行解释器 |
|---|---|---|
| `bc_train.py` | BC 训练（state→action，按 Episode 划分） | openpi_v2/.venv |
| `bc_eval.py` | 离线评测（MAE/RMSE + 基线） | openpi_v2/.venv |
| `bc_server.py` | BC 推理服务（标准库 http.server） | openpi_v2/.venv |
| `import_datasets.py` | 扫描并登记全部 LeRobot 数据集 | .venv |
| `import_logs.py` | 扫描并索引日志 | .venv |
| `backfill_episodes.py` | 回填 Episode 帧数（幂等，默认 dry-run） | .venv |
| `lint_page.py` | 页面静态校验（函数/元素/路由/锚点） | .venv |

## 四、一次请求怎么走

```
浏览器 /workbench/
  → FastAPI (apps/api/routers/workbench.py)
      → services/*（业务逻辑）
          → persistence（读写 SQLite）
          → 需要跑东西时：写一条 job 记录返回
  → Worker (worker/main.py) 轮询领任务
      → 子进程执行 scripts/*.py（带 GPU 租约环境变量）
      → 结束写回状态、MLflow run、模型版本
```

## 五、常见修改点

| 想改 | 改哪里 |
|---|---|
| 页面样式/文案 | `apps/web/workbench.html` 的 `<style>` 与各 `<h2>`/`.hint` |
| 加一个页面或区块 | 在 `apps/web/workbench.html` 的 `.content-stack` 内增加有唯一 ID 的 `<section>`，在 `pageViews` 映射视图，并为侧栏增加 `data-page` 导航项 |
| 加一个接口 | `apps/api/routers/workbench.py`，加 `@router.get/post` |
| 改表结构 | `persistence/models.py` + `migrate.py` 加一条 MIGRATIONS 并写分支 |
| 改训练命令 | `workbench.py` 的 `create_training` / `train_openpi` |
| 改调度策略 | `services/resources.py`（`ALLOW_PARTIAL` / `MIN_FREE_MB`） |
| 改机器人契约 | `services/edge.py`（DOF/限位/步长/看门狗时长） |

### 工作台界面结构

`apps/web/workbench.html` 包含页面的 HTML、CSS 和 JavaScript。`<style>` 中的主题变量定义颜色、卡片和响应式布局，侧栏位于 `<aside class="sidebar">`，页面卡片集中在 `<main>` 内的 `.content-stack`。`pageViews` 描述导航与区块的对应关系，`switchPage()` 负责切换和读取数据，`refreshCurrent()` 负责当前页刷新。

主题切换保存在浏览器的 `localStorage` 中。数据选择组件在数据集、数据处理和训练视图间共用。旧的页面锚点仍指向对应视图，训练、评测、部署接口调用仍由该文件中的原有函数处理。

`apps/api/routers/workbench.py` 在工作台 GET 请求时读取 `workbench.html`。只改 HTML、CSS 或页面脚本时刷新浏览器即可，修改 Python 路由后按本项目运维流程重启 API。每次修改页面运行 `.venv/bin/python scripts/lint_page.py`，再用浏览器验证导航和表单。

## 六、运维命令（`ops/`）

```bash
bash ops/start_api.sh      # API :18080
bash ops/start_worker.sh   # Worker
bash ops/start_web.sh      # Streamlit :8501
bash ops/start_mlflow.sh   # MLflow :15000
bash ops/stop_all.sh
bash ops/status.sh
bash ops/doctor.sh         # 20 项自检
```

## 七、数据位置

| 内容 | 路径 |
|---|---|
| SQLite | `embodiedflow_data/metadata/platform.db`（schema v10） |
| 任务日志 | `embodiedflow_data/runs/job_N/attempt_M/worker.log` |
| 训练产物 | `embodiedflow_data/workbench/run_*/`（checkpoint.pt / norm.json / metrics.json） |
| 基准与回放 | `embodiedflow_data/artifacts/benchmarks/` |
| π0.5 checkpoint | `~/workspace/checkpoints/<config>/<exp_name>/` |
| 数据集 | 分散在 `~/workspace/*`（856 个，见数据库 `datasets` 表） |

## 八、当前状态

schema v10 · 128 条路由 · 75 项测试 · 自检 20/20 · 工作台 12 个区块


## ??OpenPI ???????

- ???????? `apps/web/workbench.html`???? OpenPI ???????? batch??????????SigLIP LoRA rank??????GPU ????????????????/?????????????
- OpenPI ?????????? `apps/api/routers/workbench.py` ? `OpenPIIn`?`openpi_configs()`?`train_openpi()`???? Worker ?????????? metrics?? MLflow ???????
- `scripts/openpi_workbench_train.py` ? UI ????? OpenPI `TrainConfig`?W&B ???loss ??? MLflow metrics?SigLIP rank ???PaliGemma/Action Expert rank ??????
- Agilex ??????? ROS?MJPEG ?? `/home/agilex/embodiedflow_camera/camera_mjpeg_bridge.py`???? `http://172.19.11.47:8765/front.mjpg`???? `/left.mjpg`???????? Piper ? `Device.config.camera_url`??????????

## 本机模型目录

模型仓库页通过 `GET /workbench/api/registry` 返回 `available_models`。服务端在 `apps/api/routers/workbench.py::_local_model_catalog()` 中只检查目录、Python 解释器和候选权重路径，不导入模型框架，也不启动 GPU 任务；页面由 `apps/web/workbench.html::renderModelCatalog()` 渲染。

服务器当前找到 OpenPI π0.5、LingBot-VLA V2 6B、NVIDIA GR00T N1.7 3B、OpenWAM、DreamZero、NVIDIA GEAR-SONIC，以及 SuperMimic 非官方复现的源码/环境/部分 checkpoint。`training_adapter` 是训练能力的准入标记：目前只有 OpenPI 为 `workbench-train`，其他模型只是已发现本机资产，训练、评测和部署适配仍需分别实现。模型页面应以该字段显示训练就绪状态，不能仅凭权重文件存在就显示可运行。

SONIC 的本机目录为 `/home/oyjt/workspace/GR00T-WholeBodyControl`；HF 缓存有 encoder/decoder ONNX，但 planner 下载不完整，专用环境和工作台适配未接入。SuperMimic 是 `/mnt/20e266f0-6466-4ef3-a875-5959cd1015d7/oyjt/supermimic_vibed` 的非官方复现，包含 MuJoCo-Warp 环境、LAFAN1 与训练 checkpoint；工作台 RL 训练链仍未接入。

## 相机按需推流

机器人设备页打开时才为当前设备设置 MJPEG `<img src>`；`switchPage()` 离开 `sec-robot` 会清除 `src` 并终止浏览器请求。Agilex 桥接进程可常驻，但 `/front.mjpg` 或 `/left.mjpg` 没有 HTTP 客户端时不订阅 ROS 图像；第一个客户端连接时订阅，最后一个客户端退出后退订并清空当前帧。`/health` 只返回订阅/客户端状态，不触发相机读取。

相关实现：`apps/web/workbench.html::switchPage()` / `disconnectRobotCamera()`；`/home/agilex/embodiedflow_camera/camera_mjpeg_bridge.py::FrameStore.attach()` / `detach()`。
