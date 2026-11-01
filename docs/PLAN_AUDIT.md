# Phase 0 · 现有工程审计（对照《EmbodiedFlow 完整任务书》）

审计时间：2026-11-01
任务书：`D:\codex\2026-10-31\zhe\outputs\计划.txt`（1634 行，10 模块 + Phase 0–6）

## 一、约束遵守情况

| 任务书要求 | 现状 |
|---|---|
| 优先复用现有 FastAPI | ✅ 单应用 16 路由模块 / 64 接口 |
| 统一 `/workbench/` 提供工作台 | ✅ 已上线 |
| 用现有 Worker 执行耗时任务 | ✅ |
| 用现有 GPU 调度器 | ✅（已修 UUID 重编号） |
| SQLite 保存业务元数据 | ✅ 22 表 / schema v7 |
| MLflow 记录实验 | ✅ 训练自动写 run |
| 现有模型仓库管理权重 | ✅ models / model_versions |
| 大文件放数据盘不进 SQLite | ✅ DB 仅 304 KB，制品在磁盘 |
| 不重复建设 | ✅ |
| 不引入 K8s/Kafka/Ray/Redis | ✅ 零额外基础设施 |
| 前端不硬编码秘密 | ✅ 无 token/密码硬编码 |

## 二、十个模块对照

| # | 模块 | 后端 | 前端页面 | 结论 |
|---|---|---|---|---|
| 1 | Overview 系统总览 | `/api/summary`、`/api/resources/gpus` | Streamlit 有，工作台无 | 🟡 需并入工作台 |
| 2 | Datasets 数据管理 | 上传 / 登记 / 版本 / `episodes` 637 行 | 仅下拉选择 | 🟡 缺详情页、Episode 浏览器、Manifest |
| 3 | Data Processing 数据处理 | 质检 12 规则、对齐、切分 | 无 | 🟡 后端有，无页面 |
| 4 | Training 模型训练 | BC + OpenPI，job 提交/日志/取消 | ✅ 有 | ✅ 完成 |
| 5 | Experiments 实验管理 | `experiments` 表存在（0 行）、MLflow | 无 | 🟡 缺页面与 API |
| 6 | GPU Cluster 算力管理 | `gpu_resources` / `gpu_leases` / `/api/resources/*` | 无 | 🟡 缺页面 |
| 7 | Benchmark 模型评测 | L1 离线评测 ✅、L2 MuJoCo ✅、L3 真机 ❌ | 部分（③⑤ 区） | 🟡 L3 待真机 |
| 8 | Model Registry 模型仓库 | models / model_versions / 血缘 / `/api/models` | 无 | 🟡 缺页面 |
| 9 | Deployment 模型部署 | 起停 + predict + 一致性校验 ✅ | ✅ 有 | ✅ 基本完成 |
| 10 | Robot Hub 设备管理 | `devices` 2 行 + `/api/edge/*` 执行契约 | 无 | 🟡 缺页面 |

## 三、Phase 对照

| Phase | 任务书要求 | 状态 |
|---|---|---|
| 0 | 工程检查 | ✅ **本文件** |
| 1 | 工作台上线 | ✅ `/workbench/` 五区块 |
| 2 | 真实训练闭环 | ✅ BC（MAE 0.028 vs 基线 0.352）+ π0.5（job 15，3 GB ckpt） |
| 3 | 真实评测 | 🟡 L1/L2 有，L3 待真机 |
| 4 | 真实推理部署 | ✅ BC 0.85 ms，一致性 4.9e-07；π0.5 推理缺数据 |
| 5 | 系统串联与恢复 | 🟡 链路可重走；未验证"重启后按登记版本恢复推理" |
| 6 | 扩展算法与设备 | ⛔ 待真机与需求 |

## 四、本轮之后的开发顺序（建议）

1. **Overview 并入工作台**（统计卡片 + 最近活动 + 快捷入口）
2. **Datasets 详情页**（基本信息 + Episode 浏览器 + Manifest）
3. **Data Processing 页面**（清洗/对齐/切分入口 + 质量报告）
4. **Experiments 页面**（MLflow run 列表与对比）
5. **GPU Cluster 页面**（实时卡态 + 租约 + 调度记录）
6. **Model Registry 页面**（版本、血缘、下线）
7. **Robot Hub 页面**（设备列表 + 执行契约状态）
8. **Deployment 页面补全**（生命周期、一致性、性能）
9. **Phase 5**：重启后按登记版本恢复推理

## 五、已知阻塞（非代码问题）

| 项 | 原因 |
|---|---|
| π0.5 推理 | 缺 `physical-intelligence/libero` 数据（HF 缓存仅 12K 空壳），无法生成 norm stats |
| L3 真机评测 | 无真机与现场授权 |
| RDK/BPU 量化 | 无板卡工具链 |
