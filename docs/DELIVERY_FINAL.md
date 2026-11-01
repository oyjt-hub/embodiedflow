# EmbodiedFlow 交付总结（对照《完整具身智能研发平台建设任务书》）

入口：`http://<SERVER_HOST>:18080/workbench/`
代码：`~/workspace/embodiedflow`　数据：`~/workspace/embodiedflow_data`
规模：16 路由模块 / 63 接口 / schema v9 / 23 张表 / 75 项测试 / 16 份文档

## 一、十个模块

| # | 模块 | 后端 | 页面 | 实测证据 |
|---|---|---|---|---|
| 一 | Overview | ✅ | ✅ | 数据 4/5/637、任务 15、GPU 8 卡、模型 9/5 |
| 二 | Datasets | ✅ | ✅ | 版本 5：24 ep / 727 MB / 14 维 / 30 fps；Episode 768 行有真实帧数 |
| 三 | Data Processing | ✅ | ✅ | 夹具 33 项 / 12 类全中；6 项指标带分子分母 |
| 四 | Training | ✅ | ✅ | BC MAE 0.028 vs 基线 0.352；π0.5 job15 → 3.0 GB ckpt |
| 五 | Experiments | ✅ | ✅ | 平台自持 + MLflow 指标增强 |
| 六 | GPU Cluster | ✅ | ✅ | 8 卡实时态 + 租约归属 |
| 七 | Benchmark | ✅ L1/L2 | ✅ | arm_reach 8/8 vs 0/8；失败轨迹回放 jsonl |
| 八 | Model Registry | ✅ | ✅ | 血缘到数据版本；权重存在性校验；摘要已修正 |
| 九 | Deployment | ✅ | ✅ | 14 维动作 / 0.8–1.9 ms / 一致性 4.9e-07 |
| 十 | Robot Hub | ✅ | ✅ | 2 设备；看门狗/急停/限位约束 |

## 二、Phase 验收

| Phase | 要求 | 证据 |
|---|---|---|
| 0 工程检查 | 先检查现有工程 | `docs/PLAN_AUDIT.md` |
| 1 工作台上线 | `/workbench/` 可用 | 10 区块，静态校验通过 |
| 2 真实训练闭环 | 训练→Checkpoint→可反查 | job#12/#14；BC 与 π0.5 双后端 |
| 3 真实评测 | 留出集指标 + 基线 | MAE 0.0344 / 基线 0.3524 / 好转 90% |
| 4 真实推理部署 | HTTP 端点返回动作 | 在线 vs 离线最大差 4.9e-07 |
| 5 串联与恢复 | 重启后恢复推理 | 杀进程→重启→自动恢复 port 43851→预测 200 |

## 三、本轮修掉的关键缺陷

| 缺陷 | 影响 | 修法 |
|---|---|---|
| `$("device")` 元素不存在 | 点训练必抛 TypeError | 补元素 + 防御读取；新增静态校验器 |
| `tree_digest` 对单文件返回空摘要 | 模型摘要全为 `e3b0c442…` | 支持单文件；重算 2 条历史数据 |
| 派生版本 Episode 归属错 | 8 ep 被记成 600 ep | v8 加 `episode_indices`，三写入点接上 |
| 部署状态存内存 | API 重启即丢 | v9 新表 + 僵尸纠正 + 启动自动恢复 |
| Worker GPU 重编号 | 申请卡 3 实际跑卡 0 | 直传 GPU UUID |
| 自检硬编码 schema=7 | 迁移到 v9 后误报 | 改 `>= 9` |

## 四、已知未完成

| 项 | 原因 |
|---|---|
| π0.5 推理 | 缺 `physical-intelligence/libero` 数据（HF 缓存仅 12K 空壳），无法生成 norm stats |
| L3 真机评测 | 无真机与现场授权 |
| 页面浏览器验收 | 只有静态校验 + HTTP 200，运行时 JS 未人工点验 |
| RDK/BPU 量化 | 无板卡工具链 |

## 五、常用命令

```bash
bash ops/doctor.sh          # 20 项自检
bash ops/start_api.sh       # API :18080
bash ops/start_worker.sh    # Worker
bash ops/start_web.sh       # Streamlit :8501
bash ops/start_mlflow.sh    # MLflow :15000
.venv/bin/python scripts/lint_page.py   # 页面静态校验
```
