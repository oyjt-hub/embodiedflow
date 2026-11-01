# 工作台（/workbench/）交付状态

计划：`EmbodiedFlow-系统建设与分阶段实施计划.md`（2026-11-01 修订，当日上线版）

## 网址

```
http://<SERVER_HOST>:18080/workbench/
```

页面同源单文件，内联 HTML/CSS/JS，无 CDN 依赖，无需构建。

## P0 五步验收

| 步骤 | 计划要求 | 实测证据 |
|---|---|---|
| ① 打开网页 | 选到样本并看到状态/动作曲线 | 下拉列出数据版本；SVG 画前 3 维 state/action（120 帧） |
| ② 点击训练 | PyTorch BC，按 Episode 划分，参数可改 | job #12/#14 完成；21 训 / 3 留；轮数/学习率/设备可填 |
| ③ 点击评测 | 留出集 MAE/RMSE + 基线，可下载 JSON | MAE 0.0344 / RMSE 0.0469 / 基线 0.3524 / 好转 90%；报告含逐 Episode |
| ④ 点击部署 | 14 维状态 → 14 维动作 + 版本 + 耗时 | latency 0.85–13.9 ms；与离线前向最大差 **4.9e-07** |
| ⑤ 串联交付 | 训练→评测→部署可重走 | 链路 ID：job 12 → eval 13 → deployment 2 |
| 附加 | Worker GPU 重编号问题 | 已改为传 GPU UUID，用非 0 号卡验证 |

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/workbench/` | 工作台页面 |
| GET | `/workbench/api/datasets` | 可选数据版本 |
| GET | `/workbench/api/preview/{vid}` | 某 Episode 前 N 帧状态/动作 |
| POST | `/workbench/api/training/jobs` | 提交 BC 训练 |
| GET | `/workbench/api/jobs/{id}` | 状态 + 日志尾部 |
| POST | `/workbench/api/evaluations` | 提交离线评测 |
| GET | `/workbench/api/evaluations/{id}/report` | 评测报告 JSON |
| POST | `/workbench/api/deployments` | 发布推理服务 |
| POST | `/workbench/api/deployments/{id}/predict` | 推理 |
| POST | `/workbench/api/deployments/{id}/stop` | 停止 |
| GET | `/workbench/api/deployments` | 部署列表 |

## 实现要点

- 平台 venv 无 torch；训练/评测/推理统一用 `openpi_v2/.venv` 解释器，通过 `spec.command` 指定，不污染平台依赖
- 推理服务是独立进程（标准库 `http.server`），API 只转发；停止用 `killpg`
- 训练产物：`checkpoint.pt` / `norm.json`（训练集归一化）/ `metrics.json`；评测复用同一 `norm.json`
- MLflow：训练作业带 `mlflow: true`，attempt 记录 `mlflow_run_id`（job #14 → `5676003e…`）

## 与计划的差异（需知悉）

| 计划假设 | 实际 |
|---|---|
| ALOHA mobile cabinet 12 Episode | 服务器无此数据，用本地 `openwam_30k/uniform24`（24 ep，同 14 维、30 fps），按 21/3 划分 |
| HTML/CSS/JS 草稿已准备 | 服务器上未找到草稿，页面为新写 |
| 端到端 4 小时 | 实际跨越多轮实施，P0 各部分均已单独验证 |

## 未做（P1/P2，计划列为按需追加）

- P1.1 自有数据与 π0.5 训练
- P1.2 MuJoCo 自动 Benchmark 接入模型页
- P1.3 首台真机 Robot Adapter
- P2 板端量化 / 多节点调度
