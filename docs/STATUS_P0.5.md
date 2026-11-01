# P0.5 交付状态（离线评测 · 设备台账 · Dashboard · 演示报告）

## 一、离线评测（已在前一轮交付，此处汇总）

`services/evaluation.py`：时长、吞吐、延迟 P50/P95、MAE/RMSE、速度/加速度方差、Jerk、
终态与干预计数；缺数据给 `null + reason`，跨 Episode 汇总带覆盖率与分母。
数值测试用手算夹具（5 步线性斜坡：MAE=2.0、RMSE=√6、P95=48.0、Jerk=0），5 条断言通过。

## 二、设备台账

| 能力 | 实测 |
|---|---|
| 登记 | `POST /api/devices`，支持 `source=real|mock|sim`、dof、control_hz、config |
| 连接测试/心跳 | `POST /api/devices/{id}/probe` → 更新 `last_seen_at`；真实设备 `online`，**Mock 显式返回 `mock-online`** |
| 在线率 | 只统计 `source=real`；`{value, numerator, denominator, note}` |
| 详情 | `GET /api/devices/{id}` 带关联部署数 |

实测：`piper-demo(real)` + `piper-mock(mock)` → probe 后
`online_rate = {value:1.0, numerator:1, denominator:1}`，Mock 不在分母内。

## 三、Dashboard 汇总

`GET /api/summary` 现返回：

```json
{"devices_real":1,"devices_online":1,"devices_online_rate":1.0,
 "quality_reports":2,"evaluations":1,"datasets":2,"dataset_versions":2,
 "jobs":5,"jobs_running":0,"jobs_failed":0,"models":3,"devices":2}
```

## 四、六项演示验收报告

`GET /api/report/mvp` / `GET /api/report/mvp.md`，按计划第 8 周口径逐项判定：

| # | 演示 | 结论 | 证据 |
|---|---|---|---|
| 1 | 数据管理 | 通过 | datasets=2, quality_reports=2 |
| 2 | 训练管理 | 通过 | completed train jobs=2（另含取消/资源不足用例） |
| 3 | 实验追踪 | 通过 | attempts 关联 mlflow run=2 |
| 4 | 评测管理 | **条件通过** | evaluations=1（合成日志，真实日志待补） |
| 5 | 模型管理 | 通过 | model_versions=3 |
| 6 | 设备管理 | 通过 | devices real=1, mock=1 |

汇总：**通过 5 / 条件通过 1 / 待验收 0**，报告落盘 `artifacts/reports/mvp_acceptance.md`。

## 五、反复踩的坑：路由重复

`catalog.py` 先后与 `models`、`evaluations`、`devices` 路由抢同一路径，
FastAPI 先注册先匹配导致后者永不生效（表现为返回结构不对，而不是 404）。
已全部移除，并在 catalog 顶部加了职责说明：**实体 CRUD 归各自路由，catalog 只留只读聚合**。

## 六、测试

```
PYTHONPATH=. .venv/bin/python -m pytest tests -q
33 passed
```

## 七、未完成（进入 P1 前需补）

| 项 | 说明 |
|---|---|
| 真实执行日志 | 六项演示第 4 项仍是合成日志，需工控机导出一份 |
| Streamlit Dashboard 未接评测 | 后端 summary 已就绪，UI 页面待接线 |
| 部署/回流（P1.2） | 需要真机与现场授权 |
