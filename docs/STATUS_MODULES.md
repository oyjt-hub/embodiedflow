# 工作台十模块交付状态

入口：`http://<SERVER_HOST>:18080/workbench/`
schema：v9　测试：72 passed

| # | 模块 | 后端接口 | 页面 | 实测证据 |
|---|---|---|---|---|
| 一 | Overview | `/api/overview` | ✅ | 数据 4/5/637、任务 15、GPU 8 卡、模型 9/5 |
| 二 | Datasets | 列表/详情/Episode/Manifest/预览 | ✅ | 版本 5：24 ep、727 MB、14 维、fps 30 |
| 三 | Data Processing | `/api/processing/options`、`/jobs` | ✅ | 夹具 33 findings / 12 类全中；干净数据 0 |
| 四 | Training | `/api/training/jobs`、`/training/openpi` | ✅ | BC MAE 0.028 vs 基线 0.352；π0.5 job15 产 3.0 GB ckpt |
| 五 | Experiments | `/api/experiments` | ✅ | 5 条 run，MLflow 在线时附指标 |
| 六 | GPU Cluster | `/api/gpu` | ✅ | 8 卡实时态 + 租约归属 |
| 七 | Benchmark | `/api/benchmarks/compare` | ✅ | arm_reach 8/8 vs 0/8；失败轨迹回放 |
| 八 | Model Registry | `/api/registry` | ✅ | 9 模型，血缘到数据版本，权重存在性校验 |
| 九 | Deployment | `/api/deployments` + predict/stop | ✅ | 14 维动作、0.8 ms、一致性 4.9e-07 |
| 十 | Robot Hub | `/api/robots` | ✅ | 2 设备、安全约束、看门狗状态 |

## Phase 5：部署持久化（本轮新增）

- v9 新增 `workbench_deployments` 表（不复用 `Deployment`：后者强制 device/model_version 外键）
- 部署记录落库，**API 重启后记录仍在**
- 推理子进程用独立会话，**API 重启不会带走推理服务**（实测重启后仍可推理）
- 进程真死 → 列表读取时自动纠正为 `stopped`，调用返回 404
- 4 条回归测试锁定行为（72 passed）

**未实现**：启动时自动重新拉起上次 running 的部署（需要端口冲突/重复实例防护，单独设计）

## 已知缺陷

| 项 | 说明 |
|---|---|
| π0.5 推理 | 缺 LIBERO 数据 → 无 norm stats → `create_trained_policy` 失败 |
| 页面未经浏览器验收 | 仅验证 HTTP 200 与关键字，运行时 JS 错误未覆盖 |
| `bc` 模型 0 版本 | 历史遗留空模型记录 |
| 4items 模型血缘缺失 | 早期直接登记，无 attempt 关联 |
| Episode 列表 task 字段 | 数据源 `meta/episodes.jsonl` 为空，任务名显示 `-` |
