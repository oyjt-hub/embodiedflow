# 评测到部署

`eval-deploy`

对训练产出的模型版本做离线评测，达标后登记并启动推理服务。

## 输入

- `project_id` (int, 必填)
- `model_version_id` (int, 必填)
- `benchmark` (str, 可选，默认 holdout)
- `deploy` (bool, 可选，默认 False)

## 步骤

1. model_list_versions() 确认模型版本与来源 Job
2. evaluation_submit(model_version_id, ...) 提交评测
3. evaluation_get_report(...) 取 MAE/RMSE 与基线改善
4. 达标后 model_register_candidate(...) 登记候选
5. 需要时 deployment_start(...) 启动推理实例

## 输出

- `evaluation_id`: int
- `metrics`: MAE/RMSE 与基线对比
- `registered`: 是否登记为候选
- `deployment`: 启动后的实例与端口
- `verdict`: 达标/不达标与理由

## 使用的 MCP 工具

`model_list_versions`、`evaluation_submit`、`evaluation_get_report`、`model_register_candidate`、`deployment_start`
