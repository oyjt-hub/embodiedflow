# 训练发起与守护

`train-launch`

用受支持的配置发起训练，跟踪到终态，失败时给出原因与重试建议。

## 输入

- `project_id` (int, 必填)
- `dataset_version_id` (int, 必填)
- `backend` (str, 必填，取值 ['bc', 'act', 'openpi'])
- `epochs` (int, 可选，默认 1)
- `lr` (float, 可选，默认 0.001)
- `batch_size` (int, 可选，默认 64)
- `device` (str, 可选，默认 cpu)

## 步骤

1. training_get_capabilities() 取允许的 backend 与超参范围
2. training_validate_config(...) 预校验，避免提交后才失败
3. training_submit(...) 创建 Job
4. training_get_job(job_id) 轮询至终态；超时用 training_get_logs(job_id) 看日志
5. 失败时 training_retry(job_id) 受 max_attempts 约束

## 输出

- `job_id`: int
- `status`: queued/running/completed/failed
- `metrics`: val_mae / val_rmse / loss
- `model_version_id`: 成功时的模型版本
- `failure_reason`: 失败时的 error 与日志摘要

## 使用的 MCP 工具

`training_get_capabilities`、`training_validate_config`、`training_submit`、`training_get_job`、`training_get_logs`、`training_retry`
