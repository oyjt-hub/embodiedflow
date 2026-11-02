# 数据就绪检查与清洗

`data-readiness`

在训练前判定一份数据版本是否可用，给出可执行的清洗建议。

## 输入

- `project_id` (int, 必填)
- `dataset_version_id` (int, 必填)
- `max_episodes` (int, 可选，默认 50)
- `min_completeness` (float, 可选，默认 0.99)

## 步骤

1. dataset_get(dataset_version_id) 确认版本存在与 episode_indices
2. dataset_list_episodes(dataset_version_id) 取 episode 列表
3. processing_quality_submit(...) 提交 CPU 质检，processing_quality_get(...) 轮询到终态
4. 按 timestamp_completeness / static_frames / joint_limit 等规则统计异常 episode
5. processing_clean_submit(ops=[...]) 生成清洗计划（apply=false 先预览）

## 输出

- `ready`: bool，是否可直接训练
- `quality`: 各规则命中数与关键指标
- `checked_episodes`: int
- `bad_episodes`: 异常 episode 列表
- `clean_plan`: 拟剔除 episode 与保留数
- `next_action`: training_submit 或先做清洗

## 使用的 MCP 工具

`dataset_get`、`dataset_list_episodes`、`processing_quality_submit`、`processing_quality_get`、`processing_clean_submit`、`processing_clean_get`
