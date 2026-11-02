# 视觉标注与审核

`vision-annotate`

检查视觉能力、取回掩码与 FLORA 审核结果，判定标注质量是否可用。

## 输入

- `project_id` (int, 必填)
- `dataset_version_id` (int, 必填)
- `episode` (int, 可选，默认 0)
- `camera` (str, 可选，默认 front)

## 步骤

1. vision_get_capabilities() 确认可用模型与操作
2. vision_list_masks(...) 取该 episode/camera 的掩码清单
3. vision_get_mask(...) 取单帧掩码与 bbox
4. vision_flora_get_result(job_id) / vision_flora_get_review(job_id) 取审核与修订

## 输出

- `masks`: 掩码清单与覆盖帧
- `review_status`: 审核状态与修订次数
- `quality_flags`: 遮挡/丢失/低置信等标记
- `next_action`: 可进入派生或需人工复核

## 使用的 MCP 工具

`vision_get_capabilities`、`vision_list_masks`、`vision_get_mask`、`vision_flora_get_result`、`vision_flora_get_review`
