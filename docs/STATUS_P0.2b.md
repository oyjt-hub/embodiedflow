# P0.2b 交付状态（FLORA 自动标注）

## 一、定位更正

FLORA 是 **NVIDIA 的自动标注模型**，不是人工标注平台。
最初按人工队列（领取/提交/复核/κ）起草，已更正为「模型推理 + 分层抽检」。
计划文件中的章节同步改为 `P0.2b · FLORA 自动标注`。

## 二、已交付（管道可运行，模型待接）

| 组件 | 落点 | 说明 |
|---|---|---|
| 标注作业 | `AnnotationRun` | 数据版本 + `model=flora:<version>` + 规范版本 + 状态机 |
| 标注记录 | `Annotation` | episode + kind + 载荷 JSON + **confidence** + model_version + 审核状态 |
| 适配器 | `adapters/annotation/flora.py` | `MockFlora` 确定性假标签；`FloraAdapter` 为真实接入点 |
| 分层抽检 | `services/annotation.sampling_plan` | <0.7 全检、0.7–0.9 抽 20%、≥0.9 抽 5%，覆盖率带分母 |
| 派生版本 | `services/annotation.publish` | 生成 `status=derived` 新版本，**原始版本只读不改** |
| 统计 | `services/annotation.stats` | 接收率带分子/分母；auto/accepted/rejected 分列 |

接口：`POST /api/annotation/run` · `GET /sampling/{vid}` · `POST /annotations/{id}/review` ·
`GET /stats/{vid}` · `POST /publish`（迁移 **v7**）

## 三、实测（真实数据集版本 2 · instructmove_merged）

```
run      : flora:mock  completed  8 episodes → 8 annotations
样例     : ep0 conf=0.595 success=false task=stack
           ep1 conf=0.946 success=false task=pick
派生版本 : real-v1-flora  derived  episodes=8
原始版本 : real-v1        ready    episodes=600   ← 未改写
```

## 四、接入真实 FLORA 所需（阻塞项）

1. **权重或推理服务地址**（本地权重 / 容器镜像 / API endpoint）
2. **输入契约**：多视角视频 / 关键帧 / 是否需关节状态
3. **输出契约**：Episode 级标签 / 时间段 / 帧级框或掩码

在拿到这三项之前，`model_version` 恒为 `mock`，界面与数据库均如实标注，不冒充真实标注。

## 五、测试

```
PYTHONPATH=. .venv/bin/python -m pytest tests -q
44 passed
```
新增 3 条：Mock 标注可复现、抽检按置信度分层、发布不改原始版本。

## 六、待办

| 项 | 说明 |
|---|---|
| 与质检合并视图 | 同一 Episode 的"规则命中"与"模型标注"尚未在同一视图对照 |
| 标注冲突单列 | 模型判成功但规则判异常的情况未单独统计 |
| 真实权重接入 | 见第四节 |
