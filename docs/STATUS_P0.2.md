# P0.2 交付状态（数据中心与质检流水线）

本次范围：LeRobot 读取器、12 类质检规则、多频对齐、可复现切分、归一化只拟合训练集，
以及"够显存即可调度"的 GPU 策略（按你 2026-10-31 的指示调整）。

## 一、数据集读取（`adapters/datasets/lerobot.py`）

- 解析 `meta/info.json`、`meta/tasks.jsonl`、`meta/episodes.jsonl`
- 读 `data/chunk-*/episode_*.parquet`（PyArrow），数值流自动区分 continuous / discrete
- 视频流用 `imageio-ffmpeg` **只读容器元数据**拿到真实帧数，解码按 `decode_frames` 上限（用于损坏检测），并记录 `decode_truncated` 避免把"没解完"误判成"缺帧"
- 支持 `episodes.jsonl` 里的 `camera_time_offset`（相机时钟偏移登记）
- **只读**：适配器不修改原始目录

## 二、12 类质检规则（`services/quality.py`，rules_version 1.0）

| # | 规则 | 判定 | 实测命中 |
|---|---|---|---|
| 1 | `missing_timestamp` | 时间戳为空 | ep1, t=[0.300, 0.367] |
| 2 | `time_reversal` | Δt < 0 | ep2, Δt=-0.0333s |
| 3 | `irregular_gap` | Δt > max(0.25s, 5×中位数) | ep3, 最大 5.033s @ t=1.63→6.67 |
| 4 | `missing_frames` | 流长度与主时间轴偏差 >5% | ep4（视频 90/100 帧） |
| 5 | `video_corrupt` | 全零帧占比 >50% 或无法解码 | ep5 |
| 6 | `camera_action_misalign` | 相机与动作时钟偏移 >0.05s | ep6（+0.5s） |
| 7 | `joint_limit` | \|q\| > 3.2 rad | ep7（5.00 rad） |
| 8 | `action_jump` | 单步 \|Δaction\| > 2.0 | ep8（Δ=50.0） |
| 9 | `gripper_anomaly` | 夹爪越界/非有限 | ep9（5.00） |
| 10 | `episode_incomplete` | 实际帧数与声明长度偏差 >5% | ep10（30/100） |
| 11 | `static_frames` | 相邻帧几乎无变化占比 >90% | ep11 |
| 12 | `trajectory_break` | 单步 \|Δstate\| > 10 | ep12（Δ=500） |

每条 finding 都带 **episode / stream / t_start / t_end / detail / count**，即"定位到 Episode 和时间点"。

**指标（均带分子/分母）**：

```
timestamp_completeness    0.999187 = 3687/3690
episode_completeness      0.923077 = 12/13
joint_violation_rate      0.041463 = 51/1230
image_missing_rate        0.000000 = 0/1220
action_discontinuity_rate 0.000822 = 1/1217
static_frame_ratio        0.081348 = 99/1217
```

## 三、多频对齐与切分（`services/align.py`）

- 100 Hz 动作/状态对齐到相机时钟（30 Hz）：连续量线性插值、离散量（gripper）保持，
  记录 `max_residual_s`、`tolerance_s`、`invalid_frames`、`aligned_rows`
- 实测：`observation.state` / `action` linear @30 Hz，残差 0.0；`observation.gripper` hold
- **原始数据不被修改**（测试断言对齐前后数组完全一致）
- 切分按 Episode，`seed=7` 下 10/1/2，重跑结果完全一致（测试断言）
- 归一化只拟合训练集：`count=930, fitted_on=train_only`

## 四、GPU 调度策略变更（按用户指示）

原计划："不以还有显存直接抢卡"。现改为：

```
只要 剩余显存 >= max(需求 + safety_margin(1024 MiB), min_free(4096 MiB)) 且未被平台租用  → 可调度
```

- `EF_GPU_ALLOW_PARTIAL=1`（默认开）、`EF_GPU_MIN_FREE_MB=4096`、`EF_GPU_SAFETY_MARGIN_MB=1024`
- GPU 状态新增 `partial`（有外部占用但够用）与 `busy`（不够用）
- 任务可声明 `gpu_memory_mb`，Worker 按剩余显存挑选并只暴露 `CUDA_VISIBLE_DEVICES`

**实测**：8 张卡全部 `partial`（剩余 24–42 GiB），提交需 2 GiB 的 1 卡任务 →
落到 GPU0，`CUDA_VISIBLE_DEVICES=0`，torch 实际分配 **1.61 GB**，任务 `completed`。

## 五、测试

```
PYTHONPATH=. .venv/bin/python -m pytest tests -q
22 passed
```
覆盖：12 类规则各自命中正确 Episode、干净 Episode 无 error、指标分母非零、
切分可复现、对齐不修改原始数据、任务生命周期、取消、未实现 kind 显式失败、API 冒烟。

## 六、未做/待办

| 项 | 说明 |
|---|---|
| 流式上传会话 | 本期用"登记目录"（`/datasets/register`）；分片上传 + 中断续传留到下一轮 |
| 引用保护删除 | 已实现（jobs/model_versions 引用则 409），但 model_versions 计数目前是全局而非按版本，需收窄 |
| 多本体 Schema | 目前按 `robot_spec` 字段占位，关节顺序/单位/标定表尚未落到实体表 |
| 真实数据验收 | 目前用 13 Episode 的合成夹具；P0.2 放行还需一份真实 LeRobot 数据 |
