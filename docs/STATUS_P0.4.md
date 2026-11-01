# P0.4 交付状态（OpenPI 适配 / 追踪 / 模型仓库）

## 一、真实数据验收（P0.2 遗留项，已完成）

| 项 | 实测 |
|---|---|
| 数据集 | `~/.cache/huggingface/lerobot/instructmove_merged` |
| 规模 | **600 Episode / 3.01 GB / 30 fps** |
| 登记摘要 | `sha256=27fcb1d3…`（meta/info.json） |
| 质检（前 20 Episode） | **0 findings**，12 类规则全部可运行 |
| 指标 | timestamp_completeness 1.0=19400/19400 · episode_completeness 1.0=20/20 · joint_violation 0/1940 · image_missing 0/5820 · action_discontinuity 0/1920 · static_frame_ratio 0.0667=128/1920 |

## 二、MLflow 追踪

- 服务：`ops/start_mlflow.sh` → `http://127.0.0.1:15000`，backend `sqlite:///$EF_DATA/metadata/mlflow.db`，artifact root `$EF_DATA/artifacts/mlflow`
- `services/tracking.py`：`log_run(experiment, run_name, params, metrics, artifacts, tags)`
- 质检接口 `/datasets/{id}/validate` 支持 `log_mlflow=true`，自动写 params（数据集/版本/规则版本/Episode 数）、6 项指标、报告 JSON 制品
- 实测产出 run：`b3bca04d5dfc41dd8c478ddd0b6077a3`（experiment `data-quality`），可在 MLflow UI 查回

## 三、模型仓库

- `POST /models`、`POST /models/{id}/versions`、`GET /models`、`GET /models/{id}/versions/{v}/lineage`
- 目录摘要 `tree_digest`：小文件取内容哈希、大文件用 size（避免扫几十 GB）；**同摘要自动去重**
- 实测登记真实权重：`checkpoints/4items_pi05base_rtc_v2_all_lora`
  - **279 个文件 / 52.5 GB**
  - `sha256=643f1b21b171fd1e41f337c613700cbff9396ef489f1db7bad0d1c36c65f564f`
- 血缘：`ModelVersion → JobAttempt → Job → DatasetVersion`（本次无 attempt，为外部权重导入，lineage 返回 null 属预期）

## 四、OpenPI 适配器（Dry Run 就绪）

`adapters/training/openpi.py`：把平台 spec 翻译成 openpi tyro CLI，并做物料校验

```python
OpenPIAdapter(repo, python).prepare(spec) → OpenPIProbe(ok, command, missing[], warnings[])
```

校验项：训练入口 `scripts/train.py`、配置名是否在 `src/openpi/training/config.py` 注册、
`dataset_path` / `weights_path` 是否存在。**默认不下发训练**，只有显式 execute 才起子进程。

计划允许的 Dry Run 语义：缺资源/缺权重时保存配置与诊断，不伪造训练结果。

## 五、未完成（下次继续）

| 项 | 状态 |
|---|---|
| π0.5 真实 1 步更新 | **未跑**。需要确认 openpi venv + 具体 config + 数据可用性；本轮只做到物料校验骨架 |
| 训练作业接入 MLflow | 未接。`JobAttempt` 与 `run_id` 的关联字段已在模型中，缺 worker 侧写入 |
| 模型版本 ↔ attempt 自动关联 | 手工传 `attempt_id`；应由训练作业结束时自动登记 |

## 六、测试

```
PYTHONPATH=. .venv/bin/python -m pytest tests -q
28 passed
```
