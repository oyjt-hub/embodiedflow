# P1.1 交付状态（自有数据与 π0.5）

## 一、已完成

| 能力 | 证据 |
|---|---|
| LeRobot 上传/登记入口 | `/api/datasets/upload`（tar.gz + sha256 + 中断安全）、`/api/datasets/register` |
| 工作台选数据 | `/workbench/` ① 区下拉列出全部数据版本，含 14 维曲线预览 |
| BC 训练（CPU） | 21 训 / 3 留，val MAE 0.0281，基线 0.3524 |
| π0.5 / OpenPI 训练 | job #15：`debug_pi05`，10 步，exit 0，checkpoint 3.0 GB @ `~/workspace/checkpoints/debug_pi05/wb-pi05-r2/9` |
| GPU 调度正确性 | 租约 UUID 直传 `CUDA_VISIBLE_DEVICES`，实测 `GPU-fbab66c0-…` |
| 训练配置清单 | `/workbench/api/openpi/configs` 返回 7 个真实配置名 |

## 二、未完成：π0.5 推理（阻塞在数据）

实测 `serve_policy.py`：

```
1) 参数顺序：--port 必须在子命令之前
2) 启动命令：
   python scripts/serve_policy.py --port 9000 policy:checkpoint \
     --policy.config <name> --policy.dir <ckpt>
3) checkpoint 恢复：成功（6.4 GiB，5.6 秒）
4) 创建策略：失败
   ValueError: Asset id is required to load norm stats.
```

**根因**：`debug_pi05` 是调试配置，`data.assets.asset_id` 为空、没有归一化统计。
推理**必须**有 norm stats，而它来自配置里写死的 repo_id（如 `physical-intelligence/libero`）。

**数据现状**：

```
~/.cache/huggingface/hub/datasets--physical-intelligence--libero   仅 12K（只有 refs/，无 snapshots）
```

即：本机没有可用的 LIBERO 数据。

## 三、要闭环需要什么

| 路径 | 代价 | 说明 |
|---|---|---|
| A. 下载 LIBERO 数据集 | 体积未知（可能 10–50 GB）+ 算 norm stats + 再训一轮 | 唯一能让 π0.5 推理真正跑通的路径 |
| B. π0.5 只做训练 | 零成本 | 平台如实标注"训练可用 / 推理待数据" |

**当前选择：未定，等用户决策。**

## 四、结论

P1.1 的**训练侧完整可用**；**推理侧受限于缺失的 LIBERO/ 匹配数据集**，
这是环境前提问题，不是代码缺陷。
