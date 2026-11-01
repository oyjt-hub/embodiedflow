# 架构说明（P0.1）

## 进程模型

```
browser ──HTTP──► FastAPI (uvicorn) ──SQL──► SQLite(WAL)
                     │                        ▲
                     │ enqueue job            │ claim/lease/heartbeat
                     ▼                        │
                  jobs 表 ──────────────► Worker 进程
                                              │ subprocess(start_new_session=True)
                                              ▼
                                    runs/job_<id>/attempt_<n>/worker.log
```

- API 与 Worker **分进程**：API 只做鉴权、校验、入队；重活由 Worker 领取。
- 任务态机：`queued → running → completed | failed | cancelled`。
- 每次执行生成独立 `JobAttempt`，编号从 1 递增，失败重试不覆盖历史。

## 为什么用统一 `Job` 而不是每类任务一张表

计划里的 `TrainingJob → JobAttempt` 是一类任务的说法。实现上把
`pipeline / train / eval / sim` 统一为 `jobs.kind`，`spec` 存各自的配置
（`JobSpec`），`JobAttempt` 通用。好处是队列、租约、取消、日志只有一份实现；
代价是 `spec` 需要按 kind 做校验（`worker.build_command` 里对未实现的 kind
显式失败，而不是假装成功）。

## GPU 调度

- `nvidia-smi` 采样写入 `gpu_resources`（uuid/index/显存/利用率）。
- 判定：**有活跃租约** 或 **显存占用 > 1 GiB** → 视为不可调度（外部占用）。
- 领取任务时按 `requested_gpus` 从空闲卡里取，写 `gpu_leases`，子进程只暴露
  `CUDA_VISIBLE_DEVICES`；attempt 结束（含取消/崩溃）即释放。
- 不以"还有显存"直接抢卡，这是计划里点明的约束。

## 一致性与恢复

- SQLite `journal_mode=WAL` + `busy_timeout=30s`，多进程读写安全。
- 领取用乐观更新（`UPDATE ... WHERE status='queued'`），`rowcount=1` 才算抢到。
- 取消：Worker 每秒查一次任务状态，命中即 `SIGTERM` 整个进程组，2 秒后 `SIGKILL`。
- Worker 崩溃留下的 `running` 任务：`job_attempts.heartbeat_at` 可判活，恢复逻辑列在 P0.3 收尾项。
