# P0.3 交付状态（异步任务与算力）

计划放行条件：**"两个任务竞争一张卡时串行执行，取消或失败后下一任务可接手"** —— 已通过。

## 一、资源感知抢占（`services/jobs.py` + `services/resources.py`）

抢占前先规划，规划不到就**不抢、留队**（不消耗 attempt）：

```python
def claim_next(session, worker_id, allocator=None):
    for job in candidates:                      # 按 priority, id 排序
        allocation = allocator(session, job)    # → list[uuid] | None
        if allocator and allocation is None:
            continue                            # 资源不够，跳过，留给下一轮
        ...                                     # 乐观 UPDATE 抢状态 + 建 attempt + 落租约
```

`plan_allocation(session, job)` 支持两个 spec 字段：

| 字段 | 作用 |
|---|---|
| `gpu_memory_mb` | 每卡需要的**剩余**显存 |
| `gpu_indices` | 只允许这些索引的卡（竞争压测、独占隔离用） |

## 二、失败重试与 Worker 失联恢复（`services/recovery.py`）

`recover(stale_after_s=30, current_worker=...)` 扫描心跳过期的 `running` attempt：

1. **自己是 owner** → 跳过（绝不自杀）
2. **同一进程组** → 跳过（安全护栏，见下）
3. **进程已消失** → 释放租约 → 有预算则回队，否则判 failed
4. **进程还活着但属于已失联的 Worker** → `SIGTERM` 进程组回收孤儿，再走 3

### 踩到的坑（已修）

第一版没有"同进程组"护栏，测试里 `recover()` 没传 `current_worker`，
把一个存活 pid 当孤儿 `killpg`，**把 pytest 自己打死了**（SSH 返回 rc -1）。
现在：`os.getpgid(pid) == os.getpgid(0)` 一律跳过 —— 因为作业是
`start_new_session=True` 起的独立进程组，"同组"只可能是 Worker 自身或父进程。

## 三、实测（真实服务，非模拟）

```
[A] 两任务争同一张卡（都 pin 到 GPU7）
    j1=2  j2=3
    3s:  j1=running    j2=queued        ← 串行核心断言
    j1 完成，j2 等待约 7s 后才开始
    j2 完成                            → PASS

[B] 取消运行中的任务
    j3 已 running → cancel → cancelled
    取消后立刻提交 j4 → completed       → 卡位已释放 PASS

[C] Worker 失联恢复
    j5 运行中 → 杀掉 Worker → j5 状态仍是 running（无人更新心跳）
    重启 Worker（启动时执行恢复）→ 35s 后重新执行 → completed
    数据库核对：job 6 正好两次 attempt
        attempt_no=1  failed      （孤儿被回收）
        attempt_no=2  completed
                                        → PASS
```

## 四、测试

```
PYTHONPATH=. .venv/bin/python -m pytest tests -q
28 passed
```

新增 `tests/test_p03_scheduling.py`：
- 同卡两个任务：第二个必须返回 `None` 留队（不消耗 attempt）
- 资源不足的任务不能堵死队列（跳过它，领取后面的 CPU 任务）
- `gpu_indices` 指向不存在的卡时规划失败
- 恢复：进程已死 → 回队 + 释放租约；进程存活 → 跳过；预算耗尽 → failed

## 五、待办

| 项 | 说明 |
|---|---|
| `attempt_no` vs `attempt_id` 易混 | 日志里写的是 attempt_id，界面应两个都显示 |
| 孤儿回收的 pid 复用风险 | 已用"进程组归属"降低，但 pid 被复用仍可能误判；生产建议加启动时间戳比对 |
| 多 Worker 并发 | 当前单 Worker；多 Worker 时 `claim_next` 的乐观更新已支持，但需要压测 |
