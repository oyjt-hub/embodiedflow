# P1 执行决议（基于 2026-10-31 环境取证）

## 取证结果

| 解释器 | 可用模块 |
|---|---|
| `embodiedflow/.venv` | **none**（不含 mujoco） |
| `openpi_v2/.venv` | `mujoco, gymnasium, torch` |
| `RLinf/.venv` | `mujoco, gymnasium, torch` |

| 系统层 | 结果 |
|---|---|
| Isaac Sim 镜像 | **已存在**：`robo_orchard_sim:...isaacsim5.1-isaac_lab-v2.3.2-v0.1`（Docker） |
| 板卡工具链（RDK/BPU） | **未安装** |

## 决议

| 包 | 决议 | 依据 |
|---|---|---|
| **P1.1 MuJoCo Benchmark** | **可做**。平台 venv 补装 `mujoco`，或通过 `spec.command` 调用 `openpi_v2/.venv` 的 mujoco | 两个解释器已有 mujoco + gymnasium |
| **P1.1b Isaac Sim** | **暂缓**。镜像在，但需 GPU 图形栈与授权窗口；先在 MuJoCo 跑通最小闭环，Isaac 作为第二个后端接入 | 镜像存在 ≠ 可无头渲染；需单独验机 |
| **P1.2 真机 Sim2Real** | **阻塞**。需工控机（<EDGE_HOST>）+ 现场授权 + 安全复位流程 | 非本机可判定 |
| **P1.3 端侧量化** | **标 Planned** | 本机无 RDK/BPU 工具链 |
| **P1.4 只读助手** | **已完成** | `/api/assistant/ask`，写意图拒绝 |

## P1.1 最小闭环设计（下一轮实施）

```
Benchmark → BenchmarkVersion（固定本体/场景/初态/成功判据/种子/引擎版本）
        → Evaluation（跑 N 个 Episode，逐次记录终态 + 录像路径）
        → 报告（成功率带分母；失败 Episode 可回放到时间段）
```

第一步只做：`Pendulum` 或 `CartPole` 类的确定性环境 + 固定种子集合，
产出 `BenchmarkVersion` 与 `Evaluation` 记录，指标口径与 P0.5 评测器复用。

## 已确认的能力边界

- 平台所有写操作（训练/删除/部署）不经助手开放
- 仿真后端选择由 `BenchmarkVersion.engine` 字段决定（`mujoco` / `isaac`），
  未安装的引擎在能力表标 `Planned`，接口测试用 Mock 单列
