# P1 环境取证（2026-10-31 21:45:23）

| 解释器 | 可用模块 |
|---|---|
| workspace/embodiedflow | none |
| workspace/openpi_v2 | mujoco,gymnasium,torch |
| RLinf | mujoco,gymnasium,torch |

## 系统层

| 项 | 结果 |
|---|---|
| Isaac 镜像 | 已存在：docker.1ms.run/horizonrobotics/robo_orchard_sim:ubuntu22.04-py3.11-cuda12.8-isaacsim5.1-isaac_lab-v2.3.2-v0.1 |
| 板卡工具链 | 未安装（RDK/BPU 相关均未发现） |

## 结论

- P1.1（MuJoCo）：以上表为准；若无 `mujoco`，需先 `uv pip install mujoco`。
- P1.2（真机）：需工控机 + 现场授权，非本机可判定。
- P1.3（端侧量化）：本机无板卡工具链，标 `Planned`。
