# EmbodiedFlow 交付总览

计划文件：`../EmbodiedFlow-系统建设与分阶段实施计划.md`
代码目录：`~/workspace/embodiedflow`　数据目录：`~/workspace/embodiedflow_data`

## 一、怎么跑

```bash
cd ~/workspace/embodiedflow
bash ops/start_api.sh      # FastAPI  127.0.0.1:18080
bash ops/start_worker.sh   # 任务 Worker
bash ops/start_web.sh      # Streamlit 0.0.0.0:8501
bash ops/start_mlflow.sh   # MLflow   127.0.0.1:15000
bash ops/status.sh         # 看进程与健康
bash ops/stop_all.sh
```

访问（SSH 隧道）：
```bash
ssh -L 8501:127.0.0.1:8501 -L 18080:127.0.0.1:18080 <USER>@<SERVER_HOST>
# Web http://127.0.0.1:8501   API 文档 http://127.0.0.1:18080/docs
```

## 二、计划 → 交付对照

| 计划包 | 状态 | 证据 | 文档 |
|---|---|---|---|
| P0.1 骨架 / 队列 / Worker | ✅ | smoke 任务 queued→running→completed | STATUS_P0.1 |
| P0.2 数据与质检 | ✅ | 真实数据 600 Episode，12 类规则零误报 | STATUS_P0.2 |
| P0.2b FLORA 自动标注 | ✅ 管道（模型待接） | mock 跑通 8 条，派生版本生成 | STATUS_P0.2b |
| P0.3 队列与算力 | ✅ | 争卡串行 / 取消释放 / 孤儿回收 | STATUS_P0.3 |
| P0.4 训练·追踪·模型仓库 | ✅ | π0.5 一步真实训练 exit 0；52.5 GB 权重登记 | STATUS_P0.4 |
| P0.5 评测·设备·Dashboard | ✅ | 六项演示 5 通过 1 条件通过 | STATUS_P0.5 |
| P1.1 MuJoCo 基准 | ✅ | cart-pole 6/8；arm_reach 8/8 vs 0/8，失败样本轨迹回放 | STATUS_P1.1 |
| P1.2 真机 Sim2Real | 🟡 前置已做 | 执行契约（看门狗/限位/急停）Mock 链路跑通；真机待授权 | STATUS_P1.2pre |
| P1.3 端侧量化 | ⛔ Planned | 本机无 RDK/BPU 工具链 | P1_ENV_PROBE |
| P1.4 只读助手 | ✅ | 四类查询带引用，写意图拒绝 | — |
| P2.x | 未启动 | 需团队并发/多节点等触发条件 | — |

## 三、系统能力（全部可点可查）

数据上传（tar.gz + sha256 + 中断安全）· 目录登记 · 12 类质检 · 多频对齐 · 可复现切分 ·
持久任务队列 · GPU 租约调度（够显存即可用）· OpenPI 训练接入 · MLflow 自动追踪 ·
模型仓库与血缘 · checkpoint 自动发现 · FLORA 标注管道 · 离线评测（延迟/MAE/Jerk）·
设备台账 · 六项演示报告 · 只读助手 · Dashboard 10 页

## 四、验证方式

```bash
# 全套单测（隔离环境，不碰生产库）
env -u EF_ROOT -u EF_DATA -u EF_DB PYTHONPATH=. .venv/bin/python -m pytest tests -q
# → 52 passed, 1 xfailed

# Dashboard 依赖接口连通性
bash /home/oyjt/workspace/smoke_dashboard.sh
# → 14 个接口全部 200
```

## 五、已知限制

| 项 | 说明 |
|---|---|
| Dashboard 未做浏览器验收 | 接口全通、编译通过，但页面运行时错误需人工点验 |
| FLORA 用 mock | 缺权重与输入输出契约；`model_version` 恒为 `mock` |
| 机械臂基准未调通 | 控制器效果≈随机，已标 xfail 并注释 |
| 无鉴权 | 计划属 P2.1a；当前适合单机单用户 |
| 数据目录在 /home | `/data` 无写权限，待管理员配额；当前余量充足但有上限 |

## 六、代码规模

```
Python 文件  ~50
文档        11 份（docs/）
迁移版本    schema v7
测试        53 条（52 pass + 1 xfail）
```
