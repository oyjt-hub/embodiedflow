# EmbodiedFlow 交接说明

最后更新：2026-11-02

## 一、系统入口

| 入口 | 地址 |
| --- | --- |
| 工作台页面 | `http://<SERVER_HOST>:18080/workbench/` |
| API 文档 | `http://<SERVER_HOST>:18080/docs` |
| MCP 服务 | `http://<SERVER_HOST>:18081/mcp`（Streamable HTTP） |
| Agent 页面 | `http://<SERVER_HOST>:18082/` |

进程：API（18080）、Worker、MCP（18081）、Agent（18082）。
启动脚本在 `ops/`，全部服务重启用 `bash ops/restart_all.sh`，状态用 `bash ops/status.sh`。

## 二、工作包进度

按 `outputs/EmbodiedFlow-统一研发工作包与排期.csv`：

- 已完成 18 项
- 部分完成 10 项
- 规划待实施 15 项

明细与"还差什么"见 `outputs/EmbodiedFlow-剩余工作与前置条件.md`。

## 三、本会话新增的后端能力

| 接口 | 作用 |
| --- | --- |
| `GET /workbench/api/capabilities` | 五模块能力聚合（schema `embodiedflow.capabilities/1`） |
| `GET /workbench/api/agent/tools` | MCP 工具清单与指纹（43 个 `@mcp.tool`） |
| `GET /workbench/api/agent/confirmations` | 确认记录（只读，与 MCP 共用 platform.db） |
| `GET /workbench/api/agent/workflows` | Workflow 历史 |
| `GET /workbench/api/agent/workflows/{id}` | Workflow 多步编排（步骤 + 对应 Job 状态） |
| `GET /workbench/api/links/job/{id}` | 任务 → 模型/制品/评测 |
| `GET /workbench/api/links/model/{id}` | 模型 → 任务/评测/部署 |
| `GET /workbench/api/vision/profiles` | 观测画像（image_only 0 维 / image_mask_geometry 9 维） |
| `POST /workbench/api/vision/profile/vector` | sidecar → 固定顺序向量 |
| `GET /workbench/api/vision/assets` | 视觉资产目录（体积/下载时间/摘要） |
| `GET /workbench/api/vision/hardware` | 本机硬件画像（nvidia-smi，不加载 torch） |
| `GET /workbench/api/vision/benchmarks/compare` | 多模型视觉 benchmark 对比 |
| `GET /workbench/api/vision/ground-batches/{id}/quality` | 批量标注质量门槛（置信度/面积像素²） |
| `GET /workbench/api/vision/tracks/{id}/pts.csv` | 逐帧 PTS 轨迹导出（含遮挡状态） |
| `POST /workbench/api/processing/interpolate` | 时间轴插值重采样（可物化为派生版本） |
| `GET/POST /workbench/api/admin/agent-config` | Agent Provider 配置（密钥只回掩码） |
| `POST/GET /workbench/api/research/sources` | 研究来源登记与检索 |
| `GET /workbench/api/research/search` | FTS5 全文检索 |
| `GET /workbench/api/research/report` | 对比矩阵报告（md/html/pdf） |
| `GET /workbench/api/running` | 运行中作业（总览页用） |

## 四、页面已有区块

导航「研究来源」；日志页「三入口互查」「Agent 确认与 Workflow」；
评测页「视觉模型对比」；视觉页「批量标注质量门槛」「观测画像」；
算力页「系统信息」。

`/admin/agent-config` **有意不接前端**：工作台是内网无鉴权开放（登录仅校验用户名存在），
把改密钥界面放上去等于向局域网暴露凭证。

## 五、验证方式

```bash
# 静态校验（元素 id / fetch 路径 / 函数引用）
.venv/bin/python scripts/lint_page.py

# JS 语法校验（node v12 不支持 ?. 与 ??，需先降级）
python - <<'PY'
import re
from pathlib import Path
h = Path("apps/web/workbench.html").read_text(encoding="utf-8")
m = max(re.findall(r"<script[^>]*>(.*?)</script>", h, re.S), key=len)
Path("/tmp/chk.js").write_text(
    m.replace("?.[", "[").replace("?.", ".").replace(" ?? ", " || "), encoding="utf-8")
PY
node --check /tmp/chk.js

# 接口冒烟
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:18080/workbench/api/capabilities
```

**两者缺一不可**：`lint_page.py` 查引用，`node --check` 查语法。
本会话曾因只跑前者，导致页面 JS 失效两轮未被发现。

## 六、已知限制

- 登录仅校验用户名存在于 `/etc/passwd`，**不是身份认证**；后端接口未做鉴权。
- 服务无 systemd 自启（无 sudo），重启机器需手动执行 `ops/restart_all.sh`。
- Vision 指标只有 bbox 真值（`AnnotationRun(kind=bbox_gt)`），**没有 mask 标注类型**，
  Mask/视频 IoU 无从计算；不建议用 bbox 冒充 mask。
- 插值物化只处理状态与动作，未重编码视频，不可用于视觉策略（`info.json` 已注明）。

## 七、需要决策的三件事

1. **实跑闭环**（关闭 A10 + A14）：需要一个已保存 Pipeline 草稿 id，并同意真实跑一轮训练链。
2. **定义 Skills**（推进 A07 + A15）：仓库、`embodiedflow_data/agent`、`dsh_home` 内均无
   skills 目录或 `SKILL.md`，计划所述的 Skills 在服务器上没有对应产物。
3. **视觉指标边界**（V08 + V09）：接受只评 bbox，或投入人工标注。
