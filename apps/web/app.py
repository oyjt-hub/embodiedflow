from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API = os.environ.get("EF_API", "http://127.0.0.1:18080")

st.set_page_config(page_title="EmbodiedFlow", page_icon="🌊", layout="wide")

PAGES = [
    "Overview",
    "Jobs",
    "Datasets",
    "Annotations",
    "Benchmarks",
    "Compute Resources",
    "Models",
    "Model Families",
    "Devices",
    "Evaluation",
    "Planned",
]


def api(path: str, method: str = "GET", **kwargs):
    url = f"{API}{path}"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def err(obj) -> bool:
    if isinstance(obj, dict) and "__error__" in obj:
        st.error(f"API 不可用：{obj['__error__']}  （{API}）")
        return True
    return False


st.sidebar.title("🌊 EmbodiedFlow")
st.sidebar.caption("单机研发平台 · P0 MVP")
page = st.sidebar.radio("导航", PAGES, label_visibility="collapsed")
st.sidebar.divider()
health = api("/api/health")
if not err(health):
    st.sidebar.success(f"API 在线 · schema v{health.get('schema_version')}")
    st.sidebar.caption(f"host `{health.get('host')}` · py {health.get('python')}")
else:
    st.sidebar.error("API 离线")

if page == "Overview":
    st.title("Overview")
    s = api("/api/summary")
    if not err(s):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Datasets", s.get("datasets", 0))
        c2.metric("Jobs", s.get("jobs", 0), f"{s.get('jobs_running', 0)} running")
        c3.metric("Models", s.get("models", 0))
        c4.metric("Devices", s.get("devices", 0))
        c5, c6, c7 = st.columns(3)
        c5.metric("质检报告", s.get("quality_reports", 0))
        c6.metric("评测记录", s.get("evaluations", 0))
        rate = s.get("devices_online_rate")
        c7.metric("设备在线率", "N/A" if rate is None else f"{rate:.0%}")
        if s.get("jobs_failed"):
            st.warning(f"失败任务：{s['jobs_failed']}")
    gpus = api("/api/resources/gpus")
    if not err(gpus) and gpus:
        st.subheader("GPU")
        df = pd.DataFrame(gpus)
        df["mem"] = df.apply(lambda r: f"{r.memory_used_mb}/{r.memory_total_mb} MiB", axis=1)
        st.dataframe(df[["index", "name", "mem", "utilization", "state"]], use_container_width=True, hide_index=True)
        fig = go.Figure(go.Bar(x=df["index"].astype(str), y=df["utilization"], marker_color="#4C8DFF"))
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="util %", xaxis_title="GPU")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Jobs":
    st.title("Jobs")
    with st.form("submit"):
        c1, c2, c3 = st.columns([2, 1, 1])
        kind = c1.selectbox("任务类型", ["smoke", "gpu_probe", "train", "pipeline", "eval", "sim"])
        gpus = c2.number_input("所需 GPU", 0, 8, 0)
        name = c3.text_input("名称", value="first-task")
        cmd = st.text_input("自定义命令（可选，非 smoke/gpu_probe 时必填）", value="")
        submitted = st.form_submit_button("提交任务", type="primary")
    if submitted:
        spec = {"command": cmd} if cmd else {}
        res = api("/api/jobs", "POST", json={"kind": kind, "name": name, "spec": spec, "requested_gpus": int(gpus), "created_by": "web"})
        if not err(res):
            st.success(f"已提交 job #{res['id']}（{res['status']}）")

    jobs = api("/api/jobs?limit=30")
    if not err(jobs) and jobs:
        st.dataframe(pd.DataFrame(jobs)[["id", "kind", "name", "status", "requested_gpus", "created_by", "created_at"]], use_container_width=True, hide_index=True)
        jid = st.selectbox("查看 / 取消", [j["id"] for j in jobs])
        c1, c2 = st.columns(2)
        if c1.button("刷新日志"):
            st.session_state["show_logs"] = jid
        if c2.button("取消任务", type="secondary"):
            r = api(f"/api/jobs/{jid}/cancel", "POST")
            st.info("已请求取消" if not err(r) else "")
        if st.session_state.get("show_logs"):
            logs = api(f"/api/jobs/{jid}/logs?tail=200")
            if not err(logs):
                st.code("\n".join(logs.get("lines") or ["(暂无日志)"]) or "(空)", language="text")
                st.caption(logs.get("path"))
    elif not err(jobs):
        st.info("还没有任务。用上面的表单提交第一个。")

elif page == "Datasets":
    st.title("Datasets")
    rows = api("/api/datasets")
    if not err(rows):
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("尚无数据集。P0.2 将接入 LeRobot 上传与质检；接口 `/api/datasets` 已就位。")
    st.caption("计划对应：P0.2 数据中心与处理流水线")

elif page == "Compute Resources":
    st.title("Compute Resources")
    gpus = api("/api/resources/gpus")
    if not err(gpus):
        st.dataframe(pd.DataFrame(gpus), use_container_width=True, hide_index=True)
    running = api("/api/resources/jobs")
    if not err(running):
        st.subheader("运行中的任务")
        st.dataframe(pd.DataFrame(running) if running else pd.DataFrame([{"info": "无"}]), use_container_width=True, hide_index=True)

elif page == "Models":
    st.title("Models")
    rows = api("/api/models")
    if not err(rows):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True) if rows else st.info("尚无登记模型（P0.4）")

elif page == "Devices":
    st.title("Devices")
    rows = api("/api/devices")
    if not err(rows):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True) if rows else st.info("尚无设备台账（P0.5）")

elif page == "Evaluation":
    st.title("Evaluation")
    rows = api("/api/evaluations")
    if not err(rows):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True) if rows else st.info("尚无评测记录（P0.5）")

elif page == "Annotations":
    st.title("Annotations · FLORA 自动标注")
    st.caption("NVIDIA FLORA")
    versions = api("/api/annotation/versions")
    opts = {}
    if not err(versions) and versions:
        opts = {f"{v['label']}  [{v['status']} · {v['episodes']} ep]": v["id"] for v in versions}
    if not opts:
        st.info("还没有数据版本，先到 Datasets 页登记数据集。")
        st.stop()
    with st.form("ann_run"):
        c1, c2 = st.columns([1, 2])
        chosen = c1.selectbox("数据集版本", list(opts.keys()))
        vid = opts[chosen]
        eps = c2.text_input("Episode 索引（逗号分隔）", value="0,1,2,3,4,5,6,7")
        camera = st.text_input("相机", value="front")
        profile = st.selectbox("动作配置", ["yam", "dex", "tianji", "human"])
        if st.form_submit_button("运行自动标注", type="primary"):
            try:
                idx = [int(x) for x in eps.replace("，", ",").split(",") if x.strip()]
                r = api("/api/annotation/run", "POST",
                        json={"dataset_version_id": int(vid), "episode_indices": idx,
                              "camera": camera, "profile": profile})
                if not err(r):
                    st.session_state["flora_job_id"] = r["job_id"]
                    st.success(f"Job #{r['job_id']} · {r['status']}")
            except ValueError:
                st.error("Episode 索引应为整数")
    history = api(f"/workbench/api/vision/flora/jobs?dataset_version_id={vid}")
    if not err(history) and history.get("jobs"):
        job_map = {f"#{row['job_id']} · {row['status']}": row["job_id"] for row in history["jobs"]}
        selected_job = st.selectbox("FLORA 任务", list(job_map))
        job_id = job_map[selected_job]
        st.button("刷新任务")
        job = api(f"/api/jobs/{job_id}")
        if not err(job):
            st.write(f"Job #{job_id} · {job['status']}")
            if job["status"] == "completed":
                result = api(f"/workbench/api/vision/flora/jobs/{job_id}/result")
                if not err(result):
                    for item in result.get("annotations", []):
                        st.write(f"Episode {item['episode']}")
                        st.dataframe(item.get("annotation", {}).get("segments", []), use_container_width=True)
                st.markdown(f"[审核与导出](http://172.19.5.252:18080/workbench/#sec-vision)")
            elif job.get("error"):
                st.error(job["error"])
    st.subheader("历史 AnnotationRun 统计")
    stats = api(f"/api/annotation/stats/{vid}")
    if not err(stats):
        st.json(stats)
    plan = api(f"/api/annotation/sampling/{vid}")
    if not err(plan):
        st.caption(f"分层抽检：共 {plan['total']} 条，选中 {plan['review_selected']} 条，"
                   f"覆盖率 {plan['coverage']['value']}（{plan['coverage']['numerator']}/{plan['coverage']['denominator']}）")

elif page == "Benchmarks":
    st.title("Benchmarks · 仿真自动评测")
    with st.form("bm"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("基准名", value="cartpole-balance")
        ver = c2.text_input("版本", value="v1")
        seeds = c3.text_input("种子", value="0,1,2,3,4,5,6,7")
        if st.form_submit_button("运行基准", type="primary"):
            sd = [int(x) for x in seeds.replace("，", ",").split(",") if x.strip()]
            r = api("/api/benchmarks/run", "POST",
                    json={"name": name, "version": ver, "seeds": sd})
            if not err(r):
                s_ = r["success"]
                st.success(f"成功 {s_['numerator']}/{s_['denominator']} = {s_['value']:.2f} · "
                           f"benchmark_version #{r['benchmark_version_id']}")
                st.dataframe(pd.DataFrame(r["episodes"]), use_container_width=True, hide_index=True)
    st.caption("判据冻结在 BenchmarkVersion.spec 中；同一 seed 集合必须得到同一结果")

elif page == "Model Families":
    st.title("Model Families")
    fams = api("/api/families")
    if not err(fams):
        for f in fams["families"]:
            badge = {"available": "✅", "eval_only": "🟡", "missing": "❌"}.get(f["status"], "•")
            with st.expander(f"{badge} {f['label']}  ·  {f['kind']}  ·  {f['status']}"):
                st.write(f["notes"])
                if f.get("weights_found"):
                    st.write("**已找到权重**", f["weights_found"])
                if f.get("weights_missing"):
                    st.warning(f"缺失权重：{f['weights_missing']}")
                if f.get("train_entry"):
                    st.code(f["train_entry"], language="bash")

elif page == "Planned":
    st.title("Planned")
    st.info("Simulation · P1.1（MuJoCo 先跑通，再接 Isaac Sim）")
    st.info("Deployment · P1.2（需要真机与现场授权）")
    st.info("Agent Assistant · P1.4（只读查询，依赖 P0.5 的权限）")
