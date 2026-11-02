"""SLAM 轨迹对齐与精度评测（V14 计算层）。

提供：
- umeyama_alignment：估计相似变换 (s, R, t)，把估计轨迹对齐到真值
- ate：绝对轨迹误差（对齐后逐点平移误差的 RMSE）
- rpe：相对位姿误差（相邻帧相对平移/旋转误差的均值与 RMSE）
- selftest：用可解析计算的合成轨迹验证实现

不依赖 SLAM 算法本身；真实序列到位后直接喂轨迹即可。
"""

from __future__ import annotations

import math

Pts = "list[list[float]]"


def _centroid(P):
    n = len(P)
    return [sum(p[i] for p in P) / n for i in range(3)]


def umeyama_alignment(model: Pts, data: Pts) -> dict:
    """估计把 data 对齐到 model 的相似变换 x_model ≈ s*R*x_data + t。

    只做平移+旋转+均匀缩放（不含反射）。
    """
    n = len(model)
    if n == 0 or n != len(data):
        raise ValueError("两条轨迹点数必须相同且非空")
    mu_m = _centroid(model)
    mu_d = _centroid(data)
    # 协方差矩阵（无 numpy 依赖，手工 3x3）
    cov = [[0.0] * 3 for _ in range(3)]
    var_d = 0.0
    for pm, pd in zip(model, data):
        a = [pm[i] - mu_m[i] for i in range(3)]
        b = [pd[i] - mu_d[i] for i in range(3)]
        for i in range(3):
            for j in range(3):
                cov[i][j] += a[i] * b[j]
        var_d += sum(v * v for v in b)
    for i in range(3):
        for j in range(3):
            cov[i][j] /= n
    var_d /= n

    from services.linalg3 import det3, matmul3, svd3

    U, sv, Vt = svd3(cov)
    # 修正反射：保证 det(R)=+1
    D = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if det3(matmul3(U, Vt)) < 0:
        D[2][2] = -1.0
    R = matmul3(matmul3(U, D), Vt)
    s = 1.0
    if var_d > 0:
        trace_ds = sum(sv[i] * D[i][i] for i in range(3))
        s = trace_ds / var_d
    mu_m_t = _centroid(model)
    Rmu_d = [sum(R[i][j] * mu_d[j] for j in range(3)) for i in range(3)]
    t = [mu_m_t[i] - s * Rmu_d[i] for i in range(3)]
    return {"scale": round(s, 9),
            "rotation": [[round(v, 9) for v in row] for row in R],
            "translation": [round(v, 9) for v in t],
            "note": "完整 SVD 对齐（含反射修正），旋转可用"}


def apply_transform(P: Pts, s: float, R, t) -> Pts:
    out = []
    for p in P:
        q = [sum(R[i][j] * p[j] for j in range(3)) for i in range(3)]
        out.append([round(s * q[i] + t[i], 9) for i in range(3)])
    return out


def ate(model: Pts, estimate: Pts, align: bool = True) -> dict:
    """绝对轨迹误差：对齐后逐点平移误差的 RMSE 与最大值。"""
    est = estimate
    info = {}
    if align:
        info = umeyama_alignment(model, estimate)
        est = apply_transform(estimate, info["scale"], info["rotation"],
                              info["translation"])
    errs = [math.dist(m, e) for m, e in zip(model, est)]
    rmse = math.sqrt(sum(x * x for x in errs) / len(errs)) if errs else 0.0
    return {"rmse": round(rmse, 9), "max": round(max(errs), 9) if errs else 0.0,
            "n": len(errs), "alignment": info or None}


def rpe(model: Pts, estimate: Pts, delta: int = 1) -> dict:
    """相对位姿误差：对每对相隔 delta 帧计算相对平移误差。"""
    if delta < 1:
        raise ValueError("delta 必须 >= 1")
    errs = []
    for i in range(len(model) - delta):
        dm = [model[i + delta][k] - model[i][k] for k in range(3)]
        de = [estimate[i + delta][k] - estimate[i][k] for k in range(3)]
        errs.append(math.dist(dm, de))
    rmse = math.sqrt(sum(x * x for x in errs) / len(errs)) if errs else 0.0
    return {"delta": delta, "rmse": round(rmse, 9),
            "mean": round(sum(errs) / len(errs), 9) if errs else 0.0,
            "n": len(errs)}


def selftest() -> dict:
    cases = []
    traj = [[float(i), 0.0, 0.0] for i in range(10)]

    # 1) 完全相同 -> ATE 0
    r = ate(traj, traj)
    cases.append({"case": "identical", "passed": abs(r["rmse"]) < 1e-12,
                  "value": r["rmse"]})

    # 2) 纯平移偏移 1.0，对齐后应回 0
    off = [[p[0] + 1.0, p[1], p[2]] for p in traj]
    r = ate(traj, off)
    cases.append({"case": "translated_aligned", "passed": abs(r["rmse"]) < 1e-6,
                  "value": r["rmse"]})

    # 3) 尺度 2 倍，对齐后应回 0
    sc = [[p[0] * 2.0, p[1], p[2]] for p in traj]
    r = ate(traj, sc)
    cases.append({"case": "scaled_aligned", "passed": abs(r["rmse"]) < 1e-6,
                  "value": r["rmse"]})

    # 4) 每点 z 方向固定误差 0.5 -> ATE = 0.5（对齐会吸收平移，故不对齐）
    jitter = [[p[0], p[1], p[2] + 0.5] for p in traj]
    r = ate(traj, jitter, align=False)
    cases.append({"case": "constant_bias_noalign", "passed": abs(r["rmse"] - 0.5) < 1e-9,
                  "value": r["rmse"]})

    # 5) RPE 对相同轨迹为 0
    r = rpe(traj, traj, delta=1)
    cases.append({"case": "rpe_identical", "passed": abs(r["rmse"]) < 1e-12,
                  "value": r["rmse"]})

    # 6) RPE 对整体平移不变（相对量）
    r = rpe(traj, off, delta=1)
    cases.append({"case": "rpe_translation_invariant",
                  "passed": abs(r["rmse"]) < 1e-9, "value": r["rmse"]})

    # 7) 整体旋转 90 度 + 平移，对齐后 ATE 应回 0
    # 注意：轨迹必须三维散开，否则共线轨迹绕自身轴的旋转不可观测（退化）
    traj3 = [[float(i), float(i % 3), float(i % 2)] for i in range(10)]
    rot = [[-p[1] + 3.0, p[0] - 2.0, p[2] + 1.0] for p in traj3]
    r = ate(traj3, rot)
    cases.append({"case": "rotated_aligned", "passed": abs(r["rmse"]) < 1e-6,
                  "value": r["rmse"]})

    return {"all_pass": all(c["passed"] for c in cases), "cases": cases}
