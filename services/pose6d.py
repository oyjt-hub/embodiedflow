"""6D Pose 的表示转换与坐标映射（V13 计算层）。

覆盖：四元数 <-> 旋转矩阵、欧拉角 <-> 旋转矩阵、齐次变换复合、
相机系位姿映射到机器人基座、位姿的坐标轴端点（用于可视化）。

全部为纯数学，不依赖硬件；selftest 用已知解析解验证。
"""

from __future__ import annotations

import math

Vec3 = "list[float]"


def quat_to_matrix(q: "list[float]") -> "list[list[float]]":
    """四元数 (x, y, z, w) -> 3x3 旋转矩阵。"""
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def matrix_to_quat(m: "list[list[float]]") -> "list[float]":
    """3x3 旋转矩阵 -> 四元数 (x, y, z, w)，取 w >= 0 的分支。"""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    return [round(x, 9), round(y, 9), round(z, 9), round(w, 9)]


def euler_to_matrix(rx: float, ry: float, rz: float) -> "list[list[float]]":
    """XYZ 顺序内旋欧拉角（弧度）-> 旋转矩阵。"""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
    tmp = _matmul(Rz, Ry)
    return _matmul(tmp, Rx)


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _matvec(m, v):
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


def compose(t1: "list[float]", q1: "list[float]", t2: "list[float]",
            q2: "list[float]") -> dict:
    """位姿复合：T = T1 * T2（均以 (t, quat) 表示）。"""
    R1 = quat_to_matrix(q1)
    t2r = _matvec(R1, [float(v) for v in t2])
    t = [round(t1[i] + t2r[i], 9) for i in range(3)]
    R2 = quat_to_matrix(q2)
    R = _matmul(R1, R2)
    return {"translation": t, "quaternion": matrix_to_quat(R)}


def camera_to_base(point_cam, base_to_cam_t, base_to_cam_q) -> "list[float]":
    """相机坐标系下的点 -> 机器人基座坐标。

    约定 base_to_cam 表示「相机原点在基座中的位姿」，即 p_base = R * p_cam + t。
    """
    R = quat_to_matrix(base_to_cam_q)
    pc = _matvec(R, [float(v) for v in point_cam])
    return [round(pc[i] + float(base_to_cam_t[i]), 9) for i in range(3)]


def pose_axes(translation, quaternion, length: float = 0.1) -> dict:
    """位姿的三个坐标轴端点（基座系），用于三轴可视化。"""
    R = quat_to_matrix(quaternion)
    t = [float(v) for v in translation]
    out = {}
    for idx, name in enumerate(("x", "y", "z")):
        axis = [R[0][idx], R[1][idx], R[2][idx]]
        out[name] = {
            "origin": [round(v, 6) for v in t],
            "end": [round(t[i] + axis[i] * length, 6) for i in range(3)],
        }
    return out


def selftest() -> dict:
    cases = []

    # 1) 单位四元数 -> 单位矩阵
    I = quat_to_matrix([0, 0, 0, 1])
    cases.append({"case": "identity_quat", "passed":
                  all(abs(I[i][j] - (1 if i == j else 0)) < 1e-12
                      for i in range(3) for j in range(3))})

    # 2) 绕 z 轴 90 度：(1,0,0) -> (0,1,0)
    qz = [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    p = _matvec(quat_to_matrix(qz), [1, 0, 0])
    cases.append({"case": "rot_z90", "passed":
                  abs(p[0]) < 1e-9 and abs(p[1] - 1) < 1e-9 and abs(p[2]) < 1e-9})

    # 3) 欧拉角 (0,0,pi/2) 与 qz 等价
    R = euler_to_matrix(0, 0, math.pi / 2)
    q2 = matrix_to_quat(R)
    cases.append({"case": "euler_matches_quat", "passed":
                  all(abs(abs(q2[i]) - abs(qz[i])) < 1e-7 for i in range(4))})

    # 4) 四元数往返
    q3 = [0.1, -0.2, 0.3, 0.9]
    n = math.sqrt(sum(v * v for v in q3))
    q3n = [v / n for v in q3]
    back = matrix_to_quat(quat_to_matrix(q3n))
    cases.append({"case": "quat_roundtrip", "passed":
                  all(abs(abs(back[i]) - abs(q3n[i])) < 1e-7 for i in range(4))})

    # 5) 复合：先平移 (1,0,0)，再绕 z 90 度后平移 (1,0,0) 应得 (1,1,0)
    qz2 = [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    comp = compose([0, 0, 0], qz2, [1, 0, 0], [0, 0, 0, 1])
    # 绕 z 轴 90 度后 (1,0,0) 应变成 (0,1,0)
    cases.append({"case": "compose_translate", "passed":
                  abs(comp["translation"][0]) < 1e-7
                  and abs(comp["translation"][1] - 1) < 1e-7
                  and abs(comp["translation"][2]) < 1e-7})

    # 6) 相机->基座：无旋转仅平移
    pb = camera_to_base([0, 0, 0], [0.5, 0.2, 0.1], [0, 0, 0, 1])
    cases.append({"case": "cam_to_base_translate", "passed":
                  abs(pb[0] - 0.5) < 1e-9 and abs(pb[1] - 0.2) < 1e-9
                  and abs(pb[2] - 0.1) < 1e-9})

    # 7) 坐标轴长度正确
    axes = pose_axes([0, 0, 0], [0, 0, 0, 1], 0.1)
    d = math.dist(axes["x"]["origin"], axes["x"]["end"])
    cases.append({"case": "axes_length", "passed": abs(d - 0.1) < 1e-6})

    return {"all_pass": all(c["passed"] for c in cases), "cases": cases}
