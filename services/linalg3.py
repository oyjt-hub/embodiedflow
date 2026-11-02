"""3x3 数值线性代数（纯 Python，供 SLAM 对齐使用）。

提供 Jacobi 特征分解与 3x3 SVD，用于 Umeyama 相似变换的旋转估计。
"""

from __future__ import annotations

import math


def jacobi_eigen(a: "list[list[float]]", iters: int = 60) -> tuple:
    """对称矩阵特征分解：返回 (特征值列表, 特征向量按列组成的矩阵)。"""
    n = len(a)
    A = [row[:] for row in a]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(iters):
        off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off += A[i][j] * A[i][j]
        if off < 1e-24:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-18:
                    continue
                theta = (A[q][q] - A[p][p]) / (2 * A[p][q])
                t = 1.0 if theta >= 0 else -1.0
                t = t / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p] = c * akp - s * akq
                    A[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k] = c * apk - s * aqk
                    A[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq
    vals = [A[i][i] for i in range(n)]
    return vals, V


def svd3(a):
    """3x3 矩阵 SVD：返回 (U, s, Vt)，满足 a = U * diag(s) * Vt。"""
    At = [[a[j][i] for j in range(3)] for i in range(3)]
    AtA = [[sum(At[i][k] * a[k][j] for k in range(3)) for j in range(3)]
           for i in range(3)]
    vals, V = jacobi_eigen(AtA)
    order = sorted(range(3), key=lambda i: vals[i], reverse=True)
    s = [math.sqrt(max(vals[i], 0.0)) for i in order]
    V = [[V[r][order[c]] for c in range(3)] for r in range(3)]

    U = [[0.0] * 3 for _ in range(3)]
    for c in range(3):
        col = [V[r][c] for r in range(3)]
        if s[c] > 1e-12:
            u = [sum(a[r][k] * col[k] for k in range(3)) / s[c] for r in range(3)]
        else:
            u = [0.0, 0.0, 0.0]
            u[c] = 1.0
        for r in range(3):
            U[r][c] = u[r]
    return U, s, [[V[r][c] for r in range(3)] for c in range(3)]


def det3(m) -> float:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def matmul3(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]
