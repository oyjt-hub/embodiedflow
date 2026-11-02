"""Mask 标注的编解码与评测（V09 计算链路）。

约定：
- Mask 以 COCO 风格 RLE 表示（counts 为 0/1 交替游程长度，首段对应取值 0）。
- 评测只看形状，不依赖具体模型。示例数据用于验证算法正确性，
  结果必须带 sample=True 标记，不得当作真实模型质量。
"""

from __future__ import annotations


def rle_encode(bits: "list[int]") -> dict:
    """0/1 序列 -> {size, counts}。"""
    if not bits:
        return {"size": 0, "counts": []}
    counts: "list[int]" = []
    cur = 0
    run = 0
    for b in bits:
        v = 1 if b else 0
        if v == cur:
            run += 1
        else:
            counts.append(run)
            cur = v
            run = 1
    counts.append(run)
    if bits[0]:
        counts.insert(0, 0)
    return {"size": len(bits), "counts": counts}


def rle_decode(rle: dict) -> "list[int]":
    size = int(rle.get("size") or 0)
    counts = list(rle.get("counts") or [])
    out: "list[int]" = []
    val = 0
    for c in counts:
        out.extend([val] * int(c))
        val = 1 - val
    return out[:size] if size else out


def mask_iou(gt_bits: "list[int]", pred_bits: "list[int]") -> dict:
    """返回 IoU / precision / recall / tp / fp / fn / tn。"""
    n = max(len(gt_bits), len(pred_bits))
    g = list(gt_bits) + [0] * (n - len(gt_bits))
    p = list(pred_bits) + [0] * (n - len(pred_bits))
    tp = fp = fn = tn = 0
    for a, b in zip(g, p):
        if a and b:
            tp += 1
        elif b and not a:
            fp += 1
        elif a and not b:
            fn += 1
        else:
            tn += 1
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"iou": round(iou, 6), "precision": round(precision, 6),
            "recall": round(recall, 6), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _rect(w: int, h: int, x0: int, y0: int, x1: int, y1: int) -> "list[int]":
    bits = [0] * (w * h)
    for y in range(h):
        if y0 <= y < y1:
            for x in range(w):
                if x0 <= x < x1:
                    bits[y * w + x] = 1
    return bits


def selftest() -> dict:
    """内置用例：矩形重叠面积可手算，验证算法而非模型。"""
    w = h = 10
    a = _rect(w, h, 2, 2, 6, 6)      # 4x4 = 16
    b = _rect(w, h, 7, 7, 9, 9)      # 不相交
    c = _rect(w, h, 3, 3, 5, 5)      # 2x2 = 4，被 a 包含
    cases = [
        {"case": "identical", "expect_iou": 1.0, "got": mask_iou(a, a)["iou"]},
        {"case": "disjoint", "expect_iou": 0.0, "got": mask_iou(a, b)["iou"]},
        {"case": "contained_quarter", "expect_iou": 0.25, "got": mask_iou(a, c)["iou"]},
        {"case": "rle_roundtrip", "expect_iou": 1.0,
         "got": 1.0 if rle_decode(rle_encode(a)) == a else 0.0},
    ]
    ok = all(abs(x["got"] - x["expect_iou"]) < 1e-9 for x in cases)
    return {"sample": True,
            "note": "示例数据，仅验证计算链路正确性，不代表真实模型质量",
            "all_pass": ok, "cases": cases}
