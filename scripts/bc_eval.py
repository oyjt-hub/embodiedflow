"""离线评测：复用训练集的归一化参数，在留出 Episode 上算 MAE / RMSE。

同时给出"预测训练集平均动作"的基线，用来判断模型是否真的学到了映射。
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch


def load(data_dir: str):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(data_dir, "data", "**", "*.parquet"), recursive=True))
    S, A, E = [], [], []
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "observation.state", "action"]).to_pydict()
        for s, a, e in zip(t["observation.state"], t["action"], t["episode_index"]):
            S.append(np.asarray(s, dtype=np.float32))
            A.append(np.asarray(a, dtype=np.float32))
            E.append(int(e))
    return np.stack(S), np.stack(A), np.asarray(E)


class BC(torch.nn.Module):
    def __init__(self, i, o, h=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(i, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="训练输出目录（含 checkpoint.pt / norm.json）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", type=int, default=3)
    a = ap.parse_args()

    ck = torch.load(os.path.join(a.model, "checkpoint.pt"), map_location="cpu")
    norm = json.load(open(os.path.join(a.model, "norm.json")))
    net = BC(ck["dim_in"], ck["dim_out"])
    net.load_state_dict(ck["state_dict"])
    net.eval()

    X, Y, ep = load(a.data)
    uniq = sorted(set(ep.tolist()))
    held = set(uniq[-a.holdout:]) if len(uniq) > a.holdout else set()
    mask = np.array([e in held for e in ep])

    mu = np.array(norm["state_mean"], dtype=np.float32)
    sd = np.array(norm["state_std"], dtype=np.float32)
    ym = np.array(norm["action_mean"], dtype=np.float32)
    ys = np.array(norm["action_std"], dtype=np.float32)

    with torch.no_grad():
        pred = net(torch.tensor((X[mask] - mu) / sd)).numpy() * ys + ym
    gt = Y[mask]
    err = np.abs(pred - gt)
    base = np.tile(ym, (gt.shape[0], 1))

    per_ep = []
    for e in sorted(held):
        m = ep[mask] == e
        if not m.any():
            continue
        pe = np.abs(pred[m] - gt[m])
        per_ep.append({
            "episode_index": int(e), "samples": int(m.sum()),
            "mae": round(float(pe.mean()), 6),
            "rmse": round(float(np.sqrt(((pred[m] - gt[m]) ** 2).mean())), 6),
        })

    report = {
        "model_dir": a.model,
        "holdout_episodes": sorted(held),
        "samples": int(mask.sum()),
        "mae": round(float(err.mean()), 6),
        "rmse": round(float(np.sqrt(((pred - gt) ** 2).mean())), 6),
        "baseline_mean_action_mae": round(float(np.abs(base - gt).mean()), 6),
        "improvement_vs_baseline": round(
            float(1 - err.mean() / max(float(np.abs(base - gt).mean()), 1e-9)), 4),
        "per_episode": per_ep,
    }
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"[eval] mae={report['mae']} rmse={report['rmse']} "
          f"baseline={report['baseline_mean_action_mae']} "
          f"improve={report['improvement_vs_baseline']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
