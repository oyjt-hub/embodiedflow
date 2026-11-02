"""PyTorch 行为克隆：state(14) -> action(14)。

- 按 Episode 固定划分：前 N-3 个训练、最后 3 个留出
- 归一化参数只用训练集拟合，写入 norm.json 供评测/推理复用
- 产出 checkpoint.pt / norm.json / metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import json as _json
import os as _os

import numpy as np
import torch
import torch.nn as nn


def load(data_dir: str, episode_indices: set[int] | None = None,
         profile_name: str | None = None):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(data_dir, "data", "**", "*.parquet"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    from services.observation_profile import get_profile, vectorize
    prof = get_profile(profile_name) if profile_name else None
    sidecar = _load_sidecar(data_dir) if prof and prof.fields else {}

    states, actions, eps = [], [], []
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "observation.state", "action"]).to_pydict()
        for s, a, e in zip(t["observation.state"], t["action"], t["episode_index"]):
            if episode_indices is not None and int(e) not in episode_indices:
                continue
            states.append(np.asarray(s, dtype=np.float32))
            actions.append(np.asarray(a, dtype=np.float32))
            eps.append(int(e))
    return np.stack(states), np.stack(actions), np.asarray(eps)


class BC(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episode-indices-file")
    a = ap.parse_args()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    os.makedirs(a.out, exist_ok=True)
    allowed = None
    if a.episode_indices_file:
        allowed = set(int(x) for x in json.load(open(a.episode_indices_file, encoding="utf-8")))
        if not allowed:
            raise ValueError("Episode 选择为空")
    X, Y, ep = load(a.data, allowed, getattr(a, "profile", None))
    uniq = sorted(set(ep.tolist()))
    if len(uniq) < 2:
        raise ValueError("训练至少需要 2 个 Episode，才能分出训练集和验证集")
    holdout_count = min(max(1, a.holdout), len(uniq) - 1)
    held = set(uniq[-holdout_count:])
    tr = np.array([e not in held for e in ep])
    va = ~tr
    print(f"[bc] samples={len(X)} dims={X.shape[1]}->{Y.shape[1]} "
          f"train={int(tr.sum())} val={int(va.sum())} holdout_eps={sorted(held)}", flush=True)

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    Xn = torch.tensor((X - mu) / sd)
    Yn = torch.tensor((Y - ym) / ys)

    dev = torch.device(a.device)
    model = BC(X.shape[1], Y.shape[1]).to(dev)
    optimizer_cls = torch.optim.AdamW if a.optimizer == "adamw" else torch.optim.Adam
    opt = optimizer_cls(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    lossf = nn.MSELoss()

    idx = np.where(tr)[0]
    hist, val_hist = [], []
    for epoch in range(a.epochs):
        model.train()
        np.random.shuffle(idx)
        tot, n = 0.0, 0
        for i in range(0, len(idx), a.batch):
            b = idx[i:i + a.batch]
            xb = Xn[b].to(dev); yb = Yn[b].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += float(loss) * len(b); n += len(b)
        hist.append(round(tot / max(n, 1), 6))
        model.eval()
        with torch.no_grad():
            val_loss = float(lossf(model(Xn[va].to(dev)), Yn[va].to(dev)))
        val_hist.append(round(val_loss, 6))
        print(f"[bc] epoch {epoch + 1}/{a.epochs} loss={hist[-1]:.6f} "
              f"val_loss={val_hist[-1]:.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred = model(Xn[va].to(dev)).cpu().numpy() * ys + ym
    gt = Y[va]
    mae = float(np.abs(pred - gt).mean())
    rmse = float(np.sqrt(((pred - gt) ** 2).mean()))
    base = np.tile(Y[tr].mean(0), (gt.shape[0], 1))
    b_mae = float(np.abs(base - gt).mean())

    torch.save({"state_dict": model.state_dict(), "dim_in": X.shape[1],
                "dim_out": Y.shape[1]}, os.path.join(a.out, "checkpoint.pt"))
    json.dump({"state_mean": mu.tolist(), "state_std": sd.tolist(),
               "action_mean": ym.tolist(), "action_std": ys.tolist(),
               "profile": a.profile},
              open(os.path.join(a.out, "norm.json"), "w"))
    metrics = {
        "epochs": a.epochs, "lr": a.lr, "device": a.device,
        "batch_size": a.batch, "optimizer": a.optimizer,
        "weight_decay": a.weight_decay, "holdout_episodes_count": holdout_count,
        "seed": a.seed,
        "train_samples": int(tr.sum()), "val_samples": int(va.sum()),
        "train_episodes": len(uniq) - len(held), "holdout_episodes": sorted(held),
        "loss_history": hist, "val_loss_history": val_hist,
        "train_loss": hist[-1], "val_loss": val_hist[-1],
        "val_mae": round(mae, 6), "val_rmse": round(rmse, 6),
        "baseline_mean_action_mae": round(b_mae, 6),
    }
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print(f"[bc] val_mae={mae:.6f} rmse={rmse:.6f} baseline_mae={b_mae:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
