"""Train PVCGN / PT-PVCGN ablation variants on the training split, select the epoch on validation.

The test split is never read here; it is scored once at pipeline step 14.

    python -m metro.train --city hz --variant pt --seed 1 --epochs 100
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from metro.batches import ROOT, load_city
from metro.metrics import mean_over_horizons, per_horizon
from metro.model import Net

VARIANTS = {
    "pvcgn": dict(periodic=False, temporal=False),   # published model
    "p": dict(periodic=True, temporal=False),        # + periodic reference inputs
    "pt": dict(periodic=True, temporal=True),        # + calendar embeddings = PT-PVCGN
}
RUNS = ROOT / "ml_pipeline" / "runs"
CKPT = ROOT / "checkpoints"


def build_model(ct, variant: str, units: int = 256) -> Net:
    return Net(ct.num_nodes, ct.static_adj, units=units, **VARIANTS[variant])


@torch.no_grad()
def predict(model: Net, ct, idx: np.ndarray, batch_size: int) -> np.ndarray:
    """Raw-count predictions [B, 4, N, 2] for the given (day, window) pairs."""
    model.eval()
    out = []
    for s in range(0, len(idx), batch_size):
        out.append(model(ct.batch(idx[s:s + batch_size])).float().cpu().numpy())
    return np.concatenate(out) * ct.std + ct.mean


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["hz", "sh"], required=True)
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--milestones", type=int, nargs="+", default=[50, 75])
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--units", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args(argv)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    name = f"{a.city}_{a.variant}_s{a.seed}{a.tag}"
    RUNS.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    ct = load_city(a.city, a.device)
    tr_idx, va_idx = ct.windows("train"), ct.windows("val")
    va_true = ct.targets(va_idx)
    model = build_model(ct, a.variant, a.units).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, a.milestones, gamma=0.1)
    loss_fn = nn.L1Loss()
    rng = np.random.default_rng(a.seed)

    log = {"config": vars(a), "n_train_windows": len(tr_idx), "n_val_windows": len(va_idx),
           "n_params": sum(p.numel() for p in model.parameters()), "epochs": []}
    best, stale = float("inf"), 0
    for epoch in range(a.epochs):
        t0 = time.perf_counter()
        model.train()
        order = tr_idx[rng.permutation(len(tr_idx))]
        total = 0.0
        for s in range(0, len(order), a.batch):
            b = ct.batch(order[s:s + a.batch])
            pred = model(b) * ct.std + ct.mean
            loss = loss_fn(pred, b["y"] * ct.std + ct.mean)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(b["x"])
        sched.step()
        val = mean_over_horizons(per_horizon(predict(model, ct, va_idx, a.batch), va_true))
        rec = {"epoch": epoch + 1, "train_mae": total / len(tr_idx), **{f"val_{k}": v for k, v in val.items()},
               "seconds": round(time.perf_counter() - t0, 1)}
        log["epochs"].append(rec)
        if val["mae"] < best:
            best, stale = val["mae"], 0
            torch.save(model.state_dict(), CKPT / f"{name}.pt")
            log["best"] = rec
        else:
            stale += 1
        print(json.dumps(rec), flush=True)
        (RUNS / f"{name}.json").write_text(json.dumps(log, indent=1))
        if stale >= a.patience:
            break

    model.load_state_dict(torch.load(CKPT / f"{name}.pt", map_location=a.device))
    val_h = per_horizon(predict(model, ct, va_idx, a.batch), va_true)
    log["best_val_per_horizon"] = val_h
    log["finished"] = True
    (RUNS / f"{name}.json").write_text(json.dumps(log, indent=1))
    print("best", json.dumps(log["best"]))


if __name__ == "__main__":
    main()
