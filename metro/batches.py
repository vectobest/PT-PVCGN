"""Turn the engineered city arrays (ml_pipeline/<city>/features.npz) into model batches."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from metro.data import SLOTS_PER_DAY, load_graphs

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_PER_DAY = SLOTS_PER_DAY - 7


@dataclass
class CityTensors:
    city: str
    z: torch.Tensor            # [D, 73, N, 2] scaled counts
    raw: np.ndarray            # [D, 73, N, 2] raw counts (numpy, for metrics)
    split: np.ndarray          # [D]
    dates: np.ndarray          # [D] datetime64
    dow: torch.Tensor
    workday: torch.Tensor
    ref_type: torch.Tensor
    ref_week: torch.Tensor
    static_adj: np.ndarray     # [3, N, N] physical, similarity, correlation (released graphs)
    mean: float
    std: float

    @property
    def num_nodes(self) -> int:
        return self.raw.shape[2]

    def windows(self, split: str) -> np.ndarray:
        """(day, window) pairs in chronological order; training days without a reference are skipped."""
        ref = self.ref_type.cpu().numpy()
        days = np.where((self.split == split) & (ref >= 0))[0]
        return np.array([(d, w) for d in days for w in range(WINDOWS_PER_DAY)], dtype=np.int64)

    def batch(self, idx: np.ndarray) -> dict:
        dev = self.z.device
        d = torch.as_tensor(idx[:, 0], device=dev)
        w = torch.as_tensor(idx[:, 1], device=dev)
        steps = w[:, None] + torch.arange(8, device=dev)            # [B, 8] absolute slots
        return {
            "x": self.z[d[:, None], steps[:, :4]],
            "y": self.z[d[:, None], steps[:, 4:]],
            "slots": steps,
            "dow": self.dow[d],
            "workday": self.workday[d],
            "ref_type": self.z[self.ref_type[d][:, None], steps],
            "ref_week": self.z[self.ref_week[d][:, None], steps],
        }

    def targets(self, idx: np.ndarray) -> np.ndarray:
        """Raw-count targets [B, 4, N, 2]."""
        return self.raw[idx[:, 0][:, None], idx[:, 1][:, None] + np.arange(4, 8)]


def load_city(city: str, device: str = "cpu") -> CityTensors:
    f = np.load(ROOT / "ml_pipeline" / city / "features.npz", allow_pickle=False)
    sc = json.loads((ROOT / "ml_pipeline" / city / "scaler.json").read_text())
    g = load_graphs(city)
    raw = f["series"]
    as_t = lambda a: torch.as_tensor(np.asarray(a), device=device)
    return CityTensors(
        city=city,
        z=torch.as_tensor((raw - sc["mean"]) / sc["std"], dtype=torch.float32, device=device),
        raw=raw,
        split=f["split"],
        dates=f["dates"],
        dow=as_t(f["dow"]).long(),
        workday=as_t(f["workday"]).long(),
        ref_type=as_t(f["ref_type"]).long(),
        ref_week=as_t(f["ref_week"]).long(),
        static_adj=np.stack([g["conn"], g["sml"], g["cor"]]),
        mean=sc["mean"],
        std=sc["std"],
    )
