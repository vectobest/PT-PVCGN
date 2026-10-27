"""Calendar and reference-day selection (the periodic context PVCGN does not use).

Everything here is computed from information available at forecast time: the calendar is
published in advance and reference days are strictly earlier days.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metro.data import SLOTS_PER_DAY, SPLITS, load_split, windows_to_days

# Official public holidays and make-up working days (State Council notices) inside the data range.
HOLIDAYS = {
    "hz": {"2019-01-01"},
    "sh": {"2016-09-15", "2016-09-16", "2016-09-17"},
}
MAKEUP_WORKDAYS = {
    "hz": set(),
    "sh": {"2016-09-18"},
}

@dataclass
class CityData:
    city: str
    series: np.ndarray        # [D, 73, N, 2] raw counts, all days in chronological order
    dates: pd.DatetimeIndex   # [D]
    split: np.ndarray         # [D] 'train' / 'val' / 'test' (released split)
    dow: np.ndarray           # [D] 0=Mon
    workday: np.ndarray       # [D] bool, official working day
    holiday: np.ndarray       # [D] bool
    ref_type: np.ndarray      # [D] index of most recent earlier day of the same day type, -1 if none
    ref_week: np.ndarray      # [D] index of the same weekday 7 days earlier, fallback ref_type


def build_city(city: str) -> CityData:
    series, dates, split = [], [], []
    for s in SPLITS:
        days, stamps = windows_to_days(load_split(city, s))
        series.append(days)
        dates.extend(pd.to_datetime(stamps[:, 0]).normalize())
        split.extend([s] * len(days))
    series = np.concatenate(series)
    dates = pd.DatetimeIndex(dates)
    assert dates.is_monotonic_increasing and dates.is_unique
    keys = dates.strftime("%Y-%m-%d")
    holiday = np.array([k in HOLIDAYS[city] for k in keys])
    makeup = np.array([k in MAKEUP_WORKDAYS[city] for k in keys])
    dow = dates.dayofweek.to_numpy()
    workday = ((dow < 5) & ~holiday) | makeup

    n = len(dates)
    ref_type = np.full(n, -1)
    ref_week = np.full(n, -1)
    for d in range(n):
        earlier = [e for e in range(d) if workday[e] == workday[d]]
        if earlier:
            ref_type[d] = earlier[-1]
        week = np.where(dates == dates[d] - pd.Timedelta(days=7))[0]
        if len(week) and workday[week[0]] == workday[d]:
            ref_week[d] = week[0]
        else:
            ref_week[d] = ref_type[d]
    return CityData(city, series, dates, np.array(split), dow, workday, holiday, ref_type, ref_week)


def sample_index(cd: CityData, split: str, seq_len: int = 4, horizon: int = 4) -> np.ndarray:
    """(day, window) pairs for a split, in the released order. Days without a reference day are dropped."""
    days = np.where((cd.split == split) & (cd.ref_type >= 0))[0]
    windows = np.arange(SLOTS_PER_DAY - seq_len - horizon + 1)
    return np.array([(d, w) for d in days for w in windows], dtype=np.int64)

