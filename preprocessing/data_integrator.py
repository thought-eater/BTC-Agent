from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass
class WindowSpec:
    overlap_start: pd.Timestamp
    overlap_end: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def resolve_adapted_window(
    btc_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    overlap_days: int = 1505,
    test_hours: int = 720,
) -> WindowSpec:
    btc_min, btc_max = btc_df["timestamp"].min(), btc_df["timestamp"].max()
    tw_min, tw_max = sentiment_df["timestamp"].min(), sentiment_df["timestamp"].max()

    overlap_start = max(btc_min, tw_min)
    overlap_end_target = overlap_start + timedelta(days=overlap_days) - timedelta(hours=1)
    overlap_end = min(overlap_end_target, btc_max, tw_max)

    if overlap_end < overlap_end_target:
        raise ValueError("Not enough overlap to satisfy 1505-day adapted window.")

    test_end = overlap_end
    test_start = test_end - timedelta(hours=test_hours - 1)
    train_start = overlap_start
    train_end = test_start - timedelta(hours=1)

    return WindowSpec(
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )


def build_base_dataset(btc_df: pd.DataFrame, sentiment_df: pd.DataFrame, window: WindowSpec) -> pd.DataFrame:
    rng = pd.date_range(window.overlap_start, window.overlap_end, freq="h", tz="UTC")
    base = pd.DataFrame({"timestamp": rng})

    btc = btc_df[["timestamp", "close"]].rename(columns={"close": "ap_t"})
    sent = sentiment_df[["timestamp", "sentiment"]].rename(columns={"sentiment": "ts_t"})

    merged = base.merge(btc, on="timestamp", how="left").merge(sent, on="timestamp", how="left")
    merged["ap_t"] = merged["ap_t"].ffill().bfill()
    merged["ts_t"] = merged["ts_t"].fillna(0.0)

    merged["split"] = "train"
    merged.loc[merged["timestamp"].between(window.test_start, window.test_end), "split"] = "test"
    return merged


def merge_x1_x2(base_df: pd.DataFrame, x1_df: pd.DataFrame, x2_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.merge(x1_df[["timestamp", "x1"]], on="timestamp", how="left")
    out = out.merge(x2_df[["timestamp", "x2"]], on="timestamp", how="left")
    out["x1"] = out["x1"].fillna(0).astype(int)
    out["x2"] = out["x2"].fillna(0.0)
    return out
