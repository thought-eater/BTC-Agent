from __future__ import annotations

import pandas as pd


def clean_btc_hourly_csv(csv_path: str) -> pd.DataFrame:
    """Loads CryptoDataDownload BTC hourly file and returns sorted hourly close data."""
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {
        "volume_btc": "volume_btc",
        "volume_usd": "volume_usd",
    }
    df = df.rename(columns=rename_map)

    expected = ["unix", "date", "symbol", "open", "high", "low", "close"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing BTC columns: {missing}")

    numeric_cols = ["open", "high", "low", "close"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "close"]) 
    df = df[df["close"] > 0].copy()

    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df[["timestamp", "close"]].reset_index(drop=True)


def save_btc_clean(df: pd.DataFrame, output_path: str) -> str:
    df.to_csv(output_path, index=False)
    return output_path
