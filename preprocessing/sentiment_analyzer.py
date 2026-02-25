from __future__ import annotations

import pandas as pd


def aggregate_hourly_sentiment(vader_csv_path: str, output_path: str, chunksize: int = 200_000) -> str:
    """Aggregates tweet-level VADER scores into hourly sentiment."""
    required_cols = ["date", "vader_compound"]
    hourly_sum = {}
    hourly_count = {}

    for chunk in pd.read_csv(vader_csv_path, sep=";", usecols=required_cols, chunksize=chunksize, low_memory=True, on_bad_lines="skip"):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce", utc=True)
        chunk["vader_compound"] = pd.to_numeric(chunk["vader_compound"], errors="coerce")
        chunk = chunk.dropna(subset=["date", "vader_compound"])
        chunk["hour"] = chunk["date"].dt.floor("h")

        grouped = chunk.groupby("hour")["vader_compound"].agg(["sum", "count"]).reset_index()
        for _, row in grouped.iterrows():
            key = row["hour"]
            hourly_sum[key] = hourly_sum.get(key, 0.0) + float(row["sum"])
            hourly_count[key] = hourly_count.get(key, 0) + int(row["count"])

    rows = []
    for hour, s in hourly_sum.items():
        c = hourly_count[hour]
        rows.append({"timestamp": hour, "sentiment": s / max(c, 1)})

    out_df = pd.DataFrame(rows).sort_values("timestamp")
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True)
    out_df.to_csv(output_path, index=False)
    return output_path
