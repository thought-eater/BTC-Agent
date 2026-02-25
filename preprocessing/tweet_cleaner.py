from __future__ import annotations

from typing import Iterable

import pandas as pd


def _clean_text_columns(df: pd.DataFrame, text_cols: Iterable[str]) -> pd.DataFrame:
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("\r", " ", regex=False).str.replace("\n", " ", regex=False)
    return df


def stream_clean_tweets_csv(input_path: str, output_path: str, chunksize: int = 200_000) -> str:
    """Lightweight cleaner for large semicolon-separated tweet dumps."""
    first = True
    for chunk in pd.read_csv(input_path, sep=";", chunksize=chunksize, low_memory=True, on_bad_lines="skip"):
        chunk = _clean_text_columns(chunk, ["text"]) 
        if "date" in chunk.columns:
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce", utc=True)
            chunk = chunk.dropna(subset=["date"])
            chunk["date"] = chunk["date"].dt.strftime("%Y-%m-%d %H:%M:%S%z")

        mode = "w" if first else "a"
        chunk.to_csv(output_path, sep=";", index=False, mode=mode, header=first)
        first = False

    return output_path
