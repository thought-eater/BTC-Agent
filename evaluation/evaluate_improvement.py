from __future__ import annotations

from pathlib import Path

import pandas as pd


def run(results: list, results_dir: str = "results") -> dict:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    if df.empty:
        return {"generated": []}

    baseline = df[(df["branch"] == "baseline") & (df["method"] == "proposed") & (df["variant_id"] == "paper")].copy()
    improved = df[(df["branch"] == "improved") & (df["method"] == "proposed")].copy()
    if baseline.empty or improved.empty:
        return {"generated": []}

    merged = improved.merge(
        baseline[["alpha", "omega", "roi", "sr", "trades", "final_cash"]],
        on=["alpha", "omega"],
        how="left",
        suffixes=("_improved", "_baseline"),
    )

    merged["delta_roi"] = merged["roi_improved"] - merged["roi_baseline"]
    merged["delta_sr"] = merged["sr_improved"] - merged["sr_baseline"]
    merged["delta_trades"] = merged["trades_improved"] - merged["trades_baseline"]
    merged["delta_final_cash"] = merged["final_cash_improved"] - merged["final_cash_baseline"]

    out_path = out_dir / "table_improvement_delta.csv"
    merged.to_csv(out_path, index=False)

    summary = (
        merged.groupby("variant_id")[["delta_roi", "delta_sr", "delta_final_cash"]]
        .mean()
        .reset_index()
        .sort_values(["delta_roi", "delta_sr"], ascending=False)
    )
    summary_path = out_dir / "table_improvement_summary.csv"
    summary.to_csv(summary_path, index=False)

    return {"generated": [str(out_path), str(summary_path)]}
