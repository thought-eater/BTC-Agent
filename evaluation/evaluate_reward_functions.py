from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def run(results: List[dict], results_dir: str = "results") -> dict:
    """Generates a focused variant snapshot at paper midpoint (alpha=0.55, omega=16)."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    if df.empty:
        return {"generated": []}

    base = df[(df["alpha"].round(2) == 0.55) & (df["omega"] == 16)].copy()
    if base.empty:
        return {"generated": []}

    selected_variants = {"paper", "policy_gradient"}
    out = base[base["variant_id"].isin(selected_variants)].copy()
    if out.empty:
        return {"generated": []}

    variant_label = {
        "paper": "Paper baseline",
        "policy_gradient": "Main policy-gradient",
    }
    out["variant_label"] = out["variant_id"].map(variant_label).fillna(out["variant_id"])
    cols = ["branch", "method", "variant_id", "variant_label", "alpha", "omega", "roi", "sr", "trades", "final_cash"]
    out = out[cols].sort_values(["branch", "method", "variant_id"])

    out_path = out_dir / "table7_reward_functions.csv"
    out.to_csv(out_path, index=False)
    return {"generated": [str(out_path)]}
