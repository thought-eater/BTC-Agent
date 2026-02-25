from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def run(results: List[dict], results_dir: str = "results") -> dict:
    """Creates Table 4/5/6 style CSVs from grid results."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    if df.empty:
        return {"generated": []}

    generated = []
    for alpha, fname in [(0.30, "table4_risk30.csv"), (0.55, "table5_risk55.csv"), (0.80, "table6_risk80.csv")]:
        subset = df[df["alpha"].round(2) == round(alpha, 2)].copy()
        if subset.empty:
            continue

        cols = ["method", "alpha", "omega", "roi", "sr", "trades", "final_cash"]
        for optional_col in ("branch", "variant_id"):
            if optional_col in subset.columns:
                cols.insert(0, optional_col)
        subset = subset[cols]
        sort_cols = [c for c in ("branch", "variant_id", "method", "omega") if c in subset.columns]
        subset = subset.sort_values(sort_cols if sort_cols else ["omega"])
        out_path = out_dir / fname
        subset.to_csv(out_path, index=False)
        generated.append(str(out_path))

    return {"generated": generated}
