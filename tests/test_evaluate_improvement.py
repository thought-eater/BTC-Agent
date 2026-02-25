from pathlib import Path

import pandas as pd

from evaluation.evaluate_improvement import run


def test_improvement_delta_generation(tmp_path):
    rows = [
        {"alpha": 0.55, "omega": 16, "roi": 10, "sr": 1.0, "trades": 100, "final_cash": 1100, "branch": "baseline", "variant_id": "paper"},
        {"alpha": 0.55, "omega": 16, "roi": 12, "sr": 1.2, "trades": 98, "final_cash": 1120, "branch": "improved", "variant_id": "dueling_double"},
    ]
    out = run(rows, str(tmp_path))
    assert out["generated"]
    delta = pd.read_csv(Path(tmp_path) / "table_improvement_delta.csv")
    assert "delta_roi" in delta.columns
