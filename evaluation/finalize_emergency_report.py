from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from evaluation.evaluate_improvement import run as run_improvement_eval
from evaluation.evaluate_thresholds import run as run_threshold_eval
from visualization.plot_improvement_comparison import plot_improvement_comparison
from visualization.plot_price_history import plot_price_history
from visualization.plot_trading_signals import plot_trading_signals


def _build_table8(results_dir: str = "results") -> str:
    rows = [
        {"study": "DNA-S (Betancourt et al., 2021)", "roi": ">24%", "sr": "N/A"},
        {"study": "SharpeD-DQN (Lucarelli et al., 2019)", "roi": "26.14%", "sr": "N/A"},
        {"study": "Double Q-network + Boltzmann (Bu et al., 2018)", "roi": "27.87%", "sr": "N/A"},
        {"study": "DQN (Theate et al., 2021)", "roi": "29.4%", "sr": "N/A"},
        {"study": "TD3 (Majidi et al., 2022)", "roi": "57.5%", "sr": "1.53"},
        {"study": "M-DQN (this run)", "roi": "runtime_output", "sr": "runtime_output"},
    ]
    out = Path(results_dir) / "table8_sota_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)


def run(results_rows: List[dict], results_dir: str, base_df: pd.DataFrame, proposed_df: pd.DataFrame) -> dict:
    out = {"generated": []}
    threshold_out = run_threshold_eval(results_rows, results_dir)
    improvement_out = run_improvement_eval(results_rows, results_dir)
    table8_path = _build_table8(results_dir)

    out["generated"].extend(threshold_out.get("generated", []))
    out["generated"].extend(improvement_out.get("generated", []))
    out["generated"].append(table8_path)

    plot1 = plot_price_history(base_df, str(Path(results_dir) / "price_history.png"))
    plot2 = plot_trading_signals(proposed_df, str(Path(results_dir) / "trading_signals.png"))
    out["generated"].extend([plot1, plot2])

    delta_path = str(Path(results_dir) / "table_improvement_delta.csv")
    if Path(delta_path).exists():
        plot3 = plot_improvement_comparison(delta_path, str(Path(results_dir) / "improvement_comparison.png"))
        out["generated"].append(plot3)

    return out
