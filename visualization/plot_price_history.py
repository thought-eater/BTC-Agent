from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_price_history(base_df: pd.DataFrame, output_path: str = "results/price_history.png") -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(base_df["timestamp"], base_df["ap_t"], label="BTC Price", color="steelblue", linewidth=1)
    plt.title("BTC Price History (Adapted Window)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
