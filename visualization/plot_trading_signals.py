from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_trading_signals(dataset_df: pd.DataFrame, output_path: str = "results/trading_signals.png") -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(dataset_df["timestamp"], dataset_df["ap_t"], color="royalblue", linewidth=1, label="BTC Price")

    buys = dataset_df[dataset_df["x1"] == 1]
    sells = dataset_df[dataset_df["x1"] == -1]

    if not buys.empty:
        plt.scatter(buys["timestamp"], buys["ap_t"], c="red", s=8, label="Buy", alpha=0.7)
    if not sells.empty:
        plt.scatter(sells["timestamp"], sells["ap_t"], c="green", s=8, label="Sell", alpha=0.7)

    plt.title("Trading Signals (x1) over BTC Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
