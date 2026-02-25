from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_improvement_comparison(delta_csv_path: str, output_path: str = "results/improvement_comparison.png") -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(delta_csv_path)
    if df.empty:
        return output_path

    agg = df.groupby("variant_id")[["delta_roi", "delta_sr"]].mean().reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.bar(agg["variant_id"], agg["delta_roi"], alpha=0.7, color="teal", label="Delta ROI")
    ax2.plot(agg["variant_id"], agg["delta_sr"], color="darkorange", marker="o", label="Delta SR")

    ax1.set_ylabel("Delta ROI (%)")
    ax2.set_ylabel("Delta SR")
    ax1.set_title("Improvement vs Baseline (mean across alpha/omega)")
    ax1.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
