import numpy as np


def roi(initial_cash: float, final_cash: float) -> float:
    return float(((final_cash - initial_cash) / max(initial_cash, 1e-8)) * 100.0)


def sharpe_ratio(profits, p0: float = 0.0, pN: float = 0.0) -> float:
    """
    Paper-inspired SR from mean/std of period profits with price-adjustment term.
    """
    profits = np.asarray(profits, dtype=np.float64)
    if profits.size == 0:
        return 0.0

    n = max(int(profits.size), 1)
    mean_profit = float(np.mean(profits))
    std_profit = float(np.std(profits))
    if std_profit <= 1e-12:
        return 0.0

    return float((mean_profit - ((p0 - pN) / n)) / std_profit)
