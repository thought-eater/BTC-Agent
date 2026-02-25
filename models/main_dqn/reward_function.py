import numpy as np


def compute_main_reward(
    action: int,
    pnl: float,
    active_trade_count_day: int,
    omega: int,
    current_investment_ratio: float,
    alpha: float,
    reward_mode: str = "paper",
    drawdown_penalty: float = 0.0,
) -> float:
    """Paper reward."""
    if active_trade_count_day > omega:
        return -1.0

    if action == 2 and current_investment_ratio < (1.0 - alpha):
        return -1.0

    base = float(pnl) if action == 0 else 0.0

    return base
