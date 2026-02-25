import numpy as np

from models.main_dqn.reward_function import compute_main_reward


class MainDQNEnvironment:
    """Main-DQN environment with paper-aligned transaction mechanics."""

    def __init__(
        self,
        x1,
        x2,
        prices,
        initial_investment: float,
        transaction_fee: float,
        alpha: float,
        omega: int,
        reward_mode: str = "paper",
    ):
        self.x1 = np.asarray(x1, dtype=np.float32)
        self.x2 = np.asarray(x2, dtype=np.float32)
        self.prices = np.asarray(prices, dtype=np.float32)
        self.initial_investment = float(initial_investment)
        self.transaction_fee = float(transaction_fee)
        self.alpha = float(alpha)
        self.omega = int(omega)
        self.reward_mode = reward_mode
        self.reset()

    def reset(self):
        self.idx = 0
        self.cash = self.initial_investment
        self.btc_held = 0.0
        self.buy_lots = []  # each lot: (buy_price, qty, buy_fee)
        self.trades = []
        self.active_trades_in_day = 0
        self.current_day = 0
        self.peak_value = self.initial_investment
        return self._state()

    def _state(self):
        return np.array([self.x1[self.idx], self.x2[self.idx]], dtype=np.float32)

    def _roll_day(self):
        day_idx = self.idx // 24
        if day_idx != self.current_day:
            self.current_day = day_idx
            self.active_trades_in_day = 0

    def _buy_one_btc(self, price: float) -> bool:
        fee = price * self.transaction_fee
        total_cost = price + fee
        if self.cash >= total_cost:
            self.cash -= total_cost
            self.btc_held += 1.0
            self.buy_lots.append((price, 1.0, fee))
            self.active_trades_in_day += 1
            self.trades.append(("buy", self.idx, price, 1.0))
            return True
        return False

    def _sell_one_btc(self, price: float) -> float:
        if self.btc_held < 1.0 or not self.buy_lots:
            return 0.0

        buy_price, qty, buy_fee = self.buy_lots.pop(0)
        gross = price * qty
        sell_fee = gross * self.transaction_fee
        revenue = gross - sell_fee
        self.cash += revenue
        self.btc_held -= qty
        self.active_trades_in_day += 1
        self.trades.append(("sell", self.idx, price, qty))

        # Paper Eq. (3): PnL_k = P_sell - P_buy - c_buy - c_sell
        pnl = (price * qty) - (buy_price * qty) - buy_fee - sell_fee
        return float(pnl)

    def step(self, action: int):
        # action: 0=sell, 1=hold, 2=buy
        self._roll_day()
        price = float(self.prices[self.idx])

        pnl = 0.0
        if action == 2:
            self._buy_one_btc(price)
        elif action == 0:
            pnl = self._sell_one_btc(price)

        current_value = self.cash + self.btc_held * price
        self.peak_value = max(self.peak_value, current_value)
        drawdown = max(0.0, (self.peak_value - current_value) / max(self.peak_value, 1e-8))

        # Paper risk condition operationalized as capital floor: I_current < (1-alpha) * I_initial
        investment_ratio = current_value / max(self.initial_investment, 1e-8)

        reward = compute_main_reward(
            action=action,
            pnl=pnl,
            active_trade_count_day=self.active_trades_in_day,
            omega=self.omega,
            current_investment_ratio=investment_ratio,
            alpha=self.alpha,
            reward_mode=self.reward_mode,
            drawdown_penalty=drawdown,
        )

        self.idx += 1
        done = self.idx >= len(self.prices) - 1
        next_state = self._state() if not done else np.array([0.0, 0.0], dtype=np.float32)
        info = {
            "portfolio_value": current_value,
            "cash": self.cash,
            "btc_held": self.btc_held,
            "active_trades_in_day": self.active_trades_in_day,
            "drawdown": drawdown,
        }
        return next_state, float(reward), done, info
