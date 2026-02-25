import numpy as np


class TradeDQNEnvironment:
    """Trade-DQN preprocessing environment based on AP_t only (paper-aligned)."""

    def __init__(self, prices, hold_penalty_limit: int = 20, buy_penalty_limit: int = 20):
        self.prices = np.asarray(prices, dtype=np.float32)
        self.hold_penalty_limit = hold_penalty_limit
        self.buy_penalty_limit = buy_penalty_limit
        self.reset()

    def reset(self):
        self.idx = 0
        self.last_buy_price = None
        self.hold_streak = 0
        self.buy_streak = 0
        return self._state()

    def _state(self):
        return np.array([round(float(self.prices[self.idx]), 2)], dtype=np.float32)

    def step(self, action: int):
        # action: 0=sell, 1=hold, 2=buy
        reward = 0.0
        price = float(self.prices[self.idx])

        if action == 1:  # hold
            self.hold_streak += 1
            self.buy_streak = 0
            reward = -1.0 if self.hold_streak >= self.hold_penalty_limit else 0.0

        elif action == 2:  # buy
            self.buy_streak += 1
            self.hold_streak = 0
            if self.last_buy_price is None:
                self.last_buy_price = price
            reward = -1.0 if self.buy_streak > self.buy_penalty_limit else 0.0

        elif action == 0:  # sell
            self.hold_streak = 0
            self.buy_streak = 0
            if self.last_buy_price is not None:
                # Paper Trade-DQN reward term: Psell - Pbuy
                reward = price - self.last_buy_price
                self.last_buy_price = None

        self.idx += 1
        done = self.idx >= len(self.prices) - 1
        next_state = self._state() if not done else np.array([round(price, 2)], dtype=np.float32)
        return next_state, float(reward), done, {}
