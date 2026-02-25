import numpy as np


class PredictiveDQNEnvironment:
    """Predictive-DQN with CDR-style reward and discretized action space."""

    def __init__(self, prices, sentiments, action_min: float = -100.0, action_step: float = 0.01):
        self.prices = np.asarray(prices, dtype=np.float32)
        self.sentiments = np.asarray(sentiments, dtype=np.float32)
        self.action_min = action_min
        self.action_step = action_step
        self.reset()

    def reset(self):
        self.idx = 1
        self.prev_pred_price = float(self.prices[self.idx - 1])
        return self._state(self.idx)

    def _state(self, idx: int):
        return np.array(
            [
                round(float(self.prices[idx]), 2),
                round(float(self.sentiments[idx]), 2),
            ],
            dtype=np.float32,
        )

    def action_to_pct(self, action_idx: int) -> float:
        return self.action_min + action_idx * self.action_step

    def step(self, action_idx: int):
        # AP_t and AP_{t-1}
        ap_t = float(self.prices[self.idx])
        ap_prev = float(self.prices[self.idx - 1])

        # Predicted % action -> predicted price at t
        pct = self.action_to_pct(action_idx)
        pp_t = ap_t * (1.0 + pct / 100.0)

        # CDR-style zero-reward boundaries based on previous prediction context
        alpha = (ap_t - ap_prev) / max(abs(ap_prev), 1e-8)
        l = abs(ap_t - self.prev_pred_price * (1.0 + alpha)) + 1e-8
        zr1 = ap_t - l
        zr2 = ap_t + l

        if pp_t < ap_t:
            reward = ((pp_t - zr1) / max(ap_t - zr1, 1e-8)) * 100.0
        else:
            reward = ((pp_t - zr2) / min(ap_t - zr2, -1e-8)) * 100.0

        # Update rolling predicted anchor
        self.prev_pred_price = pp_t

        self.idx += 1
        done = self.idx >= len(self.prices) - 1
        next_state = self._state(self.idx if not done else len(self.prices) - 1)
        return next_state, float(np.clip(reward, -100.0, 100.0)), done, {}
