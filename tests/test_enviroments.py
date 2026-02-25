import numpy as np

from models.main_dqn.enviroment import MainDQNEnvironment


def test_main_environment_step():
    x1 = np.array([0, 1, -1, 0], dtype=np.float32)
    x2 = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)
    prices = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float32)

    env = MainDQNEnvironment(
        x1=x1,
        x2=x2,
        prices=prices,
        initial_investment=1000.0,
        transaction_fee=0.015,
        alpha=0.55,
        omega=16,
        reward_mode="reward_v2",
    )
    state = env.reset()
    assert state.shape == (2,)
    next_state, reward, done, info = env.step(2)
    assert next_state.shape == (2,)
    assert isinstance(reward, float)
    assert "portfolio_value" in info
    assert done is False
