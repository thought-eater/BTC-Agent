import numpy as np

from models.main_dqn.model import build_main_dqn
from models.predictive_dqn.model import build_predictive_dqn
from models.trade_dqn.model import build_trade_dqn


def test_trade_shape():
    model = build_trade_dqn()
    y = model(np.zeros((2, 1), dtype=np.float32))
    assert y.shape == (2, 3)


def test_predictive_shape():
    model = build_predictive_dqn()
    y = model(np.zeros((2, 2), dtype=np.float32))
    assert y.shape == (2, 20001)


def test_main_shape():
    model = build_main_dqn()
    y = model(np.zeros((2, 2), dtype=np.float32))
    assert y.shape == (2, 3)
