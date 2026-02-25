import pandas as pd

from preprocessing.data_integrator import build_base_dataset, resolve_adapted_window


def test_window_and_split_lengths():
    ts = pd.date_range("2020-01-01", periods=40000, freq="H", tz="UTC")
    btc = pd.DataFrame({"timestamp": ts, "close": range(len(ts))})
    sent = pd.DataFrame({"timestamp": ts, "sentiment": 0.1})

    window = resolve_adapted_window(btc, sent, overlap_days=1505, test_hours=720)
    base = build_base_dataset(btc, sent, window)

    assert len(base) == 36120
    assert (base["split"] == "test").sum() == 720
