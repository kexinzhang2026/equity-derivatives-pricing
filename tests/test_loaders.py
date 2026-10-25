"""Vendor loaders must map into the common schema identically."""
import pandas as pd

from derivpricing.data import loaders


def test_optionmetrics_mapping(tmp_path):
    raw = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02"], "exdate": ["2024-03-15", "2024-03-15"],
        "cp_flag": ["C", "P"], "strike_price": [4750000, 4500000],
        "best_bid": [120.5, 30.1], "best_offer": [121.5, 30.9],
        "volume": [10, 5], "open_interest": [100, 50],
    })
    path = tmp_path / "om.csv"
    raw.to_csv(path, index=False)
    spot = pd.Series({pd.Timestamp("2024-01-02"): 4742.83})
    df = loaders.load_optionmetrics(path, spot)
    assert list(df.columns) == loaders.COLUMNS
    assert df["strike"].tolist() == [4750.0, 4500.0]          # strike_price is x1000
    assert df["is_call"].tolist() == [True, False]
    assert (df["spot"] == 4742.83).all()


def test_yfinance_snapshot_roundtrip(tmp_path):
    snap = pd.DataFrame({
        "quote_date": ["2024-01-02"], "expiry": ["2024-03-15"], "strike": [4750.0],
        "is_call": [True], "bid": [120.5], "ask": [121.5], "volume": [10],
        "open_interest": [100], "spot": [4742.83],
    })
    snap.to_csv(tmp_path / "spx_20240102.csv", index=False)
    df = loaders.load_yfinance_snapshots(tmp_path)
    assert list(df.columns) == loaders.COLUMNS and len(df) == 1
    assert pd.api.types.is_datetime64_any_dtype(df["expiry"])
