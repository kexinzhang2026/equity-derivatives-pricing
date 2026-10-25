"""Load option quotes from different vendors into one common schema.

Common schema (one row per quote):
    quote_date  datetime64   date the quote was observed
    expiry      datetime64   option expiration date
    strike      float
    is_call     bool
    bid, ask    float
    volume, open_interest   float (may be NaN)
    spot        float        underlying level at quote time (NaN if vendor does not give it)

Calibration code only ever sees this schema, so switching from daily yfinance
snapshots to OptionMetrics history changes nothing downstream.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["quote_date", "expiry", "strike", "is_call", "bid", "ask",
           "volume", "open_interest", "spot"]


def load_yfinance_snapshots(folder: str | Path) -> pd.DataFrame:
    """Read every CSV written by scripts/fetch_spx_chain.py (already in the common schema)."""
    files = sorted(Path(folder).glob("spx_*.csv"))
    if not files:
        raise FileNotFoundError(f"no spx_*.csv snapshots in {folder}")
    df = pd.concat([pd.read_csv(f, parse_dates=["quote_date", "expiry"]) for f in files],
                   ignore_index=True)
    return df[COLUMNS]


def load_optionmetrics(path: str | Path, spot: pd.Series | None = None) -> pd.DataFrame:
    """Map an OptionMetrics IvyDB `opprcd` extract (CSV from WRDS) to the common schema.

    OptionMetrics conventions handled here:
      - strike_price is quoted in strike x 1000
      - cp_flag is 'C' / 'P'
      - best_bid / best_offer are end-of-day quotes
    `spot` (optional) is a Series of underlying closes indexed by date, e.g. from the
    `secprd` table; OptionMetrics option files do not carry the spot themselves.
    """
    raw = pd.read_csv(path, parse_dates=["date", "exdate"])
    df = pd.DataFrame({
        "quote_date": raw["date"],
        "expiry": raw["exdate"],
        "strike": raw["strike_price"] / 1000.0,
        "is_call": raw["cp_flag"].str.upper().eq("C"),
        "bid": raw["best_bid"],
        "ask": raw["best_offer"],
        "volume": raw.get("volume"),
        "open_interest": raw.get("open_interest"),
    })
    df["spot"] = df["quote_date"].map(spot) if spot is not None else float("nan")
    return df[COLUMNS]
