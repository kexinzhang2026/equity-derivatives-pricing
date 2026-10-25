"""From raw option quotes to a clean implied volatility surface.

Steps, per quote date and expiry:
  1. Basic filters: two-sided market, positive mid, not-too-wide spread, T >= min_T.
  2. Implied forward F and discount factor D from put-call parity. For every strike
     with both a call and a put,  C - P = D (F - K)  is linear in K, so a regression
     of (C - P) on K over near-the-money strikes gives slope -D and intercept D F.
     This avoids trusting a (possibly stale) spot, an assumed dividend yield, or a
     rate curve: everything the options need is implied by the options themselves.
  3. Keep the out-of-the-money option at each strike (puts below F, calls above).
     ITM options carry the same information via parity but have small vega, so
     their IV is noisy (see docs/derivations/01_black_scholes.md).
  4. Invert mid, bid and ask to implied vols. The bid/ask IV band says how well
     a model can be expected to fit each point.

Output columns: quote_date, expiry, T, K, F, D, k = ln(K/F), is_call, bid, ask,
mid, iv, iv_bid, iv_ask, vega.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from derivpricing.models import black_scholes as bs


def implied_forward(chain: pd.DataFrame, spot_guess: float, band: float = 0.1):
    """Fit C - P = D (F - K) on strikes within ±band of spot_guess. Returns (F, D, n_pairs)."""
    calls = chain[chain["is_call"]].set_index("strike")["mid"]
    puts = chain[~chain["is_call"]].set_index("strike")["mid"]
    both = calls.index.intersection(puts.index)
    both = both[np.abs(both / spot_guess - 1) <= band]
    if len(both) < 3:
        return np.nan, np.nan, len(both)
    K = both.to_numpy(float)
    y = (calls[both] - puts[both]).to_numpy(float)
    slope, intercept = np.polyfit(K, y, 1)
    D = -slope
    return intercept / D, D, len(both)


def build_surface(quotes: pd.DataFrame, min_T: float = 7 / 365, max_rel_spread: float = 0.5,
                  k_range: tuple[float, float] = (-4.0, 2.5), atm_vol_guess: float = 0.2,
                  min_mid: float = 0.05) -> pd.DataFrame:
    """Clean quotes (common schema, see data/loaders.py) into an OTM implied vol surface.

    k_range bounds the standardised log-moneyness ln(K/F) / (atm_vol_guess sqrt(T)):
    by default from 4 "standard deviations" below the forward to 2.5 above, where
    SPX quotes are reasonably liquid. Deeper wings are mostly stale or one-sided.
    """
    q = quotes.copy()
    q["mid"] = 0.5 * (q["bid"] + q["ask"])
    q = q[(q["bid"] > 0) & (q["ask"] > q["bid"]) & (q["mid"] >= min_mid)]
    q = q[(q["ask"] - q["bid"]) / q["mid"] <= max_rel_spread]
    # ACT/365 to the expiry date. (SPX standard monthlies settle at the open, weeklies at
    # the close; the ~6.5h difference matters only for the shortest expiries, cut by min_T.)
    q["T"] = (q["expiry"] - q["quote_date"]).dt.days / 365.0
    q = q[q["T"] >= min_T]

    out = []
    for (qd, exp), chain in q.groupby(["quote_date", "expiry"]):
        spot_guess = chain["spot"].iloc[0]
        F, D, n = implied_forward(chain, spot_guess)
        if not np.isfinite(F) or not (0.5 < D <= 1.0 + 1e-3):
            continue
        T = chain["T"].iloc[0]
        otm = chain[(chain["is_call"] & (chain["strike"] >= F)) |
                    (~chain["is_call"] & (chain["strike"] < F))].copy()
        otm["F"], otm["D"] = F, D
        out.append(otm)
    if not out:
        return pd.DataFrame()
    s = pd.concat(out, ignore_index=True).rename(columns={"strike": "K"})
    s["k"] = np.log(s["K"] / s["F"])
    z = s["k"] / (atm_vol_guess * np.sqrt(s["T"]))
    s = s[(z >= k_range[0]) & (z <= k_range[1])].copy()

    # Black-76 in BS clothing: S_eff = D F, r = -ln(D)/T, q = 0 reproduces forward F.
    S_eff, r = s["D"] * s["F"], -np.log(s["D"]) / s["T"]
    args = (S_eff.to_numpy(), s["K"].to_numpy(), s["T"].to_numpy(), r.to_numpy(), 0.0,
            s["is_call"].to_numpy())
    s["iv"] = bs.implied_vol_vec(s["mid"].to_numpy(), *args)
    s["iv_bid"] = bs.implied_vol_vec(s["bid"].to_numpy(), *args)
    s["iv_ask"] = bs.implied_vol_vec(s["ask"].to_numpy(), *args)
    s["vega"] = bs.greeks(args[0], args[1], args[2], args[3], s["iv"].to_numpy(), 0.0)["vega"]
    s = s[np.isfinite(s["iv"])]
    cols = ["quote_date", "expiry", "T", "K", "F", "D", "k", "is_call", "bid", "ask", "mid",
            "iv", "iv_bid", "iv_ask", "vega"]
    return s[cols].sort_values(["quote_date", "T", "K"]).reset_index(drop=True)


def calibration_subset(surf: pd.DataFrame, tenors=(1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0),
                       strikes_per_expiry: int = 25) -> pd.DataFrame:
    """Pick one listed expiry nearest to each target tenor and thin each smile to
    `strikes_per_expiry` strikes evenly spaced in log-moneyness.

    SPX lists 40+ expiries and hundreds of strikes per day. Weekly expiries crowd the
    short end, so fitting all of them would let the first month dominate the objective;
    a fixed tenor grid also makes fits on different dates comparable.
    """
    expiries = np.sort(surf["T"].unique())
    chosen = sorted({expiries[np.argmin(np.abs(expiries - t))] for t in tenors})
    parts = []
    for T in chosen:
        sl = surf[surf["T"] == T].sort_values("k")
        if len(sl) > strikes_per_expiry:
            targets = np.linspace(sl["k"].min(), sl["k"].max(), strikes_per_expiry)
            idx = np.unique([np.argmin(np.abs(sl["k"].to_numpy() - t)) for t in targets])
            sl = sl.iloc[idx]
        parts.append(sl)
    return pd.concat(parts, ignore_index=True)
