"""Delta-hedging P&L: simulation engine and daily P&L attribution.

Sign convention throughout: we are SHORT one option (the dealer's position after
selling it) and hold delta units of the underlying, financed in a cash account.

Daily P&L attribution for the hedged short option (docs/derivations/10_greeks_hedging.md):
    P&L ≈ ½ Γ S² (σ_imp² dt − (dS/S)²)      gamma/theta: implied vs realised variance
        − vega · dσ_imp                       vega: the market re-marks implied vol
        − vanna · dS · dσ_imp − ½ volga · dσ_imp²   second-order spot-vol and vol-vol terms
        + residual                            third-order terms, data noise
The first line comes from the Black-Scholes PDE: for a delta-hedged, self-financed
book, theta and financing combine into ½ Γ S² σ_imp² dt.
"""
from __future__ import annotations

import numpy as np

from scipy.stats import norm

from derivpricing.models import black_scholes as bs


def simulate_short_call_hedge(paths, K, T, r, q, sigma_hedge, cost=0.0):
    """Short one call sold at the BS price with vol sigma_hedge, delta-hedged at every column
    of `paths` (shape (n_paths, n + 1)) with BS delta at sigma_hedge.

    cost: proportional transaction cost on the traded notional (e.g. 2e-4 = 2 bp).
    Returns the terminal P&L per path, discounted to t = 0.
    """
    n_paths, n_cols = paths.shape
    dt = T / (n_cols - 1)
    S = paths[:, 0]
    premium = bs.price(S, K, T, r, sigma_hedge, q)
    delta = bs.greeks(S, K, T, r, sigma_hedge, q)["delta"]
    cash = premium - delta * S - cost * np.abs(delta) * S
    for i in range(1, n_cols):
        S_new = paths[:, i]
        cash = cash * np.exp(r * dt) + delta * S_new * (np.exp(q * dt) - 1)   # interest, dividends
        tau = T - i * dt
        if i < n_cols - 1:
            new_delta = bs.greeks(S_new, K, tau, r, sigma_hedge, q)["delta"]
        else:
            new_delta = np.zeros(n_paths)                                       # unwind at expiry
        cash -= (new_delta - delta) * S_new + cost * np.abs(new_delta - delta) * S_new
        delta = new_delta
    pnl_T = cash - np.maximum(paths[:, -1] - K, 0.0)
    return pnl_T * np.exp(-r * T)


def attribute_day(S0, S1, F0, D0, K, T0, dt, iv0, iv1, V0, V1, is_call=True):
    """Explain one day's P&L of a short, delta-hedged option from market data.

    Prices are Black-76 on the forward: S_eff = D F, r = -ln D / T. The hedge is in the
    index itself; dF/dS = F/S, so the index delta is the forward delta times F/S.
    Returns dict(total, gamma_theta, vega, residual, delta_hedge_units).
    """
    r = -np.log(D0) / T0
    q = r - np.log(F0 / S0) / T0                # dividend yield implied by the forward
    S_eff0 = D0 * F0
    g = bs.greeks(S_eff0, K, T0, r, iv0, 0.0, is_call)
    delta_S = g["delta"] * S_eff0 / S0          # dV/dS (S_eff moves 1:1 in % with S)
    gamma_S = g["gamma"] * (S_eff0 / S0) ** 2
    ret = S1 / S0 - 1.0
    # short option, long delta_S index, cash financed at r; dividends are inside F/D
    total = -(V1 - V0) + delta_S * (S1 - S0) + r * (V0 - delta_S * S0) * dt + q * delta_S * S0 * dt
    gamma_theta = 0.5 * gamma_S * S0**2 * (iv0**2 * dt - ret**2)
    dvol = iv1 - iv0
    vega = -g["vega"] * dvol
    # second-order cross terms: vanna (spot x vol) and volga (vol x vol)
    v = iv0 * np.sqrt(T0)
    d1 = (np.log(F0 / K) + 0.5 * v**2) / v
    d2 = d1 - v
    # V = D Black(F, K, sigma):  d2V/dF dsigma = D n(d1) dd1/dsigma = -D n(d1) d2 / sigma,
    # and F moves proportionally with S (dF/dS = F/S)
    vanna_S = -(F0 / S0) * D0 * norm.pdf(d1) * d2 / iv0
    volga = g["vega"] * d1 * d2 / iv0
    second_order = -(vanna_S * (S1 - S0) * dvol + 0.5 * volga * dvol**2)
    return {"total": total, "gamma_theta": gamma_theta, "vega": vega, "vanna_volga": second_order,
            "residual": total - gamma_theta - vega - second_order, "delta_hedge_units": delta_S}
