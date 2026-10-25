"""Black-Scholes-Merton closed form for European options.

Model (risk-neutral measure Q), continuous dividend yield q:
    dS_t = (r - q) S_t dt + sigma S_t dW_t

Everything is written in terms of the forward F = S e^{(r-q)T} and the
discount factor D = e^{-rT}, because that form is what the Fourier (COS),
PDE and Monte Carlo engines are later checked against, and it makes
put-call parity a one-line identity:  C - P = D (F - K).

Derivation of the price: see docs/derivations/01_black_scholes.md.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, q, sigma):
    """d1 = [ln(F/K) + sigma^2 T / 2] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)."""
    F = S * np.exp((r - q) * T)
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(S, K, T, r, sigma, q=0.0, is_call=True):
    """European option price.

    Call = D [F N(d1) - K N(d2)]
    Put  = D [K N(-d2) - F N(-d1)]
    Inputs broadcast as numpy arrays, so a whole strike grid prices in one call.
    """
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    F = S * np.exp((r - q) * T)
    D = np.exp(-r * T)
    call = D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(is_call, call, put)


def greeks(S, K, T, r, sigma, q=0.0, is_call=True):
    """Analytic Greeks. Theta is per year (dV/dt, i.e. -dV/dT); vega and rho per unit (not per 1%).

    The identity  S e^{-qT} n(d1) = K e^{-rT} n(d2)  is what makes these short:
    the terms from differentiating d1 and d2 cancel.
    """
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrt_t = np.sqrt(T)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)          # same for call and put
    vega = S * disc_q * pdf_d1 * sqrt_t                      # same for call and put
    if is_call:
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
                 + q * S * disc_q * norm.cdf(d1) - r * K * disc_r * norm.cdf(d2))
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
                 - q * S * disc_q * norm.cdf(-d1) + r * K * disc_r * norm.cdf(-d2))
        rho = -K * T * disc_r * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(target_price, S, K, T, r, q=0.0, is_call=True, lo=1e-6, hi=5.0):
    """Invert price(sigma) = target with Brent's method.

    The price is strictly increasing in sigma (vega > 0), so a root in [lo, hi]
    is unique whenever the target lies inside the no-arbitrage bounds
    D*max(F-K, 0) < C < S e^{-qT}. Outside them we return NaN rather than a
    garbage vol; market data cleaning relies on that.
    """
    f = lambda s: price(S, K, T, r, s, q, is_call) - target_price
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-12, maxiter=200)


def char_func(u, T, r, q, sigma):
    """phi(u) = E[exp(i u X_T)] with X_T = ln(S_T/S_0) ~ N((r - q - sigma^2/2) T, sigma^2 T).

    Used to validate the COS engine against the closed form above.
    """
    u = np.asarray(u, dtype=complex)
    return np.exp(1j * u * (r - q - 0.5 * sigma**2) * T - 0.5 * sigma**2 * u**2 * T)
