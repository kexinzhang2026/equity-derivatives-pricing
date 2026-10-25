"""Down barrier puts: closed form (Black-Scholes, continuous monitoring), PDE and MC pricers.

The down-and-in put is the building block of an autocallable's capital-at-risk leg,
so it is validated here first, where an exact answer exists.

Closed form: Reiner & Rubinstein (1991), in the notation of Haug (2007), "The
Complete Guide to Option Pricing Formulas", Sec. 4.17.1. Derivation sketch
(reflection principle + Girsanov): docs/derivations/09_barriers_autocallables.md.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from derivpricing.engines import pde
from derivpricing.engines.monte_carlo import MCResult, plain
from derivpricing.models import black_scholes as bs

BGK_BETA = 0.5826   # -zeta(1/2)/sqrt(2*pi), Broadie-Glasserman-Kou (1997)


def down_and_out_put_bs(S, K, H, T, r, sigma, q=0.0):
    """Continuously monitored down-and-out put, H < K, H < S, no rebate.

    Haug's building blocks with phi = -1 (put), eta = 1 (down barrier):
        P_do = A - B + C - D,   P_di = B - C + D,   A = vanilla put.
    """
    b = r - q
    v = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    x1 = np.log(S / K) / v + (1 + mu) * v
    x2 = np.log(S / H) / v + (1 + mu) * v
    y1 = np.log(H**2 / (S * K)) / v + (1 + mu) * v
    y2 = np.log(H / S) / v + (1 + mu) * v
    Sq, Kr = S * np.exp((b - r) * T), K * np.exp(-r * T)
    hs1, hs2 = (H / S) ** (2 * (mu + 1)), (H / S) ** (2 * mu)
    A = -Sq * norm.cdf(-x1) + Kr * norm.cdf(-x1 + v)
    B = -Sq * norm.cdf(-x2) + Kr * norm.cdf(-x2 + v)
    C = -Sq * hs1 * norm.cdf(y1) + Kr * hs2 * norm.cdf(y1 - v)
    D = -Sq * hs1 * norm.cdf(y2) + Kr * hs2 * norm.cdf(y2 - v)
    return A - B + C - D


def down_and_in_put_bs(S, K, H, T, r, sigma, q=0.0):
    """In-out parity: knock-in + knock-out = vanilla (exactly one of them pays)."""
    return bs.price(S, K, T, r, sigma, q, is_call=False) - down_and_out_put_bs(S, K, H, T, r, sigma, q)


def bgk_shift(H, sigma, dt):
    """Continuous-barrier level that approximates discrete monitoring every dt (down barrier):
    H_eff = H exp(-beta sigma sqrt(dt)). Discrete monitoring knocks less often, which is the
    same as a continuous barrier slightly further away."""
    return H * np.exp(-BGK_BETA * sigma * np.sqrt(dt))


def down_and_out_put_pde(S, K, H, T, r, sigma, q=0.0, M=400, N=400):
    """Continuous monitoring: the barrier is the lower grid boundary with V = 0 there.
    sigma may be a local-vol function sigma(t, S)."""
    sig_ref = sigma if np.isscalar(sigma) else 0.3
    x_max = np.log(S) + 6 * sig_ref * np.sqrt(T)
    return pde.solve(S, T, r, q, sigma, lambda s: np.maximum(K - s, 0.0),
                     lambda tau, s: 0.0, lambda tau, s: 0.0, np.log(H), x_max, M=M, N=N,
                     barrier_at_lower=True)


def down_and_in_put_mc(paths, K, H, r, T) -> MCResult:
    """Discretely monitored on the simulation grid (each column of paths is a monitoring date)."""
    knocked = paths.min(axis=1) <= H
    return plain(np.exp(-r * T) * np.where(knocked, np.maximum(K - paths[:, -1], 0.0), 0.0))
