"""COS method (Fang & Oosterlee, 2008) for European options.

Idea in one line: on a truncated interval [a, b] the density of y = ln(S_T/K) is
expanded in a cosine series whose coefficients come straight from the
characteristic function, and the payoff's cosine coefficients are known in
closed form, so the price is a finite sum:

    V ≈ e^{-rT} Σ'_{k=0}^{N-1} Re[ φ(u_k) e^{i u_k (x - a)} ] V_k,   u_k = kπ/(b-a)

x = ln(S0/K), φ = CF of ln(S_T/S0), Σ' halves the k=0 term,
V_k = (2/(b-a)) ∫_a^b payoff(y) cos(u_k (y-a)) dy.

For smooth densities the error decays exponentially in N.
Derivation: docs/derivations/02_heston_cos.md.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def cumulants(cf: Callable, h: float = 0.05):
    """First, second and fourth cumulants of X = ln(S_T/S0) from its CF.

    The cumulant generating function is K(s) = ln E[e^{sX}] = ln φ(-i s), and
    c_n = K^{(n)}(0). We take central finite differences of K at small real s,
    which works for any model that supplies a CF (no model-specific formulas).
    K(0) = 0 exactly. Truncation only needs rough values, so h = 0.05 is plenty.
    """
    K = lambda s: np.log(cf(-1j * s)).real
    k1, km1, k2, km2 = K(h), K(-h), K(2 * h), K(-2 * h)
    c1 = (k1 - km1) / (2 * h)
    c2 = (k1 + km1) / h**2
    c4 = (k2 - 4 * k1 - 4 * km1 + km2) / h**4
    return c1, c2, c4


def _chi_psi(a, b, c, d, k):
    """Closed-form cosine coefficients of e^y and 1 on [c, d] ⊂ [a, b].

    chi_k = ∫_c^d e^y cos(u_k (y-a)) dy,   psi_k = ∫_c^d cos(u_k (y-a)) dy.
    """
    u = k * np.pi / (b - a)
    arg_d, arg_c = u * (d - a), u * (c - a)
    chi = (np.cos(arg_d) * np.exp(d) - np.cos(arg_c) * np.exp(c)
           + u * (np.sin(arg_d) * np.exp(d) - np.sin(arg_c) * np.exp(c))) / (1 + u**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(k == 0, d - c, (np.sin(arg_d) - np.sin(arg_c)) / np.where(k == 0, 1, u))
    return chi, psi


def price(cf: Callable, S0, K, T, r, q=0.0, is_call=True, N: int = 256, L: float = 10.0):
    """European price from a characteristic function.

    cf: u -> E[exp(i u ln(S_T/S0))], with T, r, q and model parameters already bound.
    K may be an array (a whole smile prices in one vectorised call).

    We price the put and get the call from put-call parity. The put payoff
    K(1 - e^y)^+ is bounded, so truncating the far right tail costs nothing;
    a call payoff grows like e^y and makes the result sensitive to b (F&O, Sec. 3).
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))[:, None]          # shape (nK, 1)
    c1, c2, c4 = cumulants(cf)
    half_width = L * np.sqrt(c2 + np.sqrt(abs(c4)))
    x = np.log(S0 / K)
    a, b = x + c1 - half_width, x + c1 + half_width                   # interval for y = x + X
    k = np.arange(N)[None, :]                                         # shape (1, N)

    # Put payoff is non-zero for y < 0: integrate over [a, min(b, 0)].
    upper = np.minimum(b, 0.0)
    chi, psi = _chi_psi(a, b, a, upper, k)
    Vk = 2.0 / (b - a) * K * (psi - chi)
    Vk = np.where(upper > a, Vk, 0.0)

    u = k * np.pi / (b - a)
    terms = (cf(u) * np.exp(1j * u * (x - a))).real * Vk
    terms[:, 0] *= 0.5
    put = np.exp(-r * T) * terms.sum(axis=1)
    out = put + S0 * np.exp(-q * T) - K[:, 0] * np.exp(-r * T) if is_call else put
    return out if out.size > 1 else out[0]
