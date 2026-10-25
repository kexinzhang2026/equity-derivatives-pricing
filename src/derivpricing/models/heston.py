"""Heston stochastic volatility model.

Risk-neutral dynamics:
    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t,     d<W^S, W^v>_t = rho dt

kappa: speed of mean reversion   theta: long-run variance   xi: vol of variance
rho:   spot/vol correlation (negative for equity -> downward skew)   v0: initial variance

Feller condition 2 kappa theta >= xi^2 keeps v_t strictly positive. Calibrated
equity parameters usually violate it, which is why the Monte Carlo engine uses
the QE scheme rather than a naive Euler step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    @property
    def feller_ratio(self) -> float:
        """2 kappa theta / xi^2; >= 1 means the variance process cannot hit zero."""
        return 2 * self.kappa * self.theta / self.xi**2


def char_func(u, T, r, q, p: HestonParams):
    """phi(u) = E[exp(i u X_T)],  X_T = ln(S_T / S_0).

    Written in the "little Heston trap" form (Albrecher et al., 2007): it uses
    g = (b - d)/(b + d) and e^{-dT}, so the complex log never crosses its branch
    cut for long maturities. Heston's original form (with e^{+dT}) does.
    Derivation: docs/derivations/02_heston_cos.md.
    """
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    b = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(b**2 + p.xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = (p.kappa * p.theta / p.xi**2) * ((b - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
    D = ((b - d) / p.xi**2) * (1 - e) / (1 - g * e)
    return np.exp(iu * (r - q) * T + C + D * p.v0)
