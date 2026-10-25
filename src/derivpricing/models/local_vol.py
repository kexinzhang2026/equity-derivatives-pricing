"""Dupire local volatility built from an SVI implied-variance surface.

Dupire (1994): there is exactly one diffusion dS = (r-q) S dt + sigma_LV(t, S) S dW
that reprices every European option on the surface. In terms of total implied
variance w(k, T) = sigma_imp^2 T and log-moneyness k = ln(K / F_T) (Gatheral 2006):

    sigma_LV^2(T, K) = (dw/dT) / g(k, w),
    g = (1 - k w_k / (2w))^2 - (w_k^2 / 4)(1/w + 1/4) + w_kk / 2

Derivation: docs/derivations/08_dupire.md. Numerator >= 0 is "no calendar
arbitrage"; g >= 0 is "no butterfly arbitrage" (g is proportional to the density).

Construction: one SVI fit per listed expiry, total variance interpolated
*linearly in T at fixed k* between expiries. Linear interpolation commutes with
d/dk, keeps w increasing in T if the slices are, and makes dw/dT piecewise
constant, so sigma_LV is piecewise constant in time between expiries.
"""
from __future__ import annotations

import numpy as np

from derivpricing.models import svi


class LocalVolSurface:
    def __init__(self, expiries, svi_params, S0, r, q, sigma_bounds=(0.01, 3.0)):
        order = np.argsort(expiries)
        self.T = np.asarray(expiries, float)[order]
        self.params = [svi_params[i] for i in order]
        self.S0, self.r, self.q = S0, r, q
        self.lo, self.hi = sigma_bounds

    def _slices(self, k):
        """w, w_k, w_kk of every SVI slice at the log-moneyness array k: shape (n_T, len(k))."""
        W = np.array([svi.w(k, *p) for p in self.params])
        W1 = np.array([svi.dw(k, *p) for p in self.params])
        W2 = np.array([svi.d2w(k, *p) for p in self.params])
        return W, W1, W2

    def total_variance(self, k, T):
        """Interpolated w(k, T) and its derivatives (w, w_k, w_kk, w_T)."""
        k = np.atleast_1d(np.asarray(k, float))
        W, W1, W2 = self._slices(k)
        if T <= self.T[0]:
            # before the first expiry: w scales linearly from 0 (constant implied vol per k)
            f = T / self.T[0]
            return f * W[0], f * W1[0], f * W2[0], W[0] / self.T[0]
        if T >= self.T[-1]:
            f = T / self.T[-1]
            return f * W[-1], f * W1[-1], f * W2[-1], W[-1] / self.T[-1]
        i = np.searchsorted(self.T, T) - 1
        lam = (T - self.T[i]) / (self.T[i + 1] - self.T[i])
        mix = lambda A: (1 - lam) * A[i] + lam * A[i + 1]
        wT = (W[i + 1] - W[i]) / (self.T[i + 1] - self.T[i])
        return mix(W), mix(W1), mix(W2), wT

    def implied_vol(self, K, T):
        F = self.S0 * np.exp((self.r - self.q) * T)
        w, *_ = self.total_variance(np.log(np.asarray(K) / F), T)
        return np.sqrt(w / T)

    def sigma(self, t, S):
        """sigma_LV(t, S), vectorised in S. This is the function the PDE and MC engines call."""
        t = max(t, 1e-6)
        F = self.S0 * np.exp((self.r - self.q) * t)
        k = np.log(np.asarray(S, float) / F)
        w, w1, w2, wT = self.total_variance(k, t)
        g = (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2
        var = np.where(g > 1e-8, wT / np.maximum(g, 1e-8), self.hi**2)
        return np.clip(np.sqrt(np.maximum(var, 0.0)), self.lo, self.hi)

    @classmethod
    def from_surface(cls, surface, S0, r, q, **kw):
        """Fit SVI to each expiry of a surface DataFrame (columns T, k, iv) and build the LV.

        Assumes constant r, q so that k = ln(K/F_T) with F_T = S0 e^{(r-q)T}. Market
        surfaces carry their own implied forwards; the caller passes matching r, q.
        """
        expiries, params = [], []
        for T, sl in surface.groupby("T"):
            expiries.append(T)
            params.append(svi.fit(sl["k"].to_numpy(), (sl["iv"] ** 2 * T).to_numpy()))
        return cls(expiries, params, S0, r, q, **kw)
