"""Monte Carlo: path generators and estimators.

Every estimator returns the price *and* its standard error, because a Monte Carlo
number without an error bar cannot be compared with anything.
Derivations (variance reduction, QE scheme): docs/derivations/04_monte_carlo.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from derivpricing.models.heston import HestonParams


@dataclass
class MCResult:
    price: float
    stderr: float

    def contains(self, value, n_se=3.0) -> bool:
        """True if value lies within n_se standard errors of the estimate."""
        return abs(self.price - value) <= n_se * self.stderr


def _normals(rng, shape, antithetic):
    """Standard normals; with antithetic=True the second half is the negated first half."""
    if not antithetic:
        return rng.standard_normal(shape)
    half = rng.standard_normal((shape[0] // 2,) + tuple(shape[1:]))
    return np.concatenate([half, -half])


# ---------------------------------------------------------------- path generators

def gbm_paths(S0, T, r, q, sigma, n_steps, n_paths, rng, antithetic=False):
    """Exact GBM on a uniform grid: ln S_{t+dt} = ln S_t + (r-q-sigma^2/2)dt + sigma sqrt(dt) Z.

    No discretisation bias at any step size (the log-increments are exactly Gaussian).
    Returns an array of shape (n_paths, n_steps + 1) including S0.
    """
    dt = T / n_steps
    Z = _normals(rng, (n_paths, n_steps), antithetic)
    log_inc = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.log(S0) + np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    return np.exp(log_S)


def heston_paths_qe(S0, T, r, q, p: HestonParams, n_steps, n_paths, rng, psi_c=1.5):
    """Heston paths with Andersen's (2008) Quadratic-Exponential scheme.

    Variance step: match the first two conditional moments m, s^2 of v_{t+dt} | v_t
    (known in closed form for the CIR process) with
      - a scaled non-central chi-square proxy  a (b + Z)^2      if psi = s^2/m^2 <= psi_c
      - a point mass at 0 plus an exponential tail              otherwise (v near 0)
    so v stays non-negative even when the Feller condition fails, where a plain
    Euler step would produce negative variances.

    Log-spot step: integrate ln S exactly given the variance path, approximating
    ∫ v ds by the trapezoid rule (gamma1 = gamma2 = 1/2) and eliminating the
    correlated Brownian part through the variance SDE.
    Returns (S, v), each of shape (n_paths, n_steps + 1).
    """
    dt = T / n_steps
    k, th, xi, rho = p.kappa, p.theta, p.xi, p.rho
    e = np.exp(-k * dt)
    g1 = g2 = 0.5
    K0 = -rho * k * th * dt / xi
    K1 = g1 * dt * (k * rho / xi - 0.5) - rho / xi
    K2 = g2 * dt * (k * rho / xi - 0.5) + rho / xi
    K3 = g1 * dt * (1 - rho**2)
    K4 = g2 * dt * (1 - rho**2)

    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0], v[:, 0] = S0, p.v0
    log_S = np.full(n_paths, np.log(S0))
    for n in range(n_steps):
        vn = v[:, n]
        m = th + (vn - th) * e
        s2 = vn * xi**2 * e * (1 - e) / k + th * xi**2 * (1 - e) ** 2 / (2 * k)
        psi = s2 / m**2

        v_next = np.empty(n_paths)
        quad = psi <= psi_c
        # quadratic branch
        inv = 2.0 / psi[quad]
        b2 = inv - 1 + np.sqrt(inv) * np.sqrt(inv - 1)
        a = m[quad] / (1 + b2)
        v_next[quad] = a * (np.sqrt(b2) + rng.standard_normal(quad.sum())) ** 2
        # exponential branch (inverse CDF of the mixture)
        ex = ~quad
        p_mass = (psi[ex] - 1) / (psi[ex] + 1)
        beta = (1 - p_mass) / m[ex]
        U = rng.random(ex.sum())
        v_next[ex] = np.where(U <= p_mass, 0.0,
                              np.log((1 - p_mass) / np.maximum(1 - U, 1e-300)) / beta)

        Zs = rng.standard_normal(n_paths)
        log_S = (log_S + (r - q) * dt + K0 + K1 * vn + K2 * v_next
                 + np.sqrt(K3 * vn + K4 * v_next) * Zs)
        v[:, n + 1] = v_next
        S[:, n + 1] = np.exp(log_S)
    return S, v


# ---------------------------------------------------------------- estimators

def plain(discounted_payoffs) -> MCResult:
    """Sample mean and stderr = sample std / sqrt(n)."""
    y = np.asarray(discounted_payoffs)
    return MCResult(y.mean(), y.std(ddof=1) / np.sqrt(y.size))


def antithetic(discounted_payoffs) -> MCResult:
    """For antithetic samples the two halves are dependent, so average each pair first;
    the pair means are i.i.d. and give an honest standard error."""
    y = np.asarray(discounted_payoffs)
    half = y.size // 2
    return plain(0.5 * (y[:half] + y[half:2 * half]))


def control_variate(Y, X, EX) -> MCResult:
    """Y_cv = Y - beta (X - E[X]),  beta* = Cov(Y, X) / Var(X).

    Variance shrinks by the factor (1 - corr(Y, X)^2). Estimating beta from the
    same samples adds an O(1/n) bias, negligible at the sample sizes used here.
    """
    Y, X = np.asarray(Y), np.asarray(X)
    beta = np.cov(Y, X)[0, 1] / X.var(ddof=1)
    return plain(Y - beta * (X - EX))
