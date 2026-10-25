"""Crank-Nicolson finite differences for the Black-Scholes PDE in log-price.

With x = ln S and tau = T - t (time to maturity), the BS PDE becomes constant-coefficient:

    V_tau = a V_xx + b V_x - r V,      a = sigma^2 / 2,   b = r - q - sigma^2 / 2

Discretisation (derivation and stability analysis: docs/derivations/03_crank_nicolson.md):
  - uniform grid in x with S0 exactly on a node, central differences  -> O(h^2)
  - theta-scheme in tau, theta = 1/2 (Crank-Nicolson)                 -> O(dt^2)
  - Rannacher start-up: the first two CN steps are replaced by four implicit
    Euler half-steps. CN is A-stable but not L-stable, so the high-frequency
    error from a non-smooth payoff is damped by a factor that tends to -1 and
    shows up as oscillations in delta/gamma near the strike. Implicit Euler
    kills those modes, and four half-steps keep the overall O(dt^2) rate.
  - payoff cell-averaged over each grid cell, so the kink position relative
    to the nodes does not spoil the clean second-order convergence.
  - American exercise by the penalty method (Forsyth & Vetzal, 2002).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import solve_banded


@dataclass
class PDEResult:
    price: float
    delta: float
    gamma: float
    S: np.ndarray       # spot grid
    V: np.ndarray       # option values on the grid at t = 0


def _cell_average(payoff: Callable, x: np.ndarray, h: float, n_sub: int = 32) -> np.ndarray:
    """(1/h) ∫_{x_j - h/2}^{x_j + h/2} payoff(e^y) dy by the midpoint rule on n_sub sub-cells."""
    offsets = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    y = x[:, None] + h * offsets[None, :]
    return payoff(np.exp(y)).mean(axis=1)


def solve(S0, T, r, q, sigma, payoff: Callable, lower: Callable, upper: Callable,
          x_min, x_max, M=400, N=400, american=False, rannacher=True,
          step_hook: Callable | None = None) -> PDEResult:
    """Generic solver. Products supply the payoff and the Dirichlet boundary values.

    payoff(S)        terminal payoff (vectorised)
    lower(tau, S)    value at x_min;  upper(tau, S) value at x_max
    step_hook(tau, S, V) optional: modify V after each step (e.g. discrete barrier monitoring)
    The grid is shifted so that ln S0 sits exactly on a node.
    """
    x0 = np.log(S0)
    h = (x_max - x_min) / M
    shift = (x0 - x_min) / h - round((x0 - x_min) / h)            # put x0 on a node
    x = x_min + (shift + np.arange(M + 1)) * h
    S = np.exp(x)
    i0 = int(round((x0 - x[0]) / h))

    a, b = 0.5 * sigma**2, r - q - 0.5 * sigma**2
    lo, di, up = a / h**2 - b / (2 * h), -2 * a / h**2 - r, a / h**2 + b / (2 * h)

    V = _cell_average(payoff, x, h)
    exercise = payoff(S) if american else None
    dt = T / N
    # Schedule of (theta, step size): Rannacher = 4 implicit half-steps, then CN.
    steps = [(1.0, dt / 2)] * 4 + [(0.5, dt)] * (N - 2) if rannacher else [(0.5, dt)] * N

    tau = 0.0
    n_int = M - 1
    for theta, k in steps:
        tau += k
        # Explicit part on the interior: (I + (1-theta) k L) V^n
        rhs = V[1:-1] + (1 - theta) * k * (lo * V[:-2] + di * V[1:-1] + up * V[2:])
        V_new_lo, V_new_hi = lower(tau, S[0]), upper(tau, S[-1])
        # Move the known boundary values of the implicit part to the right-hand side.
        rhs[0] += theta * k * lo * V_new_lo
        rhs[-1] += theta * k * up * V_new_hi
        # Implicit tridiagonal matrix (I - theta k L), stored in banded form.
        ab = np.zeros((3, n_int))
        ab[0, 1:] = -theta * k * up
        ab[1, :] = 1 - theta * k * di
        ab[2, :-1] = -theta * k * lo
        if american:
            V_int = _penalty_solve(ab, rhs, exercise[1:-1])
        else:
            V_int = solve_banded((1, 1), ab, rhs)
        V = np.concatenate(([V_new_lo], V_int, [V_new_hi]))
        if step_hook is not None:
            V = step_hook(tau, S, V)

    # Greeks from the grid: V_S = V_x / S,  V_SS = (V_xx - V_x) / S^2
    Vx = (V[i0 + 1] - V[i0 - 1]) / (2 * h)
    Vxx = (V[i0 + 1] - 2 * V[i0] + V[i0 - 1]) / h**2
    return PDEResult(price=V[i0], delta=Vx / S[i0], gamma=(Vxx - Vx) / S[i0] ** 2, S=S, V=V)


def _penalty_solve(ab, rhs, g, large=1e8, max_iter=50):
    """Solve the discrete LCP  min(A V - rhs, V - g) = 0  by penalty iteration.

    Add a huge diagonal weight wherever V < g, which forces V ≈ g there, and
    re-solve until the set of exercised nodes stops changing (usually 2-3 passes).
    Error from the penalty is O(1/large), far below the discretisation error.
    """
    V = solve_banded((1, 1), ab, rhs)
    active = V < g
    for _ in range(max_iter):
        ab_p = ab.copy()
        ab_p[1] += large * active
        V = solve_banded((1, 1), ab_p, rhs + large * active * g)
        new_active = V < g
        if np.array_equal(new_active, active):
            break
        active = new_active
    return V


def price_vanilla(S0, K, T, r, sigma, q=0.0, is_call=True, american=False,
                  M=400, N=400, n_sd=6.0, rannacher=True) -> PDEResult:
    """European or American call/put. Grid spans ln S0 ± n_sd standard deviations."""
    half = n_sd * sigma * np.sqrt(T)
    x_min, x_max = np.log(S0) - half, np.log(S0) + half
    if is_call:
        payoff = lambda S: np.maximum(S - K, 0.0)
        lower = lambda tau, S: 0.0
        upper = (lambda tau, S: S - K) if american else \
                (lambda tau, S: S * np.exp(-q * tau) - K * np.exp(-r * tau))
    else:
        payoff = lambda S: np.maximum(K - S, 0.0)
        lower = (lambda tau, S: K - S) if american else \
                (lambda tau, S: K * np.exp(-r * tau) - S * np.exp(-q * tau))
        upper = lambda tau, S: 0.0
    return solve(S0, T, r, q, sigma, payoff, lower, upper, x_min, x_max,
                 M=M, N=N, american=american, rannacher=rannacher)
