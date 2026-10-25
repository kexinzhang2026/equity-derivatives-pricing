"""Autocallable note on one index, priced by Monte Carlo (any model) and PDE (BS / local vol).

Term sheet (all levels relative to the initial fixing S0, notional 100):
  - observation dates t_1 < ... < t_n = T (e.g. quarterly)
  - autocall: at t_i < T, if S_{t_i} >= autocall_level * S0, pay 100 (1 + coupon * t_i), end
    ("snowball": unpaid coupons accrue and are paid on call)
  - at T if never called:
      S_T >= autocall_level * S0          -> 100 (1 + coupon * T)
      else, never touched ki_level * S0   -> 100              (capital protected)
      else (knocked in)                    -> 100 S_T / S0     (investor is long the loss)
  - knock-in monitored daily over the whole life.

Economically the investor is short a down-and-in put and short the "vol of the
path": the bank that sells the note is long that put, and its value hinges on the
joint law of the path (call probabilities and knock-in probability), not just on
the terminal distribution. That is why models that agree on every vanilla can
disagree here. See docs/derivations/09_barriers_autocallables.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from derivpricing.engines import pde
from derivpricing.engines.monte_carlo import MCResult, plain

DAYS = 252


@dataclass(frozen=True)
class Autocallable:
    T: float = 3.0
    obs_per_year: int = 4
    autocall_level: float = 1.0
    coupon: float = 0.08            # per year, accrued
    ki_level: float = 0.6
    notional: float = 100.0

    @property
    def obs_times(self) -> np.ndarray:
        n = int(round(self.T * self.obs_per_year))
        return np.arange(1, n + 1) / self.obs_per_year

    @property
    def n_steps(self) -> int:
        """Daily grid; observation dates fall exactly on grid points."""
        return int(round(self.T * DAYS))


def price_mc(spec: Autocallable, paths: np.ndarray, r: float):
    """paths: (n_paths, spec.n_steps + 1) on the daily grid, column 0 = S0.

    Returns (MCResult, diagnostics) where diagnostics holds the probability of being
    called at each observation date and the probability of a knock-in loss at maturity.
    """
    S0 = paths[0, 0]
    n_paths = paths.shape[0]
    obs_idx = np.round(spec.obs_times * DAYS).astype(int)
    pv = np.zeros(n_paths)
    alive = np.ones(n_paths, bool)
    p_call = []
    for t, idx in zip(spec.obs_times[:-1], obs_idx[:-1]):
        called = alive & (paths[:, idx] >= spec.autocall_level * S0)
        pv[called] = spec.notional * (1 + spec.coupon * t) * np.exp(-r * t)
        alive &= ~called
        p_call.append(called.mean())

    ST = paths[:, -1]
    knocked = paths.min(axis=1) <= spec.ki_level * S0
    final_call = alive & (ST >= spec.autocall_level * S0)
    loss = alive & ~final_call & knocked
    protected = alive & ~final_call & ~knocked
    disc = np.exp(-r * spec.T)
    pv[final_call] = spec.notional * (1 + spec.coupon * spec.T) * disc
    pv[protected] = spec.notional * disc
    pv[loss] = spec.notional * ST[loss] / S0 * disc
    p_call.append(final_call.mean())
    diag = {"p_call": np.array(p_call), "p_ki_loss": loss.mean(),
            "p_protected": protected.mean(), "expected_life": _expected_life(spec, p_call)}
    return plain(pv), diag


def _expected_life(spec, p_call):
    t = spec.obs_times
    p = np.asarray(p_call)
    return float(np.sum(p * t) + (1 - p.sum()) * spec.T)


def price_pde(spec: Autocallable, S0, r, q, sigma, M=600, n_sd=6.0):
    """Two coupled 1D PDEs on the same grid (works for constant sigma or local vol sigma(t, S)):

      V_ki(t, S)  value if the knock-in has already happened
      V_no(t, S)  value if it has not; on every daily monitoring step V_no = V_ki where S <= H

    V_ki does not depend on V_no, so it is solved first and its time slices are stored.
    Observation dates: wherever S >= autocall level, both values are replaced by the call
    amount. One time step per business day, so monitoring matches the Monte Carlo grid.
    The grid is aligned so that S0 (= the autocall level here) and H are both nodes.
    """
    H = spec.ki_level * S0
    AC = spec.autocall_level * S0
    sig_ref = sigma if np.isscalar(sigma) else 0.25
    x0, xH = np.log(S0), np.log(H)
    h0 = 2 * n_sd * sig_ref * np.sqrt(spec.T) / M
    m = max(1, round((x0 - xH) / h0))
    h = (x0 - xH) / m
    j_below = int(np.ceil((n_sd * sig_ref * np.sqrt(spec.T) - (x0 - xH)) / h)) + 5
    x_min = xH - j_below * h
    x_max = x_min + M * h
    N = spec.n_steps
    dt = spec.T / N
    obs_tau = {round((spec.T - t) / dt): t for t in spec.obs_times[:-1]}   # step count -> obs time
    c_amt = lambda t: spec.notional * (1 + spec.coupon * t)

    def next_obs(t):
        later = spec.obs_times[spec.obs_times >= t - 1e-12]
        return later[0] if len(later) else spec.T

    def upper(tau, S):
        t = spec.T - tau
        tn = next_obs(t)
        return c_amt(tn) * np.exp(-r * (tn - t))

    def lower(tau, S):
        return spec.notional * S / S0 * np.exp(-q * tau)

    def on_grid_step(tau):
        k = tau / dt
        return abs(k - round(k)) < 1e-9, int(round(k))

    def autocall_hook(tau, S, V):
        is_step, k = on_grid_step(tau)
        if is_step and k in obs_tau:
            # AC sits on a node; the cell around it is half above, half below the level,
            # so that node gets the average (same cell-averaging idea as for the payoff).
            at = np.isclose(S, AC, rtol=1e-10)
            V = np.where(at, 0.5 * (c_amt(obs_tau[k]) + V), np.where(S > AC, c_amt(obs_tau[k]), V))
        return V

    payoff_ki = lambda S: np.where(S >= AC, c_amt(spec.T), spec.notional * S / S0)
    hist_ki: list = []
    pde.solve(S0, spec.T, r, q, sigma, payoff_ki, lower, upper, x_min, x_max, M=M, N=N,
              step_hook=autocall_hook, barrier_at_lower=True, history=hist_ki)
    ki_by_tau = {round(tau / dt, 6): V for tau, V in hist_ki}

    def no_ki_hook(tau, S, V):
        V = autocall_hook(tau, S, V)
        is_step, k = on_grid_step(tau)
        if is_step:
            V_ki = ki_by_tau[round(tau / dt, 6)]
            at = np.isclose(S, H, rtol=1e-10)
            V = np.where(at, 0.5 * (V_ki + V), np.where(S < H, V_ki, V))
        return V

    payoff_no = lambda S: np.where(S >= AC, c_amt(spec.T),
                                   np.where(S > H, spec.notional, spec.notional * S / S0))
    return pde.solve(S0, spec.T, r, q, sigma, payoff_no, lower, upper, x_min, x_max, M=M, N=N,
                     step_hook=no_ki_hook, barrier_at_lower=True)
