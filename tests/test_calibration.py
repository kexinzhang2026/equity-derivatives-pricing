"""Calibration: recover known parameters from synthetic surfaces; SABR formula checks;
market-data cleaning (implied forward from put-call parity)."""
import numpy as np
import pandas as pd
import pytest

from derivpricing.calibration import heston_calib, surface, synthetic
from derivpricing.models import black_scholes as bs
from derivpricing.models import sabr
from derivpricing.models.heston import HestonParams

TRUE = HestonParams(v0=0.03, kappa=2.5, theta=0.05, xi=0.8, rho=-0.7)


# ------------------------------------------------------------------ Heston

def test_heston_recovers_parameters_from_clean_surface():
    fit = heston_calib.calibrate(synthetic.heston_surface(TRUE), n_random=2)
    got = np.array([fit.params.v0, fit.params.kappa, fit.params.theta, fit.params.xi, fit.params.rho])
    want = np.array([TRUE.v0, TRUE.kappa, TRUE.theta, TRUE.xi, TRUE.rho])
    assert np.allclose(got, want, rtol=1e-4)
    assert fit.rmse_iv < 1e-5


def test_heston_fit_with_noise_is_at_noise_level():
    # 0.2 vol-pt noise: residual RMSE should be about the noise level, not below
    # (overfitting) or far above (failed optimisation).
    fit = heston_calib.calibrate(synthetic.heston_surface(TRUE, noise_iv=0.002, seed=1), n_random=2)
    assert 0.001 < fit.rmse_iv < 0.003
    assert fit.params.rho == pytest.approx(TRUE.rho, abs=0.05)
    assert fit.params.v0 == pytest.approx(TRUE.v0, rel=0.05)


# ------------------------------------------------------------------ SABR

def test_sabr_zero_volvol_lognormal_is_flat():
    K = np.linspace(60, 160, 11)
    assert np.allclose(sabr.hagan_iv(K, 100.0, 1.0, 0.2, 1.0, -0.5, 1e-10), 0.2, atol=1e-8)


def test_sabr_atm_formula():
    # At K = F (beta = 1): sigma = alpha [1 + (rho nu alpha / 4 + (2 - 3 rho^2) nu^2 / 24) T]
    a, r, n, T = 0.25, -0.4, 0.9, 0.5
    expected = a * (1 + (r * n * a / 4 + (2 - 3 * r**2) * n**2 / 24) * T)
    assert sabr.hagan_iv(np.array([100.0]), 100.0, T, a, 1.0, r, n)[0] == pytest.approx(expected, abs=1e-12)


def test_sabr_hagan_matches_monte_carlo_short_expiry():
    # Hagan is an asymptotic expansion: check it against a direct simulation of the SDE.
    a, r, n, T, F = 0.2, -0.5, 0.6, 0.25, 100.0
    rng = np.random.default_rng(7)
    n_paths, n_steps = 200_000, 100
    dt = T / n_steps
    logF = np.zeros(n_paths)
    alpha = np.full(n_paths, a)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = r * z1 + np.sqrt(1 - r**2) * rng.standard_normal(n_paths)
        logF += -0.5 * alpha**2 * dt + alpha * np.sqrt(dt) * z1
        alpha *= np.exp(n * np.sqrt(dt) * z2 - 0.5 * n**2 * dt)       # exact for GBM vol
    FT = F * np.exp(logF)
    for K in (90.0, 100.0, 110.0):
        is_call = K >= F
        payoff = np.maximum(FT - K, 0) if is_call else np.maximum(K - FT, 0)
        iv_mc = bs.implied_vol(payoff.mean(), F, K, T, 0.0, is_call=is_call)
        iv_hagan = sabr.hagan_iv(np.array([K]), F, T, a, 1.0, r, n)[0]
        assert iv_hagan == pytest.approx(iv_mc, abs=0.003)            # within 0.3 vol pt


def test_sabr_slice_calibration_recovers_parameters():
    F, T = 100.0, 0.5
    K = F * np.exp(np.linspace(-0.3, 0.2, 15))
    iv = sabr.hagan_iv(K, F, T, 0.22, 1.0, -0.65, 1.1)
    fit = sabr.calibrate_slice(K, F, T, iv)
    assert fit["alpha"] == pytest.approx(0.22, rel=1e-6)
    assert fit["rho"] == pytest.approx(-0.65, abs=1e-6)
    assert fit["nu"] == pytest.approx(1.1, rel=1e-6)


# ------------------------------------------------------------------ market data cleaning

def test_implied_forward_and_discount_from_parity():
    S, r, q, T, sigma = 5000.0, 0.045, 0.013, 0.5, 0.18
    F, D = S * np.exp((r - q) * T), np.exp(-r * T)
    rows = []
    for K in np.arange(4500, 5501, 25.0):
        for is_call in (True, False):
            p = bs.price(S, K, T, r, sigma, q, is_call)
            rows.append({"strike": K, "is_call": is_call, "mid": p})
    F_hat, D_hat, n = surface.implied_forward(pd.DataFrame(rows), spot_guess=S)
    assert F_hat == pytest.approx(F, rel=1e-10) and D_hat == pytest.approx(D, rel=1e-10)


def test_build_surface_recovers_flat_vol():
    S, r, q, sigma = 5000.0, 0.045, 0.013, 0.18
    qd = pd.Timestamp("2026-01-02")
    rows = []
    for days in (30, 91, 182):
        T = days / 365
        for K in np.arange(4000, 6001, 50.0):
            for is_call in (True, False):
                p = bs.price(S, K, T, r, sigma, q, is_call)
                rows.append({"quote_date": qd, "expiry": qd + pd.Timedelta(days=days),
                             "strike": K, "is_call": is_call, "bid": p * 0.99, "ask": p * 1.01,
                             "volume": 1, "open_interest": 1, "spot": S})
    surf = surface.build_surface(pd.DataFrame(rows))
    assert len(surf) > 50
    # mid = exact price, so the IV is exact; only OTM options are kept
    assert np.allclose(surf["iv"], sigma, atol=1e-8)
    assert ((surf["K"] >= surf["F"]) == surf["is_call"]).all()
