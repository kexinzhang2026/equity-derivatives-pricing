"""SVI fitting and Dupire local volatility.

Key check (Dupire's theorem): a local vol model built from a Heston-generated
implied vol surface must reprice Heston's European options, even though the two
models have completely different dynamics. Stage 3 then shows they disagree on
path-dependent products, which is the point of the model-risk study.
"""
from functools import partial

import numpy as np
import pytest

from derivpricing.calibration import synthetic
from derivpricing.engines import cos, pde
from derivpricing.models import black_scholes as bs
from derivpricing.models import svi
from derivpricing.models.heston import HestonParams, char_func
from derivpricing.models.local_vol import LocalVolSurface

S0, R, Q = 100.0, 0.04, 0.012
P = HestonParams(v0=0.03, kappa=2.5, theta=0.05, xi=0.8, rho=-0.7)


@pytest.fixture(scope="module")
def heston_lv():
    s = synthetic.heston_surface(P, S0, R, Q, expiries=(0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
                                 n_strikes=25, z_range=(-4.0, 3.0))
    return s, LocalVolSurface.from_surface(s, S0, R, Q)


def test_svi_fit_recovers_its_own_parameters():
    true = (0.02, 0.15, -0.6, 0.05, 0.2)
    k = np.linspace(-0.6, 0.4, 30)
    got = svi.fit(k, svi.w(k, *true))
    assert np.allclose(got, true, atol=1e-6)


def test_flat_surface_gives_flat_local_vol():
    expiries = [0.25, 0.5, 1.0]
    params = [(0.2**2 * T, 1e-10, 0.0, 0.0, 1.0) for T in expiries]
    lv = LocalVolSurface(expiries, params, S0, R, Q)
    for t in (0.1, 0.4, 0.8):
        assert np.allclose(lv.sigma(t, np.array([70.0, 100.0, 140.0])), 0.2, atol=1e-6)


def test_svi_fits_heston_smiles_without_butterfly_arbitrage(heston_lv):
    s, lv = heston_lv
    err = [lv.implied_vol(np.array([row.K]), row.T)[0] - row.iv for row in s.itertuples()]
    assert np.sqrt(np.mean(np.square(err))) < 0.002            # SVI fits Heston to < 0.2 vol pt
    k = np.linspace(-1.0, 0.6, 200)
    assert all(svi.density_g(k, p).min() > 0 for p in lv.params)


@pytest.mark.parametrize("T", [0.5, 1.0])
def test_dupire_reprices_heston_vanillas(heston_lv, T):
    _, lv = heston_lv
    half = 1.25
    cf = partial(char_func, T=T, r=R, q=Q, p=P)
    for K in (80.0, 90.0, 100.0, 110.0, 120.0):
        res = pde.solve(S0, T, R, Q, lv.sigma, lambda S: np.maximum(S - K, 0.0),
                        lambda t, S: 0.0, lambda t, S: S * np.exp(-Q * t) - K * np.exp(-R * t),
                        np.log(S0) - half, np.log(S0) + half, M=400, N=200)
        iv_lv = bs.implied_vol(res.price, S0, K, T, R, Q)
        iv_heston = bs.implied_vol(cos.price(cf, S0, K, T, R, Q, N=512), S0, K, T, R, Q)
        assert iv_lv == pytest.approx(iv_heston, abs=0.001)    # within 0.1 vol pt
