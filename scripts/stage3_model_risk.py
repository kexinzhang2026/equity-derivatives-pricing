"""Stage 3: model risk on a down-and-in put and a 3-year autocallable.

All three models are fitted to the same SPX surface (default 2023-01-18):
  BS          flat vol = market ATM implied vol at the product maturity
  Heston      calibrated to the whole surface (one parameter set)
  Local vol   Dupire from SVI slices (reprices the market smiles by construction)

Pricing engines: PDE and MC for BS and local vol (cross-check), MC for Heston.
    python scripts/stage3_model_risk.py   ->  reports/stage3_model_risk.md, figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from derivpricing.calibration import heston_calib, surface
from derivpricing.data import loaders
from derivpricing.engines import monte_carlo as mc
from derivpricing.models.local_vol import LocalVolSurface
from derivpricing.products import barrier
from derivpricing.products.autocallable import Autocallable, price_mc, price_pde

ROOT = Path(__file__).resolve().parents[1]
FIG, REP = ROOT / "reports" / "figures", ROOT / "reports"
DATE = sys.argv[1] if len(sys.argv) > 1 else "2023-01-18"
N_PATHS = 100_000
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.dpi": 150, "font.size": 8.5, "axes.grid": True, "grid.color": "#ecebe7",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.edgecolor": MUTED})

# ---------------------------------------------------------------- market and models
month = loaders.load_optionsdx(ROOT / "data" / "optionsdx" / f"spx_eod_{DATE[:4]}{DATE[5:7]}.txt.gz")
full = surface.build_surface(month[month["quote_date"] == DATE])
calib = surface.calibration_subset(full, tenors=(1/12, 3/12, 6/12, 1.0, 1.5, 2.0, 3.0))
Ts = np.sort(calib["T"].unique())
F_T = calib.groupby("T")["F"].first().to_numpy()
D_T = calib.groupby("T")["D"].first().to_numpy()
# Constant r and carry that reproduce the implied discount and forward curves (least squares)
r = float(np.sum(Ts * -np.log(D_T)) / np.sum(Ts**2))
carry, logS0 = np.polyfit(Ts, np.log(F_T), 1)
S0, q = float(np.exp(logS0)), r - carry

heston = heston_calib.calibrate(calib, n_random=3).params
lv = LocalVolSurface.from_surface(calib, S0, r, q)
spec = Autocallable(T=3.0, obs_per_year=4, autocall_level=1.0, coupon=0.08, ki_level=0.6)
T_long = Ts[np.argmin(np.abs(Ts - spec.T))]
sl = calib[calib["T"] == T_long]
atm_vol = float(np.interp(0.0, sl["k"], sl["iv"]))
print(f"S0={S0:.1f} r={r:.4f} q={q:.4f} ATM vol {T_long:.2f}y={atm_vol:.4f} heston={heston}")

rng = np.random.default_rng(11)
paths = {
    "Black-Scholes": mc.gbm_paths(S0, spec.T, r, q, atm_vol, spec.n_steps, N_PATHS, rng),
    "Heston": mc.heston_paths_qe(S0, spec.T, r, q, heston, spec.n_steps, N_PATHS, rng)[0],
    "Local vol": mc.local_vol_paths(S0, spec.T, r, q, lv.sigma, spec.n_steps, N_PATHS, rng),
}

# ---------------------------------------------------------------- autocallable
rows = []
for name, P in paths.items():
    res, diag = price_mc(spec, P, r)
    # PV is affine in the coupon: PV(c) = A + c B, so the fair coupon (PV = 100) follows
    # from two prices on the same paths.
    a0, _ = price_mc(Autocallable(**{**spec.__dict__, "coupon": 0.0}), P, r)
    fair = (100.0 - a0.price) / ((res.price - a0.price) / spec.coupon)
    pde_price = np.nan
    if name == "Black-Scholes":
        pde_price = price_pde(spec, S0, r, q, atm_vol, M=600).price
    elif name == "Local vol":
        pde_price = price_pde(spec, S0, r, q, lv.sigma, M=600).price
    di = barrier.down_and_in_put_mc(P, S0, 0.6 * S0, r, spec.T)
    rows.append({"model": name, "autocall_mc": res.price, "autocall_se": res.stderr,
                 "autocall_pde": pde_price, "fair_coupon": fair,
                 "coupon_se": res.stderr / ((res.price - a0.price) / spec.coupon),
                 "p_first_call": diag["p_call"][0], "p_ki_loss": diag["p_ki_loss"],
                 "expected_life": diag["expected_life"],
                 "dip_mc": 100 * di.price / S0, "dip_se": 100 * di.stderr / S0,
                 "p_call": diag["p_call"]})
tab = pd.DataFrame(rows)

# DIP cross-check under BS: closed form with BGK discrete-monitoring shift
dt = 1 / 252
dip_bs_exact = 100 / S0 * barrier.down_and_in_put_bs(S0, S0, barrier.bgk_shift(0.6 * S0, atm_vol, dt),
                                                     spec.T, r, atm_vol, q)

# ---------------------------------------------------------------- figure: call probabilities
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
for (_, row), color in zip(tab.iterrows(), [MUTED, BLUE, AQUA]):
    axes[0].plot(spec.obs_times, 100 * np.cumsum(row["p_call"]), "o-", ms=3, color=color, label=row["model"])
axes[0].set_xlabel("observation date (years)")
axes[0].set_ylabel("cumulative probability called (%)")
axes[0].set_title("When does the note get called?", loc="left")
axes[0].legend()
# forward smile: implied vol of a 3-month option starting in 1 year, from each model's paths
i1, i2 = 252, 252 + 63
k_grid = np.linspace(-0.15, 0.1, 11)
from derivpricing.models import black_scholes as bs
for (name, P), color in zip(paths.items(), [MUTED, BLUE, AQUA]):
    ret = P[:, i2] / P[:, i1]
    fwd = np.exp((r - q) * 0.25)
    ivs = []
    for k in k_grid:
        Kr = fwd * np.exp(k)
        put = k < 0
        pay = np.maximum(Kr - ret, 0) if put else np.maximum(ret - Kr, 0)
        ivs.append(bs.implied_vol(np.exp(-r * 0.25) * pay.mean(), 1.0, Kr, 0.25, r, q, is_call=not put))
    axes[1].plot(k_grid, 100 * np.array(ivs), "o-", ms=3, color=color, label=name)
axes[1].set_xlabel("forward log-moneyness")
axes[1].set_ylabel("forward implied vol (%)")
axes[1].set_title("Forward smile: 3m option starting in 1y", loc="left")
axes[1].legend()
fig.tight_layout()
fig.savefig(FIG / "autocall_model_risk.png")
plt.close(fig)

# ---------------------------------------------------------------- vanilla repricing on the real surface
# "Fitted to the same surface" is a claim to check, not assume. Reprice every calibration
# option under each model and compare implied vols with the market mid, per expiry.
from derivpricing.engines import pde
from derivpricing.models import black_scholes as bs


def lv_vanilla_ivs(lv_model, rows):
    """Implied vols of the surface's options under a local vol model, by Crank-Nicolson.
    Strikes are placed at the same log-moneyness k relative to the model's own forward."""
    out = np.empty(len(rows))
    for j, (_, row) in enumerate(rows.iterrows()):
        T, F_m = row["T"], S0 * np.exp((r - q) * row["T"])
        K = F_m * np.exp(row["k"])
        half = 8 * 0.3 * np.sqrt(T) + abs(row["k"]) + 0.1
        if row["is_call"]:
            pay, lo_b = (lambda S, K=K: np.maximum(S - K, 0.0)), (lambda tau, S: 0.0)
            up_b = lambda tau, S, K=K: S * np.exp(-q * tau) - K * np.exp(-r * tau)
        else:
            pay, up_b = (lambda S, K=K: np.maximum(K - S, 0.0)), (lambda tau, S: 0.0)
            lo_b = lambda tau, S, K=K: K * np.exp(-r * tau) - S * np.exp(-q * tau)
        v = pde.solve(S0, T, r, q, lv_model.sigma, pay, lo_b, up_b, np.log(S0) - half, np.log(S0) + half,
                      M=500, N=max(100, int(400 * T))).price
        try:
            out[j] = bs.implied_vol(v, S0, K, T, r, q, is_call=bool(row["is_call"]))
        except ValueError:
            out[j] = np.nan
    return out


def iv_from_prices(prices, rows):
    """Black-76 implied vols for prices quoted on each row's own forward and discount factor."""
    out = np.empty(len(rows))
    for j, (_, row) in enumerate(rows.iterrows()):
        T, F, D = row["T"], row["F"], row["D"]
        try:
            out[j] = bs.implied_vol(prices[j], D * F, row["K"], T, -np.log(D) / T, 0.0, is_call=bool(row["is_call"]))
        except ValueError:
            out[j] = np.nan
    return out


rep = calib[["T", "k", "K", "F", "D", "is_call", "iv", "iv_bid", "iv_ask"]].copy()
rep["bs"] = atm_vol
rep["heston"] = iv_from_prices(heston_calib.model_prices(heston, calib, N=256), calib)
rep["lv"] = lv_vanilla_ivs(lv, rep)
rep["half_spread"] = 0.5 * (rep["iv_ask"] - rep["iv_bid"])
models = ["bs", "heston", "lv"]
rmse = lambda e: float(np.sqrt(np.nanmean(e**2)))
by_T = pd.DataFrame({"T": Ts})
for m in models:
    by_T[m] = [100 * rmse(rep.loc[rep["T"] == T, m] - rep.loc[rep["T"] == T, "iv"]) for T in Ts]
    by_T[m + "_bias"] = [100 * float(np.nanmean(rep.loc[rep["T"] == T, m] - rep.loc[rep["T"] == T, "iv"])) for T in Ts]
by_T["half_spread"] = [100 * float(rep.loc[rep["T"] == T, "half_spread"].median()) for T in Ts]
tot = {m: 100 * rmse(rep[m] - rep["iv"]) for m in models}
by_T.to_csv(REP / "stage3_vanilla_repricing.csv", index=False)
print(by_T.round(3), tot)

# ---------------------------------------------------------------- fit versus dynamics
# Heston and market local vol differ in two ways at once: (i) they do not reprice the same
# vanillas (Heston has fit error), (ii) they have different dynamics. Separate the two with a
# third model: the local vol built from HESTON's own smiles on the same expiries and strikes.
#   LV(Heston smiles) vs Heston:      same vanillas, different dynamics  -> dynamics effect
#   LV(market) vs LV(Heston smiles):  same dynamics class, different vanillas -> fit effect
calib_h = calib.copy()
calib_h["iv"] = rep["heston"].to_numpy()
lv_h = LocalVolSurface.from_surface(calib_h.dropna(subset=["iv"]), S0, r, q)
rep["lv_h"] = lv_vanilla_ivs(lv_h, rep)
lv_h_repricing = 100 * rmse(rep["lv_h"] - rep["heston"])          # how well LV_H matches Heston vanillas
P_lvh = mc.local_vol_paths(S0, spec.T, r, q, lv_h.sigma, spec.n_steps, N_PATHS, np.random.default_rng(12))
res_h, diag_h = price_mc(spec, P_lvh, r)
a0_h, _ = price_mc(Autocallable(**{**spec.__dict__, "coupon": 0.0}), P_lvh, r)
fair_lvh = (100.0 - a0_h.price) / ((res_h.price - a0_h.price) / spec.coupon)
fair = dict(zip(tab["model"], tab["fair_coupon"]))
decomp = {"heston": fair["Heston"], "lv_heston_smiles": fair_lvh, "lv_market": fair["Local vol"],
          "dynamics": fair_lvh - fair["Heston"], "fit": fair["Local vol"] - fair_lvh,
          "lv_h_repricing_volpts": lv_h_repricing, "lv_h_pv": res_h.price, "lv_h_se": res_h.stderr,
          "lv_h_p_first_call": diag_h["p_call"][0]}
# Monte Carlo error of a fair coupon: s.e.(PV) / B, with B = dPV/dc
B_h = (res_h.price - a0_h.price) / spec.coupon
decomp["coupon_se_lv_h"] = res_h.stderr / B_h
pd.Series(decomp).to_csv(REP / "stage3_fit_vs_dynamics.csv", header=False)
print(decomp)

# ---------------------------------------------------------------- report
def f(x, d=2):
    return "–" if not np.isfinite(x) else f"{x:.{d}f}"

lines = [f"# Stage 3 · Model risk: same market, three models ({DATE})", "",
         f"Market: SPX surface on {DATE}, S0 = {S0:.1f}, r = {r:.3%}, q = {q:.3%} (implied from parity).",
         f"Black-Scholes vol = market ATM vol at {T_long:.2f}y = {atm_vol:.2%}. "
         f"Heston: v0={heston.v0:.4f}, κ={heston.kappa:.2f}, θ={heston.theta:.4f}, ξ={heston.xi:.2f}, ρ={heston.rho:.2f}.",
         "", f"Product: {spec.T:.0f}y autocallable, quarterly observations, autocall at 100%, "
         f"coupon {spec.coupon:.0%} p.a. (accrued), daily knock-in at {spec.ki_level:.0%}. Notional 100.",
         "", "| Model | Autocall PV (MC ± s.e.) | Autocall PV (PDE) | Fair coupon | P(called at 1st obs) | P(knock-in loss) | Expected life (y) | 3y DIP 100/60, % of S0 (MC ± s.e.) |",
         "|---|---|---|---|---|---|---|---|"]
for _, row in tab.iterrows():
    lines.append(f"| {row['model']} | {f(row['autocall_mc'])} ± {f(row['autocall_se'])} | {f(row['autocall_pde'])} | "
                 f"{row['fair_coupon']:.2%} | {row['p_first_call']:.1%} | {row['p_ki_loss']:.1%} | "
                 f"{f(row['expected_life'])} | {f(row['dip_mc'])} ± {f(row['dip_se'])} |")
lines += ["", f"Check: BS down-and-in put closed form with Broadie–Glasserman–Kou daily-monitoring shift = "
          f"{dip_bs_exact:.2f}% of S0.", "", "![](figures/autocall_model_risk.png)",
          "", f"Calibration set: {len(calib)} OTM options on {len(Ts)} expiries from {Ts[0]:.2f}y to {Ts[-1]:.2f}y "
          "(the stage 2 showcase stops at 2y; here the longest listed expiry is added because the note runs 3y). "
          f"Beyond {Ts[-1]:.2f}y, local vol keeps each strike's implied vol constant; Heston extends by its dynamics.",
          "", "## Do the three models reprice the same vanillas?",
          "Implied-vol RMSE against market mid, vol points, every option in the calibration set "
          "(local vol repriced by Crank–Nicolson; bias = mean model − market).", "",
          "| Expiry (y) | Black-Scholes | Heston | Local vol | Heston bias | Local vol bias | Median half spread |",
          "|---|---|---|---|---|---|---|"]
for _, b in by_T.iterrows():
    lines.append(f"| {b['T']:.2f} | {b['bs']:.2f} | {b['heston']:.2f} | {b['lv']:.2f} | {b['heston_bias']:+.2f} | "
                 f"{b['lv_bias']:+.2f} | {b['half_spread']:.2f} |")
lines += [f"| All | {tot['bs']:.2f} | {tot['heston']:.2f} | {tot['lv']:.2f} | | | |", "",
          "## Fit versus dynamics",
          f"Local vol built from Heston's own smiles reprices Heston's vanillas to {decomp['lv_h_repricing_volpts']:.2f} vol pt, "
          f"so it and Heston agree on vanillas and differ only in dynamics.", "",
          "| Model | Fair coupon (± MC s.e.) |", "|---|---|",
          f"| Heston | {decomp['heston']:.2%} ± {tab.loc[1, 'coupon_se']:.2%} |",
          f"| Local vol from Heston smiles | {decomp['lv_heston_smiles']:.2%} ± {decomp['coupon_se_lv_h']:.2%} |",
          f"| Local vol from market smiles | {decomp['lv_market']:.2%} ± {tab.loc[2, 'coupon_se']:.2%} |", "",
          f"Dynamics effect (same vanillas): {100 * decomp['dynamics']:+.2f} coupon pts. "
          f"Fit effect (same model class, Heston vs market smiles): {100 * decomp['fit']:+.2f} coupon pts."]
(REP / "stage3_model_risk.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
tab.drop(columns="p_call").to_csv(REP / "stage3_model_risk.csv", index=False)
