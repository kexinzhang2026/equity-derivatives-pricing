"""Stage 2 on real data: calibrate Heston and SABR to SPX end-of-day surfaces (optionsDX).

    python scripts/stage2_calibrate_spx.py showcase     # one date: fits, residuals, local vol
    python scripts/stage2_calibrate_spx.py timeseries   # weekly Heston fits, 2020Q1 + 2023

Outputs go to reports/figures/ and reports/stage2_*.csv. Raw data is read from
data/optionsdx/ (not in git; see data/README in the PLAN for how to get it).
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
from derivpricing.models import sabr
from derivpricing.models.local_vol import LocalVolSurface

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "optionsdx"
FIG = ROOT / "reports" / "figures"
REP = ROOT / "reports"
SHOWCASE_DATE = "2023-01-18"

BLUE, ORANGE, AQUA, INK, MUTED, BAND = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#ecebe7",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False,
})


def load_month(yyyymm: str) -> pd.DataFrame:
    return loaders.load_optionsdx(DATA / f"spx_eod_{yyyymm}.txt.gz")


def surface_on(date: str) -> pd.DataFrame:
    month = load_month(date[:4] + date[5:7])
    return surface.build_surface(month[month["quote_date"] == date])


# ---------------------------------------------------------------- one date in detail

def showcase(date: str = SHOWCASE_DATE):
    full = surface_on(date)
    calib = surface.calibration_subset(full)
    fit = heston_calib.calibrate(calib, n_random=3)
    res = fit.surface
    sabr_fits = {T: sabr.calibrate_slice(sl["K"].to_numpy(), sl["F"].iloc[0], T, sl["iv"].to_numpy())
                 for T, sl in calib.groupby("T")}

    # Small multiples: market mid with bid/ask band, Heston and SABR fits.
    Ts = sorted(calib["T"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.8), sharey=False)
    for ax, T in zip(axes.flat, Ts):
        sl = res[res["T"] == T].sort_values("k")
        ax.fill_between(sl["k"], 100 * sl["iv_bid"], 100 * sl["iv_ask"], color=BAND, lw=0,
                        label="market bid–ask")
        ax.plot(sl["k"], 100 * sl["iv"], "o", color=INK, ms=2.5, label="market mid")
        ax.plot(sl["k"], 100 * sl["model_iv"], color=BLUE, lw=1.6, label="Heston (one fit, all expiries)")
        kk = np.linspace(sl["k"].min(), sl["k"].max(), 100)
        p = sabr_fits[T]
        ax.plot(kk, 100 * sabr.hagan_iv(sl["F"].iloc[0] * np.exp(kk), sl["F"].iloc[0], T,
                                         p["alpha"], 1.0, p["rho"], p["nu"]),
                color=ORANGE, lw=1.2, ls="--", label="SABR (fit per expiry)")
        ax.set_title(f"T = {T * 12:.1f} months" if T < 1 else f"T = {T:.2f} years", loc="left",
                     color=INK, fontsize=8.5)
        ax.set_xlabel("log-moneyness ln(K/F)")
    for ax in axes[:, 0]:
        ax.set_ylabel("implied vol (%)")
    axes.flat[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"SPX implied vol smiles, {date}: Heston misses the short-end skew, SABR fits each slice",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "spx_smiles_fit.png")
    plt.close(fig)

    # Error by expiry, Heston vs SABR, with the typical half bid-ask spread for scale.
    rows = []
    for T in Ts:
        sl = res[res["T"] == T]
        rows.append({"T": T, "heston_rmse": np.sqrt(np.nanmean(sl["iv_error"] ** 2)),
                     "sabr_rmse": sabr_fits[T]["rmse_iv"],
                     "half_spread": np.nanmedian(sl["iv_ask"] - sl["iv_bid"]) / 2,
                     "inside_spread_heston": np.mean((sl["model_iv"] >= sl["iv_bid"]) &
                                                     (sl["model_iv"] <= sl["iv_ask"])),
                     **{f"sabr_{k}": sabr_fits[T][k] for k in ("alpha", "rho", "nu")}})
    err = pd.DataFrame(rows)
    err.to_csv(REP / "stage2_fit_by_expiry.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(err["T"], 100 * err["heston_rmse"], "o-", color=BLUE, label="Heston")
    ax.plot(err["T"], 100 * err["sabr_rmse"], "o-", color=ORANGE, label="SABR")
    ax.plot(err["T"], 100 * err["half_spread"], "--", color=MUTED, lw=1, label="median half bid–ask")
    ax.set_xscale("log")
    ax.set_xlabel("expiry (years, log scale)")
    ax.set_ylabel("RMSE (vol points)")
    ax.set_title("Fit error by expiry", loc="left", color=INK)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "spx_fit_error_by_expiry.png")
    plt.close(fig)

    # Residual map of the Heston fit over the full (unthinned) surface.
    full_res = full[(full["T"] >= Ts[0] * 0.9) & (full["T"] <= Ts[-1] * 1.1)].copy()
    full_res["model_price"] = heston_calib.model_prices(fit.params, full_res)
    from derivpricing.models import black_scholes as bs
    S_eff, r = full_res["D"] * full_res["F"], -np.log(full_res["D"]) / full_res["T"]
    full_res["model_iv"] = bs.implied_vol_vec(full_res["model_price"].to_numpy(), S_eff.to_numpy(),
                                              full_res["K"].to_numpy(), full_res["T"].to_numpy(),
                                              r.to_numpy(), 0.0, full_res["is_call"].to_numpy())
    e = 100 * (full_res["model_iv"] - full_res["iv"])
    lim = np.nanpercentile(np.abs(e), 98)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    sc = ax.scatter(full_res["k"] / np.sqrt(full_res["T"]), full_res["T"], c=e, cmap="RdBu_r",
                    vmin=-lim, vmax=lim, s=6, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("standardised moneyness ln(K/F)/√T")
    ax.set_ylabel("expiry (years, log scale)")
    ax.set_title("Heston residual (model − market), vol points", loc="left", color=INK)
    fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIG / "spx_heston_residuals.png")
    plt.close(fig)

    # Local vol from SVI slices. Use the calibration expiries' own forwards via an
    # effective constant carry fitted to F(T) (good to a few bp over two years).
    Ts_arr = np.array(Ts)
    F_T = np.array([calib.loc[calib["T"] == T, "F"].iloc[0] for T in Ts])
    D_T = np.array([calib.loc[calib["T"] == T, "D"].iloc[0] for T in Ts])
    r_eff = float(np.polyfit(Ts_arr, -np.log(D_T), 1)[0]) if len(Ts) > 1 else 0.0
    carry, logS0 = np.polyfit(Ts_arr, np.log(F_T), 1)
    S0 = float(np.exp(logS0))
    lv = LocalVolSurface.from_surface(calib, S0, r_eff, r_eff - carry)
    T_show = Ts[min(3, len(Ts) - 1)]
    sl = calib[calib["T"] == T_show].sort_values("k")
    K = sl["K"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(sl["k"], 100 * sl["iv"], "o", color=INK, ms=2.5, label="market implied vol")
    ax.plot(sl["k"], 100 * lv.implied_vol(K, T_show), color=BLUE, lw=1.6, label="SVI fit")
    # local vol at the spot level with the same log-moneyness relative to the LV forward
    S_at_k = S0 * np.exp(carry * T_show) * np.exp(sl["k"].to_numpy())
    ax.plot(sl["k"], 100 * lv.sigma(T_show, S_at_k), color=AQUA, lw=1.6, label="Dupire local vol")
    ax.set_xlabel("log-moneyness ln(K/F)")
    ax.set_ylabel("vol (%)")
    ax.set_title(f"Local vol skew is about twice the implied skew (T = {T_show:.2f}y)",
                 loc="left", color=INK, fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "spx_local_vs_implied.png")
    plt.close(fig)

    # skew ratio near the money: d sigma_LV/dk vs d sigma_imp/dk at k = 0
    k0 = np.array([-0.02, 0.02])
    K0 = sl["F"].iloc[0] * np.exp(k0)
    skew_iv = np.diff(lv.implied_vol(K0, T_show))[0] / 0.04
    S_grid = S0 * np.exp(carry * T_show) * np.exp(k0)
    skew_lv = np.diff(lv.sigma(T_show, S_grid))[0] / 0.04
    summary = {"date": date, **{n: getattr(fit.params, n) for n in heston_calib.NAMES},
               "feller_ratio": fit.params.feller_ratio, "heston_rmse_volpts": 100 * fit.rmse_iv,
               "sabr_rmse_volpts_mean": 100 * err["sabr_rmse"].mean(),
               "heston_inside_spread": float(np.mean(err["inside_spread_heston"])),
               "lv_to_iv_skew_ratio": skew_lv / skew_iv, "n_quotes_full": len(full),
               "n_quotes_calib": len(calib)}
    pd.Series(summary).to_csv(REP / "stage2_showcase_summary.csv", header=False)
    print(pd.Series(summary).to_string())
    print(err.round(4).to_string())


# ---------------------------------------------------------------- parameter stability

def weekly_dates(quote_dates) -> list:
    """One date per calendar week: the Wednesday, or the trading day nearest to it.
    Weeks are formed over all quote dates together, so a week that straddles two monthly
    files still gets exactly one date."""
    d = pd.Series(sorted(pd.to_datetime(pd.Series(quote_dates)).unique()))
    wk = d.dt.to_period("W")
    return [g.iloc[np.argmin(np.abs(g.dt.weekday - 2))] for _, g in d.groupby(wk)]


def timeseries():
    months = [f"2020{m:02d}" for m in (1, 2, 3)] + [f"2023{m:02d}" for m in range(1, 13)]
    data = {ym: load_month(ym) for ym in months}
    targets = {str(pd.Timestamp(x).date()) for x in
               weekly_dates(pd.concat([m["quote_date"] for m in data.values()]))}
    out_path = REP / "stage2_heston_timeseries.csv"
    if out_path.exists():                       # drop dates that are not on the weekly grid
        old = pd.read_csv(out_path)
        old[old["date"].isin(targets)].to_csv(out_path, index=False)
    done = set(pd.read_csv(out_path)["date"]) if out_path.exists() else set()
    for ym, month in data.items():
        for date in sorted(month["quote_date"].unique()):
            ds = str(pd.Timestamp(date).date())
            if ds in done or ds not in targets:
                continue
            surf = surface.build_surface(month[month["quote_date"] == date])
            calib = surface.calibration_subset(surf)
            fit = heston_calib.calibrate(calib, n_random=3)
            costs = sorted(c for _, _, c in fit.starts)
            # distinct local minima: starts whose final cost is > 1% above the best
            n_worse = sum(c > costs[0] * 1.01 for c in costs)
            atm = calib.loc[calib.groupby("T")["k"].transform(lambda k: k.abs() == k.abs().min()), :]
            atm_1m = float(atm.sort_values("T")["iv"].iloc[0])
            row = {"date": ds, **{n: getattr(fit.params, n) for n in heston_calib.NAMES},
                   "rmse_volpts": 100 * fit.rmse_iv, "feller_ratio": fit.params.feller_ratio,
                   "atm_vol_1m": atm_1m, "n_starts_worse": n_worse}
            pd.DataFrame([row]).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            print(row, flush=True)
    plot_timeseries()


def plot_timeseries():
    ts = pd.read_csv(REP / "stage2_heston_timeseries.csv", parse_dates=["date"]).sort_values("date")
    periods = {"2020 Q1 (COVID crash)": ts[ts["date"].dt.year == 2020],
               "2023": ts[ts["date"].dt.year == 2023]}
    panels = [("v0", "v0 (spot variance)"), ("theta", "theta (long-run variance)"),
              ("kappa", "kappa (mean reversion)"), ("xi", "xi (vol of variance)"),
              ("rho", "rho (spot–vol correlation)"), ("rmse_volpts", "fit RMSE (vol points)")]
    fig, axes = plt.subplots(len(panels), 2, figsize=(9, 9.5), sharex="col",
                             gridspec_kw={"width_ratios": [1, 3]})
    for row, (col, label) in enumerate(panels):
        for c, (name, d) in enumerate(periods.items()):
            ax = axes[row, c]
            ax.plot(d["date"], d[col], "o-", color=BLUE if c else ORANGE, ms=3, lw=1.4)
            if row == 0:
                ax.set_title(name, loc="left", color=INK)
            if c == 0:
                ax.set_ylabel(label, fontsize=7.5)
        lo = min(periods["2023"][col].min(), periods["2020 Q1 (COVID crash)"][col].min())
        hi = max(periods["2023"][col].max(), periods["2020 Q1 (COVID crash)"][col].max())
        pad = 0.05 * (hi - lo)
        for c in (0, 1):
            axes[row, c].set_ylim(lo - pad, hi + pad)
    for ax in axes[-1]:
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Weekly Heston calibrations to SPX: rho is stable, kappa and xi move together, v0 tracks the market",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "spx_heston_param_timeseries.png")
    plt.close(fig)

    t23 = periods["2023"]
    stats = t23[["v0", "kappa", "theta", "xi", "rho", "rmse_volpts"]].agg(["mean", "std"]).T
    stats["cv"] = stats["std"] / stats["mean"].abs()
    stats["corr_with_kappa"] = t23[["v0", "kappa", "theta", "xi", "rho", "rmse_volpts"]].corr()["kappa"]
    # day-to-day changes are what matter for hedging/P&L explain; levels can trend
    chg = t23[["v0", "kappa", "theta", "xi", "rho"]].diff().dropna()
    stats.loc[chg.columns, "corr_dchange_with_dkappa"] = chg.corr()["kappa"]
    stats.to_csv(REP / "stage2_param_stability_2023.csv")
    print(stats.round(3).to_string())


if __name__ == "__main__":
    {"showcase": showcase, "timeseries": timeseries, "plot": plot_timeseries}[sys.argv[1]]()
