"""Stage 2 (real data): kappa profile on the SPX surface of the showcase date.

Weekly co-movement of kappa and xi suggests a ridge but does not prove one: both could
also move because the market moved. The direct test is the same profile as on synthetic
data (scripts/stage2_identifiability.py), now on one real surface: fix kappa on a grid,
refit (v0, theta, xi, rho), and see how much the fit error rises.
Run: python scripts/stage2_spx_kappa_profile.py [date]
  ->  reports/figures/spx_kappa_profile.png, reports/stage2_spx_kappa_profile.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from derivpricing.calibration import heston_calib, surface
from derivpricing.data import loaders

ROOT = Path(__file__).resolve().parents[1]
FIG, REP = ROOT / "reports" / "figures", ROOT / "reports"
DATE = sys.argv[1] if len(sys.argv) > 1 else "2023-01-18"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.color": "#e6e5e1",
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "legend.frameon": False, "axes.edgecolor": MUTED})

month = loaders.load_optionsdx(ROOT / "data" / "optionsdx" / f"spx_eod_{DATE[:4]}{DATE[5:7]}.txt.gz")
calib = surface.calibration_subset(surface.build_surface(month[month["quote_date"] == DATE]))
half_spread = 100 * float((0.5 * (calib["iv_ask"] - calib["iv_bid"])).median())

free_fit = heston_calib.calibrate(calib, n_random=3)
kappas = np.geomspace(0.5, 10, 11)
rows = []
for k in kappas:
    f = heston_calib.calibrate(calib, n_random=2, fixed={"kappa": k})
    rows.append({"kappa": k, "rmse_volpts": 100 * f.rmse_iv, **{n: getattr(f.params, n) for n in
                                                                ("v0", "theta", "xi", "rho")}})
    print(rows[-1], flush=True)
prof = pd.DataFrame(rows)
prof.to_csv(REP / "stage2_spx_kappa_profile.csv", index=False)

best = 100 * free_fit.rmse_iv
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))
axes[0].plot(prof["kappa"], prof["rmse_volpts"] - best, "o-", color=BLUE, ms=4)
axes[0].axhline(half_spread, color=MUTED, lw=1, ls=":")
axes[0].text(0.55, half_spread * 1.08, f"median half spread {half_spread:.2f} vol pt", color=MUTED, fontsize=8)
axes[0].set_ylabel("RMSE above best fit (vol points)")
axes[0].set_title(f"SPX {DATE}: fit error with kappa held fixed", loc="left")
axes[1].plot(prof["kappa"], prof["xi"], "o-", color=ORANGE, ms=4)
axes[1].set_ylabel("fitted xi")
axes[1].set_title("xi compensates for kappa", loc="left")
for ax in axes:
    ax.set_xscale("log")
    ax.axvline(free_fit.params.kappa, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("kappa (held fixed)")
fig.tight_layout()
fig.savefig(FIG / "spx_kappa_profile.png")
print(f"free fit kappa={free_fit.params.kappa:.2f} rmse={best:.3f}; half spread {half_spread:.3f}")
