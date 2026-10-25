"""Stage 2 (synthetic): how well can Heston parameters be identified from a smile surface?

Profile of the fit error in kappa: fix kappa on a grid, refit (v0, theta, xi, rho), plot
the RMSE. A sharp minimum means kappa is pinned down by the data; a flat valley means
other parameters compensate (here xi rises with kappa, keeping the smile similar).
Run: python scripts/stage2_identifiability.py  ->  reports/figures/heston_kappa_profile.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from derivpricing.calibration import heston_calib, synthetic
from derivpricing.models.heston import HestonParams

FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.color": "#e6e5e1",
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "legend.frameon": False, "axes.edgecolor": MUTED})

TRUE = HestonParams(v0=0.03, kappa=2.5, theta=0.05, xi=0.8, rho=-0.7)
kappas = np.geomspace(0.5, 10, 13)
cases = {"expiries up to 2 years": (0.08, 0.25, 0.5, 1.0, 2.0),
         "expiries up to 6 months": (0.08, 0.17, 0.25, 0.5)}

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))
for (label, exps), color in zip(cases.items(), [BLUE, ORANGE]):
    s = synthetic.heston_surface(TRUE, expiries=exps, noise_iv=0.002, seed=1)
    fits = [heston_calib.calibrate(s, n_random=0, fixed={"kappa": k}) for k in kappas]
    axes[0].plot(kappas, [100 * f.rmse_iv for f in fits], "o-", color=color, ms=4, label=label)
    axes[1].plot(kappas, [f.params.xi for f in fits], "o-", color=color, ms=4, label=label)
for ax in axes:
    ax.set_xscale("log")
    ax.axvline(TRUE.kappa, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("kappa (held fixed)")
axes[0].axhline(0.2, color=MUTED, lw=1, ls=":")
axes[0].text(0.55, 0.22, "noise level 0.2 vol pt", color=MUTED, fontsize=8)
axes[0].set_ylabel("fit RMSE (vol points)")
axes[0].set_title("Shorter surface: flatter valley in kappa", loc="left", color=INK)
axes[1].set_ylabel("fitted xi")
axes[1].set_title("xi compensates for kappa", loc="left", color=INK)
axes[0].legend()
fig.tight_layout()
fig.savefig(FIG / "heston_kappa_profile.png")
print("saved", FIG / "heston_kappa_profile.png")
