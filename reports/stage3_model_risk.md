# Stage 3 · Model risk: same market, three models (2023-01-18)

Market: SPX surface on 2023-01-18, S0 = 3945.1, r = 3.932%, q = 1.488% (implied from parity).
Black-Scholes vol = market ATM vol at 2.92y = 20.65%. Heston: v0=0.0393, κ=1.75, θ=0.0715, ξ=0.96, ρ=-0.71.

Product: 3y autocallable, quarterly observations, autocall at 100%, coupon 8% p.a. (accrued), daily knock-in at 60%. Notional 100.

| Model | Autocall PV (MC ± s.e.) | Autocall PV (PDE) | Fair coupon | P(called at 1st obs) | P(knock-in loss) | Expected life (y) | 3y DIP 100/60, % of S0 (MC ± s.e.) |
|---|---|---|---|---|---|---|---|
| Black-Scholes | 97.20 ± 0.04 | 97.18 | 14.03% | 50.3% | 8.2% | 0.96 | 4.93 ± 0.04 |
| Heston | 97.14 ± 0.05 | – | 15.17% | 63.8% | 8.3% | 0.73 | 9.32 ± 0.06 |
| Local vol | 96.63 ± 0.06 | 96.63 | 15.95% | 58.8% | 8.9% | 0.73 | 8.43 ± 0.06 |

Check: BS down-and-in put closed form with Broadie–Glasserman–Kou daily-monitoring shift = 4.92% of S0.

![](figures/autocall_model_risk.png)

Calibration set: 170 OTM options on 7 expiries from 0.08y to 2.92y (the stage 2 showcase stops at 2y; here the longest listed expiry is added because the note runs 3y). Beyond 2.92y, local vol keeps each strike's implied vol constant; Heston extends by its dynamics.

## Do the three models reprice the same vanillas?
Implied-vol RMSE against market mid, vol points, every option in the calibration set (local vol repriced by Crank–Nicolson; bias = mean model − market).

| Expiry (y) | Black-Scholes | Heston | Local vol | Heston bias | Local vol bias | Median half spread |
|---|---|---|---|---|---|---|
| 0.08 | 6.52 | 1.46 | 0.43 | -0.24 | +0.02 | 0.08 |
| 0.25 | 8.56 | 0.98 | 0.41 | -0.19 | +0.02 | 0.07 |
| 0.50 | 9.48 | 0.83 | 0.44 | -0.20 | -0.02 | 0.08 |
| 1.00 | 9.67 | 0.56 | 0.76 | -0.12 | -0.19 | 0.10 |
| 1.42 | 10.29 | 0.40 | 0.41 | -0.03 | -0.06 | 0.09 |
| 1.92 | 9.00 | 0.40 | 0.95 | -0.08 | -0.29 | 0.20 |
| 2.92 | 8.31 | 0.58 | 0.68 | +0.16 | -0.21 | 0.37 |
| All | 8.91 | 0.83 | 0.61 | | | |

## Fit versus dynamics
Local vol built from Heston's own smiles reprices Heston's vanillas to 0.07 vol pt, so it and Heston agree on vanillas and differ only in dynamics.

| Model | Fair coupon (± MC s.e.) |
|---|---|
| Heston | 15.17% ± 0.13% |
| Local vol from Heston smiles | 15.94% ± 0.14% |
| Local vol from market smiles | 15.95% ± 0.13% |

Dynamics effect (same vanillas): +0.77 coupon pts. Fit effect (same model class, Heston vs market smiles): +0.01 coupon pts.
