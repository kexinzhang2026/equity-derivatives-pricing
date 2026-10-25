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
