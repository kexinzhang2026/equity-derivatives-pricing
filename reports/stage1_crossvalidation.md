# Stage 1 · Three methods, one price

| Product | Closed form | COS (Fourier) | Crank–Nicolson PDE | Monte Carlo (± 1 s.e.) |
|---|---|---|---|---|
| European call, Black-Scholes | 12.0056 | 12.0056 | 12.0057 | 12.0042 ± 0.0058 |
| European call, Heston | n/a | 5.7852 | (2D ADI: not built) | 5.7838 ± 0.0071 |
| American put (S=36, K=40) | none exists | n/a | 4.4867 | 4.4758 ± 0.0065 (50 dates) |

- Black-Scholes and Heston parameters: S=100.0, K=95.0, T=0.75, r=0.04, σ=0.25, q=0.015;
  Heston from Fang & Oosterlee (2008): v0=0.0175, κ=1.5768, θ=0.0398, ξ=0.5751, ρ=−0.5711.
- American put: the PDE value is continuous exercise; LSM uses 50 exercise dates and an
  out-of-sample rule, so it is a lower bound for the 50-date Bermudan, itself below the American.

## Convergence
- **PDE**: fitted log-log slope -2.00 (theory −2). `figures/pde_convergence.png`
- **Rannacher**: with 25 time steps, plain CN gamma oscillates at the strike. `figures/rannacher_gamma.png`
- **COS**: error reaches machine precision by N≈64 for Black-Scholes. `figures/cos_convergence.png`
- **Monte Carlo** at 256k paths: variance reduction factor 2.1x (antithetic),
  8.0x (control variate e^(-rT)S_T). `figures/mc_variance_reduction.png`
