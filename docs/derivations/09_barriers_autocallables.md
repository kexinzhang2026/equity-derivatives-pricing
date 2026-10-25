# 09 · Barrier options, the autocallable, and why models disagree

Code: `src/derivpricing/products/barrier.py`, `src/derivpricing/products/autocallable.py`, `scripts/stage3_model_risk.py`

## 1. Down-and-out put: closed form
Under BS, $X_t = \ln S_t$ is Brownian motion with drift $\nu = r-q-\tfrac12\sigma^2$. For driftless BM the **reflection principle** gives the joint law of the endpoint and the running minimum: paths that touch $h$ and end at $x$ are in bijection with paths that end at $2h - x$. Adding drift by Girsanov multiplies the reflected density by $e^{2\nu(h-x_0)/\sigma^2}$, which is where the factors $(H/S)^{2\mu}$ and $(H/S)^{2(\mu+1)}$ with $\mu = \nu/\sigma^2$ come from. Integrating the put payoff against "end above $h$ without touching" gives the Reiner–Rubinstein formula (Haug's $A - B + C - D$). The down-and-in put follows from **in–out parity**: $P_{di} + P_{do} = P_{\text{vanilla}}$.

## 2. PDE: put the barrier on the grid
Knock-out = Dirichlet $V=0$ on $S=H$. If $H$ sits between nodes, the effective barrier is off by up to $h/2$ and the error is $O(h)$. Choosing $h$ so that both $\ln H$ and $\ln S_0$ are nodes (`barrier_at_lower=True`) restores clean second order: errors $6.7\times10^{-4}$, $1.7\times10^{-4}$, $4.1\times10^{-5}$ for $M = 100, 200, 400$.

## 3. Discrete monitoring: Broadie–Glasserman–Kou
Daily monitoring knocks in less often than continuous. BGK (1997): a discrete barrier at $H$ ≈ a continuous one at $H e^{-\beta\sigma\sqrt{\Delta t}}$ (down barrier), $\beta = -\zeta(\tfrac12)/\sqrt{2\pi}\approx0.5826$. Test: daily-monitored MC down-and-in put agrees with the shifted closed form, and is significantly below the continuous one.

## 4. The autocallable
Snowball autocall: called at the first quarterly date with $S \ge S_0$, paying $100(1 + c\,t_i)$; at maturity 100 if the 60% barrier was never touched, else $100\,S_T/S_0$.
Decomposition (investor's view): zero-coupon bond + digital-style coupons conditional on being called − a down-and-in put that only matters on paths never called. Everything depends on **path** probabilities: when the note is called and whether the barrier is hit on the paths that survive.

**PDE with a knock-in.** Two value functions on the same grid: $V^{ki}$ (already knocked) and $V^{no}$. $V^{ki}$ is independent of $V^{no}$, so solve it first and store every time slice; while solving $V^{no}$, after each daily step set $V^{no} = V^{ki}$ for $S \le H$. Observation dates: replace by the call amount where $S \ge S_0$. One step per business day, so the PDE monitors exactly like the daily MC.
**Discontinuities.** Each observation date reintroduces a jump (digital) at $S_0$, and the knock-in substitution creates one at $H$. With $S_0$ and $H$ on nodes, the node at the jump gets the average of the two sides (cell averaging, as for the payoff). Without this: first-order convergence (97.59, 97.51, 97.48 for $M$ = 300, 600, 1200); with it: 97.462, 97.450, 97.451, 97.450 for $M$ = 150–1200. MC: 97.43 ± 0.04.

**Fair coupon.** PV is affine in the coupon, $PV(c) = A + cB$, so two prices on the same paths give the coupon that makes PV = 100. Sanity check of its size: coupon ≈ $r$ + (value of the short put)/(expected life). With LV numbers: 3.9% + 8.4%/0.73y ≈ 15.4% vs 15.95% computed.

## 5. Model risk (SPX, 2023-01-18)
Three models fitted to the same market (table in `reports/stage3_model_risk.md`):
- **Black-Scholes with ATM vol** has no skew: it underprices the 60% down-and-in put by almost half (4.9% vs 8.4–9.3% of notional) and gives the lowest fair coupon. Pure skew risk.
- **Heston vs local vol** both match the vanilla smile (LV exactly, Heston to ~0.9 vol pt), yet the down-and-in put differs by ~0.9% of notional and the fair coupon by ~0.8%. Cause: **forward smile**. Local vol's skew is a fixed function of $(t, S)$; as time passes the smile it implies for future periods flattens. Heston's skew comes from the spot–vol correlation and stays put. Figure `autocall_model_risk.png`, right panel: the smile of a 3-month option starting in 1 year is much steeper under Heston.
- A steeper forward skew makes large down-moves later in the life more likely relative to their vanilla-implied weight: here Heston prices the knock-in put higher. Local vol has higher realised vol near the money in the forward periods (flatter, higher forward ATM) which lowers the PV of the note through the call schedule and the conditional knock-in.
- Practitioners use **stochastic local vol** (calibrate LV on top of Heston with a leverage function) so that vanillas are matched exactly while the forward smile dynamics come from the stochastic part; the mixing weight is itself a model-risk parameter. A natural extension of this project.
