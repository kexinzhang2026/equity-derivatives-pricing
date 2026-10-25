# 10 · Greeks by simulation, and the anatomy of hedging error

Code: `src/derivpricing/engines/mc_greeks.py`, `src/derivpricing/hedging/backtest.py`, `scripts/stage4_hedging.py`

## 1. Monte Carlo Greeks without bumping
$V(S_0) = \mathbb E[e^{-rT}f(S_T)]$ with $S_T = S_0\exp((r-q-\tfrac12\sigma^2)T + \sigma\sqrt T Z)$.
- **Pathwise:** $\frac{\partial V}{\partial S_0} = \mathbb E\big[e^{-rT}f'(S_T)\frac{\partial S_T}{\partial S_0}\big] = \mathbb E\big[e^{-rT}f'(S_T)\frac{S_T}{S_0}\big]$. Valid when $f$ is Lipschitz (interchange of derivative and expectation by dominated convergence). Call: $f' = \mathbf 1_{S_T>K}$. Digital: $f' = 0$ almost everywhere, so pathwise returns 0, which is wrong.
- **Likelihood ratio:** write $V = \int e^{-rT}f(s)\,p(s;S_0)\,ds$ and differentiate the density: $\frac{\partial V}{\partial S_0} = \mathbb E\big[e^{-rT}f(S_T)\,\partial_{S_0}\ln p\big]$. For the lognormal, $\partial_{S_0}\ln p = \frac{Z}{S_0\sigma\sqrt T}$. Works for any $f$, but the weight is unbounded, so variance is higher (tested: LR stderr ≈ 2× pathwise for a call).
- **Gamma:** apply LR to the pathwise delta integrand $h = \mathbf 1_{S_T>K}S_T/S_0$, keeping the explicit $1/S_0$: $\Gamma = \mathbb E\big[e^{-rT}h\,(\frac{Z}{S_0\sigma\sqrt T} - \frac1{S_0})\big]$.

## 2. The hedged P&L identity
Short one option worth $V(t,S)$, long $\Delta = V_S$ shares, cash $V - \Delta S$ at rate $r$, dividends $q$. Over $dt$, by Taylor:
$$d\Pi = -dV + \Delta\,dS + r(V-\Delta S)\,dt + q\Delta S\,dt \approx -\big(\Theta\,dt + \tfrac12\Gamma\,dS^2 + \mathcal V\,d\sigma\big) + rV\,dt - (r-q)\Delta S\,dt.$$
The BS PDE at the implied vol, $\Theta + (r-q)S\Delta + \tfrac12\sigma^2S^2\Gamma = rV$, eliminates theta and financing:
$$\boxed{d\Pi \approx \tfrac12\Gamma S^2\Big(\sigma_{imp}^2\,dt - \big(\tfrac{dS}S\big)^2\Big) - \mathcal V\,d\sigma_{imp}}$$
Second order adds $-\text{vanna}\,dS\,d\sigma - \tfrac12\text{volga}\,d\sigma^2$, with vanna $= -D\,\tfrac FS\,n(d_1)\,d_2/\sigma$ and volga $= \mathcal V d_1 d_2/\sigma$ (Black-76).
Reading: a short option hedged at implied vol earns implied variance and pays realised variance, weighted by **dollar gamma** $\tfrac12\Gamma S^2$, plus whatever the market does to implied vol.

## 3. Discrete hedging error (Kamal & Derman 1999)
With rebalancing every $\Delta t = T/N$ and $\sigma_{real} = \sigma_{imp}$, the per-step error is $\tfrac12\Gamma S^2\sigma^2\Delta t\,(1 - Z^2)$, with variance $\tfrac12(\Gamma S^2\sigma^2\Delta t)^2$. Summing over steps and averaging dollar gamma over the Brownian path gives
$$\text{std}(P\&L) \approx \sqrt{\pi/4}\;\mathcal V\,\sigma/\sqrt N.$$
Simulation: 0.434 vs formula 0.441 at $N = 63$, slope −0.5 on a log-log plot.

## 4. Transaction costs
A proportional cost $k$ per trade: the expected number of shares traded grows like $\sqrt N$ (delta changes by $\Gamma\,dS \sim \sqrt{\Delta t}$ per step), so expected cost $\propto k\sqrt N$, while hedging error $\propto 1/\sqrt N$. The RMS P&L has a minimum (Leland 1985 turns this into an effective volatility $\sigma^2(1 \pm k\sqrt{8/(\pi\Delta t)}/\sigma)$).

## 5. Hedging at implied vs realised volatility
Sell at $\sigma_{imp} = 20\%$, realised $15\%$. Expected profit $= C(\sigma_{imp}) - C(\sigma_{real})$ either way.
- Delta at realised vol: P&L is locked in (std 0.16), but you must know realised vol in advance.
- Delta at implied vol: P&L accrues as $\int\tfrac12\Gamma_{imp}S^2(\sigma_{imp}^2 - \sigma_{real}^2)dt$ and depends on where the path spends time relative to the strike (std 0.35).
Desks hedge at implied vol because their P&L is marked at implied vol.
