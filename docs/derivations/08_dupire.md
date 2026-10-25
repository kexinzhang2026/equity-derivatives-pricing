# 08 · Dupire local volatility

Code: `src/derivpricing/models/svi.py`, `src/derivpricing/models/local_vol.py`, `engines/pde.py` (local-vol support)

## 1. Dupire's equation
Model: $dS = (r-q)S\,dt + \sigma(t,S)S\,dW$. The density $p(t,S)$ satisfies the forward Kolmogorov (Fokker–Planck) equation. Breeden–Litzenberger gives $\partial_{KK}C(T,K) = e^{-rT}p(T,K)$. Differentiating $C(T,K) = e^{-rT}\mathbb E[(S_T-K)^+]$ in $T$, substituting Fokker–Planck and integrating by parts twice:
$$\partial_T C = \tfrac12\sigma^2(T,K)K^2\,\partial_{KK}C - (r-q)K\,\partial_K C - qC.$$
Solved for $\sigma$: given a full, arbitrage-free surface of call prices there is exactly **one** local vol function consistent with it.

## 2. In implied-variance coordinates (Gatheral 2006)
Differentiating prices numerically is unstable (second derivative of noisy data). Change variables to total implied variance $w(k,T) = \sigma_{imp}^2T$, $k = \ln(K/F_T)$, and push the derivatives through the Black formula:
$$\sigma_{LV}^2(T,K) = \frac{\partial_T w}{\Big(1 - \frac{k\,\partial_k w}{2w}\Big)^2 - \frac{(\partial_k w)^2}{4}\Big(\frac1w+\frac14\Big) + \frac{\partial_{kk}w}{2}}.$$
Sanity check: flat smile ($w = \sigma^2T$, no $k$-dependence) ⇒ $\sigma_{LV} = \sigma$ (tested).
- Numerator $\ge 0$ ⇔ no calendar arbitrage (total variance non-decreasing in $T$).
- Denominator $g(k)\ge0$ ⇔ no butterfly arbitrage ($g$ is proportional to the density).

## 3. Smooth surface: SVI per expiry
Raw SVI $w(k) = a + b[\rho(k-m) + \sqrt{(k-m)^2+s^2}]$: linear wings (consistent with Lee's moment formula), analytic $w_k, w_{kk}$. Between expiries $w$ is interpolated linearly in $T$ at fixed $k$: it commutes with $\partial_k$, preserves calendar monotonicity, and makes $\partial_T w$ (hence $\sigma_{LV}$) piecewise constant between expiries.

## 4. Pricing under local vol
The Crank–Nicolson engine accepts $\sigma(t,S)$; coefficients are evaluated at the midpoint of each step (keeps second order in time).

## 5. The key test: Dupire reprices Heston's vanillas
Generate a surface from Heston, fit SVI (RMSE < 0.1 vol pt, no butterfly arbitrage), build the local vol, price calls with the PDE: matches Heston-COS within 0.03 vol pt at $T = 0.5, 1$ for strikes 80–120.
Two models with very different dynamics (stochastic vs deterministic vol) agree on every European option, which is what Dupire's theorem says. Stage 3 prices barriers and autocallables under both: there they disagree, because path-dependent payoffs depend on the joint law of the path (forward smile dynamics), not just the marginals. That gap is the model risk.
