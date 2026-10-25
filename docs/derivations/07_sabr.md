# 07 · SABR and Hagan's formula

Code: `src/derivpricing/models/sabr.py`

## 1. Model
For one expiry, under the $T$-forward measure: $dF = \alpha F^\beta dW_1$, $d\alpha = \nu\alpha\,dW_2$, $d\langle W_1, W_2\rangle = \rho\,dt$.
- $\alpha$: vol level; $\beta$: backbone (how ATM vol moves with $F$); $\rho$: skew; $\nu$: vol-of-vol, smile curvature.
- $\beta$ and $\rho$ both tilt the smile, so a single smile cannot separate them: fix $\beta$ (here $\beta=1$, lognormal backbone, natural for an equity index) and fit $(\alpha,\rho,\nu)$ per expiry.

## 2. Hagan et al. (2002)
Singular perturbation expansion for small $\varepsilon = T\cdot$(vol)². To leading orders the Black implied vol is
$$\sigma_B(K) = \frac{\alpha}{(FK)^{\frac{1-\beta}2}\big[1 + \frac{(1-\beta)^2}{24}L^2 + \frac{(1-\beta)^4}{1920}L^4\big]}\cdot\frac{z}{x(z)}\cdot\Big[1 + \Big(\frac{(1-\beta)^2\alpha^2}{24(FK)^{1-\beta}} + \frac{\rho\beta\nu\alpha}{4(FK)^{\frac{1-\beta}2}} + \frac{2-3\rho^2}{24}\nu^2\Big)T\Big]$$
with $L=\ln(F/K)$, $z = \frac\nu\alpha(FK)^{\frac{1-\beta}2}L$, $x(z) = \ln\frac{\sqrt{1-2\rho z+z^2}+z-\rho}{1-\rho}$.

Reading it: $z/x(z)$ carries the smile shape in moneyness ($z/x \approx 1 - \tfrac12\rho z + \dots$, so $\rho<0$ raises vols at low strikes); the last bracket is the time correction to the ATM level.
**At the money** ($z\to0$, use the series to avoid 0/0): $\sigma_{ATM} = \alpha\,[1 + (\rho\nu\alpha/4 + (2-3\rho^2)\nu^2/24)T]$ for $\beta=1$ (tested exactly).

## 3. Accuracy and arbitrage
It is an asymptotic expansion, so error grows with $\nu^2 T$. Test: against a direct Monte Carlo simulation of the SDE ($T=0.25$, $\nu=0.6$, $\rho=-0.5$) it agrees within 0.3 vol pt at 90/100/110.
For long expiries and large $\nu$ the formula can produce a negative implied density at low strikes (butterfly arbitrage). Remedies: the arbitrage-free SABR PDE of Hagan et al. (2014), or fitting SVI / Heston instead for long expiries. Our real-data report checks the implied density of each fitted slice.

## 4. SABR vs Heston in this project
SABR fits each smile separately, so it fits better slice by slice but says nothing consistent about dynamics across expiries. Heston is one set of 5 parameters for the whole surface: worse slice fits, but a model you can price path-dependent products with. The calibration report compares the two fit errors per expiry.
