# 05 · Longstaff–Schwartz least-squares Monte Carlo

Code: `src/derivpricing/engines/lsm.py`

## 1. The problem
A Bermudan option exercisable at $t_1<\dots<t_N=T$ is worth
$V_0 = \sup_\tau \mathbb E[e^{-r\tau}g(S_\tau)]$ over stopping times. Dynamic programming:
$V_{t_n} = \max\big(g(S_{t_n}),\ C_n(S_{t_n})\big)$, $C_n(s) = \mathbb E[e^{-r\Delta}V_{t_{n+1}}\mid S_{t_n}=s]$.
Monte Carlo simulates forward, but the recursion runs backward and needs a *conditional* expectation at every date, which plain MC cannot give.

## 2. The idea
A conditional expectation is the $L^2$ projection onto functions of $S_{t_n}$. Approximate it by least squares on a finite basis: regress the realised discounted future cash flow of each path on $\{\phi_k(S_{t_n})\}$. The fitted value $\hat C_n(s) = \sum_k\hat\beta_k\phi_k(s)$ is used **only to decide** whether to exercise; the cash flow assigned to the path is the realised one, not the fitted value. That keeps the estimator from compounding regression error through the recursion.

## 3. Algorithm (backward pass)
1. Cash flow at $T$: $g(S_T)$.
2. For $n = N-1,\dots,1$: discount cash flows one step; among in-the-money paths regress cash flow on $\phi(S_{t_n})$; where $g(S_{t_n}) > \hat C_n(S_{t_n})$, replace the cash flow by $g(S_{t_n})$.
3. Price = mean of discounted cash flows.

**Only in-the-money paths** enter the regression: exercise is never optimal when $g=0$, and the basis is better spent fitting $C$ where the decision is made.

**Basis.** Weighted Laguerre polynomials $e^{-x/2}L_k(x)$, $k\le3$, in moneyness $x = S/K$, as in the paper. Scaling by $K$ keeps the design matrix well conditioned (raw powers of $S\approx40$ span $1$ to $6\times10^4$).

## 4. Bias
- Any exercise rule is suboptimal, so applying the fitted rule to **independent** paths gives a **lower bound** (`fit_paths` argument).
- Pricing on the same paths used to fit gives a small upward "foresight" bias (the rule has seen those paths' futures).
- 50 exercise dates price a Bermudan, worth slightly less than the continuously exercisable American.
Andersen–Broadie duality gives a matching upper bound; a natural extension for later.

**Check.** S=36, K=40, r=6%, σ=20%, T=1, 50 dates, 200k paths: LSM 4.4758 ± 0.0065, below the PDE American 4.4867 as theory says; Longstaff–Schwartz report 4.472.
