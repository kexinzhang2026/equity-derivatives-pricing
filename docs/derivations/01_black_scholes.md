# 01 · Black-Scholes price and Greeks

Code: `src/derivpricing/models/black_scholes.py`

## Setup
Under the risk-neutral measure Q with continuous dividend yield q:

$$dS_t = (r-q)S_t\,dt + \sigma S_t\,dW_t \quad\Rightarrow\quad S_T = S_0\exp\!\Big((r-q-\tfrac12\sigma^2)T + \sigma\sqrt T\,Z\Big),\ Z\sim N(0,1).$$

(Itô on $\ln S$: the $-\tfrac12\sigma^2$ comes from the second-order term $\tfrac12 \cdot (-1/S^2)\cdot\sigma^2S^2$.)

Define forward $F = \mathbb E^Q[S_T] = S_0 e^{(r-q)T}$ and discount factor $D = e^{-rT}$.

## Call price
$$C = D\,\mathbb E^Q[(S_T-K)^+] = D\big(\mathbb E^Q[S_T\mathbf 1_{S_T>K}] - K\,Q(S_T>K)\big).$$

**Second term.** $S_T > K \iff Z > -d_2$ with
$d_2 = \dfrac{\ln(F/K) - \tfrac12\sigma^2T}{\sigma\sqrt T}$, so $Q(S_T>K) = N(d_2)$.

**First term.** Change measure to the share measure $Q^S$, with $\frac{dQ^S}{dQ} = S_T/F$.
Under $Q^S$, $Z$ has mean $\sigma\sqrt T$ (Girsanov), so
$\mathbb E^Q[S_T\mathbf 1_{S_T>K}] = F\,Q^S(S_T>K) = F\,N(d_1)$, with $d_1 = d_2 + \sigma\sqrt T$.

$$\boxed{C = D\,[F N(d_1) - K N(d_2)]}\qquad \boxed{P = D\,[K N(-d_2) - F N(-d_1)]}$$

Parity check: $C-P = D[F(N(d_1)+N(-d_1)) - K(\cdot)] = D(F-K)$. This is tested in `test_put_call_parity`.

## The identity that simplifies every Greek
$$S e^{-qT} n(d_1) = K e^{-rT} n(d_2).$$
Proof: $n(d_2)/n(d_1) = \exp\big(\tfrac12(d_1^2-d_2^2)\big) = \exp\big(\tfrac12(d_1+d_2)(d_1-d_2)\big) = \exp(\ln(F/K)) = F/K$.

## Delta
$\partial C/\partial S = e^{-qT}N(d_1) + \big[Se^{-qT}n(d_1) - Ke^{-rT}n(d_2)\big]\,\partial d_1/\partial S$.
The bracket is zero by the identity (since $\partial d_1/\partial S = \partial d_2/\partial S$), so $\Delta_C = e^{-qT}N(d_1)$.

## Gamma, vega
$\Gamma = \dfrac{e^{-qT}n(d_1)}{S\sigma\sqrt T}$, $\ \mathcal V = Se^{-qT}n(d_1)\sqrt T$. Both are the same for calls and puts (parity: $C-P$ does not depend on σ and is linear in S).

## Theta, rho
Theta (calendar, $\partial V/\partial t = -\partial V/\partial T$):
$\Theta_C = -\dfrac{Se^{-qT}n(d_1)\sigma}{2\sqrt T} + qSe^{-qT}N(d_1) - rKe^{-rT}N(d_2)$.
Rho: $\rho_C = KTe^{-rT}N(d_2)$.

Consistency check worth knowing: plugging these into the BS PDE
$\Theta + (r-q)S\Delta + \tfrac12\sigma^2S^2\Gamma = rV$ holds exactly.

## Implied volatility
$\sigma \mapsto C(\sigma)$ is strictly increasing (vega > 0), so the inverse is unique inside the no-arbitrage bounds. But conditioning is $\delta\sigma \approx \delta C/\mathcal V$: when vega is tiny (deep ITM, short-dated, low vol) small price noise becomes huge vol noise. Hence: invert on the OTM side (OTM puts below the forward, OTM calls above). See `test_implied_vol_roundtrip`.
