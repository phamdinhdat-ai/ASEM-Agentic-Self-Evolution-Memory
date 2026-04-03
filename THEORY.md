# ASEM Theory Notes

This document summarizes two key theoretical results referenced in the ASEM paper.

## Theorem 5.1 (Utility EMA Convergence)

Given a reward sequence $r_t$ with bounded variance $\sigma^2$ and the EMA update

$$
q_{t+1} = q_t + \alpha (r_t - q_t),
$$

the expected utility converges to the mean reward:

$$
\mathbb{E}[q_t] \to \mathbb{E}[r_t] \quad \text{as } t \to \infty,
$$

with convergence rate $(1-\alpha)^t$.

## Proposition 5.1 (Asymptotic Variance)

Under the same assumptions, the asymptotic variance of $q_t$ is bounded by

$$
\mathrm{Var}(q_t) \le \frac{\alpha \sigma^2}{2 - \alpha}.
$$

This bound motivates the trade-off between fast adaptation (large $\alpha$) and
stable utility estimates (small $\alpha$).
