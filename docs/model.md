# Model and likelihood

## Two crossover pathways

Each meiosis (one chromosome in one gamete, of genetic length $L$ Morgans) carries crossovers
from two independent pathways. Write $q = 1 - p$.

**Interference pathway.** A stationary renewal process with rate $q$ per Morgan. Gaps have
density $q\, f^*(qx; \nu)$, where

$$
f^*(x; \nu) = \sum_{k \ge 1} 2^{-k}\, \mathrm{Gamma}\!\left(x;\ \text{shape } k\nu,\ \text{rate } 2\nu\right).
$$

This is the chiasma (gamma) model thinned to one chromatid: chiasmata form a gamma renewal
process with shape $\nu$, each chiasma is passed to a given chromatid with probability 1/2, and
the gap between retained chiasmata is a geometric sum of gamma gaps. $f^*$ has mean 1, so the
pathway has rate $q$ after rescaling by $q$.

**Escape pathway.** A homogeneous Poisson process with rate $p$ per Morgan.

The observed crossovers are the superposition of the two. Which pathway produced each crossover
is not observed. The expected count per meiosis is $L$ for every $(\nu, p)$.

Three functions of the gap distribution appear in the likelihood:

| Symbol | Definition | Role |
|---|---|---|
| $f^*(u)$ | gap density | between consecutive interference crossovers |
| $S^*(u) = 1 - F^*(u)$ | survival function | first crossover (stationarity) and last crossover (no more before the end) |
| $G(u) = \int_u^\infty S^*(t)\,dt$ | integrated tail | meiosis with no interference crossover; $G(0) = 1$ |

All three are evaluated on the interference scale $u = qx$.

## Single-meiosis likelihood

For one labeling of observed positions $x_1 < \dots < x_n$, let $y_1 < \dots < y_m$ be those
assigned to the interference pathway and $n - m$ the number assigned to escape. The joint density
of that labeling is

$$
p^{\,n-m} e^{-pL} \times
\begin{cases}
G(qL) & m = 0,\\[4pt]
q S^*(q y_1) \displaystyle\prod_{i=2}^{m} q f^*\!\big(q (y_i - y_{i-1})\big)\, S^*\!\big(q (L - y_m)\big) & m \ge 1.
\end{cases}
$$

The first factor in the $m \ge 1$ case is the stationary density of the first renewal point.
The likelihood sums this over all $2^n$ labelings.

**Forward recursion.** The sum factorises over the position of the most recent interference
crossover. Let $\alpha_j(s)$ be the summed density of the labels of $x_1, \dots, x_j$, where
$s = 0$ means no interference crossover so far and $s = i$ means the latest one is $x_i$. Then

$$
\alpha_j(s) = p\, \alpha_{j-1}(s) \quad (s < j), \qquad
\alpha_j(j) = \alpha_{j-1}(0)\, q S^*(q x_j) + \sum_{i<j} \alpha_{j-1}(i)\, q f^*\!\big(q(x_j - x_i)\big),
$$

and the likelihood is $e^{-pL}\big[\alpha_n(0)\, G(qL) + \sum_{i} \alpha_n(i)\, S^*(q(L - x_i))\big]$.
This costs $O(n^2)$ per meiosis instead of $O(2^n)$. Everything is computed in log space.

**Closed-form checks.** With $\nu = 1$, $f^*$ is the unit exponential and both pathways are
Poisson, so the log-likelihood is $-L$ whatever the positions. With $p = 1$ it is $-L$ as well
($n \log 1 - L$). Both are used in the tests.

## Phase-unknown (quartet) likelihood

In a nuclear family with two offspring and no grandparents, a parent's crossovers are seen as
switches in the siblings' shared inheritance, but which of the two meioses carried each crossover
is unknown. The data for one chromosome are then the pooled crossovers of two independent meioses
with the same $(\nu, p)$ and $L$.

Each observed crossover is an interference crossover of meiosis A, an interference crossover of
meiosis B, or an escape crossover. The two escape pathways superpose to a Poisson process of rate
$2p$, so escape crossovers need no meiosis label. There are $3^n$ labelings, and each contributes

$$
(2p)^{n_E} e^{-2pL} \times R_A \times R_B,
$$

where $R_A$ and $R_B$ are the interference-pathway densities above for the crossovers assigned to
A and to B (each $G(qL)$ if empty).

**Forward recursion.** The state is now the pair $(a, b)$ of latest interference crossovers in A
and B. At each crossover, one of three moves is possible: escape (state unchanged, factor $2p$),
assign to A (sum over $a$), or assign to B (sum over $b$). Because A and B are exchangeable,
the state array is symmetric and only one of the two assignment sums needs computing. The cost is
$O(n^3)$ per quartet: a quartet with 10 crossovers takes about 1 ms, against 0.17 s for brute
force over $3^{10}$ labelings.

**Closed-form check.** With $\nu = 1$ or $p = 1$ the pooled data are a rate-2 Poisson process, so
the log-likelihood is $n \log 2 - 2L$.

## Parameters per meiosis

Both likelihoods accept $\nu$, $p$ and $L$ either as single values or as one value per meiosis
(or per quartet). This supports models where the parameters depend on covariates, such as sex or
age, and per-individual analyses. Arrays whose entries are all equal take the pooled path, so a
pooled fit costs the same either way.

## References

- Housworth, E. A. and Stahl, F. W. (2003). Crossover interference in humans. *American Journal of
  Human Genetics* 73, 188–197.
- Campbell, C. L., Furlotte, N. A., Eriksson, N., Hinds, D. and Auton, A. (2015). Escape from
  crossover interference increases with maternal age. *Nature Communications* 6, 6260.
- Broman, K. W. and Weber, J. L. (2000). Characterization of human crossover interference.
  *American Journal of Human Genetics* 66, 1911–1926.
- The `xoi` R package (K. W. Broman) provides `stahlLoglik`, which computes the same
  single-meiosis likelihood by enumerating labelings; see [Numerics](numerics.md) for a timing
  comparison.
