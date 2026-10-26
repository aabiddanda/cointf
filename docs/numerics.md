# Numerics, accuracy and speed

## Exact series (`series_kernel`)

$f^*$, $S^*$ and $G$ are infinite gamma series. With $z = 2\nu u$ and shape $a = k\nu$, term $k$
of $S^*$ is $2^{-k} Q(k\nu, z)$, where $Q$ is the regularised upper incomplete gamma function.

**Truncation with explicit error bounds.** For each distance, only a window of series indices is
evaluated:

- Terms below the window are dropped, and terms above it are replaced by their limit $Q = 1$
  (summed in closed form).
- The window is found by bisection, using the Chernoff bound $e(a, z) = z - a + a\log(a/z)$ on
  each gamma tail.
- The total error from both is kept below $e^{-40}$ relative to a lower bound on the result.
- For $f^*$ the log terms are concave in $k$, so the window is every term within a fixed margin of
  the largest one.
- Indices are capped at 200.

Typically about 10 terms per distance need an incomplete-gamma or log-gamma evaluation.

**Only what is needed.** The recursion needs $\log f^*$ only at gaps between crossovers and
$\log S^*$ only at the distances to the chromosome ends, so each is evaluated only there. $G$ is
needed once per distinct value of $qL$.

**Vectorisation.** Meioses are grouped by crossover count, and the forward recursion runs once per
group on a dense array. All kernel values for one likelihood call are computed in a single
vectorised call, which is what allows the kernel to be swapped.

## Chebyshev kernel (`ChebyshevKernel`)

For a fixed $\nu$, $\log f^*(u)$ and $\log S^*(u)$ are smooth for $u > 0$, so each can be replaced
by a piecewise polynomial built once from the exact series.

- **What is fitted.** $\log f^*(u) - (\nu - 1)\log u$ (the singular part near zero removed) and
  $\log S^*(u)$. Fitting in log space makes the error relative in $f^*$ and $S^*$.
- **Pieces.** 16 Chebyshev nodes per piece. Pieces below $u = 0.25$ use the variable $\log u$,
  where the functions are nearly linear. Pieces above use $u$, where $f^*$ has bumps near
  $u = 1/2, 1, \dots$ for large $\nu$. A piece is halved until its last three Chebyshev
  coefficients fall below the tolerance.
- **Evaluation.** Each point's piece is found by binary search, and the series is summed with the
  Clenshaw recurrence: 16 multiply-adds per point.
- **Exact fallback.** Distances outside $[10^{-6}, 5]$ use the exact series. $G$ is always exact.
- **Caching.** Fits are cached per $\nu$. With per-meiosis $\nu$, the kernel builds one fit per
  distinct value up to `max_distinct`; beyond that, it falls back to the exact series.

Measured accuracy (maximum absolute error in the log values, $u \in [10^{-6}, 5]$) and build cost:

| $\nu$ | pieces ($f^*$ / $S^*$) | error, $\log f^*$ | error, $\log S^*$ | build time |
|---|---|---|---|---|
| 0.3 | 11 / 10 | 1e-14 | 8e-15 | 6 ms |
| 1.5 | 10 / 12 | 1e-14 | 1e-14 | 4 ms |
| 4.3 | 15 / 16 | 4e-14 | 2e-14 | 5 ms |
| 8.5 | 22 / 22 | 1e-13 | 2e-14 | 6 ms |
| 20.3 | 36 / 32 | 4e-13 | 2e-14 | 7 ms |
| 100 | 78 / 58 | 3e-12 | 2e-14 | 9 ms |

**Caveats.**

- `tol` is an acceptance threshold on trailing coefficients, not a proven error bound. The table
  above is the empirical check.
- The piece layout changes discretely with $\nu$, so the likelihood has steps of about $10^{-13}$
  as $\nu$ varies. This is far below what affects an optimiser, but the surface is not smooth to
  every digit.

## How correctness is tested

The test suite (about 185 tests, 100% line and branch coverage) builds a chain of independent
checks:

1. **Series.** The fast series match the 200-term reference in `cointf.naive` to $10^{-10}$
   across $\nu \in [0.2, 150]$. $\nu = 1$ reproduces the exponential distribution, and $G$ matches
   numerical integration of $S^*$.
2. **Reference likelihood.** `loglik_meiosis_naive` matches brute-force enumeration of all $2^n$
   labelings, the $\nu = 1$ Poisson case, and crossover-count probabilities $P(N = 0, 1, 2)$
   estimated from 40,000 simulated meioses.
3. **Fast likelihood.** `loglik_fast` matches the reference to about $10^{-13}$ per meiosis,
   including random per-meiosis $L$, $\nu$ and $p$ (property-based tests), crossovers at the
   chromosome ends, and the edge cases $p = 0$, $p = 1$ and $\nu = 1$.
4. **Quartets.** `loglik_quartet` matches brute force over $3^n$ labelings and, independently,
   the sum of single-meiosis likelihoods over all $2^n$ splits of the crossovers between the two
   meioses. It also matches the rate-2 Poisson closed form and simulated $P(N = 0, 1)$.
   Two-parameter maximum-likelihood fits recover $(\nu, p)$ within 3 standard errors, including
   at the $p = 0$ boundary (see [Inference from quartet data](#inference-from-quartet-data)).
5. **Chebyshev kernel.** Compared with the exact kernel on dense grids over
   $\nu \in [0.3, 100]$, and end to end on the test bed, per-meiosis $\nu$ and long maps.

The fixed test bed (`tests/data`, 200 meioses simulated with $L = 1.5$, $\nu = 4$, $p = 0.05$,
seed 7) anchors regression values: the exact log-likelihood is about $-269.79$ at the truth and
exactly $-300$ at $\nu = 1$.

## Speed

All timings are single-threaded on a laptop, per likelihood evaluation.

**Against `xoi::stahlLoglik`** (R, v0.76; exact series kernel; timed 2026-10-25). Both agree to
within $2 \times 10^{-4}$ in total log-likelihood, consistent with xoi's 25-term series.

| Data | xoi | `loglik_fast` | Speed-up |
|---|---|---|---|
| 2000 meioses, $L = 1.5$ | 34 ms | 8.6 ms | 4× |
| 2000 meioses, $L = 2.5$ | 115 ms | 22 ms | 5× |
| 1000 meioses, $L = 3.5$ | 203 ms | 21 ms | 10× |
| 2000 meioses, $L$ from 0.5 to 3 | 1036 ms | 18 ms | 57× |

The gap widens with the number of crossovers per meiosis, because xoi enumerates all $2^n$
labelings.

**Chebyshev kernel** (20,000 meioses, $p = 0.05$; "warm" reuses a cached fit):

| $L$ | $\nu$ | exact | Chebyshev, warm | Speed-up |
|---|---|---|---|---|
| 1.5 | 1.5 | 115 ms | 11 ms | 10× |
| 1.5 | 4.3 | 80 ms | 11 ms | 7× |
| 1.5 | 20.3 | 70 ms | 12 ms | 6× |
| 3.5 | 4.3 | 381 ms | 47 ms | 8× |

**Quartets** (20,000 quartets, $\nu = 4.3$):

| $L$ | exact | Chebyshev |
|---|---|---|
| 1.5 | 207 ms | 44 ms |
| 2.5 | 553 ms | 116 ms |

## Inference from quartet data

Phase-unknown data carry less information than the same meioses with phase known, because the
likelihood must average over which meiosis each crossover came from. A simulation study measured
how much, and whether standard inference still works. It used 40 replicates per setting. Each
replicate fitted (log $\nu$, logit $p$) by Nelder–Mead to the pooled quartets, and to the same
meioses with phase known for comparison. Coverage is for 95% Wald intervals on the fitting scale.

| $L$, $\nu$, $p$, quartets | median $\hat\nu$ (quartet / phase known) | SD of $\log\hat\nu$ (quartet / phase known) | $\nu$ coverage | $p$ coverage |
|---|---|---|---|---|
| 2.0, 4, 0.05, 500 | 4.03 / 3.89 | 0.179 / 0.096 | 0.82 / 0.90 | 0.78 / 0.97 |
| 2.0, 4, 0.05, 2000 | 3.99 / 3.98 | 0.087 / 0.041 | 0.90 / 0.95 | 0.97 / 0.97 |
| 1.5, 10, 0.1, 1000 | 10.00 / 9.95 | 0.089 / 0.070 | 1.00 / 1.00 | 0.95 / 0.88 |
| 2.5, 6, 0.02, 1000 | 6.11 / 6.05 | 0.066 / 0.039 | 0.93 / 0.95 | 0.97 / 0.93 |

- **No bias.** Quartet estimates are centred on the truth in every setting.
- **Precision.** The SD of $\log\hat\nu$ is 1.3–2.1× that of phase-known fits to the same
  meioses. Since information scales with 1/SD², a quartet carries roughly 25–60% of the
  information about $\nu$ that its two phased meioses would.
- **Intervals.** Wald intervals are close to nominal from about 1000 quartets, but cover only
  0.78–0.82 with 500. For small datasets, prefer profile-likelihood intervals (as in the
  [example](examples/mle.ipynb)). With 40 replicates, the Monte Carlo SE of a coverage near 0.95
  is about 0.035.
- **At $p = 0$.** In 10 replicates with no escape, $\hat p$ was exactly 0 in 6, $\hat\nu$
  ranged from 5.6 to 7.0 (truth 6), and the likelihood-ratio statistic for $p = 0$ never reached
  2.71, the 5% critical value at a boundary.
