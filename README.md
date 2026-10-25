# `cointf`

Fast, exact likelihoods for the Housworth-Stahl model of crossover interference.

Crossovers come from two pathways: an interference pathway, a stationary renewal process with
gamma-mixture gaps governed by `nu`, and an escape pathway, a Poisson process carrying a fraction
`p` of crossovers. The likelihood sums over the unobserved pathway labels with an O(n²) forward
recursion per meiosis. For phase-unknown (quartet) data, where the crossovers of two meioses are
pooled, an O(n³) recursion sums over the assignment of each crossover to either meiosis.

## Usage

```python
from cointf import Meioses, loglik_fast, loglik_naive, simulate_meioses

data = simulate_meioses(L=1.5, nu=4.0, p=0.05, n=200, seed=7)  # positions in Morgans

loglik_fast(data, 1.5, nu=4.0, p=0.05)          # summed log-likelihood
loglik_fast(data, 1.5, nu=4.0, p=0.05, per_meiosis=True)

# Reuse the data structure across many evaluations (e.g. inside an optimiser)
mz = Meioses(data, 1.5)
loglik_fast(mz, None, nu=4.0, p=0.05)

# Chromosome length, nu and p can each be one value or one value per meiosis
loglik_fast(mz, None, nu=[4.0] * 100 + [8.0] * 100, p=0.05)

loglik_naive(data, 1.5, nu=4.0, p=0.05)         # slow reference implementation

# Faster approximate kernel: piecewise Chebyshev fits of f* and S*, cached per nu
from cointf import ChebyshevKernel
kernel = ChebyshevKernel()
loglik_fast(mz, None, nu=4.0, p=0.05, kernel=kernel)

# Phase-unknown data: crossovers of two meioses pooled
from cointf import loglik_quartet, simulate_quartets
quartets = simulate_quartets(L=1.5, nu=4.0, p=0.05, n=200, seed=7)
loglik_quartet(quartets, 1.5, nu=4.0, p=0.05)
```

- `cointf.fastlik`: vectorised implementation (`Meioses`, `loglik_fast`). The series functions
  for `f*`, `S*` and the integrated tail are available there too.
- `cointf.naive`: one-meiosis-at-a-time reference with a 200-term series, used for validation.
- `cointf.chebyshev`: `ChebyshevKernel`, an approximate drop-in `kernel=` for `loglik_fast` and
  `loglik_quartet`. Errors in `log f*` and `log S*` are about 1e-14 to 3e-12 for `nu` in
  [0.3, 100], and it is 6–10× faster when `nu` is shared. Distances outside its fitted range use
  the exact series.
- `cointf.quartet`: phase-unknown likelihood `loglik_quartet`, checked against a 3ⁿ brute force
  (`cointf.naive.loglik_quartet_naive`).
- `cointf.simintf`: simulation (`simulate_meiosis`, `simulate_meioses`, `simulate_quartets`).

## Accuracy and speed

`loglik_fast` agrees with `loglik_naive` to about 1e-13 per meiosis, including when every meiosis
has its own `L`, `nu` and `p`.

Against `xoi::stahlLoglik` (R, v0.76), single-threaded, per (`nu`, `p`) evaluation (exact
series kernel; timed 2026-10-25):

| Data | xoi | `loglik_fast` |
|---|---|---|
| 2000 meioses, L = 1.5 | 34 ms | 8.6 ms |
| 2000 meioses, L = 2.5 | 115 ms | 22 ms |
| 1000 meioses, L = 3.5 | 203 ms | 21 ms |
| 2000 meioses, L varies (0.5–3) | 1036 ms | 18 ms |

The gap widens with crossovers per meiosis, because xoi enumerates all 2ⁿ pathway labelings.
`loglik_fast` also evaluates only the series terms that matter for each distance (about 10),
with explicit error bounds, and computes `f*` only at gaps and `S*` only at chromosome ends.

## Development

```bash
pip install -e ".[tests]"
pytest
```

Several of the coding conventions were contributed by Claude and Claude Code. As such, code and
series truncations should be reviewed as necessary.

## Contact

@aabiddanda
