# `cointf`

Fast, exact likelihoods for the Housworth-Stahl model of crossover interference.

Crossovers come from two pathways: an interference pathway, a stationary renewal process with
gamma-mixture gaps governed by `nu`, and an escape pathway, a Poisson process carrying a fraction
`p` of crossovers. The likelihood sums over the unobserved pathway labels with an O(n²) forward
recursion per meiosis.

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
```

- `cointf.fastlik`: vectorised implementation (`Meioses`, `loglik_fast`). The series functions
  for `f*`, `S*` and the integrated tail are available there too.
- `cointf.naive`: one-meiosis-at-a-time reference with a 200-term series, used for validation.
- `cointf.simintf`: simulation (`simulate_meiosis`, `simulate_meioses`).

## Accuracy and speed

`loglik_fast` agrees with `loglik_naive` to about 1e-13 per meiosis, including when every meiosis
has its own `L`, `nu` and `p`.

Against `xoi::stahlLoglik` (R, v0.76), single-threaded, per (`nu`, `p`) evaluation:

| Data | xoi | `loglik_fast` |
|---|---|---|
| 2000 meioses, L = 1.5 | 34 ms | 7.6 ms |
| 2000 meioses, L = 2.5 | 115 ms | 24 ms |
| 1000 meioses, L = 3.5 | 203 ms | 18 ms |
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
