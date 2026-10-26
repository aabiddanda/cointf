# User guide

This guide uses Python. The R interface has the same functions and arguments; see
[Using cointf from R](r.md) for what differs.

## Installation

From a clone of the repository, on the `dev` branch:

```bash
pip install .               # numpy and scipy are installed as dependencies
pip install -e ".[tests]"   # for development: editable install plus the test tools
pytest                      # optional: run the test suite
```

## Data

Crossover data are a list with one entry per meiosis. Each entry holds that meiosis's crossover
positions in Morgans, and may be empty.

```python
data = [[0.31, 1.12], [0.70], []]   # three meioses; the third has no crossovers
L = 1.5                             # chromosome length in Morgans (or one value per meiosis)
```

Positions must be finite, lie in $[0, L]$ and be distinct within a meiosis. Order does not matter.
Invalid data raise `ValueError` with the index of the offending meiosis.

### Preparing data once: `Meioses`

`Meioses` validates and sorts the positions, groups meioses by crossover count and precomputes
every distance the likelihood needs. Build it once and reuse it for every parameter value:

```python
from cointf import Meioses, loglik_fast

mz = Meioses(data, L)               # L may be a scalar or one value per meiosis
mz                                  # Meioses(M=3, crossovers=3, max_count=2)
loglik_fast(mz, None, nu=4.0, p=0.05)
```

When passing a `Meioses`, the length argument must be `None`, because the object carries its own
lengths.

## Single-meiosis likelihood: `loglik_fast`

```python
loglik_fast(data, L, nu, p, *, kernel=None, per_meiosis=False)
```

- `nu > 0`: interference parameter. `nu = 1` is no interference.
- `0 <= p <= 1`: fraction of crossovers escaping interference.
- `per_meiosis=True` returns one log-likelihood per meiosis instead of the sum.

```python
total = loglik_fast(data, 1.5, nu=4.0, p=0.05)
each = loglik_fast(data, 1.5, nu=4.0, p=0.05, per_meiosis=True)   # shape (3,)
```

### Parameters that vary between meioses

`nu`, `p` and `L` each accept a scalar or an array with one value per meiosis. For example, `nu`
can depend on a covariate:

```python
import numpy as np

age = np.array([25.0, 31.0, 40.0])
nu = np.exp(1.2 + 0.01 * age)       # one nu per meiosis
loglik_fast(data, 1.5, nu=nu, p=0.05, per_meiosis=True)
```

An array whose entries are all equal is treated as the scalar, so a pooled fit costs the same
whichever form is passed.

## Kernels

The likelihood needs $\log f^*$ at the gaps between crossovers and $\log S^*$ at the distances to
the chromosome ends. These come from a *kernel*, a function

```python
kernel(u_gap, nu_gap, u_end, nu_end) -> (log_f, log_S)
```

The default, `series_kernel`, evaluates the exact gamma series. `ChebyshevKernel` replaces it
with piecewise Chebyshev fits that are built once per `nu` and cached:

```python
from cointf import ChebyshevKernel

kernel = ChebyshevKernel()          # keep one instance, so its cache is reused
loglik_fast(mz, None, nu=4.0, p=0.05, kernel=kernel)
```

When to use it:

- **Shared `nu`** (pooled fits, profiling over `p`, grids): 6–10× faster, with log-likelihood
  differences around 1e-13 per meiosis. Building a fit for a new `nu` takes 3–9 ms, which is paid
  once per distinct `nu`.
- **Many distinct `nu` values in one call** (more than `max_distinct`, default 16): the kernel
  falls back to the exact series, so there is no speed-up yet.
- Distances outside `[u_min, u_max]` (default `[1e-6, 5]`, on the interference scale) always use
  the exact series. Longer maps are therefore still correct, but gain less.

Any callable with the kernel signature can be passed, for example to experiment with other
approximations. The outputs must have the same shapes as `u_gap` and `u_end`.

## Phase-unknown data: `loglik_quartet`

For a parent in a family with two offspring and no grandparents, each chromosome yields the pooled
crossovers of two meioses, with no record of which meiosis each crossover came from. Pass one
entry per quartet:

```python
from cointf import loglik_quartet, simulate_quartets

quartets = simulate_quartets(L=1.5, nu=4.0, p=0.05, n=200, seed=7)
loglik_quartet(quartets, 1.5, nu=4.0, p=0.05)
loglik_quartet(quartets, 1.5, nu=4.0, p=0.05, kernel=kernel)   # same kernels work
```

The interface matches `loglik_fast`. `per_meiosis=True` returns one value per **quartet** (the
name is kept so the two functions are interchangeable), and a `Meioses` built from quartet data
holds one entry per quartet.

## Simulation

```python
from cointf import simulate_meiosis, simulate_meioses, simulate_quartets

simulate_meioses(L=1.5, nu=4.0, p=0.05, n=1000, seed=1)    # list of sorted arrays
simulate_quartets(L=1.5, nu=4.0, p=0.05, n=1000, seed=1)   # two meioses pooled per entry
```

The simulator draws the first interference crossover from the stationary distribution, so
simulated data follow exactly the model the likelihood assumes.

## Reference implementations

`cointf.naive` contains slow, direct implementations used to validate the fast code:

- `loglik_meiosis_naive` and `loglik_naive`: the $O(n^2)$ recursion one meiosis at a time, with a
  fixed 200-term series;
- `loglik_quartet_naive`: brute force over all $3^n$ labelings of a quartet.

Use them for spot checks, not for fitting.

## Fitting

`cointf` provides likelihoods, not optimisers. Wrap `loglik_fast` in any optimiser; working on
$\log \nu$ and $\operatorname{logit} p$ keeps the parameters in range. See
[the maximum-likelihood example](examples/mle.ipynb) for a complete fit with standard errors.
