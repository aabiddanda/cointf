# cointf

Fast, exact likelihoods for the Housworth–Stahl model of crossover interference.

`cointf` is a Python package with an R interface. It evaluates the likelihood of observed
crossover positions under a two-pathway model:
crossovers that obey interference (a gamma-type renewal process governed by $\nu$) and crossovers
that escape it (a Poisson process carrying a fraction $p$ of all crossovers). It is built for
repeated evaluation inside optimisers and samplers, and for parameters that vary between meioses.

## Contents

| Page | What it covers |
|---|---|
| [Model and likelihood](model.md) | The generative model, the forward recursion for single meioses, and the $O(n^3)$ recursion for phase-unknown (quartet) data |
| [User guide](usage.md) | Data format, pooled and per-meiosis parameters, kernels, quartets and simulation |
| [Numerics, accuracy and speed](numerics.md) | How the series are truncated, the Chebyshev kernel, how correctness is tested, and benchmarks |
| [Using cointf from R](r.md) | Installing the R package, how it finds Python, data conversion, an R maximum-likelihood fit |
| [Example: maximum-likelihood fit](examples/mle.ipynb) | Estimating $\nu$ and $p$ from a dataset, with standard errors and a likelihood surface |

## Installation

`cointf` is installed from a clone of the GitHub repository, not from PyPI or CRAN. The current
code is on the `dev` branch.

```bash
git clone https://github.com/aabiddanda/cointf.git && cd cointf && git checkout dev
pip install .        # Python
R CMD INSTALL r      # R (needs the reticulate package); see "Using cointf from R"
```

## At a glance

```python
from cointf import Meioses, loglik_fast, simulate_meioses

data = simulate_meioses(L=1.5, nu=4.0, p=0.05, n=200, seed=7)  # positions in Morgans
mz = Meioses(data, 1.5)                                         # prepare once
loglik_fast(mz, None, nu=4.0, p=0.05)                           # evaluate many times
```

| Function | Purpose |
|---|---|
| `loglik_fast` | Exact log-likelihood of single meioses (vectorised) |
| `loglik_quartet` | Log-likelihood of phase-unknown data: crossovers of two meioses pooled |
| `ChebyshevKernel` | Faster approximate kernel for either likelihood (errors ~1e-14 to 3e-12 in the log kernel terms) |
| `loglik_naive`, `loglik_meiosis_naive`, `loglik_quartet_naive` | Slow reference implementations used for validation |
| `simulate_meioses`, `simulate_quartets` | Simulation from the same model |

The R functions have the same names and arguments (`meioses()` and `chebyshev_kernel()` stand in
for `Meioses` and `ChebyshevKernel`); see [Using cointf from R](r.md).

## Conventions

- Positions and lengths are in **Morgans** (1 M = 100 cM), not base pairs.
- A "meiosis" is one chromosome in one gamete. The expected number of crossovers per meiosis is
  $L$, for every $(\nu, p)$.
- $\nu = 1$ means no interference; $\nu > 1$ means positive interference. $p = 0$ is the pure
  interference model and $p = 1$ a Poisson process.
