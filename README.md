# `cointf`

[![Documentation](https://img.shields.io/badge/docs-aabiddanda.github.io%2Fcointf-blue)](https://aabiddanda.github.io/cointf/)
[![Docs build](https://github.com/aabiddanda/cointf/actions/workflows/docs.yml/badge.svg)](https://github.com/aabiddanda/cointf/actions/workflows/docs.yml)
[![Release](https://img.shields.io/github/v/release/aabiddanda/cointf?include_prereleases&sort=semver)](https://github.com/aabiddanda/cointf/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.11-3776AB?logo=python&logoColor=white)](#python)
[![R](https://img.shields.io/badge/R-reticulate-276DC3?logo=r&logoColor=white)](#r)

Fast, exact likelihoods for the Housworth-Stahl model of crossover interference.

Crossovers come from two pathways: an interference pathway, a stationary renewal process with
gamma-mixture gaps governed by `nu`, and an escape pathway, a Poisson process carrying a fraction
`p` of crossovers. The likelihood sums over the unobserved pathway labels with an O(n²) forward
recursion per meiosis. For phase-unknown (quartet) data, where the crossovers of two meioses are
pooled, an O(n³) recursion sums over the assignment of each crossover to either meiosis.

## Installation

`cointf` is not on PyPI or CRAN. Install it from a clone of this repository.

```bash
git clone https://github.com/aabiddanda/cointf.git
cd cointf
```

### Python

```bash
pip install .                # numpy and scipy are installed as dependencies
pip install -e ".[tests]"    # for development: editable install plus pytest and hypothesis
```

### R

The R package in `r/` is a thin interface to the Python package through
[reticulate](https://rstudio.github.io/reticulate/). From the root of the clone:

```bash
R CMD INSTALL r
```

R needs the `reticulate` package (version 1.41 or later). Python then comes in one of two ways:

- **Automatic (default).** On first use, reticulate builds a private Python environment and
  installs `cointf` from the GitHub repository into it. This needs internet access
  once, and takes about 20 seconds.
- **Your own environment.** If you already ran `pip install .` in a Python environment, point
  reticulate at it before loading the package:

  ```r
  Sys.setenv(RETICULATE_PYTHON = "/path/to/venv/bin/python")
  library(cointf)
  ```

To install from GitHub without a clone, use `pip install "git+https://github.com/aabiddanda/cointf"`
for Python, or `remotes::install_github("aabiddanda/cointf", subdir = "r")` for R.

## Documentation

- [Model and likelihood](docs/model.md): the model, both recursions and their closed-form checks
- [User guide](docs/usage.md): data format, per-meiosis parameters, kernels, quartets, simulation
- [Numerics, accuracy and speed](docs/numerics.md): truncation, the Chebyshev kernel, the test
  strategy and benchmarks
- [Using cointf from R](docs/r.md): installation, how R finds Python, data conversion, an R
  maximum-likelihood fit
- [Example: maximum-likelihood fit of `nu` and `p`](docs/examples/mle.ipynb), with standard errors
  and profile likelihood
- API reference: [Python](docs/api/python.md) (every public class and function, from the
  docstrings) and [R](docs/api/r.md) (from the R help pages)

The pages read on GitHub as they are. They are also published as an HTML site at
<https://aabiddanda.github.io/cointf/>, rebuilt from `main`. To build the site locally, see
[Building the documentation](#building-the-documentation).

## Usage

### Python

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

### R

The R functions have the same names and arguments as the Python ones. Crossover data are a list
with one numeric vector of positions (in Morgans) per meiosis.

```r
library(cointf)

data <- simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 200, seed = 7)  # same data as Python
loglik_fast(data, 1.5, nu = 4, p = 0.05)
loglik_fast(data, 1.5, nu = 4, p = 0.05, per_meiosis = TRUE)

mz <- meioses(data, 1.5)                  # prepare once, evaluate many times
kernel <- chebyshev_kernel()              # optional faster kernel
loglik_fast(mz, NULL, nu = 4, p = 0.05, kernel = kernel)

quartets <- simulate_quartets(L = 1.5, nu = 4, p = 0.05, n = 200, seed = 7)
loglik_quartet(quartets, 1.5, nu = 4, p = 0.05)

# Long-format data (one row per crossover; NA position for a meiosis without crossovers)
meiosis_list(position = c(0.2, 0.9, NA, 0.5), meiosis = c("a", "a", "b", "c"))

# Maximum likelihood for (nu, p)
negloglik <- function(theta) -loglik_fast(mz, NULL, nu = exp(theta[1]), p = plogis(theta[2]), kernel = kernel)
fit <- optim(c(log(2), -2), negloglik)
c(nu = exp(fit$par[1]), p = plogis(fit$par[2]))
```

### Modules

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

## Development and testing

All commands below run from the root of a clone. There are two test suites: one for the Python
package (`tests/`) and one for the R interface (`r/tests/`).

### Python tests

```bash
pip install -e ".[tests]"     # editable install plus pytest, pytest-cov, pytest-xdist, hypothesis
pytest                        # full suite, about 10 s
pytest tests/test_quartet.py  # one file
pytest -k chebyshev           # tests whose names match a pattern
pytest -n auto                # in parallel
pytest --cov=cointf --cov-branch --cov-report=term-missing   # with coverage
```

### R tests

The R tests call the Python package, so first install the R package and choose which Python it
uses. To test your local changes, point `RETICULATE_PYTHON` at the environment where you ran
`pip install -e .` above:

```bash
Rscript -e 'install.packages(c("reticulate", "testthat"))'   # once
R CMD INSTALL r
export RETICULATE_PYTHON=$(which python)                      # the Python with your cointf install
Rscript -e 'testthat::test_local("r")'                        # full R suite
Rscript -e 'testthat::test_local("r", filter = "reticulate")' # one file (test-reticulate.R)
```

Without `RETICULATE_PYTHON`, the R package installs `cointf` from GitHub (`main`), so the
tests would run against that version rather than your local code.

Two optional checks:

```bash
COINTF_TEST_INSTALL=true Rscript -e 'testthat::test_local("r")'   # also test the automatic GitHub install (needs network)
R CMD build r && R CMD check --no-manual cointf_*.tar.gz           # full package check, including the tests
```

The R documentation in `r/man` is generated from the roxygen comments in `r/R` with
`Rscript -e 'roxygen2::roxygenise("r")'` (needs the `roxygen2` package).

### Building the documentation

```bash
pip install -e ".[docs]"                          # Sphinx, MyST-NB, numpydoc, Furo
sphinx-build -b html docs docs/_build/html        # then open docs/_build/html/index.html
sphinx-build -n -b html docs docs/_build/html     # same, warning about any broken reference
```

The Python API pages are generated from the docstrings at build time. The R API page
(`docs/api/r.md`) is generated from `r/man`; after changing roxygen comments, regenerate both:

```bash
Rscript -e 'roxygen2::roxygenise("r")'
Rscript docs/api/make_r_reference.R
```

The site is published to GitHub Pages by `.github/workflows/docs.yml`. It rebuilds on every push to
`main`, using the same strict build as above (any warning fails it), and can also be run by hand
with `gh workflow run docs.yml --ref main`. It does not run R, so commit the regenerated
`docs/api/r.md` along with any roxygen changes.

Several of the coding conventions were contributed by Claude and Claude Code. As such, code and
series truncations should be reviewed as necessary.

## Contact

@aabiddanda
