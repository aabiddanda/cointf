# Using cointf from R

The R package in `r/` is a thin interface to the Python package through
[reticulate](https://rstudio.github.io/reticulate/). Every R function calls the Python function of
the same name, so results are identical to Python, down to the simulated data for a given seed.

## Installation

From the root of a clone of the repository (on the `dev` branch until it is merged into `main`):

```bash
Rscript -e 'install.packages("reticulate")'   # version 1.41 or later
R CMD INSTALL r
```

### How R finds Python

The R package needs a Python environment that has `cointf` installed. There are two ways to get
one.

- **Automatic (default).**
  - When the package is loaded, it declares the requirement
    `cointf @ git+https://github.com/aabiddanda/cointf@dev` with `reticulate::py_require()`.
  - The first time Python is needed, reticulate builds a private environment and installs `cointf`
    and its dependencies into it. This needs internet access once and takes about 20 seconds.
    Later sessions reuse the cached environment.
- **Your own environment.**
  - If you installed the Python package with `pip install .`, point reticulate at that Python
    before loading the R package.
  - The GitHub requirement is then not declared, so local changes to the Python code are what R
    uses.

```r
Sys.setenv(RETICULATE_PYTHON = "/path/to/venv/bin/python")
library(cointf)
```

Two options adjust this:

- `options(cointf.python = "<requirement>")`, set before `library(cointf)`, changes the declared
  requirement, e.g. to another branch or a local path.
- `options(cointf.python = NULL)` declares nothing. Use it when you select the environment another
  way, such as `reticulate::use_virtualenv()`.

Loading the R package does not start Python; that happens on the first call. To see which Python
is in use, run `reticulate::py_config()`.

## Functions

| R | Python | Purpose |
|---|---|---|
| `loglik_fast(data, L, nu, p, kernel, per_meiosis)` | `loglik_fast` | Exact log-likelihood of single meioses |
| `loglik_quartet(data, L, nu, p, kernel, per_meiosis)` | `loglik_quartet` | Log-likelihood of phase-unknown (quartet) data |
| `meioses(data, L)` | `Meioses` | Prepare data once for repeated evaluation |
| `chebyshev_kernel(...)` | `ChebyshevKernel` | Faster approximate kernel, cached per `nu` |
| `simulate_meioses(L, nu, p, n, seed)` | `simulate_meioses` | Simulate meioses |
| `simulate_quartets(L, nu, p, n, seed)` | `simulate_quartets` | Simulate quartet data |
| `meiosis_list(position, meiosis)` | (R only) | Convert long-format data to the list format |
| `cointf_python()` | the `cointf` module | Direct access to anything without a wrapper |

Each function has a help page, e.g. `?loglik_fast`.

## Data

Crossover data are a list with one numeric vector of positions (in Morgans) per meiosis. Use
`numeric(0)` for a meiosis without crossovers.

```r
data <- list(c(0.31, 1.12), 0.70, numeric(0))
loglik_fast(data, 1.5, nu = 4, p = 0.05)
loglik_fast(data, 1.5, nu = 4, p = 0.05, per_meiosis = TRUE)
```

Data often come as one row per crossover. `meiosis_list()` converts that format. A meiosis without
crossovers is kept by giving it one row with `NA` position:

```r
df <- data.frame(id = c("a", "a", "b", "c"), pos = c(0.2, 0.9, NA, 0.5))
data <- meiosis_list(df$pos, df$id)   # list(a = c(0.2, 0.9), b = numeric(0), c = 0.5)
```

`L`, `nu` and `p` each accept one value or one value per meiosis, as in Python.

## Repeated evaluation

Each call from an R list sends the data to Python and builds the internal data structure
(`Meioses`). For 20,000 meioses that takes about 80 ms, most of it in building the structure.
When the same data are evaluated many times, as in an optimiser, build the structure once with
`meioses()` and pass it with `L = NULL`. Each evaluation then costs only the likelihood itself
(a few ms for 20,000 meioses with the Chebyshev kernel).

```r
mz <- meioses(data, 1.5)
kernel <- chebyshev_kernel()          # keep one kernel so its cache is reused
loglik_fast(mz, NULL, nu = 4, p = 0.05, kernel = kernel)
loglik_quartet(mz, NULL, nu = 4, p = 0.05, kernel = kernel)   # same object works for quartets
```

The object returned by `meioses()` and the kernel are Python objects. They stay in Python between
calls; only numbers cross back into R.

## Example: maximum-likelihood fit

The same fit as the [Python example](examples/mle.ipynb), working on $(\log\nu, \operatorname{logit} p)$:

```r
library(cointf)

data <- simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 2000, seed = 1)
mz <- meioses(data, 1.5)
kernel <- chebyshev_kernel()

negloglik <- function(theta) {
  -loglik_fast(mz, NULL, nu = exp(theta[1]), p = plogis(theta[2]), kernel = kernel)
}

fit <- optim(c(log(2), -2), negloglik, hessian = TRUE)
se <- sqrt(diag(solve(fit$hessian)))
est <- fit$par
ci <- rbind(est - 1.96 * se, est + 1.96 * se)

data.frame(
  estimate = c(nu = exp(est[1]), p = plogis(est[2])),
  lower = c(exp(ci[1, 1]), plogis(ci[1, 2])),
  upper = c(exp(ci[2, 1]), plogis(ci[2, 2]))
)
#>      estimate      lower     upper
#> nu 3.77266971 3.30427871 4.3074565
#> p  0.04922224 0.02937439 0.0813569
```

The fit takes about 0.06 s and agrees with the Python notebook to every printed digit (the data
are identical for the same seed). For phase-unknown data, replace `loglik_fast` with
`loglik_quartet`.

## Testing

```bash
export RETICULATE_PYTHON=$(which python)                      # Python with your local install
Rscript -e 'testthat::test_local("r")'                        # full R suite
COINTF_TEST_INSTALL=true Rscript -e 'testthat::test_local("r")'   # also test the GitHub install
```

The R tests cover:

- the Python reference values, the quartet closed form and brute force, and the Chebyshev kernel;
- that results are identical to calling Python directly;
- conversion of awkward inputs (empty, named, integer, unsorted, `NA`/`Inf`) and per-meiosis
  parameters;
- Python objects surviving round trips through R;
- a bound on conversion time;
- in fresh R sessions, that loading the package does not start Python and that
  `RETICULATE_PYTHON` is honoured without a warning.
