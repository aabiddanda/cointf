# Tests of the R <-> Python layer itself: conversions, object round trips and
# what happens when the package is loaded.

# The same data passed straight to Python, bypassing the R conversion.
py_direct <- function(data) {
  reticulate::r_to_py(lapply(data, function(x) reticulate::np_array(as.numeric(x), dtype = "float64")))
}

test_that("converted data are identical to data built in Python", {
  skip_if_no_python_cointf()
  data <- simulate_meioses(L = 2, nu = 5, p = 0.1, n = 300, seed = 8)
  py <- cointf_python()
  for (nu in c(1.5, 5, 20)) {
    expect_identical(
      loglik_fast(data, 2, nu = nu, p = 0.1, per_meiosis = TRUE),
      as.numeric(py$loglik_fast(py_direct(data), 2, nu, 0.1, per_meiosis = TRUE))
    )
  }
})

test_that("list shapes convert correctly", {
  skip_if_no_python_cointf()
  cases <- list(
    list(),
    list(numeric(0)),
    list(0.5),
    list(numeric(0), c(0.1, 0.9), numeric(0)),
    list(c(0.9, 0.1, 0.5)),
    list(a = 0.3, b = numeric(0), c = c(0.2, 1.1))
  )
  for (data in cases) {
    got <- cointf:::as_py_data(data)
    back <- lapply(reticulate::py_to_r(got), as.numeric)
    expect_equal(back, unname(lapply(data, as.numeric)))
  }
  # Unsorted positions are sorted by Meioses, as in Python.
  expect_equal(
    loglik_fast(list(c(0.9, 0.1, 0.5)), 1.5, nu = 4, p = 0.05),
    loglik_fast(list(c(0.1, 0.5, 0.9)), 1.5, nu = 4, p = 0.05)
  )
  expect_identical(loglik_fast(list(), 1.5, nu = 4, p = 0.05), 0)
})

test_that("integer and named inputs are accepted", {
  skip_if_no_python_cointf()
  data <- list(c(0.25, 1.5), numeric(0))
  expect_equal(
    loglik_fast(data, 2L, nu = 4L, p = 0, per_meiosis = TRUE),
    loglik_fast(unname(data), 2, nu = 4, p = 0, per_meiosis = TRUE)
  )
  expect_equal(
    loglik_fast(list(m1 = c(1L, 2L)), 3, nu = 2, p = 0.1),
    loglik_fast(list(c(1, 2)), 3, nu = 2, p = 0.1)
  )
})

test_that("per-meiosis parameters convert for one and several meioses", {
  skip_if_no_python_cointf()
  one <- list(c(0.2, 0.9))
  expect_length(loglik_fast(one, 1.2, nu = 4, p = 0.1, per_meiosis = TRUE), 1)
  data <- list(0.3, c(0.2, 0.8), numeric(0))
  # A vector of equal values is the same as the scalar.
  expect_identical(
    loglik_fast(data, 1, nu = c(4, 4, 4), p = 0.1),
    loglik_fast(data, 1, nu = 4, p = 0.1)
  )
  expect_error(loglik_fast(data, 1, nu = c(4, 5), p = 0.1), "nu must be a scalar or have length 3")
})

test_that("bad positions are rejected by Python with clear messages", {
  skip_if_no_python_cointf()
  expect_error(loglik_fast(list(c(0.2, NA)), 1, nu = 4, p = 0.1), "finite")
  expect_error(loglik_fast(list(c(0.2, Inf)), 1, nu = 4, p = 0.1), "finite")
  expect_error(loglik_fast(list(c(0.2, 0.2)), 1, nu = 4, p = 0.1), "duplicated")
  expect_error(loglik_fast(data.frame(x = 1), 1, nu = 4, p = 0.1), "list of numeric vectors")
  expect_error(loglik_fast(list(0.5), 1, nu = 4, p = 2), "p must lie")
})

test_that("simulations match the Python simulator exactly", {
  skip_if_no_python_cointf()
  py <- cointf_python()
  direct <- lapply(reticulate::py_to_r(py$simulate_meioses(1.5, 4, 0.05, 50L, 11L)), as.numeric)
  expect_identical(simulate_meioses(1.5, 4, 0.05, 50, 11), direct)
  direct_q <- lapply(reticulate::py_to_r(py$simulate_quartets(1.5, 4, 0.05, 50L, 11L)), as.numeric)
  expect_identical(simulate_quartets(1.5, 4, 0.05, 50, 11), direct_q)
})

test_that("Python objects survive round trips through R", {
  skip_if_no_python_cointf()
  data <- simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 100, seed = 2)
  kernel <- chebyshev_kernel()
  mz <- meioses(data, 1.5)
  expect_s3_class(kernel, "python.builtin.object")
  expect_match(format(mz), "Meioses\\(M=100")
  # The kernel's cache lives in Python and persists between R calls.
  loglik_fast(mz, NULL, nu = 4, p = 0.05, kernel = kernel)
  loglik_fast(mz, NULL, nu = 4, p = 0.2, kernel = kernel)
  loglik_quartet(mz, NULL, nu = 6, p = 0.05, kernel = kernel)
  # Reading the attribute converts the Python dict to a named R list.
  expect_equal(sort(as.numeric(names(kernel$cache))), c(4, 6))
  # The same Meioses object serves both likelihoods.
  expect_true(is.finite(loglik_quartet(mz, NULL, nu = 4, p = 0.05)))
})

test_that("a large dataset converts quickly", {
  skip_if_no_python_cointf()
  skip_on_cran()
  data <- simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 20000, seed = 3)
  elapsed <- system.time(cointf:::as_py_data(data))[["elapsed"]]
  # About 10 ms in practice; the bound only catches a per-meiosis regression
  # (which took about 1.4 s).
  expect_lt(elapsed, 0.5)
})

# Run a short R script in a fresh process and return its output lines.
run_fresh <- function(code, env = character()) {
  script <- tempfile(fileext = ".R")
  writeLines(code, script)
  on.exit(unlink(script))
  rscript <- file.path(R.home("bin"), "Rscript")
  suppressWarnings(system2(rscript, script, stdout = TRUE, stderr = TRUE, env = env))
}

test_that("loading the package does not start Python", {
  skip_on_cran()
  out <- run_fresh(c(
    "library(cointf)",
    "cat('started:', reticulate::py_available(initialize = FALSE), '\\n')"
  ))
  expect_true(any(grepl("started: FALSE", out)))
})

test_that("RETICULATE_PYTHON is used without a requirement warning", {
  skip_if_no_python_cointf()
  skip_on_cran()
  python <- reticulate::py_config()$python
  out <- run_fresh(
    c(
      "library(cointf)",
      "x <- loglik_fast(list(0.5), 1, nu = 1, p = 0.2)",
      "cat('python:', reticulate::py_config()$python, '\\n')",
      "cat('value:', x, '\\n')"
    ),
    env = paste0("RETICULATE_PYTHON=", python)
  )
  expect_true(any(grepl(paste0("python: ", python), out, fixed = TRUE)))
  expect_true(any(grepl("value: -1", out, fixed = TRUE)))
  expect_false(any(grepl("py_require", out)))
})

test_that("the automatic GitHub install works in a clean environment", {
  # Opt-in: needs network access and builds a fresh Python environment.
  skip_if_not(identical(Sys.getenv("COINTF_TEST_INSTALL"), "true"), "set COINTF_TEST_INSTALL=true to run")
  out <- run_fresh(
    c(
      "Sys.unsetenv(c('RETICULATE_PYTHON', 'VIRTUAL_ENV'))",
      "library(cointf)",
      "data <- simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 200, seed = 7)",
      "cat('value:', round(loglik_fast(data, 1.5, nu = 4, p = 0.05), 2), '\\n')"
    ),
    env = c("RETICULATE_PYTHON=", "VIRTUAL_ENV=")
  )
  expect_true(any(grepl("value: -269.79", out, fixed = TRUE)))
})
