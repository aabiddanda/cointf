# Reference values match the Python test suite: the test bed is 200 meioses
# simulated with L = 1.5, nu = 4, p = 0.05, seed = 7.

test_that("the test bed reproduces the Python reference values", {
  skip_if_no_python_cointf()
  data <- simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 200, seed = 7)
  expect_length(data, 200)
  expect_close(loglik_fast(data, 1.5, nu = 4, p = 0.05), -269.79, 0.01)
  # nu = 1 is a rate-1 Poisson process: exactly -n * L.
  expect_close(loglik_fast(data, 1.5, nu = 1, p = 0.05), -300, 1e-8)
})

test_that("per_meiosis returns one value per meiosis and sums to the total", {
  skip_if_no_python_cointf()
  data <- simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 50, seed = 3)
  each <- loglik_fast(data, 1.5, nu = 4, p = 0.05, per_meiosis = TRUE)
  expect_type(each, "double")
  expect_length(each, 50)
  expect_equal(sum(each), loglik_fast(data, 1.5, nu = 4, p = 0.05))
})

test_that("a prepared Meioses object gives the same result as a list", {
  skip_if_no_python_cointf()
  data <- simulate_meioses(L = 2, nu = 6, p = 0.1, n = 40, seed = 5)
  mz <- meioses(data, 2)
  expect_equal(loglik_fast(mz, NULL, nu = 6, p = 0.1), loglik_fast(data, 2, nu = 6, p = 0.1))
  expect_error(loglik_fast(mz, 2, nu = 6, p = 0.1), "L must be None")
})

test_that("parameters can vary between meioses", {
  skip_if_no_python_cointf()
  data <- list(c(0.2, 0.9), numeric(0), 0.5)
  each <- loglik_fast(data, c(1.2, 1.0, 1.5), nu = c(4, 6, 2), p = c(0.1, 0, 0.3), per_meiosis = TRUE)
  single <- vapply(seq_along(data), function(i) {
    loglik_fast(data[i], c(1.2, 1.0, 1.5)[i], nu = c(4, 6, 2)[i], p = c(0.1, 0, 0.3)[i])
  }, numeric(1))
  expect_equal(each, single, tolerance = 1e-12)
})

test_that("empty meioses and single crossovers are handled", {
  skip_if_no_python_cointf()
  data <- list(numeric(0), 0.7, c(0.1, 1.4))
  expect_length(loglik_fast(data, 1.5, nu = 4, p = 0.05, per_meiosis = TRUE), 3)
  expect_close(loglik_fast(list(numeric(0)), 1.5, nu = 1, p = 0.3), -1.5, 1e-10)
})

test_that("the Chebyshev kernel agrees with the exact series", {
  skip_if_no_python_cointf()
  data <- simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 200, seed = 7)
  kernel <- chebyshev_kernel()
  exact <- loglik_fast(data, 1.5, nu = 4.3, p = 0.05, per_meiosis = TRUE)
  approx <- loglik_fast(data, 1.5, nu = 4.3, p = 0.05, kernel = kernel, per_meiosis = TRUE)
  expect_close(approx, exact, 1e-11)
})

test_that("quartet likelihood matches the Poisson closed form at nu = 1", {
  skip_if_no_python_cointf()
  quartets <- simulate_quartets(L = 1.8, nu = 3, p = 0.1, n = 30, seed = 4)
  counts <- lengths(quartets)
  got <- loglik_quartet(quartets, 1.8, nu = 1, p = 0.2, per_meiosis = TRUE)
  expect_close(got, counts * log(2) - 3.6, 1e-10)
})

test_that("quartet likelihood matches the Python brute-force reference", {
  skip_if_no_python_cointf()
  naive <- cointf_python()$naive$loglik_quartet_naive
  for (xs in list(numeric(0), 0.7, c(0.05, 0.6, 1.3, 1.95))) {
    expect_close(
      loglik_quartet(list(xs), 2, nu = 4, p = 0.1),
      naive(reticulate::np_array(xs, dtype = "float64"), 2, 4, 0.1), 1e-10)
  }
})

test_that("invalid input raises informative errors", {
  skip_if_no_python_cointf()
  expect_error(loglik_fast(c(0.2, 0.5), 1, nu = 4, p = 0.1), "list of numeric vectors")
  expect_error(loglik_fast(list(1.5), 1, nu = 4, p = 0.1), "positions must lie")
  expect_error(loglik_fast(list(0.5), 1, nu = -1, p = 0.1), "nu must be")
  expect_error(simulate_meioses(L = 1, nu = 4, p = 0.1, n = 2.5, seed = 1), "whole number")
})
