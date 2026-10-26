# Skip tests that need Python when the 'cointf' Python package is unavailable
# (e.g. no network for reticulate to install it).
skip_if_no_python_cointf <- function() {
  ok <- tryCatch(reticulate::py_module_available("cointf"), error = function(e) FALSE)
  if (!ok) skip("Python package 'cointf' is not available")
}

# Absolute-error comparison (testthat's expect_equal tolerance is relative).
expect_close <- function(actual, expected, tol) {
  expect_equal(length(actual), length(expected))
  expect_lte(max(abs(actual - expected)), tol)
}
