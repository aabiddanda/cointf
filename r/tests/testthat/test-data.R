test_that("meiosis_list converts long-format data", {
  out <- meiosis_list(c(0.9, 0.2, NA, 0.5), c("a", "a", "b", "c"))
  expect_equal(names(out), c("a", "b", "c"))
  expect_equal(out$a, c(0.2, 0.9))
  expect_equal(out$b, numeric(0))
  expect_equal(out$c, 0.5)
  expect_error(meiosis_list(1:3, 1:2), "same length")
})

test_that("simulated data are lists of sorted numeric vectors", {
  skip_if_no_python_cointf()
  data <- simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 20, seed = 1)
  quartets <- simulate_quartets(L = 1.5, nu = 4, p = 0.05, n = 20, seed = 1)
  for (x in c(data, quartets)) {
    expect_type(x, "double")
    expect_false(is.unsorted(x))
    expect_true(all(x >= 0 & x <= 1.5))
  }
  expect_identical(simulate_meioses(1.5, 4, 0.05, 20, 1), data)
})
