#' Prepare crossover data for repeated likelihood evaluation
#'
#' Validates and sorts the positions, groups meioses by crossover count and
#' precomputes every distance the likelihood needs. Build it once and pass it
#' to [loglik_fast()] or [loglik_quartet()] (with `L = NULL`) for each
#' parameter value, for example inside an optimiser.
#'
#' @param data A list with one numeric vector of crossover positions (in
#'   Morgans) per meiosis; an entry may be empty (`numeric(0)`). For quartet
#'   data, one entry per quartet.
#' @param L Chromosome length in Morgans: one value, or one per entry of
#'   `data`.
#' @return A Python `Meioses` object.
#' @examples
#' \dontrun{
#' mz <- meioses(list(c(0.2, 0.9), numeric(0), 0.5), L = 1.2)
#' loglik_fast(mz, NULL, nu = 4, p = 0.05)
#' }
#' @export
meioses <- function(data, L) {
  cointf_py$Meioses(as_py_data(data), as_py_param(L))
}

#' Log-likelihood of crossover positions under the Housworth-Stahl model
#'
#' Crossovers come from an interference pathway (a stationary renewal process
#' with gamma-mixture gaps governed by `nu`) and an escape pathway (a Poisson
#' process carrying a fraction `p` of crossovers). The likelihood sums over
#' the unobserved pathway labels exactly.
#'
#' @param data A list with one numeric vector of crossover positions (in
#'   Morgans) per meiosis, or an object from [meioses()].
#' @param L Chromosome length in Morgans: one value or one per meiosis. Must
#'   be `NULL` when `data` comes from [meioses()].
#' @param nu Interference parameter, `> 0`: one value or one per meiosis.
#'   `nu = 1` is no interference.
#' @param p Fraction of crossovers escaping interference, in `[0, 1]`: one
#'   value or one per meiosis.
#' @param kernel Optional kernel, e.g. from [chebyshev_kernel()]; the default
#'   is the exact series.
#' @param per_meiosis If `TRUE`, return one log-likelihood per meiosis
#'   instead of the sum.
#' @return The total log-likelihood, or a numeric vector with one value per
#'   meiosis if `per_meiosis = TRUE`.
#' @examples
#' \dontrun{
#' data <- simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 200, seed = 7)
#' loglik_fast(data, 1.5, nu = 4, p = 0.05)
#' }
#' @export
loglik_fast <- function(data, L, nu, p, kernel = NULL, per_meiosis = FALSE) {
  out <- cointf_py$loglik_fast(
    as_py_data(data), as_py_param(L), as_py_param(nu), as_py_param(p),
    kernel = kernel, per_meiosis = per_meiosis
  )
  as.numeric(out)
}

#' Log-likelihood of phase-unknown (quartet) crossover data
#'
#' For a parent in a family with two offspring and no grandparents, each
#' chromosome yields the pooled crossovers of two meioses, without knowing
#' which meiosis each crossover came from. The likelihood sums over every
#' assignment of crossovers to the two meioses or to the escape pathway.
#'
#' @inheritParams loglik_fast
#' @param data A list with one numeric vector of pooled crossover positions
#'   (in Morgans) per quartet, or an object from [meioses()].
#' @param nu,p As for [loglik_fast()]; both meioses of a quartet share them.
#' @param per_meiosis If `TRUE`, return one value per quartet (one entry of
#'   `data`) instead of the sum. The name matches [loglik_fast()].
#' @return The total log-likelihood, or a numeric vector with one value per
#'   quartet if `per_meiosis = TRUE`.
#' @examples
#' \dontrun{
#' quartets <- simulate_quartets(L = 1.5, nu = 4, p = 0.05, n = 200, seed = 7)
#' loglik_quartet(quartets, 1.5, nu = 4, p = 0.05)
#' }
#' @export
loglik_quartet <- function(data, L, nu, p, kernel = NULL, per_meiosis = FALSE) {
  out <- cointf_py$loglik_quartet(
    as_py_data(data), as_py_param(L), as_py_param(nu), as_py_param(p),
    kernel = kernel, per_meiosis = per_meiosis
  )
  as.numeric(out)
}

#' Faster approximate likelihood kernel
#'
#' Piecewise Chebyshev fits of the log gap density and log survival function,
#' built once per value of `nu` and cached in the returned object. Keep one
#' kernel and reuse it across calls so the cache is reused. With a shared
#' `nu` it is 6-10 times faster than the exact series, with per-meiosis
#' log-likelihood differences around 1e-13.
#'
#' @param tol Acceptance threshold for each fitted piece (not an error bound).
#' @param u_min,u_max Distances (on the interference scale) inside this range
#'   use the fit; the rest use the exact series.
#' @param u_split Boundary between pieces fitted in log distance and in
#'   distance.
#' @param n_nodes Chebyshev nodes per piece.
#' @param max_distinct Largest number of distinct `nu` values in one call for
#'   which fits are built; above it the exact series is used.
#' @param max_cached Fits kept before the cache is cleared.
#' @return A Python `ChebyshevKernel` object, to pass as `kernel`.
#' @examples
#' \dontrun{
#' kernel <- chebyshev_kernel()
#' data <- simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 200, seed = 7)
#' loglik_fast(data, 1.5, nu = 4, p = 0.05, kernel = kernel)
#' }
#' @export
chebyshev_kernel <- function(tol = 1e-12, u_min = 1e-6, u_max = 5, u_split = 0.25,
                             n_nodes = 16, max_distinct = 16, max_cached = 256) {
  cointf_py$ChebyshevKernel(
    tol = tol, u_min = u_min, u_max = u_max, u_split = u_split,
    n_nodes = as_py_count(n_nodes, "n_nodes"),
    max_distinct = as_py_count(max_distinct, "max_distinct"),
    max_cached = as_py_count(max_cached, "max_cached")
  )
}

#' Crossover positions from long-format data
#'
#' Turns one-row-per-crossover data into the list format the likelihood
#' functions take. Meioses without crossovers are kept by giving them a row
#' with `NA` position.
#'
#' @param position Crossover positions in Morgans; `NA` for a meiosis without
#'   crossovers.
#' @param meiosis Meiosis identifier for each row.
#' @return A named list with one numeric vector per meiosis, in order of
#'   first appearance.
#' @examples
#' meiosis_list(c(0.2, 0.9, NA, 0.5), c("a", "a", "b", "c"))
#' @export
meiosis_list <- function(position, meiosis) {
  if (length(position) != length(meiosis)) {
    stop("`position` and `meiosis` must have the same length", call. = FALSE)
  }
  ids <- unique(meiosis)
  out <- lapply(ids, function(id) {
    x <- position[meiosis == id]
    sort(as.numeric(x[!is.na(x)]))
  })
  names(out) <- as.character(ids)
  out
}
