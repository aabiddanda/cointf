#' Simulate crossover positions under the Housworth-Stahl model
#'
#' `simulate_meioses()` draws independent meioses. `simulate_quartets()`
#' draws phase-unknown data: each entry pools the crossovers of two
#' independent meioses.
#'
#' @param L Chromosome length in Morgans, `> 0`.
#' @param nu Interference parameter, `> 0`.
#' @param p Fraction of crossovers escaping interference, in `[0, 1]`.
#' @param n Number of meioses (or quartets).
#' @param seed Random seed, `> 0`. The same seed gives the same data as the
#'   Python package.
#' @return A list of sorted numeric vectors of crossover positions in Morgans.
#' @examples
#' \dontrun{
#' simulate_meioses(L = 1.5, nu = 4, p = 0.05, n = 3, seed = 7)
#' simulate_quartets(L = 1.5, nu = 4, p = 0.05, n = 3, seed = 7)
#' }
#' @export
simulate_meioses <- function(L, nu, p, n, seed) {
  as_r_positions(cointf_py$simulate_meioses(
    L = as.numeric(L), nu = as.numeric(nu), p = as.numeric(p),
    n = as_py_count(n, "n"), seed = as_py_count(seed, "seed")
  ))
}

#' @rdname simulate_meioses
#' @export
simulate_quartets <- function(L, nu, p, n, seed) {
  as_r_positions(cointf_py$simulate_quartets(
    L = as.numeric(L), nu = as.numeric(nu), p = as.numeric(p),
    n = as_py_count(n, "n"), seed = as_py_count(seed, "seed")
  ))
}
