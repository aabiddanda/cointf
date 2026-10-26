#' cointf: Housworth-Stahl likelihoods from R
#'
#' An R interface to the Python package 'cointf'. Each R function calls the
#' Python function of the same name through 'reticulate', so results are
#' identical to the Python package.
#'
#' @section Python environment:
#' When the package is loaded it declares its Python requirement with
#' [reticulate::py_require()], and reticulate installs it into a managed
#' environment the first time Python is used. The requirement defaults to the
#' GitHub repository and can be changed with the option `cointf.python`, set
#' before loading the package:
#'
#' ```r
#' options(cointf.python = "cointf @ git+https://github.com/aabiddanda/cointf@dev")
#' ```
#'
#' To use an existing Python environment instead (for example one where
#' 'cointf' was installed with `pip install .` from a clone), point
#' reticulate at it before loading the package, e.g.
#' `Sys.setenv(RETICULATE_PYTHON = "/path/to/venv/bin/python")`. The GitHub
#' requirement is then not declared. Setting `options(cointf.python = NULL)`
#' also skips it, for environments selected another way (e.g.
#' [reticulate::use_virtualenv()]).
#'
#' @keywords internal
"_PACKAGE"

# The Python module, imported lazily so that loading the R package does not
# start Python.
cointf_py <- NULL

default_requirement <- "cointf @ git+https://github.com/aabiddanda/cointf@dev"

# Required name: R calls .onLoad when the namespace is loaded.
.onLoad <- function(libname, pkgname) {
  # With RETICULATE_PYTHON set, the user has chosen an environment (e.g. a
  # local pip install); declaring the GitHub requirement would only make
  # reticulate warn that it is not installed there.
  requirement <- getOption("cointf.python", default_requirement)
  if (!nzchar(Sys.getenv("RETICULATE_PYTHON")) && !is.null(requirement)) {
    reticulate::py_require(requirement)
  }
  cointf_py <<- reticulate::import("cointf", delay_load = TRUE)
}

#' The underlying Python module
#'
#' Gives direct access to the Python 'cointf' module, for functions without
#' an R wrapper (for example `cointf_python()$naive$loglik_quartet_naive`).
#'
#' @return The 'cointf' Python module, as a reticulate object.
#' @export
cointf_python <- function() {
  cointf_py
}
