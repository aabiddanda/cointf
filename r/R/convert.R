# Conversions between R values and the arguments the Python functions expect.

# A list of numeric vectors, one per meiosis, as a Python list of float64
# arrays. A prepared Meioses object passes through unchanged.
as_py_data <- function(data) {
  if (inherits(data, "python.builtin.object")) {
    return(data)
  }
  if (!is.list(data) || is.data.frame(data)) {
    stop(
      "`data` must be a list of numeric vectors (one per meiosis) or a ",
      "Meioses object; see meiosis_list() for long-format data",
      call. = FALSE
    )
  }
  reticulate::r_to_py(lapply(data, function(x) {
    reticulate::np_array(as.numeric(x), dtype = "float64")
  }))
}

# A scalar stays a scalar; a longer vector becomes a float64 array.
as_py_param <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  x <- as.numeric(x)
  if (length(x) == 1L) x else reticulate::np_array(x, dtype = "float64")
}

as_py_count <- function(x, name) {
  if (length(x) != 1L || is.na(x) || x != round(x)) {
    stop("`", name, "` must be a single whole number", call. = FALSE)
  }
  as.integer(x)
}

# A Python list of arrays (one per meiosis) as an R list of numeric vectors.
as_r_positions <- function(x) {
  lapply(reticulate::py_to_r(x), as.numeric)
}
