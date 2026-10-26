# Generate docs/api/r.md from the R help pages in r/man.
#
# Run from the repository root after changing roxygen comments in r/R:
#   Rscript -e 'roxygen2::roxygenise("r")'
#   Rscript docs/api/make_r_reference.R

topics <- c(
  "loglik_fast", "loglik_quartet", "meioses", "chebyshev_kernel",
  "simulate_meioses", "meiosis_list", "cointf_python", "cointf-package"
)
headings <- c(
  "cointf-package" = "Package overview and Python setup",
  simulate_meioses = "simulate_meioses(), simulate_quartets()"
)
# Names that Rd links point at, mapped to the section holding them.
targets <- c(
  setNames(topics, topics),
  simulate_quartets = "simulate_meioses", `cointf-package` = "cointf-package"
)

rd_to_html <- function(topic) {
  out <- textConnection("html", "w", local = TRUE)
  tools::Rd2HTML(file.path("r", "man", paste0(topic, ".Rd")), out = out, standalone = FALSE)
  close(out)
  html <- paste(html, collapse = "\n")
  # The title becomes the lead paragraph; the section heading names the function.
  html <- sub("<h2[^>]*>(.*?)</h2>", "<p><strong>\\1</strong></p>", html, perl = TRUE)
  # Examples are wrapped in \dontrun{} (they need Python); drop the markers.
  html <- gsub("## Not run: *\n|\n?## End\\(Not run\\)", "", html)
  # Links between help pages render as plain code; point them at this page.
  for (name in names(targets)) {
    pattern <- paste0("<code>", name, "\\(\\)</code>")
    link <- paste0("<a href=\"#r-", targets[[name]], "\"><code>", name, "()</code></a>")
    html <- gsub(pattern, link, html)
  }
  html
}

lines <- c(
  "# R API",
  "",
  "Reference for the R package in `r/`, generated from its help pages (the same text as",
  "`?loglik_fast` and friends in R) by `docs/api/make_r_reference.R`. Each function calls the",
  "Python function of the same name; see [Using cointf from R](../r.md) for setup.",
  ""
)
for (topic in topics) {
  heading <- if (topic %in% names(headings)) headings[[topic]] else paste0(topic, "()")
  lines <- c(
    lines,
    paste0("## ", heading),
    "",
    "```{raw} html",
    paste0("<span id=\"r-", topic, "\"></span>"),
    rd_to_html(topic),
    "```",
    ""
  )
}
writeLines(lines, file.path("docs", "api", "r.md"))
cat("wrote docs/api/r.md\n")
