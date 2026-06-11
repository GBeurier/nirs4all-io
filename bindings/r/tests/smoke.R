# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
# Smoke test for the R binding. Run after installing the package (see
# build_and_test.sh). Exercises the C-ABI JSON surface against the contract corpus.
library(nirs4allio)

corpus <- Sys.getenv("N4IO_CORPUS")
stopifnot(nzchar(corpus))

# A path is a JSON string value: "\"<path>\"".
as_json_path <- function(p) paste0('"', p, '"')

# to_spec on the train_test directory -> a canonical spec with schema_version 1.
spec <- n4io_to_spec(as_json_path(file.path(corpus, "train_test")))
stopifnot(is.character(spec), grepl('"schema_version"', spec, fixed = TRUE))
stopifnot(endsWith(spec, "\n"))

# The produced spec validates.
n4io_validate(spec)

# infer returns a plan.
plan <- n4io_infer(as_json_path(file.path(corpus, "single_combined")))
stopifnot(grepl('"resolved_spec"', plan, fixed = TRUE))

# A bad spec is rejected (error).
bad <- tryCatch({
  n4io_validate('{"partitions": {"by": "random"}}')
  FALSE
}, error = function(e) TRUE)
stopifnot(bad)

# ABI version looks like semver.
stopifnot(grepl("^[0-9]+\\.[0-9]+\\.[0-9]+", n4io_abi_version()))

# --- idiomatic layer ----------------------------------------------------------

# nio_to_spec accepts a native R path and returns a typed n4io_spec.
sp <- nio_to_spec(file.path(corpus, "train_test"))
stopifnot(inherits(sp, "n4io_spec"))
stopifnot(sp$schema_version == 1)
stopifnot(length(sp$sources) >= 1)

# the typed spec round-trips through nio_validate (no error).
stopifnot(isTRUE(nio_validate(sp)))

# print/format render without error and mention the spec class.
stopifnot(grepl("n4io_spec", format(sp), fixed = TRUE))

# nio_infer returns a typed n4io_plan; as.data.frame yields scored decisions.
pl <- nio_infer(file.path(corpus, "single_combined"))
stopifnot(inherits(pl, "n4io_plan"))
df <- as.data.frame(pl)
stopifnot(is.data.frame(df))
stopifnot(all(c("decision", "value", "score", "ambiguous") %in% names(df)))
stopifnot("structure" %in% df$decision, "task_type" %in% df$decision)
stopifnot(grepl("n4io_plan", format(pl), fixed = TRUE))

# the plan's resolved spec is an n4io_spec that validates.
rs <- nio_resolved_spec(pl)
stopifnot(inherits(rs, "n4io_spec"))
stopifnot(isTRUE(nio_validate(rs)))

# a file-list input is accepted natively (character vector -> JSON array).
flist <- file.path(corpus, "x_y_separate", c("X.csv", "Y.csv"))
fl_plan <- nio_infer(flist)
stopifnot(inherits(fl_plan, "n4io_plan"))

# nio_validate signals an error on a bad spec (list input).
bad2 <- tryCatch({
  nio_validate(list(partitions = list(by = "random")))
  FALSE
}, error = function(e) TRUE)
stopifnot(bad2)

cat("R binding smoke OK\n")
