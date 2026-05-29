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

cat("R binding smoke OK\n")
