# Test targets. The --depwarn=no is NOT optional: Pkg.test hardcodes --depwarn=yes on
# the test worker, which costs ~20x per fit here (normal_iris at the benchmark budget:
# 2.1s -> 42.9s). --check-bounds=no/--warn-overwrite=no are off to match — see the
# "Diagnostics" testset in test/runtests.jl, which runs with all three ON instead.
# julia_args is appended after Pkg's own flags, so it wins.
JULIA ?= julia
# `catch; exit(1)` swaps Pkg.test's own pkgerror stacktrace (noise — the report CSV and
# the printed pass/fail counts already say what failed) for a plain nonzero exit, which
# make still reports and CI can still key off.
TEST := $(JULIA) --project=. -e 'using Pkg; try; Pkg.test(julia_args=["--depwarn=no", "--check-bounds=no", "--warn-overwrite=no"]); catch; exit(1); end'

.PHONY: test fast benchmarks benchmarks-full

test:            ## no-MCMC tests + all families, random effects, weights, LOO, plots
	$(TEST)

fast:            ## no-MCMC tests + two small fixed-effect fits
	TR_TEST_LEVEL=fast $(TEST)

benchmarks:      ## + full-budget fits vs the brms reference, core cases
	TR_TEST_LEVEL=benchmarks $(TEST)

benchmarks-full: ## + every brms reference case, repeat seeds on the core
	TR_TEST_LEVEL=benchmarks_full $(TEST)
