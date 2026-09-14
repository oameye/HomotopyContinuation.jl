using HomotopyContinuationNext
using ParallelTestRunner: ParallelTestRunner, find_tests

# Scope discovery to this `test/` directory. The default (`find_tests(pwd())`)
# walks the current working directory, which is the repo root when run via
# `make test`, and would sweep up `src/` and the `lib/` subpackage.
testsuite = find_tests(@__DIR__)

# `extensive/` holds solves that take minutes each and has its own environment.
filter!(entry -> !startswith(first(entry), "extensive/"), testsuite)

# These three historical files load v2 and v3 in one process. After restoring
# the registered HomotopyContinuation package identity (name + UUID), Julia
# cannot install both versions in one environment. Keeping them in this suite
# would silently turn the oracle into a v3-v3 self-comparison through the
# temporary test compatibility module. The parity assertions that do not load
# v2 (`v2_parity_test.jl`, `monodromy_v2_parity_test.jl`) remain active.
const SAME_PROCESS_V2_ORACLES = Set([
    "compare_v2_primitives_test.jl",
    "compare_v2_solve_counts_test.jl",
    "compare_v2_solve_match_test.jl",
])
filter!(entry -> first(entry) ∉ SAME_PROCESS_V2_ORACLES, testsuite)

ParallelTestRunner.runtests(HomotopyContinuationNext, ARGS; testsuite = testsuite)
