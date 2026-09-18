# The ordinary suite in `test/` runs the production configuration, where
# DispatchDoctor is disabled. This environment turns it to `"error"` through
# `LocalPreferences.toml` and reruns the same files, so a type instability
# anywhere in the package fails a test rather than passing silently.
#
# Run with `make test-strict`.
using HomotopyContinuation
using DispatchDoctor: DispatchDoctor
using ParallelTestRunner: ParallelTestRunner, find_tests

# A gate that cannot fire is worse than no gate: `@stable` expands to nothing
# outside DispatchDoctor's supported version window.
DispatchDoctor.JULIA_OK ||
    error("DispatchDoctor does not instrument Julia $VERSION; this run would prove nothing.")

# These files gate contracts other than type stability, and `make test` already
# runs them against package code. AllocCheck and JET would additionally report on
# the wrappers rather than on the package. Their analysis dependencies are absent
# from this environment, so a file added here without thought fails loudly.
const OTHER_CONTRACTS = [
    "alloc_check_test",
    "aqua_test",
    "concrete_structs_test",
    "explicit_imports_test",
    "jet_test",
]

testsuite = find_tests(dirname(@__DIR__))
filter!(testsuite) do entry
    name = first(entry)
    startswith(name, "extensive/") && return false
    startswith(name, "strict/") && return false
    return !(name in OTHER_CONTRACTS)
end
ParallelTestRunner.runtests(HomotopyContinuation, ARGS; testsuite = testsuite)
