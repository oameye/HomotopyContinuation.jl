# The production lane uses `test/LocalPreferences.toml`, where DispatchDoctor is
# disabled. CI reuses this runner in the current-Julia core lane after replacing
# that preference with the hard-error configuration from `test/strict/`.
# Local developers can still run the same instrumented suite with `make test-strict`.
using HomotopyContinuation
using DispatchDoctor: DispatchDoctor
using ParallelTestRunner: ParallelTestRunner, find_tests

# A gate that cannot fire is worse than no gate: `@stable` expands to nothing
# outside DispatchDoctor's supported version window.
DispatchDoctor.JULIA_OK ||
    error("DispatchDoctor does not instrument Julia $VERSION; this run would prove nothing.")

# These files gate contracts other than runtime type stability. They must inspect
# production package code rather than DispatchDoctor wrappers, so the instrumented
# lane excludes them. The Julia 1.10 production lane and dedicated analysis
# workflows retain those contracts.
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
    startswith(name, "public_api/") && return false
    startswith(name, "strict/") && return false
    return !(name in OTHER_CONTRACTS)
end
ParallelTestRunner.runtests(HomotopyContinuation, ARGS; testsuite = testsuite)
