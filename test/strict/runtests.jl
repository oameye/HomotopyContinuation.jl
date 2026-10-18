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

# The other static contracts in `test/quality/` gate properties other than runtime
# type stability. They must inspect production package code rather than
# DispatchDoctor wrappers, so the instrumented lane excludes them. The production
# lane and dedicated analysis workflows retain those contracts.
const INSTRUMENTED_CONTRACTS = ["quality/instruction_count_test"]

testsuite = find_tests(dirname(@__DIR__))
filter!(testsuite) do entry
    name = first(entry)
    startswith(name, "extensive/") && return false
    startswith(name, "strict/") && return false
    startswith(name, "quality/") && return name in INSTRUMENTED_CONTRACTS
    return true
end
ParallelTestRunner.runtests(HomotopyContinuation, ARGS; testsuite = testsuite)
