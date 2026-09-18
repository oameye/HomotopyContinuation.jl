# Enable DispatchDoctor in the active test environment before loading the package,
# matching the supported package-testing pattern used by KeldyshContraction.jl.
using Preferences: set_preferences!
set_preferences!("HomotopyContinuation", "dispatch_doctor_mode" => "error"; force = true)

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

# These are broad semantic/regression stress suites whose large multiplicity makes
# DispatchDoctor repeatedly infer the same numerical kernels. Compact instrumented
# workloads in `quality/dispatch_doctor.jl` cover the corresponding endgame,
# witness-set, sweep, and monodromy routes. The ordinary suite still runs every file.
const SEMANTIC_STRESS = [
    "endgame_test",
    "monodromy_v2_parity_test",
    "system_sweep_test",
    "v2_parity_test",
    "witness_set_test",
]

testsuite = find_tests(dirname(@__DIR__))
filter!(testsuite) do entry
    name = first(entry)
    startswith(name, "extensive/") && return false
    startswith(name, "strict/") && return false
    name in OTHER_CONTRACTS && return false
    return !(name in SEMANTIC_STRESS)
end
ParallelTestRunner.runtests(HomotopyContinuation, ARGS; testsuite = testsuite)
