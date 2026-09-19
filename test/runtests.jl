# The ordinary suite is the user-visible behavioral contract. Compiler, package-
# hygiene, and instrumentation proofs run in dedicated quality workflows instead
# of being rediscovered as semantic tests here.
using HomotopyContinuation
using ParallelTestRunner: ParallelTestRunner, find_tests

const QUALITY_CONTRACTS = Set(
    [
        "alloc_check_test",
        "aqua_test",
        "concrete_structs_test",
        "dispatch_doctor_test",
        "explicit_imports_test",
        "instruction_count_test",
        "jet_test",
    ]
)

testsuite = find_tests(@__DIR__)
filter!(entry -> !startswith(first(entry), "extensive/"), testsuite)
filter!(entry -> !startswith(first(entry), "strict/"), testsuite)
filter!(entry -> !(first(entry) in QUALITY_CONTRACTS), testsuite)
ParallelTestRunner.runtests(HomotopyContinuation, ARGS; testsuite = testsuite)
