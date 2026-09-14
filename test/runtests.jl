using Preferences: set_preferences!

# DispatchDoctor is a production no-op by default. Tests turn the package-wide
# stability contract into a hard error before HomotopyContinuation is loaded.
set_preferences!(
    "HomotopyContinuation",
    "dispatch_doctor_mode" => "error",
    "dispatch_doctor_codegen_level" => "min";
    force = true,
)

using HomotopyContinuation
using ParallelTestRunner: ParallelTestRunner, find_tests

testsuite = find_tests(@__DIR__)
filter!(entry -> !startswith(first(entry), "extensive/"), testsuite)
ParallelTestRunner.runtests(HomotopyContinuation, ARGS; testsuite = testsuite)
