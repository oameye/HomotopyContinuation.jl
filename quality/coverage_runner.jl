using HomotopyContinuation
using ParallelTestRunner: ParallelTestRunner, find_tests

# Coverage must exercise exactly the ordinary semantic suite, not the private
# compiler/numerical contracts that live in Quality.
const QUALITY_CONTRACTS = Set(
    [
        "alloc_check_test",
        "aqua_test",
        "concrete_structs_test",
        "dispatch_doctor_test",
        "explicit_imports_test",
        "instruction_count_test",
        "jet_test",
    ],
)

repo_root = realpath(joinpath(@__DIR__, ".."))
test_root = joinpath(repo_root, "test")
testsuite = find_tests(test_root)
filter!(entry -> !startswith(first(entry), "extensive/"), testsuite)
filter!(entry -> !startswith(first(entry), "strict/"), testsuite)
filter!(entry -> !(first(entry) in QUALITY_CONTRACTS), testsuite)

coverage_dir = abspath(get(ENV, "HC_COVERAGE_DIR", joinpath(repo_root, "coverage")))
mkpath(coverage_dir)
for path in readdir(coverage_dir; join = true)
    isfile(path) && endswith(path, ".info") && rm(path; force = true)
end

# ParallelTestRunner launches isolated Julia worker processes. Instrument those
# workers explicitly; instrumenting only this parent process would miss nearly
# the entire semantic suite. `%p` gives every worker/process its own LCOV file.
# Restrict instrumentation to this checkout so dependency code does not dominate
# runtime. The PID-dependent setting is also propagated by Julia to child Julia
# processes, including those exercised by distributed tests.
tracefile = joinpath(coverage_dir, "worker-%p.info")
jobs = parse(Int, get(ENV, "HC_COVERAGE_JOBS", "4"))
ParallelTestRunner.runtests(
    HomotopyContinuation,
    ["--jobs=$jobs"];
    testsuite,
    exeflags = ["--code-coverage=$tracefile", "--code-coverage=@$repo_root"],
)
