# The ordinary suite runs the production configuration. DispatchDoctor's hard-error
# instrumentation is exercised independently by `test/strict/` so static analyzers
# such as AllocCheck and JET inspect package code, not instrumentation.
using HomotopyContinuation
using ParallelTestRunner: ParallelTestRunner, find_tests

testsuite = find_tests(@__DIR__)
filter!(entry -> !startswith(first(entry), "extensive/"), testsuite)
filter!(entry -> !startswith(first(entry), "public_api/"), testsuite)
filter!(entry -> !startswith(first(entry), "strict/"), testsuite)
ParallelTestRunner.runtests(HomotopyContinuation, ARGS; testsuite = testsuite)
