# The ordinary suite is the user-visible behavioral contract. Compiler, package-
# hygiene, and instrumentation proofs run in dedicated quality workflows instead
# of being rediscovered as semantic tests here.
using HomotopyContinuation
using ParallelTestRunner: ParallelTestRunner, find_tests

testsuite = find_tests(@__DIR__)
filter!(entry -> !startswith(first(entry), "extensive/"), testsuite)
filter!(entry -> !startswith(first(entry), "strict/"), testsuite)
filter!(entry -> !startswith(first(entry), "quality/"), testsuite)
ParallelTestRunner.runtests(HomotopyContinuation, ARGS; testsuite = testsuite)
