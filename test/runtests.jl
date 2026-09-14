using HomotopyContinuation
using ParallelTestRunner: ParallelTestRunner, find_tests

testsuite = find_tests(@__DIR__)
filter!(entry -> !startswith(first(entry), "extensive/"), testsuite)
ParallelTestRunner.runtests(HomotopyContinuation, ARGS; testsuite = testsuite)
