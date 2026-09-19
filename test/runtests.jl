# The ordinary suite runs the production configuration. DispatchDoctor's hard-error
# instrumentation is exercised independently by `test/strict/` so static analyzers
# such as AllocCheck and JET inspect package code, not instrumentation.
using HomotopyContinuation
using ParallelTestRunner: ParallelTestRunner, find_tests

# Migration diagnostic: fail immediately on private HomotopyContinuation API use
# before starting the expensive parallel semantic suite.
include("public_surface_contract_test.jl")

testsuite = find_tests(@__DIR__)
filter!(entry -> !startswith(first(entry), "extensive/"), testsuite)
filter!(entry -> !startswith(first(entry), "strict/"), testsuite)
filter!(entry -> first(entry) != "public_surface_contract_test", testsuite)
ParallelTestRunner.runtests(HomotopyContinuation, ARGS; testsuite = testsuite)
