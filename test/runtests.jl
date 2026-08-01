using HomotopyContinuationNext
using ParallelTestRunner: ParallelTestRunner, find_tests

# Scope discovery to this `test/` directory. The default (`find_tests(pwd())`)
# walks the current working directory, which is the repo root when run via
# `make test`, and would sweep up `src/` and the `lib/` subpackage. The
# certification subpackage has its own test suite (see `make test-cert`).
testsuite = find_tests(@__DIR__)
# `extensive/` holds solves that take minutes each and has its own environment
# (it certifies, so it needs the certification subpackage): `make test-extensive`.
filter!(entry -> !startswith(first(entry), "extensive/"), testsuite)
ParallelTestRunner.runtests(HomotopyContinuationNext, ARGS; testsuite = testsuite)
