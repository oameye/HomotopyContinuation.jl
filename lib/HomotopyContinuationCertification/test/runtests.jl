# The ordinary certification suite. The static contracts in `quality/` run in the
# Quality workflow instead, like the main package's.
using HomotopyContinuationCertification
using ParallelTestRunner: ParallelTestRunner, find_tests

testsuite = find_tests(@__DIR__)
filter!(entry -> !startswith(first(entry), "quality/"), testsuite)
# Some tests compare serial certification against `Threaded(4)`.
ParallelTestRunner.runtests(
    HomotopyContinuationCertification, ARGS; testsuite, exeflags = ["--threads=4"],
)
