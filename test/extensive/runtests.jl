using Test

# Solves large enough that a full run takes minutes, kept out of `make test`.
# Run with `make test-extensive`; use plenty of threads, the executors are threaded.
@testset "HomotopyContinuationNext extensive" begin
    include("fano_quintic_extensive_test.jl")
    include("steiner_extensive_test.jl")
end
