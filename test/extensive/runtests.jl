using Test

# Solves large enough that a full run takes minutes, kept out of `make test`.
# Run with `make test-extensive`; use plenty of threads, the executors are threaded.
@testset "HomotopyContinuation extensive" begin
    include("fano_quintic.jl")
    include("steiner.jl")
    include("iterator_certification.jl")
end
