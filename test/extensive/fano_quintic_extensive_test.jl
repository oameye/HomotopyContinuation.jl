using Test
using HomotopyContinuationNext
using HomotopyContinuationNextCertification: certify, ndistinct_certified
using Random: MersenneTwister

isdefined(@__MODULE__, :fano_quintic_system) || include("../test_systems.jl")

const FANO_SEED = 0x9a8b7c6d

@testset "Lines on a quintic surface in 3-space" begin
    equations, vars, params = fano_quintic_system()
    F = System(equations; variables = vars, parameters = params)
    # Pinned draw: the count is not stable across parameter draws, one in five
    # yields 2874. `implementation_docs/02_status.md` records what is known.
    q₀ = randn(MersenneTwister(1), ComplexF64, 125)
    G = fix_parameters(F, q₀)

    @testset "total degree" begin
        res = solve(G, TotalDegree(; seed = FANO_SEED, show_progress = false))
        @test ntracked(res) == 15625
        @test nsolutions(res) == 2875
        @test ndistinct_certified(certify(F, res, q₀)) == 2875
    end

    @testset "polyhedral" begin
        res = solve(G, Polyhedral(; seed = FANO_SEED, show_progress = false))
        @test ntracked(res) == 6725
        @test nsolutions(res) == 2875
    end
end
