using Test, Random
using HomotopyContinuation
using DynamicPolynomials: @polyvar

@testset "LinearSubspace compiler contracts" begin
    @test @inferred(rand_subspace(3; dim = 1)) isa LinearSubspace{ComplexF64}
    @test @inferred(rand_subspace(ComplexF64, 3; dim = 1)) isa LinearSubspace{ComplexF64}
    @test @inferred(rand_subspace(Float64, 3; dim = 1)) isa LinearSubspace{Float64}
    @test @inferred(rand_subspace(Random.default_rng(), 3; codim = 1)) isa
        LinearSubspace{ComplexF64}
    @test @inferred(rand_subspace(Random.default_rng(), Float64, 3; codim = 1)) isa
        LinearSubspace{Float64}

    @polyvar x y z
    @test @inferred(rand_subspace([x, y, z]; dim = 1)) isa LinearSubspace{ComplexF64}
    @test @inferred(rand_subspace(Float64, [x, y, z]; dim = 1)) isa
        LinearSubspace{Float64}
end
