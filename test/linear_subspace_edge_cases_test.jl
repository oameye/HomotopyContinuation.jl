using Test
using LinearAlgebra
using HomotopyContinuationNext: LinearSubspace, rand_subspace, rand_subspace!, dim, codim,
    ambient_dim, extrinsic, translate, is_linear

@testset "LinearSubspace endpoint dimensions" begin
    for L in (
            rand_subspace(3; dim = 0),
            rand_subspace(3; codim = 3),
        )
        @test dim(L) == 0
        @test codim(L) == 3
        @test ambient_dim(L) == 3
    end

    for L in (
            rand_subspace(3; dim = 3),
            rand_subspace(3; codim = 0),
        )
        @test dim(L) == 3
        @test codim(L) == 0
        @test ambient_dim(L) == 3
        @test is_linear(L)
    end

    x = ComplexF64[1, 2, 3]
    Lpoint = rand_subspace(x; dim = 0)
    @test dim(Lpoint) == 0
    @test norm(extrinsic(Lpoint).A * x - extrinsic(Lpoint).b) < 1.0e-12

    Lfull = rand_subspace(x; codim = 0)
    @test dim(Lfull) == 3
    @test codim(Lfull) == 0

    @test_throws ArgumentError rand_subspace(x; dim = 0, affine = false)

    # The zero vector is the unique point of the zero-dimensional linear subspace.
    Lzero = rand_subspace(zeros(ComplexF64, 3); dim = 0, affine = false)
    @test dim(Lzero) == 0
    @test is_linear(Lzero)
end

@testset "rand_subspace! through the zero vector" begin
    A = zeros(ComplexF64, 2, 3)
    b = zeros(ComplexF64, 2)
    x = zeros(ComplexF64, 3)

    L = rand_subspace!(A, b, x; affine = false)
    @test is_linear(L)
    @test all(isfinite, extrinsic(L).A)
    @test norm(extrinsic(L).A * x) == 0
end

@testset "translate promotes coefficient type" begin
    L = rand_subspace(4; dim = 1, real = true)
    δb = ComplexF64[0.2 + 0.3im, -0.7im, 1.1 - 0.4im]
    Lt = translate(L, δb)

    @test Lt isa LinearSubspace{ComplexF64}
    @test extrinsic(Lt).b ≈ ComplexF64.(extrinsic(L).b) + δb
    @test L isa LinearSubspace{Float64}
end
