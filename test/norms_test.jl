using Test
using LinearAlgebra: norm as la_norm
using FixedSizeArrays: FixedSizeVector
using HomotopyContinuationNext:
    InfNorm, WeightedNorm, WeightedNormOptions,
    inf_norm, weighted_norm, inf_distance, weighted_distance,
    init!, update!, fast_abs, DoubleF64, ComplexDF64

const FSVec{T} = FixedSizeVector{T}

@testset "Norms" begin
    @testset "InfNorm" begin
        x = FSVec{ComplexF64}([1.0 + 2.0im, 3.0 + 4.0im, 0.5 + 0.0im])
        @test inf_norm(x) ≈ abs(3.0 + 4.0im)

        y = FSVec{ComplexF64}([2.0 + 2.0im, 1.0 + 4.0im, 0.5 + 1.0im])
        d = inf_distance(x, y)
        expected = maximum(abs.(Vector(x) .- Vector(y)))
        @test d ≈ expected
    end

    @testset "WeightedNorm construction" begin
        n = 4
        w = WeightedNorm(n)
        @test length(w.weights) == n
        @test all(w.weights .== 1.0)
    end

    @testset "weighted_norm and weighted_distance" begin
        weights = FSVec{Float64}([2.0, 0.5, 1.0])
        w = WeightedNorm(weights)
        x = FSVec{ComplexF64}([2.0 + 0.0im, 4.0 + 0.0im, 3.0 + 0.0im])
        # ||D⁻¹x||_∞ = max(|2/2|, |4/0.5|, |3/1|) = 8.0
        @test weighted_norm(x, w) ≈ 8.0

        y = FSVec{ComplexF64}([4.0 + 0.0im, 4.0 + 0.0im, 3.0 + 0.0im])
        # ||D⁻¹(x-y)||_∞ = max(|2/2|, |0/0.5|, |0/1|) = 1.0
        @test weighted_distance(x, y, w) ≈ 1.0
    end

    @testset "weighted_norm with complex values" begin
        weights = FSVec{Float64}([1.0, 1.0])
        w = WeightedNorm(weights)
        x = FSVec{ComplexF64}([3.0 + 4.0im, 1.0 + 0.0im])
        @test weighted_norm(x, w) ≈ 5.0
    end

    @testset "fast_abs on ComplexDF64" begin
        z = ComplexDF64(DoubleF64(3.0), DoubleF64(4.0))
        @test fast_abs(z) ≈ 5.0 atol = 1e-28
    end

    @testset "init! and update!" begin
        w = WeightedNorm(3)
        x = FSVec{ComplexF64}([100.0 + 0.0im, 1e-10 + 0.0im, 50.0 + 0.0im])
        init!(w, x)
        @test w.weights[1] > 0.0
        @test w.weights[2] > 0.0
        @test w.weights[3] > 0.0

        y = FSVec{ComplexF64}([100.0 + 0.0im, 1e-10 + 0.0im, 50.0 + 0.0im])
        update!(w, y)
        @test all(w.weights .> 0.0)
    end

    @testset "overflow handling" begin
        x = FSVec{ComplexF64}([1e300 + 1e300im, 0.0 + 0.0im])
        n = inf_norm(x)
        @test isfinite(n) || n == Inf
    end
end
