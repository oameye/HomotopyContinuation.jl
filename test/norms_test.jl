using Test
using FixedSizeArrays: FixedSizeArray
using HomotopyContinuationNext:
    WeightedNorm, inf_norm, weighted_norm, inf_distance, weighted_distance,
    init!, update!, fast_abs, DoubleF64, ComplexDF64

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}

@testset "Norms" begin
    @testset "inf_norm and inf_distance" begin
        x = FSVec{ComplexF64}([1.0 + 2.0im, 3.0 + 4.0im, 0.5 + 0.0im])
        @test inf_norm(x) ≈ abs(3.0 + 4.0im)

        y = FSVec{ComplexF64}([2.0 + 2.0im, 1.0 + 4.0im, 0.5 + 1.0im])
        @test inf_distance(x, y) ≈ maximum(abs.(Vector(x) .- Vector(y)))
        @test inf_distance(x, x) ≈ 0.0
    end

    @testset "weighted_norm and weighted_distance" begin
        weights = FSVec{Float64}([2.0, 0.5, 1.0])
        w = WeightedNorm(weights)
        x = FSVec{ComplexF64}([2.0 + 0.0im, 4.0 + 0.0im, 3.0 + 0.0im])
        # ||D⁻¹x||_∞ = max(|2/2|, |4/0.5|, |3/1|) = 8.0
        @test weighted_norm(x, w) ≈ 8.0

        y = FSVec{ComplexF64}([4.0 + 0.0im, 4.0 + 0.0im, 3.0 + 0.0im])
        @test weighted_distance(x, y, w) ≈ 1.0
        @test weighted_distance(x, x, w) ≈ 0.0
    end

    @testset "init! and update!" begin
        w = WeightedNorm(3)
        @test all(w.weights .== 1.0)

        x = FSVec{ComplexF64}([100.0 + 0.0im, 1.0e-10 + 0.0im, 50.0 + 0.0im])
        init!(w, x)
        @test all(w.weights .> 0.0)
        # Small component gets clamped to scale_min * norm, not left at 1e-10
        @test w.weights[2] > 1.0e-10

        update!(w, x)
        @test all(w.weights .> 0.0)
    end

    @testset "fast_abs on ComplexDF64" begin
        z = ComplexDF64(DoubleF64(3.0), DoubleF64(4.0))
        @test fast_abs(z) ≈ 5.0 atol = 1.0e-28
    end

    @testset "inf_norm overflow (exp2(700), triggers isinf fallback)" begin
        x = FSVec{ComplexF64}([2.0im, 3.0 - 1im, 5.0 + 2.0im])
        y = FSVec{ComplexF64}([-2.0im, 3.0 - 1im, 5.0 + 2.0im])
        huge_x = FSVec{ComplexF64}(exp2(700) .* Vector(x))
        huge_y = FSVec{ComplexF64}(exp2(700) .* Vector(y))

        @test inf_norm(huge_x) ≈ exp2(700) * abs(5.0 + 2.0im)
        @test inf_distance(huge_x, huge_x) ≈ 0.0
        @test inf_distance(huge_x, huge_y) ≈ exp2(700) * abs(4im)
    end
end
