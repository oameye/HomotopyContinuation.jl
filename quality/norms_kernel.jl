using Test
using HomotopyContinuation:
    ComplexDF64, DoubleF64, EuclideanNorm, FSVec, InfNorm, WeightedNorm,
    _vt_distance, fast_abs, inf_distance, inf_norm, init!, satisfies_triangle_inequality,
    update!

@testset "norm and scaling kernel contracts" begin
    @testset "adaptive coordinate scaling protects small components" begin
        scale = WeightedNorm(3)
        x = FSVec{ComplexF64}([100.0 + 0.0im, 1.0e-10 + 0.0im, 50.0 + 0.0im])

        init!(scale, x)
        @test all(scale.weights .> 0.0)
        @test scale.weights[2] > 1.0e-10

        previous = copy(scale.weights)
        update!(scale, FSVec{ComplexF64}([200.0 + 0.0im, 1.0e-12 + 0.0im, 25.0 + 0.0im]))
        @test all(scale.weights .> 0.0)
        @test scale.weights[1] >= previous[1]
    end

    @testset "double-double complex magnitude retains extended precision" begin
        z = ComplexDF64(DoubleF64(3.0), DoubleF64(4.0))
        @test fast_abs(z) ≈ 5.0 atol = 1.0e-28
    end

    @testset "infinity metric remains finite at extreme scale" begin
        x = FSVec{ComplexF64}(exp2(700) .* ComplexF64[2im, 3 - 1im, 5 + 2im])
        y = FSVec{ComplexF64}(exp2(700) .* ComplexF64[-2im, 3 - 1im, 5 + 2im])

        @test inf_norm(x) ≈ exp2(700) * abs(5 + 2im)
        @test inf_distance(x, x) == 0.0
        @test inf_distance(x, y) ≈ exp2(700) * abs(4im)
    end

    @testset "Voronoi pruning metrics satisfy their declared geometry" begin
        x = FSVec{ComplexF64}(ComplexF64[3, 4])
        y = FSVec{ComplexF64}(zeros(ComplexF64, 2))

        @test satisfies_triangle_inequality(InfNorm())
        @test satisfies_triangle_inequality(EuclideanNorm())
        @test _vt_distance(InfNorm(), x, y) ≈ 4.0
        @test _vt_distance(EuclideanNorm(), x, y) ≈ 5.0
    end
end
