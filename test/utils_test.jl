using Test
using HomotopyContinuationNext: fast_abs, nanmin, nanmax, nthroot

@testset "Utility functions" begin
    @testset "fast_abs" begin
        @test fast_abs(3.0 + 4.0im) ≈ 5.0
        @test fast_abs(-3.0) == 3.0
        @test fast_abs(0.0 + 0.0im) == 0.0
    end

    @testset "nanmin / nanmax" begin
        @test nanmin(1.0, 2.0) == 1.0
        @test nanmin(NaN, 2.0) == 2.0
        @test nanmin(1.0, NaN) == 1.0
        @test isnan(nanmin(NaN, NaN))
        @test nanmax(1.0, 2.0) == 2.0
        @test nanmax(NaN, 2.0) == 2.0
        @test nanmax(1.0, NaN) == 1.0
    end

    @testset "nthroot" begin
        @test nthroot(8.0, 3) ≈ 2.0
        @test nthroot(16.0, 4) ≈ 2.0
        @test nthroot(9.0, 2) ≈ 3.0
        @test nthroot(5.0, 1) == 5.0
        @test nthroot(7.0, 0) == 1.0
        @test nthroot(32.0, 5) ≈ 2.0
    end
end
