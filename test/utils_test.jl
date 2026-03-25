using Test
using HomotopyContinuationNext:
    fast_abs,
    nanmin,
    nanmax,
    nthroot,
    SegmentStepper,
    init!,
    propose_step!,
    step_success!,
    is_done,
    dist_to_target

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

@testset "SegmentStepper" begin
    @testset "forward stepping" begin
        S = SegmentStepper(0.0 + 0.0im, 1.0 + 0.0im)
        @test !is_done(S)
        @test S.t ≈ 0.0 + 0.0im
        @test dist_to_target(S) ≈ 1.0

        propose_step!(S, 0.3)
        @test S.s′ ≈ 0.3
        step_success!(S)
        @test S.s ≈ 0.3

        propose_step!(S, 10.0)  # clamps to target
        step_success!(S)
        @test is_done(S)
        @test S.t ≈ 1.0 + 0.0im
    end

    @testset "backward stepping" begin
        S = SegmentStepper(1.0 + 0.0im, 0.0 + 0.0im)
        @test S.forward == false
        @test !is_done(S)

        propose_step!(S, 0.5)
        step_success!(S)
        @test !is_done(S)

        propose_step!(S, 10.0)
        step_success!(S)
        @test is_done(S)
    end

    @testset "reinit (returns new stepper)" begin
        S = SegmentStepper(0.0 + 0.0im, 1.0 + 0.0im)
        propose_step!(S, 0.5)
        step_success!(S)

        # init! returns NEW SegmentStepper because start/target/abs_Δ/forward are const
        S = init!(S, 0.0 + 0.0im, 2.0 + 0.0im)
        @test S.abs_Δ ≈ 2.0
        @test S.s ≈ 0.0
        @test !is_done(S)
    end

    @testset "Δs and Δt" begin
        S = SegmentStepper(0.0 + 0.0im, 1.0 + 0.0im)
        propose_step!(S, 0.25)
        @test S.Δs ≈ 0.25
        @test S.Δt ≈ 0.25 + 0.0im
    end
end
