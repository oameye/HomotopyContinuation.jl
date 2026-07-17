using Test
using HomotopyContinuationNext:
    fast_abs, nanmin, nanmax, nthroot, _stable_sort!, _stable_sort_by!,
    SegmentStepper, reinit!, propose_step!, step_success!, is_done, dist_to_target
using Random: MersenneTwister

@testset "Utility functions" begin
    @testset "_stable_sort! / _stable_sort_by!: sorted and stable at all sizes" begin
        rng = MersenneTwister(11)
        # Sizes straddling the small-input cutoff, including empty and degenerate.
        for n in (0, 1, 2, 17, 32, 33, 100, 1000)
            v = rand(rng, UInt32(1):UInt32(50), n)
            sorted = _stable_sort!(copy(v), isless)
            @test sorted == sort(v)

            # Stability: sort (key, id) pairs by key only; ids within equal keys
            # must keep their original order.
            pairs = [(rand(rng, 1:5), i) for i in 1:n]
            bykey = _stable_sort_by!(copy(pairs), first)
            @test bykey == sort(pairs; by = first)  # Base sort is stable by default
            ltkey = _stable_sort!(copy(pairs), (a, b) -> isless(a[1], b[1]))
            @test ltkey == bykey
        end
    end

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
        @test nthroot(7.0, 0) == 1.0
        @test nthroot(5.0, 1) == 5.0
        @test nthroot(9.0, 2) ≈ 3.0
        @test nthroot(8.0, 3) ≈ 2.0
        @test nthroot(16.0, 4) ≈ 2.0
        @test nthroot(32.0, 5) ≈ 2.0
    end
end

@testset "SegmentStepper" begin
    @testset "reinit! resets stepper in-place" begin
        S = SegmentStepper(0.0 + 0.0im, 1.0 + 0.0im)
        propose_step!(S, 0.5)
        step_success!(S)

        reinit!(S, 0.0 + 0.0im, 2.0 + 0.0im)
        @test S.abs_Δ ≈ 2.0
        @test S.s ≈ 0.0
        @test !is_done(S)
    end

    @testset "forward with extreme steps" begin
        seg = SegmentStepper(0, 1)
        @test seg.forward
        @test seg.abs_Δ == 1.0
        @test dist_to_target(seg) ≈ 1.0

        propose_step!(seg, exp2(-60))
        @test seg.Δs == exp2(-60)
        @test seg.t′ == exp2(-60)
        @test seg.t == 0
        @test seg.Δt == exp2(-60)
        step_success!(seg)
        @test seg.s == exp2(-60)

        propose_step!(seg, exp2(-50))
        @test seg.s′ == exp2(-60) + exp2(-50)

        # Clamping to target
        propose_step!(seg, 4)
        @test seg.s′ == 1.0
        @test seg.t′ == 1.0
        @test !is_done(seg)
        step_success!(seg)
        @test is_done(seg)
    end

    @testset "complex backward segment" begin
        seg = SegmentStepper(im, 0)
        @test !seg.forward
        @test seg.abs_Δ == 1.0

        propose_step!(seg, exp2(-20))
        @test seg.Δs == exp2(-20)
        @test seg.t′ ≈ im * (1 - exp2(-20))
        @test seg.t == im
        @test seg.Δt ≈ -im * exp2(-20)
        step_success!(seg)
        @test seg.s == 1 - exp2(-20)

        # Clamping to target
        propose_step!(seg, 2)
        @test seg.s′ == 0.0
        @test seg.t′ == 0.0
        @test !is_done(seg)
        step_success!(seg)
        @test is_done(seg)
    end
end
