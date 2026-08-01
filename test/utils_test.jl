using Test
using HomotopyContinuationNext:
    fast_abs, nanmin, nanmax, nthroot, _stable_sort!, _stable_sort_by!,
    SegmentStepper, reinit!, propose_step!, step_success!, is_done, dist_to_target,
    write_solutions, read_solutions, write_parameters, read_parameters
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

@testset "writing and reading" begin
    dir = mktempdir()
    rng = MersenneTwister(1234)

    @testset "round trip is exact" begin
        S = [rand(rng, ComplexF64, 3) for _ in 1:2]
        path = joinpath(dir, "sols.txt")
        write_solutions(path, S)
        @test read_solutions(path) == S

        p = rand(rng, ComplexF64, 14)
        ppath = joinpath(dir, "params.txt")
        write_parameters(ppath, p)
        @test read_parameters(ppath) == p
    end

    @testset "integer input reads back as complex" begin
        path = joinpath(dir, "ints.txt")
        write_solutions(path, [[1, 1], [-1, 2]])
        @test read_solutions(path) == [[1.0 + 0im, 1.0 + 0im], [-1.0 + 0im, 2.0 + 0im]]
        @test read(path, String) == "2\n\n1 0\n1 0\n\n-1 0\n2 0\n"
    end

    @testset "empty and single" begin
        path = joinpath(dir, "empty.txt")
        write_solutions(path, Vector{ComplexF64}[])
        @test isempty(read_solutions(path))

        ppath = joinpath(dir, "one.txt")
        write_parameters(ppath, [2.0 - 3.0im])
        @test read_parameters(ppath) == [2.0 - 3.0im]
    end

    @testset "a missing imaginary part is zero" begin
        path = joinpath(dir, "real_only.txt")
        write(path, "2\n\n1.5\n-2.5\n")
        @test read_parameters(path) == [1.5 + 0im, -2.5 + 0im]
    end

    @testset "declared count is checked" begin
        path = joinpath(dir, "bad_count.txt")
        write(path, "3\n\n1.0 0.0\n2.0 0.0\n")
        @test_throws ArgumentError read_parameters(path)

        write(path, "")
        @test_throws ArgumentError read_solutions(path)

        write(path, "1\n\n1.0 0.0 2.0\n")
        @test_throws ArgumentError read_parameters(path)
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
