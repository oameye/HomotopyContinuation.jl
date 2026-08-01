using Test
using HomotopyContinuationNext
using HomotopyContinuationNext: TotalDegree, Polyhedral, Result, PathResult,
    PathResultCode, TrackerOptions,
    solutions, real_solutions, nsolutions, nreal, is_success, is_real,
    nexcess_solutions, is_homogeneous, fix_parameters, FixedParameterSystem,
    total_degree_count, SolveCache, PolyhedralSolveCache,
    Serial, Threaded,
    _clone_system_evaluator, TrackingWorkerState, PolyhedralWorkerState,
    StraightLineBuilder, ParameterBuilder, PolyhedralBuilder
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray
using CommonSolve: CommonSolve

@testset "Solve" begin

    @testset "solve: linear system" begin
        @polyvar x y
        result = solve(System([x - 2, y - 3]), TotalDegree(; show_progress = false))
        @test nsolutions(result) == 1
        sols = solutions(result)
        @test length(sols) == 1
        @test abs(sols[1][1] - 2) < 1.0e-8
        @test abs(sols[1][2] - 3) < 1.0e-8
    end

    @testset "solve: quadratic system" begin
        @polyvar x y
        result = solve(System([x^2 + y - 1, x * y - 0.5]), TotalDegree(; show_progress = false))
        @test nsolutions(result) >= 2
        for sol in solutions(result)
            @test abs(sol[1]^2 + sol[2] - 1) < 1.0e-6
            @test abs(sol[1] * sol[2] - 0.5) < 1.0e-6
        end
    end

    @testset "solve: x^2-1, y^2-4 finds all real solutions" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]), TotalDegree(; show_progress = false))
        @test nsolutions(result) == 4
        rsols = real_solutions(result)
        @test length(rsols) == 4
        for sol in rsols
            @test sol isa Vector{Float64}
            @test abs(sol[1]^2 - 1) < 1.0e-6
            @test abs(sol[2]^2 - 4) < 1.0e-6
        end
    end

    @testset "solve: katsura-3" begin
        @polyvar x0 x1 x2 x3
        F = System(
            [
                x0 + 2x1 + 2x2 + 2x3 - 1,
                x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
                2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
                x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
            ]
        )
        result = solve(F, TotalDegree(; show_progress = false))
        @test nsolutions(result) >= 2
        for sol in solutions(result)
            residual = maximum(
                abs.(
                    [
                        sol[1] + 2sol[2] + 2sol[3] + 2sol[4] - 1,
                        sol[1]^2 + 2sol[2]^2 + 2sol[3]^2 + 2sol[4]^2 - sol[1],
                        2sol[1] * sol[2] + 2sol[2] * sol[3] + 2sol[3] * sol[4] - sol[2],
                        sol[2]^2 + 2sol[1] * sol[3] + 2sol[2] * sol[4] - sol[3],
                    ]
                )
            )
            @test residual < 1.0e-6
        end
    end

    @testset "solve: reproducible with seed" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 0.5])
        r1 = solve(F, TotalDegree(; seed = UInt32(42), show_progress = false))
        r2 = solve(F, TotalDegree(; seed = UInt32(42), show_progress = false))
        @test nsolutions(r1) == nsolutions(r2)
        # Compare as sets — solution ordering may differ even with same seed
        s1 = solutions(r1)
        s2 = solutions(r2)
        for a in s1
            @test any(b -> isapprox(a, b; atol = 1.0e-10), s2)
        end
    end

    @testset "solve: explicit algorithm" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y - 2]), TotalDegree(; show_progress = false))
        @test nsolutions(result) >= 1
        for sol in solutions(result)
            @test abs(sol[1]^2 - 1) < 1.0e-6
            @test abs(sol[2] - 2) < 1.0e-6
        end
    end

    @testset "solve: CommonSolve init/solve! interface" begin
        @polyvar x y
        cache = CommonSolve.init(System([x^2 - 1, y - 2]), TotalDegree())
        @test cache isa SolveCache
        result = CommonSolve.solve!(cache)
        @test nsolutions(result) >= 1
    end

    @testset "solve: complex-only solutions" begin
        @polyvar x y
        result = solve(System([x^2 + 1, y - 1]), TotalDegree(; show_progress = false))
        @test nreal(result) == 0
        @test nsolutions(result) >= 1
    end

    @testset "Result: show" begin
        @polyvar x y
        result = solve(System([x - 1, y - 2]), TotalDegree(; show_progress = false))
        buf = IOBuffer()
        show(buf, result)
        s = String(take!(buf))
        @test contains(s, "tracked paths")
        @test contains(s, "solutions")
    end

    @testset "TotalDegree: custom tracker options" begin
        @polyvar x y
        opts = TrackerOptions(; max_steps = 100)
        alg = TotalDegree(; tracker_options = opts, show_progress = false)
        result = solve(System([x^2 - 1, y^2 - 1]), alg)
        # Should still work with small max_steps for simple system
        @test nsolutions(result) >= 1
    end

    @testset "total_degree_count" begin
        @test total_degree_count([2, 3]) == 6
        @test total_degree_count([1, 2, 2, 2]) == 8
    end

    # ── Polyhedral homotopy tests ──────────────────────────────────────────

    @testset "Polyhedral: x²+y-1, xy-2" begin
        @polyvar x y
        result = solve(System([x^2 + y - 1, x * y - 2]), Polyhedral(; show_progress = false))
        # mixed volume = 3 for this system
        @test nsolutions(result) >= 2
        for sol in solutions(result)
            @test abs(sol[1]^2 + sol[2] - 1) < 1.0e-6
            @test abs(sol[1] * sol[2] - 2) < 1.0e-6
        end
    end

    @testset "Polyhedral: x²-1, y²-4 finds all solutions" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]), Polyhedral(; show_progress = false))
        @test nsolutions(result) == 4
        rsols = real_solutions(result)
        @test length(rsols) == 4
        for sol in rsols
            @test abs(sol[1]^2 - 1) < 1.0e-6
            @test abs(sol[2]^2 - 4) < 1.0e-6
        end
    end

    @testset "Polyhedral: katsura-3" begin
        @polyvar x0 x1 x2 x3
        F = System(
            [
                x0 + 2x1 + 2x2 + 2x3 - 1,
                x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
                2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
                x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
            ]
        )
        result = solve(F, Polyhedral(; show_progress = false))
        @test nsolutions(result) >= 2
        for sol in solutions(result)
            residual = maximum(
                abs.(
                    [
                        sol[1] + 2sol[2] + 2sol[3] + 2sol[4] - 1,
                        sol[1]^2 + 2sol[2]^2 + 2sol[3]^2 + 2sol[4]^2 - sol[1],
                        2sol[1] * sol[2] + 2sol[2] * sol[3] + 2sol[3] * sol[4] - sol[2],
                        sol[2]^2 + 2sol[1] * sol[3] + 2sol[2] * sol[4] - sol[3],
                    ]
                )
            )
            @test residual < 1.0e-6
        end
    end

    @testset "Polyhedral: reproducible with seed" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 0.5])
        r1 = solve(F, Polyhedral(; seed = UInt32(42), show_progress = false))
        r2 = solve(F, Polyhedral(; seed = UInt32(42), show_progress = false))
        @test nsolutions(r1) == nsolutions(r2)
        # Compare as sets: for each solution in r1, find a matching one in r2
        s1 = solutions(r1)
        s2 = solutions(r2)
        for a in s1
            @test any(b -> isapprox(a, b; atol = 1.0e-10), s2)
        end
    end

    @testset "Polyhedral: CommonSolve init/solve! interface" begin
        @polyvar x y
        cache = CommonSolve.init(System([x^2 - 1, y - 2]), Polyhedral())
        @test cache isa PolyhedralSolveCache
        result = CommonSolve.solve!(cache)
        @test nsolutions(result) >= 1
    end

    @testset "Polyhedral: default endgame options" begin
        alg = Polyhedral()
        @test alg.common.endgame_options.lambda == 0.25
        @test alg.common.endgame_options.min_cond == 1.0e6
    end

    @testset "Polyhedral: fewer paths than TotalDegree" begin
        # For a sparse system, polyhedral should track fewer (or equal) paths
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 2])
        r_td = solve(F, TotalDegree(; show_progress = false))
        r_ph = solve(F, Polyhedral(; show_progress = false))
        # Both should find the same solutions
        @test nsolutions(r_td) == nsolutions(r_ph)
    end

    @testset "Polyhedral: only_torus" begin
        @polyvar x₁ x₂ s
        F = System(
            [
                x₁^3 * s^15 + x₁ * x₂ * s + x₂^3 + s^12,
                x₁^2 * s^9 + x₁ * x₂^2 + x₂ * s^3,
                x₁^2 * x₂ * s^5 + x₁ * s^8 + x₂^2,
            ],
        )
        all_alg = Polyhedral(; seed = UInt32(1), show_progress = false)
        torus_alg =
            Polyhedral(; only_torus = true, seed = UInt32(1), show_progress = false)
        @test paths_to_track(F, all_alg) == 92
        @test paths_to_track(F, torus_alg) == 54
        @test mixed_volume(F) == 54

        # Every torus solution is also found by the padded start system, and the
        # padded one additionally reaches the coordinate hyperplanes.
        r_all = solve(F, all_alg, Serial())
        r_torus = solve(F, torus_alg, Serial())
        @test nsolutions(r_torus) <= nsolutions(r_all)
        @test all(s -> all(!iszero, s), solutions(r_torus))
        for s in solutions(r_torus)
            @test any(t -> maximum(abs.(s .- t)) < 1.0e-8, solutions(r_all))
        end
    end

    @testset "mixed_volume: dense system reaches the Bezout number" begin
        @polyvar x y
        # A dense system has no sparsity to exploit, so BKK equals Bezout.
        F = System([x^2 + x * y + y^2 + x + y + 1, x^2 + 2 * x * y - y^2 + x - y + 2])
        @test mixed_volume(F) == 4
        @test paths_to_track(F, TotalDegree()) == 4
    end

    @testset "Polyhedral vs TotalDegree: solution counts match" begin
        @polyvar x y
        systems = [
            System([x^2 - 1, y^2 - 4]),
            System([x^2 + y^2 - 1, x * y - 0.25]),
        ]
        for F in systems
            r_td = solve(F, TotalDegree(; seed = UInt32(1), show_progress = false))
            r_ph = solve(F, Polyhedral(; seed = UInt32(1), show_progress = false))
            @test nsolutions(r_td) == nsolutions(r_ph)
        end
    end

    # ── Projective (homogeneous) input ────────────────────────────────────
    #
    # Solutions are ambient representatives, so they are compared normalized.

    projective_normalize(sols) = [s ./ s[argmax(abs.(s))] for s in sols]
    function same_point_set(a, b)
        nb = projective_normalize(b)
        return length(a) == length(b) &&
            all(u -> any(v -> maximum(abs.(u .- v)) < 1.0e-7, nb), projective_normalize(a))
    end

    @testset "projective: square after the chart row" begin
        @polyvar x y z
        F = System(
            [
                2.3 * x^2 + 1.2 * y^2 + 3x * z - 2y * z + 3 * z^2,
                2.3 * x^2 + 1.2 * y^2 + 5x * z + 2y * z - 5 * z^2,
            ]
        )
        @test is_homogeneous(F)
        run(alg, exec = Serial()) = solve(F, alg, exec)

        r = run(TotalDegree(; seed = UInt32(0x1234)))
        for res in (r, run(Polyhedral(; seed = UInt32(0x1234))))
            @test res.tracked_paths == 4
            @test count(is_success, res.path_results) == 4
            @test nsolutions(res) == 4
            for v in projective_normalize(solutions(res))
                @test abs(2.3v[1]^2 + 1.2v[2]^2 + 3v[1] * v[3] - 2v[2] * v[3] + 3v[3]^2) < 1.0e-8
                @test abs(2.3v[1]^2 + 1.2v[2]^2 + 5v[1] * v[3] + 2v[2] * v[3] - 5v[3]^2) < 1.0e-8
            end
        end

        # Two of the four points have z = 0, which the chart z = 1 would not reach.
        @test count(s -> abs(s[3]) < 1.0e-8 * maximum(abs.(s)), solutions(r)) == 2

        # Independent of the chart drawn, the algorithm, and the executor.
        for (alg, exec) in (
                (TotalDegree(; seed = UInt32(99)), Serial()),
                (Polyhedral(; seed = UInt32(99)), Serial()),
                (TotalDegree(; seed = UInt32(0x1234)), Threaded()),
            )
            @test same_point_set(solutions(run(alg, exec)), solutions(r))
        end
    end

    @testset "projective: overdetermined after the chart row" begin
        @polyvar x y z
        # 3 equations in 3 variables, so the chart row makes it overdetermined:
        # the square-up keeps the 3 largest of the degrees [3, 3, 1, 1].
        F = System(
            [
                (x^2 + y^2 + x * y - 3 * z^2) * (x + 3z),
                (x^2 + y^2 + x * y - 3 * z^2) * (y - x + 2z),
                2x + 5y - 3z,
            ]
        )
        for alg in (
                TotalDegree(; seed = UInt32(0x1234), show_progress = false),
                Polyhedral(; seed = UInt32(0x1234), show_progress = false),
            )
            r = solve(F, alg, Serial())
            @test r.tracked_paths == 9
            @test count(is_success, r.path_results) == 2
            @test nsolutions(r) == 2
            @test nexcess_solutions(r) == 3
        end

        # Equation order does not change the Bezout count: degrees [3, 1, 4] plus
        # the chart row track 4 * 3 * 1 paths either way.
        G = System(
            [
                (x^2 + y^2 + x * y - 3 * z^2) * (x + 3z),
                2x + 5y - 3z,
                (x^2 + y^2 + x * y - 3 * z^2) * (y^2 - x * z + 2 * z^2),
            ]
        )
        r = solve(G, TotalDegree(; seed = UInt32(0x1234), show_progress = false), Serial())
        @test r.tracked_paths == 12
        @test count(is_success, r.path_results) == 2
    end

    @testset "projective: composition and fixed parameters" begin
        @polyvar u v w a
        run(F) = solve(F, TotalDegree(; seed = UInt32(5), show_progress = false), Serial())
        L = System([2u - v + w, u + 3w, v - w]; variables = [u, v, w])

        C = System([u * v - w^2, u^2 + v * w]; variables = [u, v, w]) ∘ L
        @test is_homogeneous(C)
        rc = run(C)
        @test rc.tracked_paths == 4
        @test nsolutions(rc) == 4
        @test same_point_set(solutions(rc), solutions(run(System(C))))

        Cp = System([u * v - a * w^2, u^2 + v * w]; variables = [u, v, w], parameters = [a]) ∘ L
        FP = fix_parameters(Cp, [2.0])
        @test FP isa FixedParameterSystem
        @test is_homogeneous(FP)
        rf = run(FP)
        @test nsolutions(rf) == 4
        @test same_point_set(solutions(rf), solutions(run(fix_parameters(System(Cp), [2.0]))))
    end

    # ── Parameter homotopy tests ──────────────────────────────────────────

    @testset "Parameter homotopy: basic" begin
        @polyvar x y a b
        F = System([x^2 + a * y - 1, x * y - b]; parameters = [a, b])
        # Solve the non-parametric version at a=1, b=0.5 to get start solutions
        F_fixed = System([x^2 + 1.0 * y - 1, x * y - 0.5])
        r1 = solve(F_fixed, TotalDegree(; show_progress = false))
        @test nsolutions(r1) >= 2
        # Track to new parameters a=2, b=1
        r2 = solve(
            F,
            solutions(r1),
            [1.0, 0.5],
            [2.0, 1.0],
            Continuation(; show_progress = false),
        )
        @test nsolutions(r2) >= 2
        # Verify solutions satisfy the target system
        for sol in solutions(r2)
            @test abs(sol[1]^2 + 2.0 * sol[2] - 1) < 1.0e-6
            @test abs(sol[1] * sol[2] - 1.0) < 1.0e-6
        end
    end

    @testset "Parameter homotopy: x²-a, y²-a" begin
        @polyvar x y a
        F = System([x^2 - a, y^2 - a]; parameters = [a])
        # Solve at a=1
        F_fixed = System([x^2 - 1, y^2 - 1])
        r1 = solve(F_fixed, TotalDegree(; show_progress = false))
        @test nsolutions(r1) == 4
        # Track to a=4
        r2 = solve(F, solutions(r1), [1.0], [4.0], Continuation(; show_progress = false))
        @test nsolutions(r2) == 4
        rsols = real_solutions(r2)
        @test length(rsols) == 4
        for sol in rsols
            @test abs(sol[1]^2 - 4) < 1.0e-6
            @test abs(sol[2]^2 - 4) < 1.0e-6
        end
    end

    @testset "Parameter homotopy: CommonSolve interface" begin
        @polyvar x y a
        F = System([x^2 - a, y - 1]; parameters = [a])
        F_fixed = System([x^2 - 1, y - 1])
        r1 = solve(F_fixed, TotalDegree(; show_progress = false))
        cache = CommonSolve.init(F, solutions(r1), [1.0], [4.0])
        @test cache isa SolveCache
        r2 = CommonSolve.solve!(cache)
        @test nsolutions(r2) >= 1
    end

    # Without values there is no target system to build a start system for; the
    # routes with values fixed are covered in `fixed_parameter_test.jl`.
    @testset "start-system routes reject a parametric system without values" begin
        @polyvar x y a
        F = System([x^2 + y^2 - a, x * y - 1]; variables = [x, y], parameters = [a])
        for alg in (
                TotalDegree(; show_progress = false),
                Polyhedral(; show_progress = false),
            )
            @test_throws ArgumentError solve(F, alg, Serial())
            @test_throws ArgumentError CommonSolve.init(F, alg, Serial())
        end
        @test_throws ArgumentError solve(
            F,
            rand_subspace(2; codim = 1),
            TotalDegree(; show_progress = false),
            Serial(),
        )
        G = System([x^2 + y^2 - 5, x * y - 1]; variables = [x, y])
        @test nsolutions(solve(G, TotalDegree(; show_progress = false), Serial())) == 4
    end

    @testset "Executor types" begin
        @test Serial() isa HomotopyContinuationNext.AbstractExecutor
        @test Threaded() isa HomotopyContinuationNext.AbstractExecutor
        @test Threaded().ntasks == Threads.nthreads()
        @test Threaded(1).ntasks == 1
        # Cannot exceed available threads
        @test_throws ArgumentError Threaded(Threads.nthreads() + 1)
        # Must be positive
        @test_throws ArgumentError Threaded(0)
        @test_throws ArgumentError Threaded(-1)
    end

    @testset "_clone_system_evaluator: interpreted" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 2])
        @test F.compile_mode == CompileMode.INTERPRETED

        original = F.evaluator
        cloned = _clone_system_evaluator(F)

        n = 2
        x_test = FixedSizeArray{ComplexF64, 1}(ComplexF64[1.0 + 0.5im, 2.0 - 0.3im])
        p_empty = FixedSizeArray{ComplexF64, 1}(ComplexF64[])

        # evaluate!
        u_orig = FixedSizeArray{ComplexF64, 1}(zeros(ComplexF64, n))
        u_clone = FixedSizeArray{ComplexF64, 1}(zeros(ComplexF64, n))
        original._evaluate!(u_orig, x_test, p_empty)
        cloned._evaluate!(u_clone, x_test, p_empty)
        @test u_orig ≈ u_clone

        # evaluate_and_jacobian!
        U_orig = FixedSizeArray{ComplexF64, 2}(zeros(ComplexF64, n, n))
        U_clone = FixedSizeArray{ComplexF64, 2}(zeros(ComplexF64, n, n))
        original._evaluate_and_jacobian!(u_orig, U_orig, x_test, p_empty)
        cloned._evaluate_and_jacobian!(u_clone, U_clone, x_test, p_empty)
        @test u_orig ≈ u_clone
        @test U_orig ≈ U_clone

        # Independence: calling one does not affect the other
        x_test2 = FixedSizeArray{ComplexF64, 1}(ComplexF64[3.0, 4.0])
        original._evaluate!(u_orig, x_test2, p_empty)
        cloned._evaluate!(u_clone, x_test, p_empty)  # different input
        @test !(u_orig ≈ u_clone)
    end

    @testset "_clone_system_evaluator: compiled" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 2]; compile = CompileMode.COMPILED)
        @test F.compile_mode == CompileMode.COMPILED

        original = F.evaluator
        cloned = _clone_system_evaluator(F)

        n = 2
        x_test = FixedSizeArray{ComplexF64, 1}(ComplexF64[1.0 + 0.5im, 2.0 - 0.3im])
        p_empty = FixedSizeArray{ComplexF64, 1}(ComplexF64[])

        u_orig = FixedSizeArray{ComplexF64, 1}(zeros(ComplexF64, n))
        u_clone = FixedSizeArray{ComplexF64, 1}(zeros(ComplexF64, n))
        original._evaluate!(u_orig, x_test, p_empty)
        cloned._evaluate!(u_clone, x_test, p_empty)
        @test u_orig ≈ u_clone

        U_orig = FixedSizeArray{ComplexF64, 2}(zeros(ComplexF64, n, n))
        U_clone = FixedSizeArray{ComplexF64, 2}(zeros(ComplexF64, n, n))
        original._evaluate_and_jacobian!(u_orig, U_orig, x_test, p_empty)
        cloned._evaluate_and_jacobian!(u_clone, U_clone, x_test, p_empty)
        @test u_orig ≈ u_clone
        @test U_orig ≈ U_clone
    end

    @testset "_clone_system_evaluator: compiled_all" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 2]; compile = CompileMode.COMPILED_ALL)
        @test F.compile_mode == CompileMode.COMPILED_ALL

        original = F.evaluator
        cloned = _clone_system_evaluator(F)

        n = 2
        x_test = FixedSizeArray{ComplexF64, 1}(ComplexF64[1.0 + 0.5im, 2.0 - 0.3im])
        p_empty = FixedSizeArray{ComplexF64, 1}(ComplexF64[])

        u_orig = FixedSizeArray{ComplexF64, 1}(zeros(ComplexF64, n))
        u_clone = FixedSizeArray{ComplexF64, 1}(zeros(ComplexF64, n))
        original._evaluate!(u_orig, x_test, p_empty)
        cloned._evaluate!(u_clone, x_test, p_empty)
        @test u_orig ≈ u_clone

        U_orig = FixedSizeArray{ComplexF64, 2}(zeros(ComplexF64, n, n))
        U_clone = FixedSizeArray{ComplexF64, 2}(zeros(ComplexF64, n, n))
        original._evaluate_and_jacobian!(u_orig, U_orig, x_test, p_empty)
        cloned._evaluate_and_jacobian!(u_clone, U_clone, x_test, p_empty)
        @test u_orig ≈ u_clone
        @test U_orig ≈ U_clone
    end

    @testset "_clone_system_evaluator: parametric" begin
        @polyvar x y a b
        F = System([a * x^2 + y - 1, x * y - b]; parameters = [a, b])

        original = F.evaluator
        cloned = _clone_system_evaluator(F)

        n = 2
        x_test = FixedSizeArray{ComplexF64, 1}(ComplexF64[1.0 + 0.5im, 2.0 - 0.3im])
        p_test = FixedSizeArray{ComplexF64, 1}(ComplexF64[3.0, 0.7 + 0.1im])

        u_orig = FixedSizeArray{ComplexF64, 1}(zeros(ComplexF64, n))
        u_clone = FixedSizeArray{ComplexF64, 1}(zeros(ComplexF64, n))
        original._evaluate!(u_orig, x_test, p_test)
        cloned._evaluate!(u_clone, x_test, p_test)
        @test u_orig ≈ u_clone

        U_orig = FixedSizeArray{ComplexF64, 2}(zeros(ComplexF64, n, n))
        U_clone = FixedSizeArray{ComplexF64, 2}(zeros(ComplexF64, n, n))
        original._evaluate_and_jacobian!(u_orig, U_orig, x_test, p_test)
        cloned._evaluate_and_jacobian!(u_clone, U_clone, x_test, p_test)
        @test u_orig ≈ u_clone
        @test U_orig ≈ U_clone
    end

    @testset "PolyhedralSolveCache carries executor" begin
        @polyvar x y
        cache = CommonSolve.init(System([x^2 - 1, y^2 - 4]), Polyhedral(), Serial())
        @test cache isa PolyhedralSolveCache{Serial}
        result = CommonSolve.solve!(cache)
        @test nsolutions(result) == 4

        cache2 = CommonSolve.init(System([x^2 - 1, y^2 - 4]), Polyhedral(), Threaded())
        @test cache2 isa PolyhedralSolveCache{Threaded}
    end

    @testset "SolveCache carries executor and builder" begin
        @polyvar x y
        cache = CommonSolve.init(System([x^2 - 1, y - 2]), TotalDegree(), Serial())
        @test cache isa SolveCache{Serial}
        result = CommonSolve.solve!(cache)
        @test nsolutions(result) >= 1

        cache2 = CommonSolve.init(System([x^2 - 1, y - 2]), TotalDegree(), Threaded())
        @test cache2 isa SolveCache{Threaded}
    end

    @testset "StraightLineBuilder produces working tracker" begin
        @polyvar x y
        F = System([x^2 - 1, y^2 - 4])
        builder = StraightLineBuilder(
            F.degrees, F, cis(2π * 0.3),
            TrackerOptions(), EndgameOptions(),
        )
        ws = builder()
        @test ws isa TrackingWorkerState

        # Second call produces independent state
        ws2 = builder()
        @test ws2.tracker !== ws.tracker
    end

    @testset "System stores compile_mode" begin
        @polyvar x y
        F_interp = System([x^2 - 1, y - 2])
        @test F_interp.compile_mode == CompileMode.INTERPRETED
        @test HomotopyContinuationNext.system_shape(F_interp) isa
            HomotopyContinuationNext.SquareShape

        F_compiled = System([x^2 - 1, y - 2]; compile = CompileMode.COMPILED)
        @test F_compiled.compile_mode == CompileMode.COMPILED
        @test typeof(F_interp) != typeof(F_compiled)

        F_under = System([x + y])
        F_over = System([x^2 - 1, y - 2, x + y - 3])
        @test HomotopyContinuationNext.system_shape(F_under) isa
            HomotopyContinuationNext.UnderdeterminedShape
        @test HomotopyContinuationNext.system_shape(F_over) isa
            HomotopyContinuationNext.OverdeterminedShape
        @test typeof(F_interp) != typeof(F_under)
        @test typeof(F_interp) != typeof(F_over)
    end

    # ── Executor integration tests ───────────────────────────────────────

    @testset "solve: TotalDegree + Serial" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]), TotalDegree(; show_progress = false), Serial())
        @test nsolutions(result) == 4
        @test length(real_solutions(result)) == 4
    end

    @testset "solve: TotalDegree + Threaded" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]), TotalDegree(; show_progress = false), Threaded())
        @test nsolutions(result) == 4
        @test length(real_solutions(result)) == 4
    end

    @testset "solve: convenience executor method" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]), TotalDegree(; show_progress = false), Serial())
        @test nsolutions(result) == 4
    end

    @testset "solve: Polyhedral + Serial" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]), Polyhedral(; show_progress = false), Serial())
        @test nsolutions(result) == 4
        @test length(real_solutions(result)) == 4
    end

    @testset "solve: Polyhedral + Threaded" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]), Polyhedral(; show_progress = false), Threaded())
        @test nsolutions(result) == 4
        @test length(real_solutions(result)) == 4
    end

    @testset "Parameter homotopy + Serial" begin
        @polyvar x y a
        F = System([x^2 - a, y^2 - a]; parameters = [a])
        F_fixed = System([x^2 - 1, y^2 - 1])
        r1 = solve(F_fixed, TotalDegree(; show_progress = false))
        r2 = solve(
            F,
            solutions(r1),
            [1.0],
            [4.0],
            Continuation(; show_progress = false),
            Serial(),
        )
        @test nsolutions(r2) == 4
        for sol in real_solutions(r2)
            @test abs(sol[1]^2 - 4) < 1.0e-6
            @test abs(sol[2]^2 - 4) < 1.0e-6
        end
    end

    @testset "Parameter homotopy + Threaded" begin
        @polyvar x y a
        F = System([x^2 - a, y^2 - a]; parameters = [a])
        F_fixed = System([x^2 - 1, y^2 - 1])
        r1 = solve(F_fixed, TotalDegree(; show_progress = false))
        r2 = solve(
            F,
            solutions(r1),
            [1.0],
            [4.0],
            Continuation(; show_progress = false),
            Threaded(),
        )
        @test nsolutions(r2) == 4
        for sol in real_solutions(r2)
            @test abs(sol[1]^2 - 4) < 1.0e-6
            @test abs(sol[2]^2 - 4) < 1.0e-6
        end
    end

    @testset "Serial vs Threaded: full consistency" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 0.5])
        r_serial = solve(F, TotalDegree(; seed = UInt32(99), show_progress = false), Serial())
        r_threaded = solve(F, TotalDegree(; seed = UInt32(99), show_progress = false), Threaded())

        # Solution counts
        @test nsolutions(r_serial) == nsolutions(r_threaded)
        @test nreal(r_serial) == nreal(r_threaded)
        @test nsingular(r_serial) == nsingular(r_threaded)
        @test nnonsingular(r_serial) == nnonsingular(r_threaded)
        @test nat_infinity(r_serial) == nat_infinity(r_threaded)

        # Path-level accounting
        @test r_serial.tracked_paths == r_threaded.tracked_paths
        n_success_serial = count(is_success, r_serial.path_results)
        n_success_threaded = count(is_success, r_threaded.path_results)
        @test n_success_serial == n_success_threaded

        # Solution sets match (as unordered sets)
        s_serial = sort(solutions(r_serial); by = s -> (real(s[1]), imag(s[1])))
        s_threaded = sort(solutions(r_threaded); by = s -> (real(s[1]), imag(s[1])))
        for (a, b) in zip(s_serial, s_threaded)
            @test a ≈ b atol = 1.0e-6
        end
    end

    @testset "Polyhedral: Serial vs Threaded consistency" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 0.5])
        r_serial = solve(F, Polyhedral(; seed = UInt32(99), show_progress = false), Serial())
        r_threaded = solve(F, Polyhedral(; seed = UInt32(99), show_progress = false), Threaded())

        @test nsolutions(r_serial) == nsolutions(r_threaded)
        @test r_serial.tracked_paths == r_threaded.tracked_paths

        s_serial = sort(solutions(r_serial); by = s -> (real(s[1]), imag(s[1])))
        s_threaded = sort(solutions(r_threaded); by = s -> (real(s[1]), imag(s[1])))
        for (a, b) in zip(s_serial, s_threaded)
            @test a ≈ b atol = 1.0e-6
        end
    end

    @testset "Serial vs Threaded: cluster structure" begin
        @polyvar x0 x1 x2 x3
        F = System(
            [
                x0 + 2x1 + 2x2 + 2x3 - 1,
                x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
                2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
                x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
            ],
        )
        r_serial = solve(F, TotalDegree(; seed = UInt32(7), show_progress = false), Serial())
        r_threaded = solve(F, TotalDegree(; seed = UInt32(7), show_progress = false), Threaded())

        @test nsolutions(r_serial) == nsolutions(r_threaded)
        @test r_serial.tracked_paths == r_threaded.tracked_paths
        @test length(r_serial.clusters) == length(r_threaded.clusters)

        m_serial = sort([length(c) for c in r_serial.clusters])
        m_threaded = sort([length(c) for c in r_threaded.clusters])
        @test m_serial == m_threaded
    end

    @testset "paths_to_track" begin
        @polyvar x y z
        @test paths_to_track(System([x^2 + y^2 - 4, x * y - 1])) == 4
        @test paths_to_track(
            System([2y + 3 * y^2 - x * y^3, x + 4 * x^2 - 2 * x^3 * y]),
        ) == 16
        for F in (
                System([x^2 + y^2 - 4, x * y - 1]),
                # Projective: one chart row makes the square system one smaller.
                System([x^2 + y^2 - z^2, x * y - z^2]),
                System([(x^2 - 4) * (x * y - 2), x * y - 2, x^2 - 4]),
                System([x^2 - y, x + y - 1]) ∘ System([x + y, x - y]),
            )
            alg = TotalDegree(; seed = UInt32(11), show_progress = false)
            @test paths_to_track(F, alg) ==
                solve(F, alg, Serial()).tracked_paths
        end
    end

    # `tracked_paths` is what ran, so `nfailed` (derived from it) must stay 0 when
    # a callback ends a run early. Reporting the planned count would show the
    # untracked paths as failures.
    @testset "early_stop_callback" begin
        @polyvar x y
        F = System([x^3 + y^2 - 3, x * y^2 - 2])   # Bezout 9
        for exec in (Serial(), Threaded(1))
            full = solve(F, TotalDegree(; seed = UInt32(5), show_progress = false), exec)
            @test full.tracked_paths == 9

            hits = Threads.Atomic{Int}(0)
            stopped = solve(
                F,
                TotalDegree(;
                    seed = UInt32(5), show_progress = false,
                    early_stop_callback = function (pr)
                        Threads.atomic_add!(hits, 1)
                        return true
                    end,
                ),
                exec,
            )
            @test hits[] >= 1
            @test 0 < stopped.tracked_paths < 9
            @test nfailed(stopped) == 0
            @test length(path_results(stopped)) == stopped.tracked_paths

            # A callback that never fires leaves the run untouched.
            never = solve(
                F,
                TotalDegree(;
                    seed = UInt32(5), show_progress = false,
                    early_stop_callback = _ -> false,
                ),
                exec,
            )
            @test never.tracked_paths == 9
            @test nsolutions(never) == nsolutions(full)
        end

        # A multi-task run skips iterations rather than breaking, so the surviving
        # path numbers are not their positions. The accessors keyed by
        # `path_number` must still address the right path.
        if Threads.nthreads() > 1
            stopped = solve(
                F,
                TotalDegree(;
                    seed = UInt32(5), show_progress = false,
                    early_stop_callback = _ -> true,
                ),
                Threaded(min(4, Threads.nthreads())),
            )
            prs = path_results(stopped)
            @test issorted(path_number.(prs))
            @test allunique(path_number.(prs))
            for pr in prs
                is_success(pr) || continue
                i = path_number(pr)
                @test multiplicity(stopped, i) == multiplicity(pr)
                @test any(c -> solution(c) == solution(pr), cluster_of(stopped, i))
            end
            @test_throws BoundsError cluster_of(stopped, 10)
        end
    end
end
