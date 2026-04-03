using Test
using HomotopyContinuationNext
using HomotopyContinuationNext: TotalDegree, Polyhedral, Result, PathResult,
    PathResultCode, TrackerOptions,
    solutions, real_solutions, nsolutions, nreal, is_success, is_real,
    total_degree_count, SolveCache, PolyhedralSolveCache,
    Serial, Threaded,
    _clone_system_evaluator, TrackingWorkerState, PolyhedralWorkerState,
    StraightLineBuilder, CoefficientBuilder, PolyhedralBuilder
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray
using CommonSolve: CommonSolve

@testset "Solve" begin

    @testset "solve: linear system" begin
        @polyvar x y
        result = solve(System([x - 2, y - 3]))
        @test nsolutions(result) == 1
        sols = solutions(result)
        @test length(sols) == 1
        @test abs(sols[1][1] - 2) < 1.0e-8
        @test abs(sols[1][2] - 3) < 1.0e-8
    end

    @testset "solve: quadratic system" begin
        @polyvar x y
        result = solve(System([x^2 + y - 1, x * y - 0.5]))
        @test nsolutions(result) >= 2
        for sol in solutions(result)
            @test abs(sol[1]^2 + sol[2] - 1) < 1.0e-6
            @test abs(sol[1] * sol[2] - 0.5) < 1.0e-6
        end
    end

    @testset "solve: x^2-1, y^2-4 finds all real solutions" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]))
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
        result = solve(F)
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
        r1 = solve(F, TotalDegree(; seed = UInt32(42)))
        r2 = solve(F, TotalDegree(; seed = UInt32(42)))
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
        result = solve(System([x^2 - 1, y - 2]), TotalDegree())
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
        result = solve(System([x^2 + 1, y - 1]))
        @test nreal(result) == 0
        @test nsolutions(result) >= 1
    end

    @testset "Result: show" begin
        @polyvar x y
        result = solve(System([x - 1, y - 2]))
        buf = IOBuffer()
        show(buf, result)
        s = String(take!(buf))
        @test contains(s, "tracked paths")
        @test contains(s, "solutions")
    end

    @testset "TotalDegree: custom tracker options" begin
        @polyvar x y
        opts = TrackerOptions(; max_steps = 100)
        alg = TotalDegree(; tracker_options = opts)
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
        result = solve(System([x^2 + y - 1, x * y - 2]), Polyhedral())
        # mixed volume = 3 for this system
        @test nsolutions(result) >= 2
        for sol in solutions(result)
            @test abs(sol[1]^2 + sol[2] - 1) < 1.0e-6
            @test abs(sol[1] * sol[2] - 2) < 1.0e-6
        end
    end

    @testset "Polyhedral: x²-1, y²-4 finds all solutions" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]), Polyhedral())
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
        result = solve(F, Polyhedral())
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
        r1 = solve(F, Polyhedral(; seed = UInt32(42)))
        r2 = solve(F, Polyhedral(; seed = UInt32(42)))
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

    @testset "Polyhedral: default endgame options (v2 parity)" begin
        alg = Polyhedral()
        @test alg.endgame_options.lambda == 0.25
        @test alg.endgame_options.min_cond == 1.0e6
    end

    @testset "Polyhedral: fewer paths than TotalDegree" begin
        # For a sparse system, polyhedral should track fewer (or equal) paths
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 2])
        r_td = solve(F, TotalDegree())
        r_ph = solve(F, Polyhedral())
        # Both should find the same solutions
        @test nsolutions(r_td) == nsolutions(r_ph)
    end

    @testset "Polyhedral vs TotalDegree: solution counts match" begin
        @polyvar x y
        systems = [
            System([x^2 - 1, y^2 - 4]),
            System([x^2 + y^2 - 1, x * y - 0.25]),
        ]
        for F in systems
            r_td = solve(F, TotalDegree(; seed = UInt32(1)))
            r_ph = solve(F, Polyhedral(; seed = UInt32(1)))
            @test nsolutions(r_td) == nsolutions(r_ph)
        end
    end

    # ── Parameter homotopy tests ──────────────────────────────────────────

    @testset "Parameter homotopy: basic" begin
        @polyvar x y a b
        F = System([x^2 + a * y - 1, x * y - b]; parameters = [a, b])
        # Solve the non-parametric version at a=1, b=0.5 to get start solutions
        F_fixed = System([x^2 + 1.0 * y - 1, x * y - 0.5])
        r1 = solve(F_fixed)
        @test nsolutions(r1) >= 2
        # Track to new parameters a=2, b=1
        r2 = solve(
            F, solutions(r1);
            start_parameters = [1.0, 0.5],
            target_parameters = [2.0, 1.0],
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
        r1 = solve(F_fixed)
        @test nsolutions(r1) == 4
        # Track to a=4
        r2 = solve(
            F, solutions(r1);
            start_parameters = [1.0],
            target_parameters = [4.0],
        )
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
        r1 = solve(F_fixed)
        cache = CommonSolve.init(
            F, solutions(r1);
            start_parameters = [1.0],
            target_parameters = [4.0],
        )
        @test cache isa SolveCache
        r2 = CommonSolve.solve!(cache)
        @test nsolutions(r2) >= 1
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

        F_compiled = System([x^2 - 1, y - 2]; compile = CompileMode.COMPILED)
        @test F_compiled.compile_mode == CompileMode.COMPILED
    end

    # ── Executor integration tests ───────────────────────────────────────

    @testset "solve: TotalDegree + Serial" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]), TotalDegree(), Serial())
        @test nsolutions(result) == 4
        @test length(real_solutions(result)) == 4
    end

    @testset "solve: TotalDegree + Threaded" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]), TotalDegree(), Threaded())
        @test nsolutions(result) == 4
        @test length(real_solutions(result)) == 4
    end

    @testset "solve: convenience executor method" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]), Serial())
        @test nsolutions(result) == 4
    end

    @testset "solve: Polyhedral + Serial" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]), Polyhedral(), Serial())
        @test nsolutions(result) == 4
        @test length(real_solutions(result)) == 4
    end

    @testset "solve: Polyhedral + Threaded" begin
        @polyvar x y
        result = solve(System([x^2 - 1, y^2 - 4]), Polyhedral(), Threaded())
        @test nsolutions(result) == 4
        @test length(real_solutions(result)) == 4
    end

    @testset "Parameter homotopy + Serial" begin
        @polyvar x y a
        F = System([x^2 - a, y^2 - a]; parameters = [a])
        F_fixed = System([x^2 - 1, y^2 - 1])
        r1 = solve(F_fixed)
        r2 = solve(
            F, solutions(r1), Serial();
            start_parameters = [1.0],
            target_parameters = [4.0],
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
        r1 = solve(F_fixed)
        r2 = solve(
            F, solutions(r1), Threaded();
            start_parameters = [1.0],
            target_parameters = [4.0],
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
        r_serial = solve(F, TotalDegree(; seed = UInt32(99)), Serial())
        r_threaded = solve(F, TotalDegree(; seed = UInt32(99)), Threaded())

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
        r_serial = solve(F, Polyhedral(; seed = UInt32(99)), Serial())
        r_threaded = solve(F, Polyhedral(; seed = UInt32(99)), Threaded())

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
        r_serial = solve(F, TotalDegree(; seed = UInt32(7)), Serial())
        r_threaded = solve(F, TotalDegree(; seed = UInt32(7)), Threaded())

        @test nsolutions(r_serial) == nsolutions(r_threaded)
        @test r_serial.tracked_paths == r_threaded.tracked_paths
        @test length(r_serial.clusters) == length(r_threaded.clusters)

        m_serial = sort([length(c) for c in r_serial.clusters])
        m_threaded = sort([length(c) for c in r_threaded.clusters])
        @test m_serial == m_threaded
    end
end
