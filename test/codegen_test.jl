using Test
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions
import HomotopyContinuationNext as HC
using HomotopyContinuationNext:
    System, Interpreter, InstructionSequence, CompileMode,
    _instruction_sequence_to_eval_expr,
    _instruction_sequence_to_jac_expr,
    _instruction_sequence_to_taylor_expr,
    _instruction_sequence_to_taylor_param_expr,
    execute!, evaluate!, evaluate_and_jacobian!,
    execute_taylor!, TaylorVector, TruncatedTaylorSeries, taylor!
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

_rgf(expr) = RuntimeGeneratedFunctions.RuntimeGeneratedFunction(HC, HC, expr)

@testset "Code Generation" begin

    # ── Eval expr ────────────────────────────────────────────────────────

    @testset "eval expr matches interpreter: katsura-3" begin
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        sys = System(F)
        seq = sys._interp_f64.sequence

        expr = _instruction_sequence_to_eval_expr(seq)
        @test expr isa Expr

        fn = _rgf(expr)

        x = ComplexF64[0.3, 0.5, -0.2, 0.1]
        u_compiled = zeros(ComplexF64, 4)
        fn(u_compiled, x, ComplexF64[])

        u_interp = zeros(ComplexF64, 4)
        execute!(u_interp, sys._interp_f64, x)

        @test u_compiled ≈ u_interp atol = 1.0e-14
    end

    @testset "eval expr matches interpreter: parametric" begin
        @polyvar x a b
        sys = System([a * x^2 + b * x - 1]; parameters = [a, b])
        seq = sys._interp_f64.sequence

        expr = _instruction_sequence_to_eval_expr(seq)
        fn = _rgf(expr)

        u_compiled = zeros(ComplexF64, 1)
        fn(u_compiled, ComplexF64[0.5], ComplexF64[2.0, 3.0])

        u_interp = zeros(ComplexF64, 1)
        execute!(u_interp, sys._interp_f64, ComplexF64[0.5], ComplexF64[2.0, 3.0])

        @test u_compiled ≈ u_interp atol = 1.0e-14
    end

    # ── Jac expr ─────────────────────────────────────────────────────────

    @testset "jac expr matches interpreter: katsura-3" begin
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        sys = System(F)
        seq_jac = sys._interp_jac.sequence

        expr = _instruction_sequence_to_jac_expr(seq_jac)
        fn = _rgf(expr)

        x = ComplexF64[0.3, 0.5, -0.2, 0.1]
        u_c = zeros(ComplexF64, 4)
        U_c = zeros(ComplexF64, 4, 4)
        fn(u_c, U_c, x, ComplexF64[])

        u_i = zeros(ComplexF64, 4)
        U_i = zeros(ComplexF64, 4, 4)
        execute!(u_i, U_i, sys._interp_jac, x)

        @test u_c ≈ u_i atol = 1.0e-14
        @test U_c ≈ U_i atol = 1.0e-14
    end

    # ── System compile modes ─────────────────────────────────────────────

    @testset "System(compile=COMPILED) matches INTERPRETED" begin
        @polyvar x y
        F_polys = [x^2 + y - 1, x * y - 2]

        sys_i = System(F_polys; compile = CompileMode.INTERPRETED)
        sys_c = System(F_polys; compile = CompileMode.COMPILED)

        xv = FSVec{ComplexF64}(ComplexF64[0.3, 0.7])
        pv = FSVec{ComplexF64}(ComplexF64[])

        u_i = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        u_c = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u_i, sys_i.evaluator, xv, pv)
        evaluate!(u_c, sys_c.evaluator, xv, pv)
        @test u_i ≈ u_c atol = 1.0e-14

        U_i = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        U_c = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        evaluate_and_jacobian!(u_i, U_i, sys_i.evaluator, xv, pv)
        evaluate_and_jacobian!(u_c, U_c, sys_c.evaluator, xv, pv)
        @test u_i ≈ u_c atol = 1.0e-14
        @test U_i ≈ U_c atol = 1.0e-14
    end

    @testset "System(compile=COMPILED) zero allocations" begin
        @polyvar x y
        sys = System([x^2 + y - 1, x * y - 2]; compile = CompileMode.COMPILED)

        xv = FSVec{ComplexF64}(ComplexF64[0.3, 0.7])
        pv = FSVec{ComplexF64}(ComplexF64[])
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))

        # @allocated needs a function barrier to avoid top-level scope artifacts
        function _test_compiled_allocs(sys, u, xv, pv)
            evaluate!(u, sys.evaluator, xv, pv)  # warmup
            return @allocated evaluate!(u, sys.evaluator, xv, pv)
        end
        @test _test_compiled_allocs(sys, u, xv, pv) == 0
    end

    @testset "System default is INTERPRETED" begin
        @polyvar x y
        sys = System([x^2 + y - 1, x * y - 2])
        xv = FSVec{ComplexF64}(ComplexF64[0.3, 0.7])
        pv = FSVec{ComplexF64}(ComplexF64[])
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u, sys.evaluator, xv, pv)
        @test all(isfinite, Vector(u))
    end

    @testset "System(compile=COMPILED) solves correctly" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 2]; compile = CompileMode.COMPILED)
        result = HC.solve(F)
        @test HC.nsolutions(result) >= 2
        for sol in HC.solutions(result)
            res = abs(sol[1]^2 + sol[2] - 1) + abs(sol[1] * sol[2] - 2)
            @test res < 1.0e-6
        end
    end

    # ── Taylor expr ──────────────────────────────────────────────────────

    @testset "taylor expr matches interpreter: katsura-3, order $K" for K in 1:3
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        sys = System(F)
        seq = sys._interp_f64.sequence
        N = K + 1  # TTS order: taylor_1 uses TTS{2}, etc.

        expr = _instruction_sequence_to_taylor_expr(seq, Val(K))
        @test expr isa Expr

        fn = _rgf(expr)

        nvars = 4
        data = FSMat{ComplexF64}(randn(ComplexF64, N, nvars))
        tx = TaylorVector{N, ComplexF64}(data)
        p = ComplexF64[]

        u_compiled = zeros(ComplexF64, 4)
        fn(u_compiled, tx, p)

        interp = K == 1 ? sys._interp_t1 : K == 2 ? sys._interp_t2 : sys._interp_t3
        u_interp = zeros(ComplexF64, 4)
        execute_taylor!(u_interp, Val(K), interp, tx, p)

        @test u_compiled ≈ u_interp atol = 1.0e-12
    end

    @testset "taylor expr matches interpreter: parametric, order $K" for K in 1:3
        @polyvar x a b
        sys = System([a * x^2 + b * x - 1]; parameters = [a, b])
        seq = sys._interp_f64.sequence
        N = K + 1

        expr = _instruction_sequence_to_taylor_expr(seq, Val(K))
        fn = _rgf(expr)

        data = FSMat{ComplexF64}(randn(ComplexF64, N, 1))
        tx = TaylorVector{N, ComplexF64}(data)
        p = ComplexF64[2.0, 3.0]

        u_compiled = zeros(ComplexF64, 1)
        fn(u_compiled, tx, p)

        interp = K == 1 ? sys._interp_t1 : K == 2 ? sys._interp_t2 : sys._interp_t3
        u_interp = zeros(ComplexF64, 1)
        execute_taylor!(u_interp, Val(K), interp, tx, p)

        @test u_compiled ≈ u_interp atol = 1.0e-12
    end

    # ── Taylor param expr (TaylorVector parameters) ─────────────────────

    @testset "taylor param expr matches interpreter: parametric, order $K" for K in 1:3
        @polyvar x a b
        sys = System([a * x^2 + b * x - 1]; parameters = [a, b])
        seq = sys._interp_f64.sequence
        N = K + 1

        expr = _instruction_sequence_to_taylor_param_expr(seq, Val(K))
        fn = _rgf(expr)

        data_x = FSMat{ComplexF64}(randn(ComplexF64, N, 1))
        tx = TaylorVector{N, ComplexF64}(data_x)
        data_p = FSMat{ComplexF64}(randn(ComplexF64, N, 2))
        tp = TaylorVector{N, ComplexF64}(data_p)

        u_compiled = zeros(ComplexF64, 1)
        fn(u_compiled, tx, tp)

        interp = K == 1 ? sys._interp_t1 : K == 2 ? sys._interp_t2 : sys._interp_t3
        u_interp = zeros(ComplexF64, 1)
        execute_taylor!(u_interp, Val(K), interp, tx, tp)

        @test u_compiled ≈ u_interp atol = 1.0e-12
    end

    @testset "taylor param expr matches interpreter: katsura-3 parametric, order $K" for K in 1:3
        @polyvar x0 x1 x2 x3 a b c d
        F = [a * x0^2 + b * x1 - c, d * x0 * x1 + a * x2 - b, c * x1^2 + d * x3 - a, b * x2 * x3 + c * x0 - d]
        sys = System(F; parameters = [a, b, c, d])
        seq = sys._interp_f64.sequence
        N = K + 1

        expr = _instruction_sequence_to_taylor_param_expr(seq, Val(K))
        fn = _rgf(expr)

        data_x = FSMat{ComplexF64}(randn(ComplexF64, N, 4))
        tx = TaylorVector{N, ComplexF64}(data_x)
        data_p = FSMat{ComplexF64}(randn(ComplexF64, N, 4))
        tp = TaylorVector{N, ComplexF64}(data_p)

        u_compiled = zeros(ComplexF64, 4)
        fn(u_compiled, tx, tp)

        interp = K == 1 ? sys._interp_t1 : K == 2 ? sys._interp_t2 : sys._interp_t3
        u_interp = zeros(ComplexF64, 4)
        execute_taylor!(u_interp, Val(K), interp, tx, tp)

        @test u_compiled ≈ u_interp atol = 1.0e-12
    end

    # ── COMPILED_ALL system-level tests ──────────────────────────────────

    @testset "System(compile=COMPILED_ALL) matches INTERPRETED" begin
        @polyvar x y
        F_polys = [x^2 + y - 1, x * y - 2]

        sys_i = System(F_polys; compile = CompileMode.INTERPRETED)
        sys_c = System(F_polys; compile = CompileMode.COMPILED_ALL)

        xv = FSVec{ComplexF64}(ComplexF64[0.3, 0.7])
        pv = FSVec{ComplexF64}(ComplexF64[])

        u_i = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        u_c = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u_i, sys_i.evaluator, xv, pv)
        evaluate!(u_c, sys_c.evaluator, xv, pv)
        @test u_i ≈ u_c atol = 1.0e-14

        U_i = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        U_c = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        evaluate_and_jacobian!(u_i, U_i, sys_i.evaluator, xv, pv)
        evaluate_and_jacobian!(u_c, U_c, sys_c.evaluator, xv, pv)
        @test u_i ≈ u_c atol = 1.0e-14
        @test U_i ≈ U_c atol = 1.0e-14

        # Taylor correctness through SystemEvaluator
        for K in 1:3
            N = K + 1
            data = FSMat{ComplexF64}(randn(ComplexF64, N, 2))
            tx = TaylorVector{N, ComplexF64}(data)
            u_ti = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            u_tc = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            taylor!(u_ti, Val(K), sys_i.evaluator, tx, pv)
            taylor!(u_tc, Val(K), sys_c.evaluator, tx, pv)
            @test u_ti ≈ u_tc atol = 1.0e-12
        end
    end

    @testset "System(compile=COMPILED_ALL) parametric Taylor matches INTERPRETED" begin
        @polyvar x y a b c d
        # Parametric system similar to what CoefficientHomotopy builds:
        # coefficients multiply monomials, parameters change along the homotopy path
        F_polys = [a * x^2 + b * x * y + c * y - d, b * x + a * y^2 - c]

        sys_i = System(F_polys; parameters = [a, b, c, d], compile = CompileMode.INTERPRETED)
        sys_c = System(F_polys; parameters = [a, b, c, d], compile = CompileMode.COMPILED_ALL)

        pv = FSVec{ComplexF64}(ComplexF64[1.5, -0.3, 2.1, 0.7])

        # Eval + jac correctness with parameters
        xv = FSVec{ComplexF64}(ComplexF64[0.3, 0.7])
        u_i = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        u_c = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u_i, sys_i.evaluator, xv, pv)
        evaluate!(u_c, sys_c.evaluator, xv, pv)
        @test u_i ≈ u_c atol = 1.0e-14

        # Taylor correctness with scalar parameters
        for K in 1:3
            N = K + 1
            data = FSMat{ComplexF64}(randn(ComplexF64, N, 2))
            tx = TaylorVector{N, ComplexF64}(data)
            u_ti = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            u_tc = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            taylor!(u_ti, Val(K), sys_i.evaluator, tx, pv)
            taylor!(u_tc, Val(K), sys_c.evaluator, tx, pv)
            @test u_ti ≈ u_tc atol = 1.0e-12
        end

        # Taylor correctness with TaylorVector parameters — the production
        # CoefficientHomotopy/ToricHomotopy path
        for K in 1:3
            N = K + 1
            data_x = FSMat{ComplexF64}(randn(ComplexF64, N, 2))
            tx = TaylorVector{N, ComplexF64}(data_x)
            data_p = FSMat{ComplexF64}(randn(ComplexF64, N, 4))
            tp = TaylorVector{N, ComplexF64}(data_p)
            u_ti = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            u_tc = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            taylor!(u_ti, Val(K), sys_i.evaluator, tx, tp)
            taylor!(u_tc, Val(K), sys_c.evaluator, tx, tp)
            @test u_ti ≈ u_tc atol = 1.0e-12
        end
    end

    @testset "System(compile=COMPILED_ALL) zero allocations: eval + jac + taylor" begin
        @polyvar x y
        sys = System([x^2 + y - 1, x * y - 2]; compile = CompileMode.COMPILED_ALL)

        xv = FSVec{ComplexF64}(ComplexF64[0.3, 0.7])
        pv = FSVec{ComplexF64}(ComplexF64[])
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))

        function _test_compiled_all_allocs(sys, u, xv, pv)
            evaluate!(u, sys.evaluator, xv, pv)  # warmup
            return @allocated evaluate!(u, sys.evaluator, xv, pv)
        end
        @test _test_compiled_all_allocs(sys, u, xv, pv) == 0

        # Taylor zero-allocation checks
        for K in 1:3
            N = K + 1
            data = FSMat{ComplexF64}(randn(ComplexF64, N, 2))
            tx = TaylorVector{N, ComplexF64}(data)
            ut = FSVec{ComplexF64}(zeros(ComplexF64, 2))

            # Use function barrier to avoid top-level scope artifacts
            function _test_taylor_allocs(sys, ut, tx, pv, ::Val{K}) where {K}
                taylor!(ut, Val(K), sys.evaluator, tx, pv)  # warmup
                return @allocated taylor!(ut, Val(K), sys.evaluator, tx, pv)
            end
            @test _test_taylor_allocs(sys, ut, tx, pv, Val(K)) == 0
        end
    end

    @testset "System(compile=COMPILED_ALL) solves correctly: residual check" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 2]; compile = CompileMode.COMPILED_ALL)
        result = HC.solve(F)
        @test HC.nsolutions(result) >= 2
        for sol in HC.solutions(result)
            res = abs(sol[1]^2 + sol[2] - 1) + abs(sol[1] * sol[2] - 2)
            @test res < 1.0e-10
        end
    end

    @testset "System(compile=COMPILED_ALL) katsura-3: solution agreement" begin
        @polyvar x0 x1 x2 x3
        F_polys = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        r_i = HC.solve(System(F_polys; compile = CompileMode.INTERPRETED))
        r_c = HC.solve(System(F_polys; compile = CompileMode.COMPILED_ALL))
        @test HC.nsolutions(r_i) == HC.nsolutions(r_c)

        # Compare solution sets via nearest-neighbor matching
        sols_i = HC.solutions(r_i)
        sols_c = HC.solutions(r_c)
        used = falses(length(sols_c))
        for si in sols_i
            dists = [used[j] ? Inf : maximum(abs.(si .- sols_c[j])) for j in eachindex(sols_c)]
            j = argmin(dists)
            @test dists[j] < 1.0e-6
            used[j] = true
        end
    end

    @testset "System(compile=COMPILED_ALL) parameter homotopy: solution agreement" begin
        @polyvar x y a
        F_i = System([x^2 - a, y^2 - a]; parameters = [a], compile = CompileMode.INTERPRETED)
        F_c = System([x^2 - a, y^2 - a]; parameters = [a], compile = CompileMode.COMPILED_ALL)

        F_start = System([x^2 - 1, y^2 - 1]; compile = CompileMode.INTERPRETED)
        starts = HC.solutions(HC.solve(F_start))

        r_i = HC.solve(F_i, starts; start_parameters = [1.0], target_parameters = [4.0])
        r_c = HC.solve(F_c, starts; start_parameters = [1.0], target_parameters = [4.0])
        @test HC.nresults(r_i) == HC.nresults(r_c)

        sols_i = sort(HC.real_solutions(r_i); by = s -> (s[1], s[2]))
        sols_c = sort(HC.real_solutions(r_c); by = s -> (s[1], s[2]))
        @test length(sols_i) == length(sols_c)
        for (si, sc) in zip(sols_i, sols_c)
            @test si ≈ sc atol = 1.0e-6
        end
    end
end
