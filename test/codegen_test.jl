using Test
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions, @RuntimeGeneratedFunction
RuntimeGeneratedFunctions.init(@__MODULE__)
import HomotopyContinuationNext as HC
using HomotopyContinuationNext:
    System, Interpreter, InstructionSequence, CompileMode,
    _instruction_sequence_to_eval_expr,
    _instruction_sequence_to_jac_expr,
    execute!, evaluate!, evaluate_and_jacobian!
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

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

        fn = @RuntimeGeneratedFunction(HC, expr)

        x = ComplexF64[0.3, 0.5, -0.2, 0.1]
        u_compiled = zeros(ComplexF64, 4)
        fn(u_compiled, x, ComplexF64[])

        u_interp = zeros(ComplexF64, 4)
        execute!(u_interp, sys._interp_f64, x)

        @test u_compiled ≈ u_interp atol = 1e-14
    end

    @testset "eval expr matches interpreter: parametric" begin
        @polyvar x a b
        sys = System([a * x^2 + b * x - 1]; parameters = [a, b])
        seq = sys._interp_f64.sequence

        expr = _instruction_sequence_to_eval_expr(seq)
        fn = @RuntimeGeneratedFunction(HC, expr)

        u_compiled = zeros(ComplexF64, 1)
        fn(u_compiled, ComplexF64[0.5], ComplexF64[2.0, 3.0])

        u_interp = zeros(ComplexF64, 1)
        execute!(u_interp, sys._interp_f64, ComplexF64[0.5], ComplexF64[2.0, 3.0])

        @test u_compiled ≈ u_interp atol = 1e-14
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
        fn = @RuntimeGeneratedFunction(HC, expr)

        x = ComplexF64[0.3, 0.5, -0.2, 0.1]
        u_c = zeros(ComplexF64, 4)
        U_c = zeros(ComplexF64, 4, 4)
        fn(u_c, U_c, x, ComplexF64[])

        u_i = zeros(ComplexF64, 4)
        U_i = zeros(ComplexF64, 4, 4)
        execute!(u_i, U_i, sys._interp_jac, x)

        @test u_c ≈ u_i atol = 1e-14
        @test U_c ≈ U_i atol = 1e-14
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
        @test u_i ≈ u_c atol = 1e-14

        U_i = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        U_c = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        evaluate_and_jacobian!(u_i, U_i, sys_i.evaluator, xv, pv)
        evaluate_and_jacobian!(u_c, U_c, sys_c.evaluator, xv, pv)
        @test u_i ≈ u_c atol = 1e-14
        @test U_i ≈ U_c atol = 1e-14
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
            @test res < 1e-6
        end
    end
end
