using Test
import HomotopyContinuationNext as Next
using HomotopyContinuationNext: Interpreter, InstructionSequence, Instruction,
    OpType, execute!, execute_taylor!,
    compile_to_instructions, SExpr, SExprT, cse,
    TruncatedTaylorSeries, TaylorVector,
    DoubleF64, ComplexDF64
using FixedSizeArrays: FixedSizeArray

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

# Helper: f(x1,x2) = x1*x2 + x1
function make_test_sequence()
    exprs = SExprT[SExpr.SAdd(SExprT[SExpr.SMul(SExprT[SExpr.SVar(1), SExpr.SVar(2)]), SExpr.SVar(1)])]
    replacements, reduced = cse(exprs)
    return compile_to_instructions(
        replacements, reduced, 2, 0, 1,
    )
end

# Helper: f(x1,x2; p1) = p1*x1^2 + x2
function make_param_sequence()
    exprs = SExprT[SExpr.SAdd(SExprT[SExpr.SMul(SExprT[SExpr.SParam(1), SExpr.SPow(SExpr.SVar(1), 2)]), SExpr.SVar(2)])]
    replacements, reduced = cse(exprs)
    return compile_to_instructions(
        replacements, reduced, 2, 1, 1,
    )
end

# Helper: f1=x1*x2, f2=x1+x2
function make_two_output_sequence()
    exprs = SExprT[SExpr.SMul(SExprT[SExpr.SVar(1), SExpr.SVar(2)]), SExpr.SAdd(SExprT[SExpr.SVar(1), SExpr.SVar(2)])]
    replacements, reduced = cse(exprs)
    return compile_to_instructions(
        replacements, reduced, 2, 0, 2,
    )
end

# Low-level unary sequence used to keep the interpreter surface covered before
# the non-polynomial expression frontend starts emitting these operations.
function make_unary_sequence(op::OpType.T)
    input = (Int32(1), Int32(1), Int32(1), Int32(1))
    stop_input = (Int32(2), Int32(2), Int32(2), Int32(2))
    instructions = Instruction[
        Instruction(input, op, Int32(2)),
        Instruction(stop_input, OpType.OP_STOP, Int32(2)),
    ]
    return InstructionSequence(
        instructions,
        ComplexF64[],
        1:0,
        1:0,
        1:1,
        1,
        2,
        [(1, 2)],
        Tuple{Int, Int}[],
        true,
        false,
    )
end

@testset "Interpreter" begin
    @testset "execute! with parameters" begin
        seq = make_param_sequence()
        I = Interpreter(Vector{ComplexF64}, seq)
        u = zeros(ComplexF64, 1)
        execute!(u, I, ComplexF64[3.0, 1.0], ComplexF64[2.0])
        @test u[1] ≈ 19.0 + 0im  # 2*9 + 1
    end

    @testset "execute! two outputs" begin
        seq = make_two_output_sequence()
        I = Interpreter(Vector{ComplexF64}, seq)
        u = zeros(ComplexF64, 2)
        execute!(u, I, ComplexF64[2.0, 3.0], ComplexF64[])
        @test u[1] ≈ 6.0 + 0im
        @test u[2] ≈ 5.0 + 0im
    end

    @testset "execute! repeated calls" begin
        seq = make_test_sequence()
        I = Interpreter(Vector{ComplexF64}, seq)
        u = zeros(ComplexF64, 1)
        execute!(u, I, ComplexF64[2.0, 3.0], ComplexF64[])
        @test u[1] ≈ 8.0 + 0im
        execute!(u, I, ComplexF64[4.0, 5.0], ComplexF64[])
        @test u[1] ≈ 24.0 + 0im
    end

    @testset "ComplexDF64 tape" begin
        seq = make_test_sequence()
        I = Interpreter(Vector{ComplexDF64}, seq)
        u = zeros(ComplexDF64, 1)
        execute!(u, I, ComplexDF64[ComplexDF64(2.0), ComplexDF64(3.0)], ComplexDF64[])
        @test real(ComplexF64(u[1])) ≈ 8.0
    end

    @testset "retained non-polynomial unary instructions" begin
        z = 1.3 + 0.4im
        series = TruncatedTaylorSeries((z, 0.3 - 0.2im, -0.15 + 0.25im))
        data = FSMat{ComplexF64}(zeros(ComplexF64, 3, 1))
        for k in 0:2
            data[k + 1, 1] = series[k]
        end
        tx = TaylorVector{3, ComplexF64}(data)

        cases = (
            (OpType.OP_SIN, sin, Next.taylor_op_sin),
            (OpType.OP_COS, cos, Next.taylor_op_cos),
            (OpType.OP_SQRT, sqrt, Next.taylor_op_sqrt),
        )
        for (op, scalar_fn, taylor_fn) in cases
            seq = make_unary_sequence(op)

            I = Interpreter(Vector{ComplexF64}, seq)
            u = zeros(ComplexF64, 1)
            execute!(u, I, ComplexF64[z], ComplexF64[])
            @test u[1] ≈ scalar_fn(z)

            I_taylor = Interpreter(
                Vector{TruncatedTaylorSeries{3, ComplexF64}},
                seq,
            )
            expected = taylor_fn(series)
            for K in 0:2
                execute_taylor!(u, Val(K), I_taylor, tx, ComplexF64[])
                @test u[1] ≈ expected[K] atol = 1.0e-12
            end
        end

        # ComplexDF64 already supports the algebraic square-root path. The
        # expression-input TODO tracks DF64 sin/cos support separately.
        I_df64 = Interpreter(Vector{ComplexDF64}, make_unary_sequence(OpType.OP_SQRT))
        u_df64 = zeros(ComplexDF64, 1)
        execute!(u_df64, I_df64, ComplexDF64[ComplexDF64(z)], ComplexDF64[])
        @test ComplexF64(u_df64[1]) ≈ sqrt(z) atol = 1.0e-14
    end

    @testset "execute_taylor! higher order" begin
        seq = make_test_sequence()
        # f(x1,x2) = x1*x2 + x1
        # x1(t) = 1+t+t², x2(t) = 2+t
        # f(t) = (1+t+t²)(2+t) + (1+t+t²) = 3+4t+4t²
        TTS = TruncatedTaylorSeries{3, ComplexF64}
        I_taylor = Interpreter(Vector{TTS}, seq)

        mat = FSMat{ComplexF64}(zeros(ComplexF64, 3, 2))
        mat[1, 1] = 1.0; mat[2, 1] = 1.0; mat[3, 1] = 1.0  # x1
        mat[1, 2] = 2.0; mat[2, 2] = 1.0; mat[3, 2] = 0.0  # x2
        tx = TaylorVector{3, ComplexF64}(mat)

        u = zeros(ComplexF64, 1)
        execute_taylor!(u, Val(0), I_taylor, tx, ComplexF64[])
        @test u[1] ≈ 3.0 + 0im
        execute_taylor!(u, Val(1), I_taylor, tx, ComplexF64[])
        @test u[1] ≈ 4.0 + 0im
        execute_taylor!(u, Val(2), I_taylor, tx, ComplexF64[])
        @test u[1] ≈ 4.0 + 0im
    end

    @testset "execute_taylor! with TaylorVector parameters (Cauchy product)" begin
        # f(x1, x2; p1) = p1*x1^2 + x2  with x1(t) = 1+t, x2(t) = 0, p1(t) = 2+3t
        # f(t) = (2+3t)*(1+t)^2 + 0 = (2+3t)(1+2t+t²) = 2+7t+8t²+3t³ (truncated at order 2)
        # [f]₀ = 2, [f]₁ = 7, [f]₂ = 8
        seq = make_param_sequence()
        TTS = TruncatedTaylorSeries{3, ComplexF64}
        I = Interpreter(Vector{TTS}, seq)

        mat = FSMat{ComplexF64}(zeros(ComplexF64, 3, 2))
        mat[1, 1] = 1.0; mat[2, 1] = 1.0; mat[3, 1] = 0.0  # x1(t) = 1+t
        mat[1, 2] = 0.0; mat[2, 2] = 0.0; mat[3, 2] = 0.0  # x2(t) = 0
        tx = TaylorVector{3, ComplexF64}(mat)

        tp_mat = FSMat{ComplexF64}(zeros(ComplexF64, 3, 1))
        tp_mat[1, 1] = 2.0  # p₀
        tp_mat[2, 1] = 3.0  # p₁
        tp_mat[3, 1] = 0.0  # p₂
        tp = TaylorVector{3, ComplexF64}(tp_mat)

        u = zeros(ComplexF64, 1)
        execute_taylor!(u, Val(0), I, tx, tp)
        @test u[1] ≈ 2.0 + 0im
        execute_taylor!(u, Val(1), I, tx, tp)
        @test u[1] ≈ 7.0 + 0im
        execute_taylor!(u, Val(2), I, tx, tp)
        @test u[1] ≈ 8.0 + 0im
    end

    @testset "SExpr compound constructors copy arg vectors" begin
        add_args = SExprT[SExpr.SVar(1), SExpr.SVar(2)]
        mul_args = SExprT[SExpr.SVar(1), SExpr.SVar(2)]
        func_args = SExprT[SExpr.SVar(1), SExpr.SVar(2)]

        add = SExpr.SAdd(add_args)
        mul = SExpr.SMul(mul_args)
        func = SExpr.SFuncSym(Next.SFuncKind.SFUNC_ADD, func_args)
        add_set = Set([add])
        mul_set = Set([mul])
        func_set = Set([func])

        push!(add_args, SExpr.SVar(3))
        push!(mul_args, SExpr.SVar(3))
        push!(func_args, SExpr.SVar(3))

        @test add.args !== add_args
        @test mul.args !== mul_args
        @test func.args !== func_args
        @test length(add.args) == 2
        @test length(mul.args) == 2
        @test length(func.args) == 2
        @test add in add_set
        @test mul in mul_set
        @test func in func_set
    end
end
