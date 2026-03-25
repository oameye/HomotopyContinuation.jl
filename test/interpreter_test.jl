using Test
using HomotopyContinuationNext: Interpreter, InstructionSequence, Instruction,
    IRStatementRef, IRStatement, IntermediateRepresentation,
    OpType, execute!, execute_taylor!, build_instruction_sequence_from_ir,
    TruncatedTaylorSeries, TaylorVector,
    DoubleF64, ComplexDF64
using FixedSizeArrays: FixedSizeArray

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

# Helper: f(x1,x2) = x1*x2 + x1
function make_test_sequence()
    stmts = [
        IRStatement(OpType.OP_MUL, IRStatementRef(1), :x1, :x2),
        IRStatement(OpType.OP_ADD, IRStatementRef(2), IRStatementRef(1), :x1),
    ]
    assignments = [(1, IRStatementRef(2))]
    ir = IntermediateRepresentation(stmts, assignments, 1)
    return build_instruction_sequence_from_ir(
        ir; nvars = 2, nparams = 0, nconstants = 0, constants = ComplexF64[],
        variables = [:x1, :x2],
    )
end

# Helper: f(x1,x2; p1) = p1*x1^2 + x2
function make_param_sequence()
    stmts = [
        IRStatement(OpType.OP_SQR, IRStatementRef(1), :x1),
        IRStatement(OpType.OP_MUL, IRStatementRef(2), :p1, IRStatementRef(1)),
        IRStatement(OpType.OP_ADD, IRStatementRef(3), IRStatementRef(2), :x2),
    ]
    assignments = [(1, IRStatementRef(3))]
    ir = IntermediateRepresentation(stmts, assignments, 1)
    return build_instruction_sequence_from_ir(
        ir; nvars = 2, nparams = 1, nconstants = 0, constants = ComplexF64[],
        variables = [:x1, :x2], parameters = [:p1],
    )
end

# Helper: f1=x1*x2, f2=x1+x2
function make_two_output_sequence()
    stmts = [
        IRStatement(OpType.OP_MUL, IRStatementRef(1), :x1, :x2),
        IRStatement(OpType.OP_ADD, IRStatementRef(2), :x1, :x2),
    ]
    assignments = [(1, IRStatementRef(1)), (2, IRStatementRef(2))]
    ir = IntermediateRepresentation(stmts, assignments, 2)
    return build_instruction_sequence_from_ir(
        ir; nvars = 2, nparams = 0, nconstants = 0, constants = ComplexF64[],
        variables = [:x1, :x2],
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
end
