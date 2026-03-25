using Test
using HomotopyContinuationNext:
    IRStatementRef, IRStatement, IntermediateRepresentation,
    Instruction, InstructionSequence,
    OpType, build_instruction_sequence_from_ir

@testset "IR data structures" begin
    stmt = IRStatement(
        OpType.OP_ADD, IRStatementRef(3), IRStatementRef(1), IRStatementRef(2),
    )
    @test stmt.op == OpType.OP_ADD
    @test stmt.args[1] == IRStatementRef(1)
    @test stmt.args[3] === nothing
end

@testset "InstructionSequence from IR" begin
    # f(x₁, x₂) = x₁ + x₂
    stmts = [
        IRStatement(OpType.OP_ADD, IRStatementRef(3), IRStatementRef(1), IRStatementRef(2)),
    ]
    assignments = [(1, IRStatementRef(3))]
    ir = IntermediateRepresentation(stmts, assignments, 1)

    seq = build_instruction_sequence_from_ir(
        ir; nvars = 2, nparams = 0, nconstants = 0, constants = ComplexF64[],
    )
    @test seq.output_dim == 1
    @test seq.tape_space_needed > 0
    @test length(seq.instructions) >= 1
    @test !isempty(seq.u_assignments)
end

@testset "InstructionSequence with constants" begin
    # f(x) = 2*x + 3, constants=[2.0, 3.0]
    stmts = [
        IRStatement(OpType.OP_MUL, IRStatementRef(4), IRStatementRef(1), IRStatementRef(3)),
        IRStatement(OpType.OP_ADD, IRStatementRef(5), IRStatementRef(4), IRStatementRef(2)),
    ]
    assignments = [(1, IRStatementRef(5))]
    ir = IntermediateRepresentation(stmts, assignments, 1)

    seq = build_instruction_sequence_from_ir(
        ir; nvars = 1, nparams = 0, nconstants = 2, constants = ComplexF64[2.0, 3.0],
    )
    @test seq.output_dim == 1
    @test length(seq.constants) == 2
    @test seq.constants_range == 1:2
    @test seq.all_u_assigned
end

@testset "InstructionSequence tape layout with params" begin
    # f(x; p) = x + p
    stmts = [
        IRStatement(OpType.OP_ADD, IRStatementRef(3), IRStatementRef(1), IRStatementRef(2)),
    ]
    assignments = [(1, IRStatementRef(3))]
    ir = IntermediateRepresentation(stmts, assignments, 1)

    seq = build_instruction_sequence_from_ir(
        ir; nvars = 1, nparams = 1, nconstants = 0, constants = ComplexF64[],
    )
    @test isempty(seq.constants_range)
    @test length(seq.parameters_range) == 1
    @test length(seq.variables_range) == 1
    @test first(seq.parameters_range) < first(seq.variables_range)
end

@testset "InstructionSequence all_u/U_assigned flags" begin
    # 2-output: f₁ = x₁, f₂ = x₂
    stmts = [
        IRStatement(OpType.OP_IDENTITY, IRStatementRef(3), IRStatementRef(1)),
        IRStatement(OpType.OP_IDENTITY, IRStatementRef(4), IRStatementRef(2)),
    ]
    assignments = [(1, IRStatementRef(3)), (2, IRStatementRef(4))]
    ir = IntermediateRepresentation(stmts, assignments, 2)

    seq = build_instruction_sequence_from_ir(
        ir; nvars = 2, nparams = 0, nconstants = 0, constants = ComplexF64[],
    )
    @test seq.output_dim == 2
    @test seq.all_u_assigned
    @test !seq.all_U_assigned
    @test length(seq.u_assignments) == 2
end
