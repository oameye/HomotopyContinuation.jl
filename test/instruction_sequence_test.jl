using Test
using HomotopyContinuationNext: Instruction, InstructionSequence, OpType,
    compile_to_instructions, SExpr, SVar, SConst, SAdd, SMul, SPow, cse

@testset "InstructionSequence from compile_to_instructions" begin
    # f(x₁, x₂) = x₁ + x₂
    exprs = SExpr[SAdd([SVar(1), SVar(2)])]
    replacements, reduced = cse(exprs)
    seq = compile_to_instructions(
        replacements, reduced;
        nvars = 2, nparams = 0, output_dim = 1, npolys = 1,
    )
    @test seq.output_dim == 1
    @test seq.tape_space_needed > 0
    @test length(seq.instructions) >= 1
    @test !isempty(seq.u_assignments)
end

@testset "InstructionSequence with constants" begin
    # f(x) = 2*x + 3, where x is var 1
    exprs = SExpr[SAdd([SMul([SConst(2.0 + 0im), SVar(1)]), SConst(3.0 + 0im)])]
    replacements, reduced = cse(exprs)
    seq = compile_to_instructions(
        replacements, reduced;
        nvars = 1, nparams = 0, output_dim = 1, npolys = 1,
    )
    @test seq.output_dim == 1
    @test length(seq.constants) >= 1
    @test seq.all_u_assigned
end

@testset "InstructionSequence tape layout with params" begin
    # f(x; p) = x + p, where x is var 1, p is param 1
    exprs = SExpr[SAdd([SVar(1), SConst(1.0 + 0im)])]
    replacements, reduced = cse(exprs)
    seq = compile_to_instructions(
        replacements, reduced;
        nvars = 1, nparams = 1, output_dim = 1, npolys = 1,
    )
    @test length(seq.parameters_range) == 1
    @test length(seq.variables_range) == 1
    @test first(seq.parameters_range) < first(seq.variables_range)
end

@testset "InstructionSequence all_u/U_assigned flags" begin
    # 2-output: f₁ = x₁, f₂ = x₂
    exprs = SExpr[SVar(1), SVar(2)]
    replacements, reduced = cse(exprs)
    seq = compile_to_instructions(
        replacements, reduced;
        nvars = 2, nparams = 0, output_dim = 2, npolys = 2,
    )
    @test seq.output_dim == 2
    @test seq.all_u_assigned
    @test !seq.all_U_assigned
    @test length(seq.u_assignments) == 2
end
