using Test
using HomotopyContinuationNext: OpType, arity,
    op_add, op_sub, op_mul, op_div, op_neg,
    op_sqr, op_cb, op_sqrt, op_inv, op_invsqr,
    op_muladd, op_mulsub, op_submul,
    op_add3, op_add4, op_mul3, op_mul4,
    op_mulmuladd, op_mulmulsub,
    op_pow_int, op_identity, op_inv_not_zero

@testset "Operations" begin
    @testset "arity" begin
        @test arity(OpType.OP_STOP) == 0
        @test arity(OpType.OP_NEG) == 1
        @test arity(OpType.OP_ADD) == 2
        @test arity(OpType.OP_MULADD) == 3
        @test arity(OpType.OP_MULMULADD) == 4
    end

    @testset "scalar ops (ComplexF64)" begin
        a, b, c, d = 2.0 + 1.0im, 3.0 - 1.0im, 1.0 + 2.0im, 0.5 + 0.5im
        @test op_add(a, b) ≈ a + b
        @test op_sub(a, b) ≈ a - b
        @test op_mul(a, b) ≈ a * b
        @test op_div(a, b) ≈ a / b
        @test op_neg(a) ≈ -a
        @test op_sqr(a) ≈ a^2
        @test op_cb(a) ≈ a^3
        @test op_sqrt(a) ≈ sqrt(a)
        @test op_inv(a) ≈ 1 / a
        @test op_invsqr(a) ≈ 1 / a^2
        @test op_identity(a) == a
        @test op_inv_not_zero(a) ≈ 1 / a
        @test op_inv_not_zero(0.0 + 0.0im) == 0.0 + 0.0im
        @test op_pow_int(a, 3) ≈ a^3
        @test op_pow_int(a, 0) ≈ 1.0
        @test op_muladd(a, b, c) ≈ a * b + c
        @test op_mulsub(a, b, c) ≈ a * b - c
        @test op_submul(a, b, c) ≈ c - a * b
        @test op_add3(a, b, c) ≈ a + b + c
        @test op_mul3(a, b, c) ≈ a * b * c
        @test op_add4(a, b, c, d) ≈ a + b + c + d
        @test op_mul4(a, b, c, d) ≈ a * b * c * d
        @test op_mulmuladd(a, b, c, d) ≈ a * b + c * d
        @test op_mulmulsub(a, b, c, d) ≈ a * b - c * d
    end
end
