using Test
import HomotopyContinuationNext as Next
using HomotopyContinuationNext: OpType, arity, op_call,
    op_add, op_sub, op_mul, op_div, op_neg,
    op_sqr, op_cb, op_sqrt, op_inv, op_invsqr,
    op_muladd, op_mulsub, op_submul,
    op_add3, op_add4, op_mul3, op_mul4,
    op_mulmuladd, op_mulmulsub,
    op_pow_int, op_identity, op_inv_not_zero,
    op_exp, op_tan, op_asin, op_acos, op_sinh, op_cosh, op_tanh, op_pow,
    Expression, TruncatedTaylorSeries, @var, expand, subs, differentiate

@testset "Operations" begin
    @testset "arity" begin
        @test arity(OpType.OP_STOP) == 0
        @test arity(OpType.OP_NEG) == 1
        @test arity(OpType.OP_ADD) == 2
        @test arity(OpType.OP_MULADD) == 3
        @test arity(OpType.OP_MULMULADD) == 4
        @test arity(OpType.OP_TANH) == 1
        @test arity(OpType.OP_POW) == 2
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
        @test op_exp(a) ≈ exp(a)
        @test op_tan(a) ≈ tan(a)
        @test op_asin(a) ≈ asin(a)
        @test op_acos(a) ≈ acos(a)
        @test op_sinh(a) ≈ sinh(a)
        @test op_cosh(a) ≈ cosh(a)
        @test op_tanh(a) ≈ tanh(a)
        @test op_pow(a, 1.5 + 0.0im) ≈ a^1.5
        @test op_pow(a, b) ≈ a^b
    end

    # Every Taylor rule against symbolic differentiation of its scalar op: with
    # `Expression` coefficients the series composition can be differentiated in ε,
    # which pins each recurrence instead of sampling it.
    @testset "Taylor rules against symbolic differentiation" begin
        @var x[0:3] y[0:3] z[0:3] w[0:3] ε

        series(t::TruncatedTaylorSeries{N}) where {N} =
            sum(t.val[k + 1] * ε^k for k in 0:(N - 1))
        function nth_term(e::Expression, k::Int)
            d = e
            for _ in 1:k
                d = differentiate(d, ε)
            end
            return expand(subs(d, ε => Expression(0)) / Expression(factorial(k)))
        end

        @testset "N=$N" for N in (1, 2, 4)
            tx = TruncatedTaylorSeries(tuple(x[1:N]...))
            ty = TruncatedTaylorSeries(tuple(y[1:N]...))
            tz = TruncatedTaylorSeries(tuple(z[1:N]...))
            tw = TruncatedTaylorSeries(tuple(w[1:N]...))
            @testset "$op" for op in instances(OpType.T)
                op === OpType.OP_STOP && continue
                scalar_op = getfield(Next, op_call(op))
                taylor_op = getfield(Next, Symbol(:taylor_, op_call(op)))
                got, scalar = if arity(op) == 1
                    taylor_op(tx), scalar_op(series(tx))
                elseif arity(op) == 2
                    if op === OpType.OP_POW_INT
                        taylor_op(tx, 5), series(tx)^5
                    elseif op === OpType.OP_POW
                        # The exponent reaches the tape as a constant slot.
                        r = TruncatedTaylorSeries(
                            tuple(Expression(1.5), (Expression(0) for _ in 2:N)...),
                        )
                        taylor_op(tx, r), scalar_op(series(tx), Expression(1.5))
                    else
                        taylor_op(tx, ty), scalar_op(series(tx), series(ty))
                    end
                elseif arity(op) == 3
                    taylor_op(tx, ty, tz), scalar_op(series(tx), series(ty), series(tz))
                else
                    taylor_op(tx, ty, tz, tw),
                        scalar_op(series(tx), series(ty), series(tz), series(tw))
                end
                for k in 0:(N - 1)
                    @test expand(got.val[k + 1] - nth_term(scalar, k)) == zero(Expression)
                end
            end
        end
    end
end
