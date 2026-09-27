using Test
import HomotopyContinuation as HC
using HomotopyContinuation:
    TruncatedTaylorSeries, OpType, op_call,
    taylor_op_add, taylor_op_sub, taylor_op_mul, taylor_op_div,
    taylor_op_neg, taylor_op_identity, taylor_op_inv, taylor_op_inv_not_zero,
    taylor_op_sqr, taylor_op_cb, taylor_op_sqrt, taylor_op_invsqr,
    taylor_op_sin, taylor_op_cos, taylor_op_pow_int,
    taylor_op_muladd, taylor_op_mulsub, taylor_op_submul,
    taylor_op_add3, taylor_op_add4, taylor_op_mul3, taylor_op_mul4,
    taylor_op_mulmuladd, taylor_op_mulmulsub,
    taylor_op_exp, taylor_op_log, taylor_op_sinh, taylor_op_cosh, taylor_op_tan,
    taylor_op_tanh, taylor_op_asin, taylor_op_acos, taylor_op_pow

include("cauchy_oracle.jl")

series_at(s::TruncatedTaylorSeries, λ::ComplexF64) = sum(s[k] * λ^k for k in 0:(length(s) - 1))

@testset "Taylor kernel mathematical oracle" begin
    K = 3
    a = TruncatedTaylorSeries((1.3 + 0.4im, 0.3 - 0.2im, -0.15 + 0.25im, 0.05 + 0.1im))
    b = TruncatedTaylorSeries((-0.7 + 0.9im, 0.2 + 0.1im, 0.3 - 0.05im, -0.2 + 0.15im))
    c = TruncatedTaylorSeries((0.5 - 1.1im, -0.25 + 0.4im, 0.1 + 0.2im, 0.3 - 0.1im))
    d = TruncatedTaylorSeries((1.1 + 0.2im, 0.15 + 0.35im, -0.2 - 0.1im, 0.05 - 0.25im))
    e = TruncatedTaylorSeries((0.3 + 0.2im, 0.1 - 0.05im, 0.02 + 0.03im, -0.01 + 0.02im))
    r15 = TruncatedTaylorSeries((1.5 + 0.0im, 0.0im, 0.0im, 0.0im))
    rm43 = TruncatedTaylorSeries((-4 / 3 + 0.0im, 0.0im, 0.0im, 0.0im))

    cases = [
        ("identity", taylor_op_identity, x -> x, (a,)),
        ("neg", taylor_op_neg, x -> -x, (a,)),
        ("inv", taylor_op_inv, inv, (a,)),
        ("inv_not_zero", taylor_op_inv_not_zero, inv, (a,)),
        ("sqr", taylor_op_sqr, x -> x^2, (a,)),
        ("cb", taylor_op_cb, x -> x^3, (a,)),
        ("sqrt", taylor_op_sqrt, sqrt, (a,)),
        ("invsqr", taylor_op_invsqr, x -> inv(x^2), (a,)),
        ("sin", taylor_op_sin, sin, (a,)),
        ("cos", taylor_op_cos, cos, (a,)),
        ("pow_int(5)", x -> taylor_op_pow_int(x, 5), x -> x^5, (a,)),
        ("pow_int(-3)", x -> taylor_op_pow_int(x, -3), x -> x^-3, (a,)),
        ("add", taylor_op_add, +, (a, b)),
        ("sub", taylor_op_sub, -, (a, b)),
        ("mul", taylor_op_mul, *, (a, b)),
        ("div", taylor_op_div, /, (a, b)),
        ("muladd", taylor_op_muladd, (x, y, z) -> x * y + z, (a, b, c)),
        ("mulsub", taylor_op_mulsub, (x, y, z) -> x * y - z, (a, b, c)),
        ("submul", taylor_op_submul, (x, y, z) -> z - x * y, (a, b, c)),
        ("add3", taylor_op_add3, (x, y, z) -> x + y + z, (a, b, c)),
        ("mul3", taylor_op_mul3, (x, y, z) -> x * y * z, (a, b, c)),
        ("add4", taylor_op_add4, (x, y, z, w) -> x + y + z + w, (a, b, c, d)),
        ("mul4", taylor_op_mul4, (x, y, z, w) -> x * y * z * w, (a, b, c, d)),
        ("mulmuladd", taylor_op_mulmuladd, (x, y, z, w) -> x * y + z * w, (a, b, c, d)),
        ("mulmulsub", taylor_op_mulmulsub, (x, y, z, w) -> x * y - z * w, (a, b, c, d)),
        ("exp", taylor_op_exp, exp, (a,)),
        ("log", taylor_op_log, log, (a,)),
        ("sinh", taylor_op_sinh, sinh, (a,)),
        ("cosh", taylor_op_cosh, cosh, (a,)),
        ("tan", taylor_op_tan, tan, (a,)),
        ("tanh", taylor_op_tanh, tanh, (a,)),
        ("asin", taylor_op_asin, asin, (e,)),
        ("acos", taylor_op_acos, acos, (e,)),
        ("pow(1.5)", x -> taylor_op_pow(x, r15), x -> x^1.5, (a,)),
        ("pow(-4/3)", x -> taylor_op_pow(x, rm43), x -> x^(-4 / 3), (a,)),
    ]

    @testset "$name" for (name, taylor_op, scalar_op, args) in cases
        got = taylor_op(args...)
        truth = cauchy_coefficients(λ -> scalar_op(map(s -> series_at(s, λ), args)...), K; r = 0.15)
        for k in 0:K
            @test got[k] ≈ truth[k + 1] atol = 1.0e-9
        end
    end

    case_names = Set(case[1] for case in cases)
    is_covered(op_name) = any(c -> c == op_name || startswith(c, op_name * "("), case_names)

    @testset "every taylor_op_* rule is covered" begin
        defined = [
            String(name)[(length("taylor_op_") + 1):end]
                for name in names(HC; all = true) if startswith(String(name), "taylor_op_")
        ]
        @test length(defined) >= 30
        for op_name in defined
            @test is_covered(op_name)
        end
        for case_name in case_names
            @test split(case_name, '(')[1] in defined
        end
    end

    @testset "every OpType is covered" begin
        for op in instances(OpType.T)
            op == OpType.OP_STOP && continue
            @test is_covered(replace(String(op_call(op)), "op_" => ""))
        end
    end

    @testset "inv_not_zero passes a zero constant term through" begin
        z = TruncatedTaylorSeries((0.0 + 0.0im, 1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im))
        @test taylor_op_inv_not_zero(z) === z
    end

    @testset "truncation order N is honored" begin
        a1 = TruncatedTaylorSeries((1.3 + 0.4im, 0.3 - 0.2im))
        b1 = TruncatedTaylorSeries((-0.7 + 0.9im, 0.2 + 0.1im))
        product = taylor_op_mul(a1, b1)
        @test length(product) == 2
        @test product[0] ≈ a1[0] * b1[0]
        @test product[1] ≈ a1[0] * b1[1] + a1[1] * b1[0]
    end
end

@testset "taylor_op_pow_int with a vanishing constant term" begin
    t = TruncatedTaylorSeries((0.0 + 0im, 1.0 + 0im, 0.0 + 0im, 0.0 + 0im))
    @testset "t^$r" for r in 1:5
        w = taylor_op_pow_int(t, r)
        for k in 0:3
            @test isfinite(w[k])
            @test w[k] ≈ (k == r ? 1.0 + 0im : 0.0 + 0im) atol = 1.0e-14
        end
    end

    b = TruncatedTaylorSeries((0.0 + 0im, 2.0 + 0im, -3.0 + 0im, 0.0 + 0im))
    w = taylor_op_pow_int(b, 3)
    @test all(isfinite, (w[0], w[1], w[2], w[3]))
    @test w[0] ≈ 0.0 + 0im atol = 1.0e-14
    @test w[1] ≈ 0.0 + 0im atol = 1.0e-14
    @test w[2] ≈ 0.0 + 0im atol = 1.0e-14
    @test w[3] ≈ 8.0 + 0im atol = 1.0e-13
end
