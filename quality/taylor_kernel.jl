using Test
using HomotopyContinuation:
    TruncatedTaylorSeries,
    taylor_op_mul,
    taylor_op_div,
    taylor_op_sqrt,
    taylor_op_sin,
    taylor_op_cos,
    taylor_op_exp,
    taylor_op_log,
    taylor_op_pow,
    taylor_op_pow_int

function cauchy_coefficients(f, series; K::Int = 3, M::Int = 96, r::Float64 = 0.12)
    acc = zeros(ComplexF64, K + 1)
    for j in 0:(M - 1)
        λ = r * cis(2π * j / M)
        values = map(series) do s
            sum(s[k] * λ^k for k in 0:(length(s) - 1))
        end
        y = f(values...)
        for k in 0:K
            acc[k + 1] += y * cis(-2π * j * k / M)
        end
    end
    return [acc[k + 1] / (M * r^k) for k in 0:K]
end

@testset "Taylor kernel mathematical oracle" begin
    a = TruncatedTaylorSeries((1.3 + 0.4im, 0.3 - 0.2im, -0.15 + 0.25im, 0.05 + 0.1im))
    b = TruncatedTaylorSeries((-0.7 + 0.9im, 0.2 + 0.1im, 0.3 - 0.05im, -0.2 + 0.15im))
    exponent = TruncatedTaylorSeries((1.5 + 0im, 0im, 0im, 0im))

    cases = [
        ("product", taylor_op_mul, (x, y) -> x * y, (a, b)),
        ("quotient", taylor_op_div, (x, y) -> x / y, (a, b)),
        ("sqrt", taylor_op_sqrt, sqrt, (a,)),
        ("sin", taylor_op_sin, sin, (a,)),
        ("cos", taylor_op_cos, cos, (a,)),
        ("exp", taylor_op_exp, exp, (a,)),
        ("log", taylor_op_log, log, (a,)),
        ("fractional power", x -> taylor_op_pow(x, exponent), x -> x^1.5, (a,)),
    ]

    @testset "$name" for (name, taylor_op, scalar_op, args) in cases
        got = taylor_op(args...)
        truth = cauchy_coefficients(scalar_op, collect(args))
        for k in 0:3
            @test got[k] ≈ truth[k + 1] atol = 1.0e-9
        end
    end

    # The important singular algebraic edge case: a zero constant term must not
    # enter the logarithmic-derivative recurrence used for nonzero constants.
    t = TruncatedTaylorSeries((0.0 + 0im, 1.0 + 0im, 0.0 + 0im, 0.0 + 0im))
    cubic = taylor_op_pow_int(t, 3)
    @test all(isfinite, (cubic[0], cubic[1], cubic[2], cubic[3]))
    @test cubic[0] ≈ 0 atol = 1.0e-14
    @test cubic[1] ≈ 0 atol = 1.0e-14
    @test cubic[2] ≈ 0 atol = 1.0e-14
    @test cubic[3] ≈ 1 atol = 1.0e-14
end
