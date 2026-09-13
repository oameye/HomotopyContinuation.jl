using Test
using HomotopyContinuationNext:
    TruncatedTaylorSeries, TaylorVector, vectors,
    taylor_op_add, taylor_op_sub, taylor_op_mul, taylor_op_div,
    taylor_op_neg, taylor_op_identity, taylor_op_inv, taylor_op_inv_not_zero,
    taylor_op_sqr, taylor_op_cb, taylor_op_sqrt, taylor_op_invsqr,
    taylor_op_sin, taylor_op_cos, taylor_op_pow_int,
    taylor_op_muladd, taylor_op_mulsub, taylor_op_submul,
    taylor_op_add3, taylor_op_add4, taylor_op_mul3, taylor_op_mul4,
    taylor_op_mulmuladd, taylor_op_mulmulsub,
    taylor_op_exp, taylor_op_sinh, taylor_op_cosh, taylor_op_tan, taylor_op_tanh,
    taylor_op_asin, taylor_op_acos, taylor_op_pow, FSMat
using HomotopyContinuationNext: OpType, op_call

@testset "TruncatedTaylorSeries" begin
    @testset "construction" begin
        t = TruncatedTaylorSeries((1.0 + 0im, 2.0 + 0im, 3.0 + 0im))
        @test t[0] == 1.0 + 0im
        @test length(t) == 3
    end

    @testset "from scalar (zero-padded)" begin
        t = TruncatedTaylorSeries{3, ComplexF64}(2.0 + 0im)
        @test t[0] == 2.0 + 0im
        @test t[1] == 0.0 + 0im
        @test t[2] == 0.0 + 0im
    end
end

@testset "taylor_op_* correctness" begin
    a = TruncatedTaylorSeries((2.0 + 0im, 1.0 + 0im, 0.5 + 0im))
    b = TruncatedTaylorSeries((3.0 + 0im, -1.0 + 0im, 0.0 + 0im))

    @testset "add" begin
        r = taylor_op_add(a, b)
        @test r[0] ≈ 5.0 + 0im
        @test r[1] ≈ 0.0 + 0im
        @test r[2] ≈ 0.5 + 0im
    end

    @testset "neg" begin
        r = taylor_op_neg(a)
        @test r[0] ≈ -a[0]
        @test r[1] ≈ -a[1]
    end

    @testset "mul (Cauchy product)" begin
        r = taylor_op_mul(a, b)
        @test r[0] ≈ a[0] * b[0]
        @test r[1] ≈ a[0] * b[1] + a[1] * b[0]
        @test r[2] ≈ a[0] * b[2] + a[1] * b[1] + a[2] * b[0]
    end

    @testset "sqr includes odd-order cross term" begin
        c = TruncatedTaylorSeries((2.0 + 0im, 3.0 + 0im, 5.0 + 0im, 7.0 + 0im))
        r = taylor_op_sqr(c)
        @test r[0] ≈ 4.0 + 0im
        @test r[1] ≈ 12.0 + 0im
        @test r[2] ≈ 29.0 + 0im
        @test r[3] ≈ 58.0 + 0im
    end

    @testset "inv: inv(b) * b ≈ (1, 0, 0)" begin
        r = taylor_op_inv(b)
        product = taylor_op_mul(r, b)
        @test product[0] ≈ 1.0 + 0im atol = 1.0e-12
        @test product[1] ≈ 0.0 + 0im atol = 1.0e-12
        @test product[2] ≈ 0.0 + 0im atol = 1.0e-12
    end

    @testset "div: (a/b) * b ≈ a" begin
        r = taylor_op_div(a, b)
        product = taylor_op_mul(r, b)
        for k in 0:2
            @test product[k] ≈ a[k] atol = 1.0e-12
        end
    end

    @testset "sqrt: sqrt(a)^2 ≈ a" begin
        r = taylor_op_sqrt(a)
        product = taylor_op_mul(r, r)
        for k in 0:2
            @test product[k] ≈ a[k] atol = 1.0e-12
        end
    end
end

# Coefficients of λ ↦ f(a(λ), …) at 0 from samples on a circle of radius r.
function cauchy_coefficients(f, series; K::Int, M::Int = 64, r::Float64 = 0.15)
    acc = zeros(ComplexF64, K + 1)
    for j in 0:(M - 1)
        λ = r * cis(2π * j / M)
        vals = map(s -> sum(s[k] * λ^k for k in 0:(length(s) - 1)), series)
        g = f(vals...)
        for k in 0:K
            acc[k + 1] += g * cis(-2π * j * k / M)
        end
    end
    return [acc[k + 1] / (M * r^k) for k in 0:K]
end

@testset "taylor_op_pow_int with a vanishing constant term" begin
    # `a(t) = t` gives `a^r = t^r`: coefficient 1 at order r, 0 below it. The
    # log-derivative recurrence divides by `a[0]`, so this is where it would
    # report NaN.
    a = TruncatedTaylorSeries((0.0 + 0im, 1.0 + 0im, 0.0 + 0im, 0.0 + 0im))
    @testset "r = $r" for r in 1:5
        w = taylor_op_pow_int(a, r)
        for k in 0:3
            @test isfinite(w[k])
            @test w[k] ≈ (k == r ? 1.0 + 0im : 0.0 + 0im) atol = 1.0e-14
        end
    end

    # A general series with a zero constant term: `(2t - 3t²)³ = 8t³ - 36t⁴ + …`,
    # so orders 0..2 vanish and order 3 is 8.
    b = TruncatedTaylorSeries((0.0 + 0im, 2.0 + 0im, -3.0 + 0im, 0.0 + 0im))
    w = taylor_op_pow_int(b, 3)
    @test all(isfinite, (w[0], w[1], w[2], w[3]))
    @test w[0] ≈ 0.0 + 0im atol = 1.0e-14
    @test w[1] ≈ 0.0 + 0im atol = 1.0e-14
    @test w[2] ≈ 0.0 + 0im atol = 1.0e-14
    @test w[3] ≈ 8.0 + 0im atol = 1.0e-13

    # A nonzero constant term still goes through the recurrence unchanged.
    c = TruncatedTaylorSeries((1.5 + 0.5im, 0.3 - 0.2im, 0.1 + 0.4im, -0.2 + 0.1im))
    @test taylor_op_pow_int(c, 4)[3] ≈ taylor_op_mul(taylor_op_sqr(c), taylor_op_sqr(c))[3] atol = 1.0e-10
end

@testset "taylor_op_* against a Cauchy-integral oracle" begin
    K = 3
    a = TruncatedTaylorSeries((1.3 + 0.4im, 0.3 - 0.2im, -0.15 + 0.25im, 0.05 + 0.1im))
    b = TruncatedTaylorSeries((-0.7 + 0.9im, 0.2 + 0.1im, 0.3 - 0.05im, -0.2 + 0.15im))
    c = TruncatedTaylorSeries((0.5 - 1.1im, -0.25 + 0.4im, 0.1 + 0.2im, 0.3 - 0.1im))
    d = TruncatedTaylorSeries((1.1 + 0.2im, 0.15 + 0.35im, -0.2 - 0.1im, 0.05 - 0.25im))
    e = TruncatedTaylorSeries((0.3 + 0.2im, 0.1 - 0.05im, 0.02 + 0.03im, -0.01 + 0.02im))
    # OP_POW takes its exponent from a tape slot, so it arrives as a constant series.
    r15 = TruncatedTaylorSeries((1.5 + 0.0im, 0.0im, 0.0im, 0.0im))
    rm43 = TruncatedTaylorSeries((-4 / 3 + 0.0im, 0.0im, 0.0im, 0.0im))

    cases = [
        ("identity", taylor_op_identity, x -> x, (a,)),
        ("neg", taylor_op_neg, x -> -x, (a,)),
        ("inv", taylor_op_inv, x -> inv(x), (a,)),
        ("inv_not_zero", taylor_op_inv_not_zero, x -> inv(x), (a,)),
        ("sqr", taylor_op_sqr, x -> x^2, (a,)),
        ("cb", taylor_op_cb, x -> x^3, (a,)),
        ("sqrt", taylor_op_sqrt, x -> sqrt(x), (a,)),
        ("invsqr", taylor_op_invsqr, x -> inv(x^2), (a,)),
        ("sin", taylor_op_sin, x -> sin(x), (a,)),
        ("cos", taylor_op_cos, x -> cos(x), (a,)),
        ("pow_int(5)", x -> taylor_op_pow_int(x, 5), x -> x^5, (a,)),
        ("pow_int(-3)", x -> taylor_op_pow_int(x, -3), x -> x^-3, (a,)),
        ("add", taylor_op_add, (x, y) -> x + y, (a, b)),
        ("sub", taylor_op_sub, (x, y) -> x - y, (a, b)),
        ("mul", taylor_op_mul, (x, y) -> x * y, (a, b)),
        ("div", taylor_op_div, (x, y) -> x / y, (a, b)),
        ("muladd", taylor_op_muladd, (x, y, z) -> x * y + z, (a, b, c)),
        ("mulsub", taylor_op_mulsub, (x, y, z) -> x * y - z, (a, b, c)),
        ("submul", taylor_op_submul, (x, y, z) -> z - x * y, (a, b, c)),
        ("add3", taylor_op_add3, (x, y, z) -> x + y + z, (a, b, c)),
        ("mul3", taylor_op_mul3, (x, y, z) -> x * y * z, (a, b, c)),
        ("add4", taylor_op_add4, (x, y, z, w) -> x + y + z + w, (a, b, c, d)),
        ("mul4", taylor_op_mul4, (x, y, z, w) -> x * y * z * w, (a, b, c, d)),
        ("mulmuladd", taylor_op_mulmuladd, (x, y, z, w) -> x * y + z * w, (a, b, c, d)),
        ("mulmulsub", taylor_op_mulmulsub, (x, y, z, w) -> x * y - z * w, (a, b, c, d)),
        ("exp", taylor_op_exp, x -> exp(x), (a,)),
        ("sinh", taylor_op_sinh, x -> sinh(x), (a,)),
        ("cosh", taylor_op_cosh, x -> cosh(x), (a,)),
        ("tan", taylor_op_tan, x -> tan(x), (a,)),
        ("tanh", taylor_op_tanh, x -> tanh(x), (a,)),
        # `asin`/`acos` need a constant term inside the unit disc, away from ±1.
        ("asin", taylor_op_asin, x -> asin(x), (e,)),
        ("acos", taylor_op_acos, x -> acos(x), (e,)),
        ("pow(1.5)", x -> taylor_op_pow(x, r15), x -> x^1.5, (a,)),
        ("pow(-4/3)", x -> taylor_op_pow(x, rm43), x -> x^(-4 / 3), (a,)),
    ]

    @testset "$name" for (name, taylor_op, scalar_op, args) in cases
        got = taylor_op(args...)
        truth = cauchy_coefficients(scalar_op, collect(args); K = K)
        for k in 0:K
            @test got[k] ≈ truth[k + 1] atol = 1.0e-9
        end
    end

    @testset "every OpType is covered" begin
        covered = Set(first.(cases))
        for op in instances(OpType.T)
            op == OpType.OP_STOP && continue
            name = replace(String(op_call(op)), "op_" => "")
            # `pow_int` carries its exponent in the case name.
            @test any(c -> c == name || startswith(c, name * "("), covered)
        end
    end

    @testset "inv_not_zero passes a zero constant term through" begin
        z = TruncatedTaylorSeries((0.0 + 0.0im, 1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im))
        @test taylor_op_inv_not_zero(z) === z
    end

    @testset "truncation order N is honored" begin
        a1 = TruncatedTaylorSeries((1.3 + 0.4im, 0.3 - 0.2im))
        b1 = TruncatedTaylorSeries((-0.7 + 0.9im, 0.2 + 0.1im))
        r = taylor_op_mul(a1, b1)
        @test length(r) == 2
        @test r[0] ≈ a1[0] * b1[0]
        @test r[1] ≈ a1[0] * b1[1] + a1[1] * b1[0]
    end
end

@testset "TaylorVector" begin
    @testset "getindex / setindex!" begin
        tv = TaylorVector{2, ComplexF64}(3)
        tv[1] = TruncatedTaylorSeries((1.0 + 0im, 2.0 + 0im))
        t = tv[1]
        @test t[0] == 1.0 + 0im
        @test t[1] == 2.0 + 0im
    end

    @testset "vectors" begin
        tv = TaylorVector{2, ComplexF64}(3)
        tv[1] = TruncatedTaylorSeries((1.0 + 0im, 10.0 + 0im))
        tv[2] = TruncatedTaylorSeries((2.0 + 0im, 20.0 + 0im))
        tv[3] = TruncatedTaylorSeries((3.0 + 0im, 30.0 + 0im))

        vs = vectors(tv)
        @test length(vs) == 2
        @test vs[1][1] == 1.0 + 0im
        @test vs[2][3] == 30.0 + 0im
    end
end
