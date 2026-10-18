using Test
using HomotopyContinuation
using HomotopyContinuation: TaylorVector

include("cauchy_oracle.jl")

@static if VERSION < v"1.11"
    const Vec{T} = Vector{T}
else
    using FixedSizeArrays: FixedSizeVectorDefault
    const Vec{T} = FixedSizeVectorDefault{T}
end

# `ParameterHomotopy(F, p₁, p₀)` evaluates `F(x(s); p(t) + (p₁ - p₀) s)`: order 1
# takes a point `x` and differentiates along the parameter only, orders 2 and 3 take
# a `TaylorVector` series for `x`. The oracle is the Cauchy integral of the same
# scalar function along the same input series.
function homotopy_taylor(H, K::Int, series::Vector{<:NTuple{4}}, t::ComplexF64)
    u = Vec{ComplexF64}(zeros(ComplexF64, size(H, 1)))
    if K == 1
        taylor!(u, Val(1), H, Vec{ComplexF64}([s[1] for s in series]), t)
    else
        tx = TaylorVector{K + 1, ComplexF64}(length(series))
        for (i, s) in enumerate(series)
            tx[i] = s[1:(K + 1)]
        end
        taylor!(u, Val(K), H, tx, t)
    end
    return collect(u)
end

truncated_series(s::NTuple{4}, K::Int, λ) = sum(s[k + 1] * λ^k for k in 0:K)

function oracle_coefficient(f, K::Int, series, a_t, da; r)
    Kx = K == 1 ? 0 : K
    g(λ) = f([truncated_series(s, Kx, λ) for s in series], a_t + da * λ)
    return cauchy_coefficients(g, K; r = r)[K + 1]
end

# Each case is one equation in the operands X = x + a, Y = y - 2a, Z = z + a/2,
# W = w + a/3 and E = e + a/5, so that every operand carries a first-order term
# along the parameter and a full series along the variables.
const CASES = [
    ("add", (X, Y, Z, W, E) -> X + Y),
    ("sub", (X, Y, Z, W, E) -> X - Y),
    ("neg", (X, Y, Z, W, E) -> -X),
    ("mul", (X, Y, Z, W, E) -> X * Y),
    ("div", (X, Y, Z, W, E) -> X / Y),
    ("inv", (X, Y, Z, W, E) -> inv(X)),
    ("sqr", (X, Y, Z, W, E) -> X^2),
    ("cube", (X, Y, Z, W, E) -> X^3),
    ("pow 5", (X, Y, Z, W, E) -> X^5),
    ("pow -2", (X, Y, Z, W, E) -> X^-2),
    ("pow -3", (X, Y, Z, W, E) -> X^-3),
    ("pow 1.5", (X, Y, Z, W, E) -> X^1.5),
    ("pow -4/3", (X, Y, Z, W, E) -> X^(-4 / 3)),
    ("sqrt", (X, Y, Z, W, E) -> sqrt(X)),
    ("exp", (X, Y, Z, W, E) -> exp(X)),
    ("log", (X, Y, Z, W, E) -> log(X)),
    ("sin", (X, Y, Z, W, E) -> sin(X)),
    ("cos", (X, Y, Z, W, E) -> cos(X)),
    ("tan", (X, Y, Z, W, E) -> tan(X)),
    ("asin", (X, Y, Z, W, E) -> asin(E)),
    ("acos", (X, Y, Z, W, E) -> acos(E)),
    ("sinh", (X, Y, Z, W, E) -> sinh(X)),
    ("cosh", (X, Y, Z, W, E) -> cosh(X)),
    ("tanh", (X, Y, Z, W, E) -> tanh(X)),
    ("muladd", (X, Y, Z, W, E) -> X * Y + Z),
    ("mulsub", (X, Y, Z, W, E) -> X * Y - Z),
    ("submul", (X, Y, Z, W, E) -> Z - X * Y),
    ("add3", (X, Y, Z, W, E) -> X + Y + Z),
    ("mul3", (X, Y, Z, W, E) -> X * Y * Z),
    ("add4", (X, Y, Z, W, E) -> X + Y + Z + W),
    ("mul4", (X, Y, Z, W, E) -> X * Y * Z * W),
    ("mulmuladd", (X, Y, Z, W, E) -> X * Y + Z * W),
    ("mulmulsub", (X, Y, Z, W, E) -> X * Y - Z * W),
    ("sin(xy)/exp(z)", (X, Y, Z, W, E) -> sin(X * Y) / exp(Z)),
    ("sqrt(x^2+y)", (X, Y, Z, W, E) -> sqrt(X^2 + Y)),
    ("log(1+xy)cos(z)", (X, Y, Z, W, E) -> log(1 + X * Y) * cos(Z)),
    ("tanh(x)^2 w/y^3", (X, Y, Z, W, E) -> tanh(X)^2 * W / Y^3),
    ("x^1.5 asin(e) - z", (X, Y, Z, W, E) -> X^1.5 * asin(E) - Z),
]

operands(x, y, z, w, e, a) = (x + a, y - 2a, z + a / 2, w + a / 3, e + a / 5)

@testset "taylor! against the Cauchy integral, $mode" for mode in
    (CompileMode.INTERPRETED, CompileMode.COMPILED)
    @var x y z w e a
    F = System(
        [f(operands(x, y, z, w, e, a)...) for (_, f) in CASES];
        variables = [x, y, z, w, e], parameters = [a], compile = mode,
    )
    scales = equation_scales(F)
    p₁, p₀, t = 0.3 + 0.1im, -0.1 + 0.05im, 0.4 + 0.1im
    H = ParameterHomotopy(F, [p₁], [p₀])
    a_t, da = t * p₁ + (1 - t) * p₀, p₁ - p₀
    series = [
        (1.3 + 0.4im, 0.3 - 0.2im, -0.15 + 0.25im, 0.05 + 0.1im),
        (-0.7 + 0.9im, 0.2 + 0.1im, 0.3 - 0.05im, -0.2 + 0.15im),
        (0.5 - 1.1im, -0.25 + 0.4im, 0.1 + 0.2im, 0.3 - 0.1im),
        (1.1 + 0.2im, 0.15 + 0.35im, -0.2 - 0.1im, 0.05 - 0.25im),
        (0.3 + 0.2im, 0.1 - 0.05im, 0.02 + 0.03im, -0.01 + 0.02im),
    ]
    scalar(v, α) = [f(operands(v..., α)...) for (_, f) in CASES]
    for K in 1:3
        got = homotopy_taylor(H, K, series, t) .* scales
        truth = oracle_coefficient(scalar, K, series, a_t, da; r = 0.15)
        @testset "$(name), order $K" for (i, (name, _)) in enumerate(CASES)
            @test got[i] ≈ truth[i] atol = 1.0e-11
        end
    end
end

@testset "every function the Expression frontend accepts is covered" begin
    @var x y z w e a
    printed = [string(f(operands(x, y, z, w, e, a)...)) for (_, f) in CASES]
    covered(pattern) = any(s -> occursin(pattern, s), printed)
    @test covered("+")
    @test covered("-")
    @test covered("*")
    @test covered(r"\^-1\b")
    @test covered(r"\^[2-9]\b")
    @test covered(r"\^-[2-9]\b")
    @test covered(r"\^-?\d+\.\d")
    @test covered(r"\^-\d+\.\d")
    for fname in ("sqrt", "exp", "log", "sin", "cos", "tan", "asin", "acos", "sinh", "cosh", "tanh")
        @test covered(fname * "(")
    end
end

@testset "integer powers of a series with a vanishing constant term" begin
    # With p₁ = 1, p₀ = 0 and t = 0 the parameter series is exactly s, so
    # (x + a)^r at x = 0 is s^r, and (y + 2a)^3 with y = -3s² is (2s - 3s²)^3.
    @var x y a
    F = System(
        [[(x + a)^r for r in 1:5]; (y + 2a)^3];
        variables = [x, y], parameters = [a],
    )
    H = ParameterHomotopy(F, [1.0], [0.0])
    series = [(0.0im, 0.0im, 0.0im, 0.0im), (0.0im, 0.0im, -3.0 + 0.0im, 0.0im)]
    scales = equation_scales(F)
    for K in 1:3
        got = homotopy_taylor(H, K, series, 0.0im) .* scales
        @test all(isfinite, got)
        @testset "s^$r, order $K" for r in 1:5
            @test got[r] ≈ (K == r ? 1.0 : 0.0) atol = 1.0e-14
        end
        @test got[6] ≈ (K == 3 ? 8.0 : 0.0) atol = 1.0e-13
    end
end
