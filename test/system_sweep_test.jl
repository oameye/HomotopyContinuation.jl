using Test, Random
using HomotopyContinuation
using HomotopyContinuation: TaylorVector, ComplexDF64
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: MultivariatePolynomials as MP
using FixedSizeArrays: FixedSizeVectorDefault, FixedSizeMatrixDefault

include("test_systems.jl")
include("minors_polys.jl")
include("cauchy_oracle.jl")

const MODES = (CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL)

# The mutating entry points take the fixed-size buffers the package tracks with.
@static if VERSION < v"1.11"
    buffer(v::AbstractVector{T}) where {T} = Vector{T}(v)
    buffer(M::AbstractMatrix{T}) where {T} = Matrix{T}(M)
else
    buffer(v::AbstractVector{T}) where {T} = FixedSizeVectorDefault{T}(v)
    buffer(M::AbstractMatrix{T}) where {T} = FixedSizeMatrixDefault{T}(M)
end

function taylor_vector(X::AbstractMatrix{ComplexF64})
    tv = TaylorVector{size(X, 1), ComplexF64}(size(X, 2))
    for i in axes(X, 2)
        tv[i] = Tuple(X[:, i])
    end
    return tv
end

_at(f, vars, x, params, p) = isempty(params) ? f(vars => x) : f(vars => x, params => p)

mp_eval(polys, vars, x, params, p) =
    ComplexF64[_at(f, vars, x, params, p) for f in polys]

function mp_jacobian(polys, vars, x, params, p)
    J = zeros(ComplexF64, length(polys), length(vars))
    for j in eachindex(vars), i in eachindex(polys)
        J[i, j] = _at(MP.differentiate(polys[i], vars[j]), vars, x, params, p)
    end
    return J
end

# Series variable of the exact Taylor ground truth.
@polyvar λ

# Order-K coefficient of λ ↦ F(x(λ); p + λ·dp) with x(λ) = Σₖ X[k+1, :] λᵏ.
function mp_taylor(polys, vars, X, params, p, dp, K::Int)
    xs = [sum(X[k + 1, i] * λ^k for k in 0:(size(X, 1) - 1)) for i in eachindex(vars)]
    ps = [p[i] + dp[i] * λ for i in eachindex(params)]
    return [ComplexF64(MP.coefficient(_at(f, vars, xs, params, ps), λ^K)) for f in polys]
end

_public_evaluate(F, x, p) = isempty(p) ? evaluate(F, x) : evaluate(F, x, p)
_public_jacobian(F, x, p) = isempty(p) ? jacobian(F, x) : jacobian(F, x, p)

# Start and target parameters placing p(t) = t·p₁ + (1 - t)·p₀ at `p` with p₁ - p₀ = dp.
endpoints(p, dp, t) = (p .+ (1 - t) .* dp, p .- t .* dp)

const TEST_SYSTEMS = [(name, build()...) for (name, build) in TEST_SYSTEM_COLLECTION]

# The mutating evaluation of a `System` is `ParameterHomotopy(F, p₁, p₀)` at a
# path parameter `t`. Its order-1 Taylor coefficient holds x fixed, and orders 2
# and 3 expand λ ↦ F(x(λ); p(t + λ)), so a system with parameters exercises
# Taylor-valued parameters in every order.
@testset "System sweep: $name" for (name, polys, vars, params) in TEST_SYSTEMS
    rng = MersenneTwister(0x00051ee7 + length(name))
    m, n, r = length(polys), length(vars), length(params)
    xvals = randn(rng, ComplexF64, n)
    pvals = randn(rng, ComplexF64, r)
    dp = randn(rng, ComplexF64, r)
    X = randn(rng, ComplexF64, 4, n)
    X[1, :] = xvals
    t = 0.37 + 0.21im
    p1, p0 = endpoints(pvals, dp, t)

    truth_u = mp_eval(polys, vars, xvals, params, pvals)
    truth_J = mp_jacobian(polys, vars, xvals, params, pvals)
    truth_taylor = [mp_taylor(polys, vars, X[1:(K + 1), :], params, pvals, dp, K) for K in 1:3]

    @testset "$mode" for mode in MODES
        F = System(polys; variables = vars, parameters = params, compile = mode)
        @test size(F) == (m, n)
        @test nvariables(F) == n
        @test nparameters(F) == r
        @test collect(variables(F)) == collect(vars)
        @test collect(parameters(F)) == collect(params)
        scales = equation_scales(F)
        @test _public_evaluate(F, xvals, pvals) .* scales ≈ truth_u rtol = 1.0e-10
        @test _public_jacobian(F, xvals, pvals) .* scales ≈ truth_J rtol = 1.0e-10

        H = ParameterHomotopy(F, p1, p0)
        @test size(H) == (m, n)
        u = buffer(zeros(ComplexF64, m))
        U = buffer(zeros(ComplexF64, m, n))

        evaluate!(u, H, buffer(xvals), t)
        @test u .* scales ≈ truth_u rtol = 1.0e-10

        fill!(u, 0)
        evaluate_and_jacobian!(u, U, H, buffer(xvals), t)
        @test u .* scales ≈ truth_u rtol = 1.0e-10
        @test U .* scales ≈ truth_J rtol = 1.0e-10

        fill!(u, 0)
        evaluate!(u, H, buffer(ComplexDF64.(xvals)), t)
        @test u .* scales ≈ truth_u rtol = 1.0e-10

        fill!(u, 0)
        taylor!(u, Val(1), H, buffer(xvals), t)
        @test u .* scales ≈ mp_taylor(polys, vars, X[1:1, :], params, pvals, dp, 1) rtol = 1.0e-9

        @testset "taylor! K=$K" for K in 2:3
            fill!(u, 0)
            taylor!(u, Val(K), H, taylor_vector(X[1:(K + 1), :]), t)
            @test u .* scales ≈ truth_taylor[K] rtol = 1.0e-9
        end
    end
end

# Real, order-one points keep every entry clear of its poles and branch cuts, and
# `ref` is the plain-arithmetic reference each collection entry carries.
@testset "Non-polynomial Taylor sweep: $name" for (name, exprs, vars, params, ref) in
    NONPOLYNOMIAL_SYSTEM_COLLECTION

    rng = MersenneTwister(0x00e8b1a5 + length(name))
    m, n, r = length(exprs), length(vars), length(params)
    X = vcat(ComplexF64.(0.7 .+ rand(rng, 1, n)), 0.15 .* randn(rng, ComplexF64, 3, n))
    pvals = ComplexF64.(0.7 .+ rand(rng, r))
    dp = 0.15 .* randn(rng, ComplexF64, r)
    t = 0.37 + 0.21im
    p1, p0 = endpoints(pvals, dp, t)
    x0 = X[1, :]
    x_at(λ) = [sum(X[k + 1, i] * λ^k for k in 0:3) for i in 1:n]

    fixed_x = cauchy_coefficients(λ -> ref(x0, pvals .+ λ .* dp), 1; M = 128, r = 0.05)
    coupled = cauchy_coefficients(λ -> ref(x_at(λ), pvals .+ λ .* dp), 3; M = 128, r = 0.05)

    @testset "$mode" for mode in MODES
        F = System(exprs; variables = vars, parameters = params, compile = mode)
        scales = equation_scales(F)
        @test _public_evaluate(F, x0, pvals) .* scales ≈ ref(x0, pvals) rtol = 1.0e-10

        H = ParameterHomotopy(F, p1, p0)
        u = buffer(zeros(ComplexF64, m))
        evaluate!(u, H, buffer(x0), t)
        @test u .* scales ≈ ref(x0, pvals) rtol = 1.0e-10

        taylor!(u, Val(1), H, buffer(x0), t)
        @test u .* scales ≈ fixed_x[2] atol = 1.0e-7

        @testset "taylor! K=$K" for K in 2:3
            fill!(u, 0)
            taylor!(u, Val(K), H, taylor_vector(X[1:(K + 1), :]), t)
            @test u .* scales ≈ coupled[K + 1] atol = 1.0e-7
        end
    end
end
