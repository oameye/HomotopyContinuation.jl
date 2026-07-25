using Test, Random
using HomotopyContinuationNext: System, CompileMode, HomotopyEvaluator,
    StraightLineHomotopy, evaluate!, evaluate_and_jacobian!, taylor!,
    TaylorVector, ComplexDF64, FSVec, FSMat, _total_degree_startevaluator
using DynamicPolynomials: @polyvar, subs
using MultivariatePolynomials: MultivariatePolynomials as MP

include("test_systems.jl")

const MODES = (CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL)

# Series variable for the exact Taylor ground truth.
@polyvar λ

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

# Order-K coefficient of λ ↦ F(x(λ), p(λ)) with x(λ) = Σ_{k=0}^{K} X[k+1,:] λ^k.
function mp_taylor(polys, vars, X, params, P, K::Int)
    tx = [sum(X[k + 1, i] * λ^k for k in 0:K) for i in eachindex(vars)]
    tp = [sum(P[k + 1, i] * λ^k for k in 0:K) for i in eachindex(params)]
    return [ComplexF64(MP.coefficient(_at(f, vars, tx, params, tp), λ^K)) for f in polys]
end

function taylor_vector(X)
    tv = TaylorVector{size(X, 1), ComplexF64}(size(X, 2))
    tv.data .= X
    return tv
end

@testset "System sweep: $name" for (name, polys, vars, params) in TEST_SYSTEM_COLLECTION
    rng = MersenneTwister(0x00051ee7 + length(name))
    m, n, r = length(polys), length(vars), length(params)
    xvals = randn(rng, ComplexF64, n)
    pvals = randn(rng, ComplexF64, r)
    X = randn(rng, ComplexF64, 4, n)
    P = randn(rng, ComplexF64, 4, r)
    # Constant parameters: every order above 0 vanishes.
    P_const = vcat(reshape(pvals, 1, r), zeros(ComplexF64, 3, r))

    truth_u = mp_eval(polys, vars, xvals, params, pvals)
    truth_J = mp_jacobian(polys, vars, xvals, params, pvals)
    truth_taylor = [mp_taylor(polys, vars, X, params, P, K) for K in 1:3]
    truth_taylor_const = [mp_taylor(polys, vars, X, params, P_const, K) for K in 1:3]

    xf = FSVec{ComplexF64}(xvals)
    pf = FSVec{ComplexF64}(pvals)

    @testset "$mode" for mode in MODES
        S = System(polys; variables = vars, parameters = params, compile = mode).evaluator
        @test size(S) == (m, n)

        u = FSVec{ComplexF64}(zeros(ComplexF64, m))
        U = FSMat{ComplexF64}(zeros(ComplexF64, m, n))

        evaluate!(u, S, xf, pf)
        @test u ≈ truth_u rtol = 1.0e-10

        fill!(u, 0)
        evaluate_and_jacobian!(u, U, S, xf, pf)
        @test u ≈ truth_u rtol = 1.0e-10
        @test U ≈ truth_J rtol = 1.0e-10

        # Extended precision evaluation agrees with the ground truth.
        u_df64 = FSVec{ComplexDF64}(zeros(ComplexDF64, m))
        evaluate!(u_df64, S, FSVec{ComplexDF64}(ComplexDF64.(xvals)), pf)
        @test ComplexF64.(Vector(u_df64)) ≈ truth_u rtol = 1.0e-10

        @testset "taylor! K=$K" for K in 1:3
            tx = taylor_vector(X[1:(K + 1), :])
            fill!(u, 0)
            taylor!(u, Val(K), S, tx, pf)
            @test u ≈ truth_taylor_const[K] rtol = 1.0e-9

            # Taylor-valued parameters go through the Cauchy-product path.
            if r > 0
                fill!(u, 0)
                taylor!(u, Val(K), S, tx, taylor_vector(P[1:(K + 1), :]))
                @test u ≈ truth_taylor[K] rtol = 1.0e-9
            end
        end
    end
end

# H(x,t) = γ·t·G(x) + (1-t)·F(x) with the total-degree start system needs one
# equation per variable.
const SQUARE_SYSTEMS = filter(t -> length(t[2]) == length(t[3]), TEST_SYSTEM_COLLECTION)

@testset "StraightLineHomotopy sweep: $name" for (name, polys, vars, params) in
    SQUARE_SYSTEMS

    rng = MersenneTwister(0x00c0ffee + length(name))
    n = length(vars)
    pvals = randn(rng, ComplexF64, length(params))
    # Substituting the parameters keeps the symbolic ground truth polynomial.
    target = isempty(params) ? polys : [subs(f, params => pvals) for f in polys]
    degrees = [MP.maxdegree(f) for f in target]
    start = [vars[i]^degrees[i] - 1 for i in 1:n]
    γ = cis(2π * 0.3)
    t = 0.37 + 0.21im
    X = randn(rng, ComplexF64, 4, n)
    x0 = X[1, :]

    F = System(target; variables = vars)
    @test F.degrees == degrees
    H = StraightLineHomotopy(
        _total_degree_startevaluator(degrees), F.evaluator; γ = ComplexF64(γ),
    )
    He = HomotopyEvaluator(H)
    u = FSVec{ComplexF64}(zeros(ComplexF64, n))

    G0 = mp_eval(start, vars, x0, [], [])
    F0 = mp_eval(target, vars, x0, [], [])

    evaluate!(u, He, FSVec{ComplexF64}(x0), ComplexF64(t))
    @test u ≈ γ * t .* G0 .+ (1 - t) .* F0 rtol = 1.0e-10

    U = FSMat{ComplexF64}(zeros(ComplexF64, n, n))
    evaluate_and_jacobian!(u, U, He, FSVec{ComplexF64}(x0), ComplexF64(t))
    @test u ≈ γ * t .* G0 .+ (1 - t) .* F0 rtol = 1.0e-10
    @test U ≈ γ * t .* mp_jacobian(start, vars, x0, [], []) .+
        (1 - t) .* mp_jacobian(target, vars, x0, [], []) rtol = 1.0e-10

    # Val(1) takes a plain point: x is constant, so only the ∂/∂t term remains.
    fill!(u, 0)
    taylor!(u, Val(1), He, FSVec{ComplexF64}(x0), ComplexF64(t))
    @test u ≈ γ .* G0 .- F0 rtol = 1.0e-10

    # Order-K coefficient of λ ↦ H(x(λ), t+λ).
    function h_taylor(K::Int)
        tx = [sum(X[k + 1, i] * λ^k for k in 0:K) for i in 1:n]
        Gs = [f(vars => tx) for f in start]
        Fs = [f(vars => tx) for f in target]
        Hs = γ .* (t + λ) .* Gs .+ (1 - (t + λ)) .* Fs
        return [ComplexF64(MP.coefficient(g, λ^K)) for g in Hs]
    end

    @testset "taylor! K=$K" for K in 2:3
        fill!(u, 0)
        taylor!(u, Val(K), He, taylor_vector(X[1:(K + 1), :]), ComplexF64(t))
        @test u ≈ h_taylor(K) rtol = 1.0e-9
    end
end
