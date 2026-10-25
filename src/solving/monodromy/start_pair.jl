"""
    StartPair

A start pair `(x, p)` for monodromy: `x` solves the system at parameters `p`.
`found` is `false` when no pair could be produced, and `p` is empty for a
parameter-free system (use [`is_parameterized`](@ref)).
"""
struct StartPair
    found::Bool
    x::Vector{ComplexF64}
    p::Vector{ComplexF64}
end

StartPair() = StartPair(false, ComplexF64[], ComplexF64[])
StartPair(x::Vector{ComplexF64}) = StartPair(true, x, ComplexF64[])
StartPair(x::Vector{ComplexF64}, p::Vector{ComplexF64}) = StartPair(true, x, p)

"""
    is_parameterized(pair::StartPair)

Whether `pair` carries parameters; `false` for a parameter-free system.
"""
is_parameterized(pair::StartPair)::Bool = !isempty(pair.p)

"""
    find_start_pair(F::SystemLike; max_tries = 1_000, atol = 0.0, rtol = 1e-12,
                    rng = Random.default_rng())

Try to find a pair `(x, p)` for the system `F` such that `F(x, p) = 0` by
sampling a random `x` from `rng` and solving the linear system in the parameters
(when `F` is linear in the parameters), or by a Newton solve of the joint system
in `(x, p)` otherwise. Returns a [`StartPair`](@ref); `pair.found` is `false` if
no pair could be found in `max_tries` tries. For a parameter-free system the
returned pair has an empty `p`, so [`is_parameterized`](@ref) is `false`.
"""
function find_start_pair(
        F::System;
        max_tries::Int = 1_000,
        atol::Float64 = 0.0,
        rtol::Float64 = 1.0e-12,
        rng::Random.AbstractRNG = Random.default_rng(),
    )::StartPair
    refine_atol = atol > 0 ? atol : 1.0e-12
    nparameters(F) == 0 &&
        return parameter_free_start_pair(F, rng, max_tries, refine_atol, rtol)
    return parameterized_start_pair(F, rng, max_tries, refine_atol, rtol)
end

# A composition keeps no equations, so the symbolic strategies do not apply and
# Newton runs on the joint system assembled from the evaluator.
function find_start_pair(
        C::CompositionSystem;
        max_tries::Int = 1_000,
        atol::Float64 = 0.0,
        rtol::Float64 = 1.0e-12,
        rng::Random.AbstractRNG = Random.default_rng(),
    )::StartPair
    refine_atol = atol > 0 ? atol : 1.0e-12
    nparameters(C) == 0 &&
        return parameter_free_start_pair(C, rng, max_tries, refine_atol, rtol)
    return composition_start_pair(C, rng, max_tries, refine_atol, rtol)
end

@noinline function composition_start_pair(
        C::CompositionSystem, rng::Random.AbstractRNG, max_tries::Int,
        refine_atol::Float64, rtol::Float64,
    )::StartPair
    m, n = size(C)
    np = nparameters(C)
    joint = SystemEvaluator(StartPairSystem(C.evaluator))
    joint_cache = newton_cache(m, n + np)
    cache = NewtonCache(C)
    for _ in 1:max_tries
        xp₀ = randn(rng, ComplexF64, n + np)
        res = newton_iterate(
            joint, joint_cache, xp₀, EMPTY_PARAMS, 1.0e-8, 1.0e-8, 20, false,
            1.0, typemax(Int), Inf, Inf,
        )
        if res.return_code == NewtonReturnCode.NEWTON_SUCCESS
            x = res.x[1:n]
            p = res.x[(n + 1):end]
            refined = newton(
                C, x; p = p, atol = refine_atol, rtol = rtol, cache = cache,
            )
            if refined.return_code == NewtonReturnCode.NEWTON_SUCCESS
                return StartPair(refined.x, p)
            end
        end
    end
    return StartPair()
end

@noinline function parameter_free_start_pair(
        F::SystemLike, rng::Random.AbstractRNG, max_tries::Int,
        refine_atol::Float64, rtol::Float64,
    )::StartPair
    nvars = nvariables(F)
    cache = NewtonCache(F)
    for _ in 1:max_tries
        x₀ = randn(rng, ComplexF64, nvars)
        res = newton(F, x₀; atol = 1.0e-8, cache = cache)
        if res.return_code == NewtonReturnCode.NEWTON_SUCCESS
            refined = newton(
                F, res.x; atol = refine_atol, rtol = rtol, cache = cache,
            )
            if refined.return_code == NewtonReturnCode.NEWTON_SUCCESS
                return StartPair(refined.x)
            end
        end
    end
    return StartPair()
end

@noinline function parameterized_start_pair(
        F::System, rng::Random.AbstractRNG, max_tries::Int,
        refine_atol::Float64, rtol::Float64,
    )::StartPair

    # 1. Linear-in-parameters fast path. Each attempt draws a
    # fresh random x₀ internally, so a bad draw should retry, not abandon the
    # fast path.
    for _ in 1:3
        pair = linear_in_params_start_pair(F, rng)
        pair.found && return pair
    end

    return joint_newton_start_pair(F, rng, max_tries, refine_atol, rtol)
end

@noinline function joint_newton_start_pair(
        F::System, rng::Random.AbstractRNG, max_tries::Int,
        refine_atol::Float64, rtol::Float64,
    )::StartPair
    nvars = nvariables(F)
    np = nparameters(F)
    G = System(
        collect(F.polys);
        variables = [collect(F.variables); collect(F.parameters)],
    )
    cache = NewtonCache(G)
    F_cache = NewtonCache(F)
    for _ in 1:max_tries
        xp₀ = randn(rng, ComplexF64, nvars + np)
        res = newton(G, xp₀; atol = 1.0e-8, cache = cache)
        if res.return_code == NewtonReturnCode.NEWTON_SUCCESS
            x = res.x[1:nvars]
            p = res.x[(nvars + 1):end]
            refined = newton(
                F, x; p = p, atol = refine_atol, rtol = rtol, cache = F_cache,
            )
            if refined.return_code == NewtonReturnCode.NEWTON_SUCCESS
                return StartPair(refined.x, p)
            end
        end
    end
    return StartPair()
end

# Fast path: sample x₀, substitute it into every polynomial and check that the
# result is linear in the parameters (every term touches at most one parameter,
# with exponent at most 1, and every equation actually contains a parameter).
# Then solve the linear system A p = b exactly.
function linear_in_params_start_pair(
        F::System, rng::Random.AbstractRNG,
    )::StartPair
    nvars = nvariables(F)
    np = nparameters(F)
    m = length(F.polys)
    m <= np || return StartPair()

    x₀ = randn(rng, ComplexF64, nvars)
    vars = collect(F.variables)
    params = collect(F.parameters)

    A = zeros(ComplexF64, m, np)
    b = zeros(ComplexF64, m)
    for (i, f) in enumerate(F.polys)
        g = subs(f, vars => x₀)
        # Not polynomial in the parameters, or of degree two or more.
        degree(g, params) == 1 || return StartPair()
        E, c = exponents_coefficients(g, params)
        has_param_term = false
        for k in eachindex(c)
            iszero(c[k]) && continue
            d = sum(view(E, :, k))
            if d == 0
                b[i] -= c[k]
            elseif d == 1
                A[i, findfirst(!iszero, view(E, :, k))] += c[k]
                has_param_term = true
            else
                return StartPair()
            end
        end
        # A parameter-free equation cannot be satisfied by choosing p.
        has_param_term || return StartPair()
    end

    p₀ = if iszero(b)
        # Only the trivial solution when the system is square; otherwise
        # sample from the nullspace.
        m == np && return StartPair()
        N = LA.nullspace(A)
        size(N, 2) == 0 && return StartPair()
        Vector{ComplexF64}(N * randn(rng, ComplexF64, size(N, 2)))
    else
        Vector{ComplexF64}(LA.qr(A, LA.ColumnNorm()) \ b)
    end
    all(isfinite, p₀) || return StartPair()
    return StartPair(x₀, p₀)
end
