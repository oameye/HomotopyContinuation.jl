## TotalDegree — Bezout bound start system for homotopy continuation.

"""
    TotalDegree(; seed, max_steps, extended_precision, ...)

Algorithm that constructs a total-degree (Bezout) start system.
The number of paths tracked is prod(degrees) — the Bezout bound.

Accepts all `TrackerOptions` fields as keyword arguments, or a pre-built
`tracker_options` object.

# Examples
```julia
@polyvar x y
F = System([x^2 + y - 1, x*y - 2])
result = solve(F, TotalDegree())

# With explicit seed for reproducibility
result = solve(F, TotalDegree(; seed=UInt32(42)))

# Tune tracker options directly
result = solve(F, TotalDegree(; max_steps=500, extended_precision=false))
```
"""
struct TotalDegree
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
    seed::UInt32
end

struct TotalDegreeStartSystem <: AbstractSystem
    degrees::Vector{Int}
end

Base.size(F::TotalDegreeStartSystem)::Tuple{Int, Int} =
    (length(F.degrees), length(F.degrees))

@inline _total_degree_value(x, degree::Int) = Base.power_by_squaring(x, degree) - one(x)
@inline _total_degree_jacobian_entry(x, degree::Int) =
    degree == 1 ? one(x) : degree * Base.power_by_squaring(x, degree - 1)

function evaluate!(
        u::FSVec{ComplexF64}, F::TotalDegreeStartSystem,
        x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    @inbounds for i in eachindex(u)
        u[i] = _total_degree_value(x[i], F.degrees[i])
    end
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, F::TotalDegreeStartSystem,
        x::FSVec{ComplexDF64}, p::FSVec{ComplexF64},
    )::Nothing
    @inbounds for i in eachindex(u)
        u[i] = ComplexF64(_total_degree_value(x[i], F.degrees[i]))
    end
    return nothing
end

function evaluate!(
        u::FSVec{ComplexDF64}, F::TotalDegreeStartSystem,
        x::FSVec{ComplexDF64}, p::FSVec{ComplexF64},
    )::Nothing
    @inbounds for i in eachindex(u)
        u[i] = _total_degree_value(x[i], F.degrees[i])
    end
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64},
        F::TotalDegreeStartSystem, x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    fill!(U, zero(ComplexF64))
    @inbounds for i in eachindex(u)
        degree = F.degrees[i]
        x_i = x[i]
        u[i] = _total_degree_value(x_i, degree)
        U[i, i] = _total_degree_jacobian_entry(x_i, degree)
    end
    return nothing
end

# TotalDegreeStartSystem ignores parameters — shared implementation for all
# taylor! dispatch variants (FSVec and TaylorVector parameter signatures).
@inline function _td_taylor!(
        u::FSVec{ComplexF64}, ::Val{K},
        F::TotalDegreeStartSystem, tx::TaylorVector
    )::Nothing where {K}
    @inbounds for i in eachindex(u)
        u[i] = taylor_op_pow_int(tx[i], F.degrees[i])[K]
    end
    return nothing
end

# FSVec parameter variants (used by StraightLineHomotopy, etc.)
taylor!(
    u::FSVec{ComplexF64}, v::Val{1}, F::TotalDegreeStartSystem,
    tx::TaylorVector{2, ComplexF64}, ::FSVec{ComplexF64}
)::Nothing = _td_taylor!(u, v, F, tx)
taylor!(
    u::FSVec{ComplexF64}, v::Val{2}, F::TotalDegreeStartSystem,
    tx::TaylorVector{3, ComplexF64}, ::FSVec{ComplexF64}
)::Nothing = _td_taylor!(u, v, F, tx)
taylor!(
    u::FSVec{ComplexF64}, v::Val{3}, F::TotalDegreeStartSystem,
    tx::TaylorVector{4, ComplexF64}, ::FSVec{ComplexF64}
)::Nothing = _td_taylor!(u, v, F, tx)

# TaylorVector parameter variants (used by CoefficientHomotopy/ToricHomotopy Cauchy product path)
taylor!(
    u::FSVec{ComplexF64}, v::Val{1}, F::TotalDegreeStartSystem,
    tx::TaylorVector{2, ComplexF64}, ::TaylorVector{2, ComplexF64}
)::Nothing = _td_taylor!(u, v, F, tx)
taylor!(
    u::FSVec{ComplexF64}, v::Val{2}, F::TotalDegreeStartSystem,
    tx::TaylorVector{3, ComplexF64}, ::TaylorVector{3, ComplexF64}
)::Nothing = _td_taylor!(u, v, F, tx)
taylor!(
    u::FSVec{ComplexF64}, v::Val{3}, F::TotalDegreeStartSystem,
    tx::TaylorVector{4, ComplexF64}, ::TaylorVector{4, ComplexF64}
)::Nothing = _td_taylor!(u, v, F, tx)

function TotalDegree(;
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        max_steps::Int = tracker_options.max_steps,
        max_step_size::Float64 = tracker_options.max_step_size,
        max_initial_step_size::Float64 = tracker_options.max_initial_step_size,
        extended_precision::Bool = tracker_options.extended_precision,
        min_step_size::Float64 = tracker_options.min_step_size,
        terminate_cond::Float64 = tracker_options.terminate_cond,
        a::Float64 = tracker_options.a,
        β_a::Float64 = tracker_options.β_a,
        β_ω::Float64 = tracker_options.β_ω,
        β_τ::Float64 = tracker_options.β_τ,
        strict_β_τ::Float64 = tracker_options.strict_β_τ,
    )
    opts = TrackerOptions(;
        max_steps, max_step_size, max_initial_step_size,
        extended_precision, min_step_size, terminate_cond,
        a, β_a, β_ω, β_τ, strict_β_τ,
    )
    return TotalDegree(opts, endgame_options, seed)
end

function _total_degree_startsystem(degrees::Vector{Int})::TotalDegreeStartSystem
    return TotalDegreeStartSystem(copy(degrees))
end

function _total_degree_startevaluator(degrees::Vector{Int})::SystemEvaluator
    return SystemEvaluator(_total_degree_startsystem(degrees))
end

function _total_degree_solutions(degrees::Vector{Int})::Vector{Vector{ComplexF64}}
    n = length(degrees)
    npaths = prod(degrees)
    result = Vector{Vector{ComplexF64}}(undef, npaths)
    for path_index in 0:(npaths - 1)
        quotient = path_index
        solution = Vector{ComplexF64}(undef, n)
        @inbounds for variable_index in 1:n
            degree = degrees[variable_index]
            root_index, quotient = divrem(quotient, degree)
            solution[variable_index] = cis(2π * quotient / degree)
            quotient = root_index
        end
        result[path_index + 1] = solution
    end
    return result
end

total_degree_count(degrees::Vector{Int})::Int = prod(degrees)
