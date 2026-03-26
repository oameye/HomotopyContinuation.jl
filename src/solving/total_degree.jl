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
    seed::UInt32
end

function TotalDegree(;
        tracker_options::TrackerOptions = TrackerOptions(),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        max_steps::Int = tracker_options.max_steps,
        max_step_size::Float64 = tracker_options.max_step_size,
        max_initial_step_size::Float64 = tracker_options.max_initial_step_size,
        extended_precision::Bool = tracker_options.extended_precision,
        min_step_size::Float64 = tracker_options.min_step_size,
        terminate_cond::Float64 = tracker_options.terminate_cond,
        a::Float64 = tracker_options.a,
        β_ω::Float64 = tracker_options.β_ω,
        β_τ::Float64 = tracker_options.β_τ,
        strict_β_τ::Float64 = tracker_options.strict_β_τ,
    )
    opts = TrackerOptions(;
        max_steps, max_step_size, max_initial_step_size,
        extended_precision, min_step_size, terminate_cond,
        a, β_ω, β_τ, strict_β_τ,
    )
    return TotalDegree(opts, seed)
end

function _total_degree_startsystem(degrees::Vector{Int})::System
    n = length(degrees)
    @polyvar _td_x[1:n]
    polys = [_td_x[i]^degrees[i] - 1 for i in 1:n]
    return System(polys; variables = collect(_td_x))
end

function _total_degree_solutions(degrees::Vector{Int})::Vector{Vector{ComplexF64}}
    roots = [cis.(2π .* (0:(d - 1)) ./ d) for d in degrees]
    result = Vector{ComplexF64}[]
    for combo in Iterators.product(roots...)
        push!(result, ComplexF64[combo...])
    end
    return result
end

total_degree_count(degrees::Vector{Int})::Int = prod(degrees)
