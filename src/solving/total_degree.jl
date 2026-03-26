## TotalDegree — Bezout bound start system for homotopy continuation.

"""
    TotalDegree(; tracker_options=TrackerOptions(), seed=nothing)

Algorithm that constructs a total-degree (Bezout) start system.
The number of paths tracked is prod(degrees) — the Bezout bound.

# Examples
```julia
@polyvar x y
result = solve([x^2 + y - 1, x*y - 2], TotalDegree())
```
"""
@kwdef struct TotalDegree
    tracker_options::TrackerOptions = TrackerOptions()
    seed::Union{Nothing, UInt32} = nothing
end

function _total_degree_startsystem(
        degrees::Vector{Int}, variables::AbstractVector,
    )::SystemEvaluator
    n = length(degrees)
    polys = [variables[i]^degrees[i] - 1 for i in 1:n]
    _, evaluator = system_eval(polys; variables = variables)
    return evaluator
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
