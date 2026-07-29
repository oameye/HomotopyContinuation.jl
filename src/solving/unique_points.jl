## UniquePoints, multiplicities, unique_points: group-action-aware point
## deduplication on top of VoronoiTree.
#
# The default distance is InfNorm; relative to a Euclidean distance,
# tolerances differ by at most sqrt(2n).

"""
    UniquePoints(d::Int; distance = InfNorm(), group_actions = nothing,
                 triangle_inequality)

A data structure to quickly check whether a point of dimension `d` is close to
an already indexed point, up to the given `distance` function and modulo the
given group actions. Points are stored by their `Int` identifiers.
"""
struct UniquePoints{T, M, GA}
    # GA is either Nothing, a GroupActions, or a callable implementing the
    # 2-arg protocol `actions(cb, s)` (e.g. the chart-normalizing wrapper used
    # by the monodromy solver); apply_actions dispatches on it.
    tree::VoronoiTree{T, M}
    group_actions::GA
    zero_vec::Vector{T}    # cached zero for the rtol distance-to-origin
end

function UniquePoints(
        d::Int;
        distance = InfNorm(),
        group_action = nothing,
        group_actions = group_action === nothing ? nothing : GroupActions(group_action),
        triangle_inequality::Bool = satisfies_triangle_inequality(distance),
    )
    group_actions = _as_group_actions(group_actions)
    tree = VoronoiTree{ComplexF64}(
        d; distance = distance, triangle_inequality = triangle_inequality,
    )
    return UniquePoints(tree, group_actions, zeros(ComplexF64, d))
end

function Base.show(io::IO, UP::UniquePoints)
    return print(io, typeof(UP), " with ", length(UP.tree), " points")
end
Base.length(UP::UniquePoints) = length(UP.tree)
Base.collect(UP::UniquePoints) = collect(UP.tree)
Base.broadcastable(UP::UniquePoints) = Ref(UP)
function Base.empty!(UP::UniquePoints)
    empty!(UP.tree)
    return UP
end

"""
    search_in_radius(unique_points, v, tol)

Search whether `unique_points` contains a point with distance at most `tol`
from `v` or from any of its orbit images. Returns `nothing` if no point exists,
otherwise the identifier of the found point.
"""
function search_in_radius(
        UP::UniquePoints{T, M, GA}, v::AbstractVector, tol::Real,
    ) where {T, M, GA}
    id = search_in_radius(UP.tree, v, tol)
    if id === nothing && UP.group_actions !== nothing
        # Ref against closure boxing (id would be reassigned inside the closure).
        id_ref = Ref{Union{Nothing, Int}}(nothing)
        let actions = UP.group_actions::GA
            apply_actions(actions, v) do w
                id′ = search_in_radius(UP.tree, w, tol)
                if id′ !== nothing
                    id_ref[] = id′
                    return true
                end
                false
            end
        end
        id = id_ref[]
    end
    return id
end

"""
    add!(unique_points, v, id; atol = 1e-14, rtol = sqrt(eps()))
    add!(unique_points, v, id, tol)

Search whether `unique_points` contains a point with distance at most
`max(atol, rtol * distance(v, 0))` (resp. `tol`) from `v` or any of its orbit
images. If so, the identifier of that point and `false` is returned. Otherwise
`v` is inserted and `(id, true)` is returned.
"""
function add!(UP::UniquePoints{T, M, GA}, v::AbstractVector, id::Int, tol::Real) where {T, M, GA}
    # search_in_radius(UP, ...) already checks the tree and every orbit image.
    found_id = search_in_radius(UP, v, tol)
    if found_id === nothing
        insert!(UP.tree, v, id)
        return (id, true)
    else
        return (found_id::Int, false)
    end
end

function add!(
        UP::UniquePoints{T, M, GA}, v::AbstractVector, id::Int;
        atol::Float64 = 1.0e-14,
        rtol::Float64 = sqrt(eps()),
    ) where {T, M, GA}
    n = _vt_distance(UP.tree.distance, v, UP.zero_vec)
    rad = max(atol, rtol * n)
    return add!(UP, v, id, rad)
end

####################
## Multiplicities ##
####################

"""
    multiplicities(vectors; distance = InfNorm(), atol = 1e-14, rtol = 1e-8, kwargs...)

Returns a `Vector{Vector{Int}}` `v`. Each vector `w` in `v` contains all
indices `i`, `j` such that `w[i]` and `w[j]` have `distance` at most
`max(atol, rtol * distance(w[i], 0))`. The remaining `kwargs` are passed to
[`UniquePoints`](@ref).
"""
function multiplicities(
        v;
        distance = InfNorm(),
        atol::Float64 = 1.0e-14,
        rtol::Float64 = 1.0e-8,
        group_action = nothing,
        group_actions = group_action === nothing ? nothing : GroupActions(group_action),
        triangle_inequality::Bool = satisfies_triangle_inequality(distance),
    )
    return multiplicities(
        identity, v;
        distance = distance, atol = atol, rtol = rtol,
        group_action = group_action, group_actions = group_actions,
        triangle_inequality = triangle_inequality,
    )
end
function multiplicities(
        f::F, v;
        distance = InfNorm(),
        atol::Float64 = 1.0e-14,
        rtol::Float64 = 1.0e-8,
        group_action = nothing,
        group_actions = group_action === nothing ? nothing : GroupActions(group_action),
        triangle_inequality::Bool = satisfies_triangle_inequality(distance),
    ) where {F <: Function}
    isempty(v) && return Vector{Vector{Int}}()
    UP = UniquePoints(
        length(f(first(v)));
        distance = distance,
        group_actions = group_actions,
        triangle_inequality = triangle_inequality,
    )
    mults = Dict{Int, Vector{Int}}()
    for (i, vᵢ) in enumerate(v)
        wᵢ = f(vᵢ)
        k, new_point = add!(UP, wᵢ, i; atol = atol, rtol = rtol)
        if !new_point
            if haskey(mults, k)
                push!(mults[k], i)
            else
                mults[k] = [k, i]
            end
        end
    end
    # Return groups in a deterministic order (by representative index) rather
    # than the arbitrary hash order of `values(::Dict)`.
    return [mults[k] for k in sort!(collect(keys(mults)))]
end

"""
    unique_points(vectors; distance = InfNorm(), atol = 1e-14, rtol = 1e-8, kwargs...)

Returns all elements of `vectors` whose pairwise `distance` exceeds
`max(atol, rtol * distance(w, 0))`. The output can depend on the order of the
elements. The remaining `kwargs` are passed to [`UniquePoints`](@ref).
"""
function unique_points(
        V;
        distance = InfNorm(),
        atol::Float64 = 1.0e-14,
        rtol::Float64 = 1.0e-8,
        group_action = nothing,
        group_actions = group_action === nothing ? nothing : GroupActions(group_action),
        triangle_inequality::Bool = satisfies_triangle_inequality(distance),
    )
    UP = UniquePoints(
        length(first(V));
        distance = distance,
        group_actions = group_actions,
        triangle_inequality = triangle_inequality,
    )
    out = Vector{eltype(V)}()
    for (i, vᵢ) in enumerate(V)
        _, new_point = add!(UP, vᵢ, i; atol = atol, rtol = rtol)
        if new_point
            push!(out, vᵢ)
        end
    end
    return out
end
