## VoronoiTree: nearest-point search tree for solution deduplication.
#
# Depth-indexed scratch buffers live on the tree, not per node: case 3 of the
# search sorts the parent's distances AFTER returning from child recursions,
# so a single shared buffer would be corrupted by the recursion; one buffer
# per depth is enough. add! does a plain search, then on miss a plain insert
# descent that recomputes distances.
#
# Dedup is a cold path; the small allocations from growing the depth scratch
# are fine.

# Distance callable indirection: InfNorm is a marker struct, custom distances
# are callables d(x, y).
@inline _vt_distance(::InfNorm, x, y) = inf_distance(x, y)
@inline _vt_distance(d, x, y) = d(x, y)

mutable struct VTNode{T}
    # Mutable: nentries grows as points are inserted; children slots fill in lazily.
    nentries::Int
    const values::Matrix{T}               # d x capacity, filled column by column
    const ids::Vector{Int}                # length capacity
    const children::Vector{VTNode{T}}     # isassigned-sparse slots (standard idiom)
end

function VTNode{T}(d::Int, capacity::Int) where {T}
    return VTNode{T}(
        0,
        Matrix{T}(undef, d, capacity),
        Vector{Int}(undef, capacity),
        Vector{VTNode{T}}(undef, capacity),
    )
end

function VTNode{T}(v::AbstractVector, id::Int, capacity::Int) where {T}
    node = VTNode{T}(length(v), capacity)
    node.nentries = 1
    node.values[:, 1] .= v
    node.ids[1] = id
    return node
end

Base.length(node::VTNode) = node.nentries
Base.isempty(node::VTNode) = length(node) == 0
capacity(node::VTNode) = length(node.children)

"""
    VoronoiTree{T}(d::Int; distance = InfNorm(), capacity = 8, triangle_inequality = nothing)

A Voronoi tree over points of element type `T` and dimension `d` with `Int`
identifiers. Distances are measured by `distance` (the default `InfNorm` is a
metric, so triangle-inequality pruning stays valid; relative to a Euclidean
distance, tolerances differ by at most `sqrt(2d)`).
`triangle_inequality = nothing` autodetects for custom distances via a
probabilistic check on random points.
"""
mutable struct VoronoiTree{T, M}
    # Mutable: nentries counts inserts; root is replaced by empty!.
    root::VTNode{T}
    nentries::Int
    const distance::M
    const scratch::Vector{Vector{Tuple{Float64, Int}}}   # one buffer per depth
    const triangle_inequality::Bool
end

function _detect_triangle_inequality(distance, d::Int)::Bool
    distance isa InfNorm && return true
    # Probabilistic check on random points.
    v₁ = randn(ComplexF64, d)
    v₂ = randn(ComplexF64, d)
    v₃ = randn(ComplexF64, d)
    return _vt_distance(distance, v₁, v₂) <=
        _vt_distance(distance, v₁, v₃) + _vt_distance(distance, v₃, v₂) &&
        _vt_distance(distance, v₁, 4 .* v₁) <=
        _vt_distance(distance, v₁, 2 .* v₁) + _vt_distance(distance, 2 .* v₁, 4 .* v₁)
end

function VoronoiTree{T}(
        d::Int;
        distance = InfNorm(),
        capacity::Int = 8,
        triangle_inequality::Union{Nothing, Bool} = nothing,
    ) where {T}
    tri = triangle_inequality === nothing ?
        _detect_triangle_inequality(distance, d) : triangle_inequality
    root = VTNode{T}(d, capacity)
    scratch = [Vector{Tuple{Float64, Int}}(undef, capacity)]
    return VoronoiTree{T, typeof(distance)}(root, 0, distance, scratch, tri)
end

Base.length(tree::VoronoiTree) = tree.nentries
Base.broadcastable(tree::VoronoiTree) = Ref(tree)

function Base.empty!(tree::VoronoiTree{T, M}) where {T, M}
    tree.root = VTNode{T}(size(tree.root.values, 1), capacity(tree.root))
    tree.nentries = 0
    return tree
end

# One distances buffer per recursion depth (1-based), grown on demand.
@inline function _scratch_at(tree::VoronoiTree, depth::Int)::Vector{Tuple{Float64, Int}}
    while length(tree.scratch) < depth
        push!(tree.scratch, Vector{Tuple{Float64, Int}}(undef, capacity(tree.root)))
    end
    return @inbounds tree.scratch[depth]
end

function _compute_distances!(
        distances::Vector{Tuple{Float64, Int}}, tree::VoronoiTree, node::VTNode, x,
    )::Nothing
    for j in 1:length(node)
        distances[j] = (_vt_distance(tree.distance, x, view(node.values, :, j)), j)
    end
    return nothing
end

"""
    search_in_radius(tree::VoronoiTree, v::AbstractVector, tol::Real)

Search whether `tree` contains a point with distance at most `tol` from `v`.
Returns `nothing` if no such point exists, otherwise its identifier.
"""
function search_in_radius(tree::VoronoiTree{T, M}, v::AbstractVector, tol::Real) where {T, M}
    return _search_in_radius!(tree, tree.root, v, Float64(tol), 1)
end

function _search_in_radius!(
        tree::VoronoiTree{T, M}, node::VTNode{T}, v, tol::Float64, depth::Int,
    )::Union{Nothing, Int} where {T, M}
    !isempty(node) || return nothing

    n = length(node)
    triangle_inequality = tree.triangle_inequality

    # Distance to every entry. If one is below tol we are done. Otherwise we
    # only need to recurse into children whose distance dᵢ satisfies
    # dᵢ - d_min < 2 tol (triangle inequality); without a triangle inequality
    # all children are checked. Track the three smallest along the way.
    m₁ = m₂ = m₃ = (Inf, 1)
    distances = _scratch_at(tree, depth)
    _compute_distances!(distances, tree, node, v)
    for i in 1:n
        dᵢ = first(distances[i])
        if dᵢ < tol
            return node.ids[i]
        end
        if dᵢ < first(m₁)
            m₃ = m₂
            m₂ = m₁
            m₁ = (dᵢ, i)
        elseif dᵢ < first(m₂)
            m₃ = m₂
            m₂ = (dᵢ, i)
        elseif dᵢ < first(m₃)
            m₃ = (dᵢ, i)
        end
    end

    # Case analysis:
    # 1) m₂ - m₁ > 2tol: the point can only be in the nearest child's subtree.
    # 2) m₃ - m₁ > 2tol: only in the two nearest subtrees.
    # 3) otherwise: sort all distances and sweep until the bound cuts off.

    if isassigned(node.children, last(m₁))
        retid = _search_in_radius!(tree, node.children[last(m₁)], v, tol, depth + 1)
        retid === nothing || return retid
    end

    if m₂[1] - m₁[1] > 2tol && triangle_inequality
        return nothing # already checked the first subtree
    end

    if isassigned(node.children, last(m₂))
        retid = _search_in_radius!(tree, node.children[last(m₂)], v, tol, depth + 1)
        retid === nothing || return retid
    end

    if m₃[1] - m₁[1] > 2tol && triangle_inequality
        return nothing # checked the first and second subtree
    end

    if isassigned(node.children, last(m₃))
        retid = _search_in_radius!(tree, node.children[last(m₃)], v, tol, depth + 1)
        retid === nothing || return retid
    end

    # Case 3: child recursions used deeper scratch buffers, so this node's
    # distances are still intact and can be sorted now.
    sort!(view(distances, 1:n); alg = Base.Sort.InsertionSort, by = first)

    # Start at 4: the three smallest were already checked.
    for k in 4:n
        dᵢ, i = distances[k]
        if dᵢ - m₁[1] < 2tol || !triangle_inequality
            if isassigned(node.children, i)
                retid = _search_in_radius!(tree, node.children[i], v, tol, depth + 1)
                retid === nothing || return retid
            end
        else
            break
        end
    end

    return nothing
end

"""
    insert!(tree::VoronoiTree, v::AbstractVector, id::Int)

Insert the point `v` with identifier `id` into the tree.
"""
function Base.insert!(tree::VoronoiTree{T, M}, v::AbstractVector, id::Int) where {T, M}
    _insert!(tree, tree.root, v, id, 1)
    tree.nentries += 1
    return tree
end

function _insert!(
        tree::VoronoiTree{T, M}, node::VTNode{T}, v, id::Int, depth::Int,
    )::Nothing where {T, M}
    # If not filled so far, just add to the current node.
    if length(node) < capacity(node)
        k = (node.nentries += 1)
        node.values[:, k] .= v
        node.ids[k] = id
        return nothing
    end

    distances = _scratch_at(tree, depth)
    _compute_distances!(distances, tree, node, v)
    dmin, minᵢ = distances[1]
    for j in 2:length(node)
        dⱼ = first(distances[j])
        if dⱼ < dmin
            dmin, minᵢ = dⱼ, j
        end
    end

    if !isassigned(node.children, minᵢ)
        node.children[minᵢ] = VTNode{T}(v, id, capacity(node))
    else
        _insert!(tree, node.children[minᵢ], v, id, depth + 1)
    end

    return nothing
end

"""
    add!(tree::VoronoiTree, v, id, tol)

Insert the point `v` with identifier `id` unless `search_in_radius(tree, v, tol)`
finds a point. Returns `(id, true)` when inserted, otherwise the found
identifier and `false`.
"""
function add!(tree::VoronoiTree{T, M}, v::AbstractVector, id::Int, tol::Real) where {T, M}
    found_id = search_in_radius(tree, v, tol)
    if found_id === nothing
        insert!(tree, v, id)
        return (id, true)
    else
        return (found_id::Int, false)
    end
end

function _identifiers!(ids::Vector{Int}, node::VTNode)::Vector{Int}
    for i in 1:length(node)
        push!(ids, node.ids[i])
        if isassigned(node.children, i)
            _identifiers!(ids, node.children[i])
        end
    end
    return ids
end

Base.collect(tree::VoronoiTree) = _identifiers!(Int[], tree.root)
