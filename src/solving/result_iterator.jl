## Lazy path tracking: one `PathResult` per iteration step.
#
# A `ResultIterator` holds a solve cache and tracks a path only when iterated,
# so nothing is stored for paths that are never asked for. Iteration is serial
# by construction (the cache's single tracker is reused), and no clustering or
# excess-solution reclassification happens; call `Result(ri)` for those.

# The type itself is in `starts.jl`, which `StartsLike` forces to be compiled
# before every route that takes start solutions.

# ── One path per cache kind ────────────────────────────────────────────────

function _path_result(cache::SolveCache, i::Int)::PathResult
    x₀ = cache.start_solutions[i]
    track!(cache.tracker, x₀)
    return PathResult(
        cache.tracker; path_number = i, start_solution = Vector{ComplexF64}(x₀),
    )
end

_path_result(cache::WorkerSolveCache, i::Int)::PathResult =
    _track_path!(cache.worker, cache.start_solutions[i], i)

function _path_result(cache::PolyhedralSolveCache, i::Int)::PathResult
    cell, x₀ = cache.start_solutions[i]
    return _track_polyhedral_path!(
        cache.toric_tracker, cache.coeff_tracker, cache.toric_homotopy,
        cache.support, cache.lifting, cache.x_buffer, cell, x₀, i,
    )
end

# ── Iteration ──────────────────────────────────────────────────────────────

Base.eltype(::Type{<:ResultIterator}) = PathResult
# `Base.IteratorSize` needs no method: its fallback is `HasLength()`, which is
# what `length` below provides.
Base.length(ri::ResultIterator)::Int = count(ri.mask)

function Base.iterate(ri::ResultIterator, i::Int = 1)
    n = length(ri.cache.start_solutions)
    while i <= n && !ri.mask[i]
        i += 1
    end
    i > n && return nothing
    return (_path_result(ri.cache, i), i + 1)
end

function Base.show(io::IO, ri::ResultIterator)
    n = length(ri.cache.start_solutions)
    k = length(ri)
    print(io, "ResultIterator over ", k, " of ", n, " start solutions")
    return
end

# ── Accessors ──────────────────────────────────────────────────────────────

"""
    seed(ri::ResultIterator)

The random seed of the underlying solve.
"""
seed(ri::ResultIterator)::UInt32 = ri.cache.seed

"""
    path_results(ri::ResultIterator)

Track every selected path and collect the results.
"""
path_results(ri::ResultIterator)::Vector{PathResult} = collect(ri)

"""
    start_solutions(ri::ResultIterator)

The start solutions of `ri`, including the ones its mask filters out.
"""
start_solutions(ri::ResultIterator) = ri.cache.start_solutions

# A copy: the constructor pins `length(mask)`, so a `push!` on the internal
# `BitVector` would break `length(ri)`.
"""
    bitmask(ri::ResultIterator)

The mask selecting which start solutions `ri` tracks.
"""
bitmask(ri::ResultIterator)::BitVector = copy(ri.mask)

"""
    bitmask(f, ri::ResultIterator)

Track every selected path and record `f(path_result)` as a `BitVector` over the
selected paths.
"""
bitmask(f, ri::ResultIterator)::BitVector = BitVector(map(f, ri))

"""
    bitmask_filter(f, ri::ResultIterator)

Return a new [`ResultIterator`](@ref) restricted to the paths for which
`f(path_result)` is `true`. The paths are tracked once here to evaluate `f`; the
returned iterator tracks the surviving ones again when iterated.
"""
function bitmask_filter(f, ri::ResultIterator)::ResultIterator
    mask = copy(ri.mask)
    selected = findall(mask)
    keep = bitmask(f, ri)
    for (j, i) in enumerate(selected)
        mask[i] = keep[j]
    end
    return ResultIterator(ri.cache, mask)
end

"""
    Result(ri::ResultIterator)

Track every selected path and assemble a full [`Result`](@ref), including
solution clustering and (for an overdetermined system) excess-solution
reclassification.
"""
function Result(ri::ResultIterator)::Result
    prs = collect(ri)
    return _finalize_result(prs, length(prs), seed(ri), _excess_checker(ri.cache))
end

_excess_checker(cache::SolveCache) = cache.excess_checker
_excess_checker(cache::PolyhedralSolveCache) = cache.excess_checker
_excess_checker(::WorkerSolveCache) = nothing

# ── Entry points ───────────────────────────────────────────────────────────

"""
    result_iterator(F::System, alg = TotalDegree()) -> ResultIterator
    result_iterator(F::System, L::LinearSubspace, alg = TotalDegree())
    result_iterator(F::System, starts, p_start, p_target, alg = Continuation())
    result_iterator(F::System, starts, L_start::LinearSubspace,
                    L_target::LinearSubspace, alg = Continuation())

Build a [`ResultIterator`](@ref) for the same problems [`solve`](@ref) accepts,
tracking paths lazily instead of all at once. Tracking is serial, so the
algorithm's `show_progress` is ignored.

A `ResultIterator` may be passed as the start solutions of another `solve` or
`result_iterator`; its successful endpoints are used.

# Example
```julia
@polyvar x y
F = System([x^2 + y^2 - 5])
L = rand_subspace(2; codim = 1)
ri = result_iterator(F, L)
first(ri)                 # tracks exactly one path
real_paths = bitmask_filter(is_real, ri)
```
"""
result_iterator(F::System, alg::TotalDegree = TotalDegree())::ResultIterator =
    ResultIterator(CommonSolve.init(F, _quiet(alg), Serial()))

result_iterator(F::System, alg::Polyhedral)::ResultIterator =
    ResultIterator(CommonSolve.init(F, _quiet(alg), Serial()))

result_iterator(
    F::System, L::LinearSubspace, alg::TotalDegree = TotalDegree(),
)::ResultIterator = ResultIterator(CommonSolve.init(F, L, _quiet(alg), Serial()))

result_iterator(
    F::System, L::LinearSubspace, alg::Polyhedral,
)::ResultIterator = ResultIterator(CommonSolve.init(F, L, _quiet(alg), Serial()))

result_iterator(
    F::System, starts::StartsLike, p_start::AbstractVector{<:Number},
    p_target::AbstractVector{<:Number}, alg::Continuation = Continuation(),
)::ResultIterator = ResultIterator(
    CommonSolve.init(F, starts, p_start, p_target, _quiet(alg), Serial()),
)

result_iterator(
    F::System, starts::StartsLike, L_start::LinearSubspace,
    L_target::LinearSubspace, alg::Continuation = Continuation(),
)::ResultIterator = ResultIterator(
    CommonSolve.init(F, starts, L_start, L_target, _quiet(alg), Serial()),
)

# A `ResultIterator` used as start solutions contributes every successful
# endpoint, tracked on the spot. Singular endpoints are kept, unlike
# `_start_points(::Result)`, which goes through `solutions` (nonsingular only).
_start_points(ri::ResultIterator)::Vector{Vector{ComplexF64}} =
    [solution(pr) for pr in ri if is_success(pr)]
