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
    j = findnext(ri.mask, i)
    j === nothing && return nothing
    return (_path_result(ri.cache, j), j + 1)
end

Base.firstindex(::ResultIterator)::Int = 1
Base.lastindex(ri::ResultIterator)::Int = length(ri)

"""
    ri[k]

Track the `k`-th path `ri` selects, and only that one.
"""
function Base.getindex(ri::ResultIterator, k::Int)::PathResult
    1 <= k <= length(ri) || throw(BoundsError(ri, k))
    i = 0
    for _ in 1:k
        i = findnext(ri.mask, i + 1)::Int
    end
    return _path_result(ri.cache, i)
end

function Base.show(io::IO, ri::ResultIterator)
    print(
        io, "ResultIterator over ", length(ri), " of ",
        length(ri.cache.start_solutions), " start solutions",
    )
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

The start solutions of the underlying solve, including the ones `ri` does not
track. Fresh vectors, so writing to one cannot change what a later `collect`
tracks from.
"""
start_solutions(ri::ResultIterator)::Vector{Vector{ComplexF64}} =
    _start_points(ri.cache.start_solutions)

"""
    selection(ri::ResultIterator)

The paths `ri` tracks, as a `BitVector` with one entry per start solution.
"""
selection(ri::ResultIterator)::BitVector = copy(ri.mask)

"""
    selection(f, ri::ResultIterator)

Track every path `ri` selects and record `f(path_result)` as a `BitVector` with
one entry per start solution, `false` for every path `ri` does not track. Pass
it to [`restrict`](@ref) to replay only the paths it selects; use
[`filter`](@ref) instead when the results themselves are wanted.
"""
function selection(f, ri::ResultIterator)::BitVector
    mask = falses(length(ri.mask))
    for i in eachindex(ri.mask)
        ri.mask[i] || continue
        mask[i] = f(_path_result(ri.cache, i))::Bool
    end
    return mask
end

"""
    restrict(ri::ResultIterator, mask::BitVector)

A [`ResultIterator`](@ref) over the paths `mask` selects, which has one entry
per start solution of `ri`. Nothing is tracked here, and a path `ri` already
skips stays skipped; `mask` usually comes from [`selection`](@ref).
"""
function restrict(ri::ResultIterator, mask::BitVector)::ResultIterator
    length(mask) == length(ri.mask) || throw(
        ArgumentError(
            string(
                "the mask has length ", length(mask), ", but the solve has ",
                length(ri.mask), " start solution(s)",
            ),
        ),
    )
    return ResultIterator(ri.cache, mask .& ri.mask)
end

"""
    filter(f, ri::ResultIterator)

Track every path `ri` selects and keep the results with `f(path_result)`. Use
`Iterators.filter` to keep the tracking lazy, or [`selection`](@ref) to record
which paths passed without keeping their results.
"""
Base.filter(f, ri::ResultIterator)::Vector{PathResult} =
    collect(Iterators.filter(f, ri))

"""
    trace(ri::ResultIterator)

The coordinate-wise sum of the solutions `ri` selects, accumulated one path at a
time so the solutions are never all held at once. A path that does not end at a
finite point contributes nothing.

For a witness set moving along a pencil of parallel slices the trace is affine in
the slice parameter, which is what the trace test exploits.
"""
function trace(ri::ResultIterator)::Vector{ComplexF64}
    t = ComplexF64[]
    started = false
    for r in ri
        if !started
            t = zeros(ComplexF64, length(solution(r)))
            started = true
        end
        isfinite(r) && (t .+= solution(r))
    end
    return t
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
    result_iterator(G::System, F::System, starts, alg = Continuation())
    result_iterator(H::AbstractHomotopy, starts, alg = Continuation())

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
first(ri)                  # tracks exactly one path
real_paths = filter(is_real, ri)

# Record which paths are worth tracking, then replay only those.
keep = selection(is_success, ri)
restrict(ri, keep)
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
    G::CloneableSystem, F::CloneableSystem, starts::StartsLike,
    alg::Continuation = Continuation(),
)::ResultIterator = ResultIterator(
    CommonSolve.init(G, F, starts, _quiet(alg), Serial()),
)

result_iterator(
    F::System, starts::StartsLike, L_start::LinearSubspace,
    L_target::LinearSubspace, alg::Continuation = Continuation(),
)::ResultIterator = ResultIterator(
    CommonSolve.init(F, starts, L_start, L_target, _quiet(alg), Serial()),
)

result_iterator(
    H::HomotopyLike, starts::StartsLike, alg::Continuation = Continuation(),
)::ResultIterator = ResultIterator(
    CommonSolve.init(H, starts, _quiet(alg), Serial()),
)

# A `ResultIterator` used as start solutions contributes every successful
# endpoint, tracked on the spot. Singular endpoints are kept, unlike
# `_start_points(::Result)`, which goes through `solutions` (nonsingular only).
_start_points(ri::ResultIterator)::Vector{Vector{ComplexF64}} =
    [solution(pr) for pr in ri if is_success(pr)]
