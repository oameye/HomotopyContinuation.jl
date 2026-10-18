## Tracking every start solution of a solve cache: serially, on threads, or (with
## the Distributed extension) on worker processes.

# ── Work units ──────────────────────────────────────────────────────────────
#
# A work unit is the whole path-tracking problem as plain data: `builder()` makes
# one worker state per task and `work(state, i)` tracks path `i` with it. Every
# executor runs the same unit, and the Distributed extension ships it to each
# worker process once.

struct TrackWork{B}
    builder::B
    starts::Vector{Vector{ComplexF64}}
end

(w::TrackWork)(state, i::Int)::PathResult = _track_path!(state, w.starts[i], i)

struct PolyhedralWork{B}
    builder::B
    starts::Vector{Tuple{MixedSubdivisions.MixedCell, Vector{ComplexF64}}}
    support::Vector{Matrix{Int32}}
    lifting::Vector{Vector{Int32}}
end

function (w::PolyhedralWork)(state, i::Int)::PathResult
    cell, x₀ = w.starts[i]
    return _track_polyhedral_path!(state, w.support, w.lifting, cell, x₀, i)
end

"""
    PathSolveCache{E}

The solve caches that track a fixed list of start solutions with executor `E`.
Each holds a `builder` for per-task worker states, the `worker` state the serial
loop and the lazy [`ResultIterator`](@ref) use, and `path_work(cache)` names the
work unit.
"""
const PathSolveCache{E} = Union{SolveCache{E}, PolyhedralSolveCache{E}, WorkerSolveCache{E}}

path_work(cache::Union{SolveCache, WorkerSolveCache}) =
    TrackWork(cache.builder, cache.start_solutions)
path_work(cache::PolyhedralSolveCache) = PolyhedralWork(
    cache.builder, cache.start_solutions, cache.support, cache.lifting,
)

_excess_checkers(cache::Union{SolveCache, PolyhedralSolveCache})::ExcessCheckers =
    cache.excess_checkers
# The subspace routes never square up, so they carry nothing to re-check.
_excess_checkers(::WorkerSolveCache)::ExcessCheckers = ExcessCheckers()

# ── Loops ───────────────────────────────────────────────────────────────────
#
# `track_paths(exec, worker, work, n, progress, stop)` tracks paths `1:n` and
# returns the results of the paths it ran, in path order. `progress` is `nothing`
# or a progress meter; `stop` ends the run after the first successful path it
# accepts.

function track_paths(
        ::Serial, worker, work, n::Int, progress, stop::EarlyStop,
    )::Vector{PathResult}
    path_results = PathResult[]
    sizehint!(path_results, n)
    stats = ProgressStats()
    for k in 1:n
        pr = work(worker, k)
        push!(path_results, pr)
        update_progress!(progress, k, stats, pr)
        is_success(pr) && stop(pr) && break
    end
    return path_results
end

# Every task builds its own worker state, so the cache's `worker` is not used.
function track_paths(
        exec::Threaded, _, work, n::Int, progress, stop::EarlyStop,
    )::Vector{PathResult}
    nt = exec.ntasks
    results = Vector{PathResult}(undef, n)
    stats = ProgressStats()
    counter = Threads.Atomic{Int}(0)
    plock = ReentrantLock()
    stopped = Threads.Atomic{Bool}(false)

    # `@tasks` cannot `break`, so a stopped run skips the remaining iterations and
    # leaves their slots unassigned.
    @tasks for i in 1:n
        @set ntasks = nt
        @local ws = work.builder()
        if !stopped[]
            pr = work(ws, i)
            results[i] = pr
            if progress !== nothing
                k = Threads.atomic_add!(counter, 1) + 1
                @lock plock update_progress!(progress, k, stats, pr)
            end
            is_success(pr) && stop(pr) && (stopped[] = true)
        end
    end
    return _assigned_results(results)
end

# The Distributed extension adds the method for the work units above.
function track_paths(
        ::DistributedExecutor, worker, work, n::Int, progress, stop::EarlyStop,
    )
    Base.@nospecialize worker work progress
    return _distributed_ext_missing()
end

function track_all_paths(cache::PathSolveCache, progress)::Result
    prs = track_paths(
        cache.executor, cache.worker, path_work(cache),
        length(cache.start_solutions), progress, cache.early_stop,
    )
    # `tracked_paths` is what ran, which is what `nfailed` is derived from.
    return _finalize_result(prs, length(prs), cache.seed, _excess_checkers(cache))
end

# ── CommonSolve.solve! ──────────────────────────────────────────────────────

# `solve!` has one other method, CommonSolve's precompile iterator returning
# `nothing`, so a caller whose cache type is not inferred names it as a
# `PathSolveCache` to keep `solve!` inferred as `Result`.
#
# The progress variant is chosen behind `Base.inferencebarrier`, so a `solve` call
# compiles only the one it runs.
function CommonSolve.solve!(cache::PathSolveCache)::Result
    runner = cache.show_progress ? track_all_paths_with_progress : track_all_paths
    return _dispatch_solve_policy(Base.inferencebarrier(runner), cache)
end

@noinline function _dispatch_solve_policy(solver::Function, cache)::Result
    Base.@nospecialize solver cache
    return solver(cache)
end

@noinline track_all_paths(cache::PathSolveCache)::Result = track_all_paths(cache, nothing)
@noinline track_all_paths_with_progress(cache::PathSolveCache)::Result =
    track_all_paths(cache, make_progress(length(cache.start_solutions)))
