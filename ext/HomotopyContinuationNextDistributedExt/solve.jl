## Per-route work units, progress reporters, and the `solve!` dispatch.

# ── Work units ──────────────────────────────────────────────────────────────
#
# A work unit is the whole problem as plain data: a builder that produces one
# per-task state, plus whatever the per-path body reads. It ships once per
# process and is called as `work(state, i)` for the global path index `i`.

struct TrackWork{B}
    builder::B
    starts::Vector{Vector{ComplexF64}}
end

(w::TrackWork)(state, i::Int)::HCN.PathResult =
    HCN._track_path!(state, w.starts[i], i)

struct PolyhedralWork{B}
    builder::B
    starts::Vector{Tuple{MixedSubdivisions.MixedCell, Vector{ComplexF64}}}
    support::Vector{Matrix{Int32}}
    lifting::Vector{Vector{Int32}}
end

function (w::PolyhedralWork)(state, i::Int)::HCN.PathResult
    cell, x₀ = w.starts[i]
    return HCN._track_polyhedral_path!(state, w.support, w.lifting, cell, x₀, i)
end

# A sweep walks the (target, path) product in target-major order, so a task
# retargets only when it crosses a target boundary. The state has to remember
# which target it currently holds, which the tracking state itself does not.
mutable struct SweepState{W}
    const worker::W
    targeted::Int
end

struct SweepBuilder{B}
    builder::B
end

(b::SweepBuilder)() = SweepState(b.builder(), 0)

struct SweepWork{B, Q}
    builder::SweepBuilder{B}
    starts::Vector{Vector{ComplexF64}}
    qs::Vector{Q}
    npaths::Int
end

function (w::SweepWork)(state::SweepState, f::Int)::HCN.PathResult
    k, i = fldmod1(f, w.npaths)
    if state.targeted != k
        HCN._retarget!(state.worker, w.qs[k])
        state.targeted = k
    end
    return HCN._track_path!(state.worker, w.starts[i], i)
end

# ── Progress reporters ──────────────────────────────────────────────────────
#
# Called once per received batch, only from the driver's collector loop, so the
# counters need no locking.

struct NoReport end

(::NoReport)(::UnitRange{Int}, ::Vector{HCN.PathResult})::Nothing = nothing

mutable struct PathReport{P}
    const progress::P
    const stats::HCN.ProgressStats
    tracked::Int
end

PathReport(progress::P) where {P} = PathReport{P}(progress, HCN.ProgressStats(), 0)

function (r::PathReport)(::UnitRange{Int}, prs::Vector{HCN.PathResult})::Nothing
    for pr in prs
        r.tracked += 1
        HCN.update_progress!(r.progress, r.tracked, r.stats, pr)
    end
    return nothing
end

# A target's paths can straddle a batch boundary, so no single batch is
# guaranteed to close it; count down per target and report a target as solved by
# whichever batch takes its last path.
mutable struct SweepReport{P}
    const progress::P
    const remaining::Vector{Int}
    const npaths::Int
    tracked::Int
    solved::Int
end

SweepReport(progress::P, n_targets::Int, npaths::Int) where {P} =
    SweepReport{P}(progress, fill(npaths, n_targets), npaths, 0, 0)

function (r::SweepReport)(batch::UnitRange{Int}, ::Vector{HCN.PathResult})::Nothing
    for f in batch
        k = fld1(f, r.npaths)
        r.tracked += 1
        r.remaining[k] -= 1
        iszero(r.remaining[k]) && (r.solved += 1)
    end
    HCN.update_many_progress!(r.progress, r.solved, r.tracked)
    return nothing
end

# ── Total degree and parameter homotopy ─────────────────────────────────────

function HCN._distributed_solve!(
        cache::HCN.SolveCache{HCN.DistributedExecutor},
    )::HCN.Result
    solver = cache.show_progress ?
        _solve_total_degree_distributed_with_progress :
        _solve_total_degree_distributed_without_progress
    solver = Base.inferencebarrier(solver)
    return HCN._dispatch_solve_policy(solver, cache)
end

@noinline _solve_total_degree_distributed_without_progress(
    cache::HCN.SolveCache{HCN.DistributedExecutor},
) = _solve_total_degree_distributed(cache, NoReport())
@noinline _solve_total_degree_distributed_with_progress(
    cache::HCN.SolveCache{HCN.DistributedExecutor},
) = _solve_total_degree_distributed(
    cache, PathReport(HCN.make_progress(length(cache.start_solutions), true)),
)

function _solve_total_degree_distributed(
        cache::HCN.SolveCache{HCN.DistributedExecutor}, report::R,
    )::HCN.Result where {R}
    starts = cache.start_solutions
    n_paths = length(starts)
    results = _distributed_map(
        cache.executor, TrackWork(cache.builder, starts), n_paths, report,
    )
    return HCN._finalize_result(
        results, n_paths, cache.seed, cache.excess_checker,
    )
end

# ── Two-phase polyhedral ────────────────────────────────────────────────────

function HCN._distributed_solve!(
        cache::HCN.PolyhedralSolveCache{HCN.DistributedExecutor},
    )::HCN.Result
    solver = cache.show_progress ?
        _solve_polyhedral_distributed_with_progress :
        _solve_polyhedral_distributed_without_progress
    solver = Base.inferencebarrier(solver)
    return HCN._dispatch_solve_policy(solver, cache)
end

@noinline _solve_polyhedral_distributed_without_progress(
    cache::HCN.PolyhedralSolveCache{HCN.DistributedExecutor},
) = _solve_polyhedral_distributed(cache, NoReport())
@noinline _solve_polyhedral_distributed_with_progress(
    cache::HCN.PolyhedralSolveCache{HCN.DistributedExecutor},
) = _solve_polyhedral_distributed(
    cache, PathReport(HCN.make_progress(length(cache.start_solutions), true)),
)

function _solve_polyhedral_distributed(
        cache::HCN.PolyhedralSolveCache{HCN.DistributedExecutor}, report::R,
    )::HCN.Result where {R}
    starts = cache.start_solutions
    n_paths = length(starts)
    work = PolyhedralWork(cache.builder, starts, cache.support, cache.lifting)
    results = _distributed_map(cache.executor, work, n_paths, report)
    return HCN._finalize_result(
        results, n_paths, cache.seed, cache.excess_checker,
    )
end

# ── Subspace to subspace, and retargeted parameter homotopies ───────────────

function HCN._distributed_solve!(
        cache::HCN.WorkerSolveCache{HCN.DistributedExecutor},
    )::HCN.Result
    solver = cache.show_progress ?
        _solve_worker_distributed_with_progress :
        _solve_worker_distributed_without_progress
    solver = Base.inferencebarrier(solver)
    return HCN._dispatch_solve_policy(solver, cache)
end

@noinline _solve_worker_distributed_without_progress(
    cache::HCN.WorkerSolveCache{HCN.DistributedExecutor},
) = _solve_worker_distributed(cache, NoReport())
@noinline _solve_worker_distributed_with_progress(
    cache::HCN.WorkerSolveCache{HCN.DistributedExecutor},
) = _solve_worker_distributed(
    cache, PathReport(HCN.make_progress(length(cache.start_solutions), true)),
)

function _solve_worker_distributed(
        cache::HCN.WorkerSolveCache{HCN.DistributedExecutor}, report::R,
    )::HCN.Result where {R}
    starts = cache.start_solutions
    n_paths = length(starts)
    results = _distributed_map(
        cache.executor, TrackWork(cache.builder, starts), n_paths, report,
    )
    return HCN._finalize_result(results, n_paths, cache.seed, nothing)
end

# ── Many-target sweeps ──────────────────────────────────────────────────────

# As in the threaded sweep, every target is tracked before any is transformed:
# it is the (target, path) product that gets batched.
function HCN._distributed_sweep_entries(
        cache::HCN.WorkerSolveCache{HCN.DistributedExecutor},
        targets::AbstractVector, idxs::Vector, first_q,
        transform_result::TR, transform_parameters::TP, progress,
    ) where {TR, TP}
    qs = [
        k == 1 ? first_q : transform_parameters(targets[idxs[k]])
            for k in eachindex(idxs)
    ]
    results = _sweep_results_distributed(cache, qs, progress)
    out = HCN._entry_vector(
        transform_result(results[1], targets[idxs[1]]), length(results),
    )
    for k in 2:length(results)
        out[k] = transform_result(results[k], targets[idxs[k]])
    end
    return out
end

function _sweep_results_distributed(
        cache::HCN.WorkerSolveCache{HCN.DistributedExecutor},
        qs::Vector{Q}, progress,
    )::Vector{HCN.Result} where {Q}
    starts = cache.start_solutions
    seed = cache.seed
    np = length(starts)
    n_targets = length(qs)

    work = SweepWork(SweepBuilder(cache.builder), starts, qs, np)
    report = progress === nothing ? NoReport() :
        SweepReport(progress, n_targets, np)
    flat = _distributed_map(cache.executor, work, n_targets * np, report)

    prs = [Vector{HCN.PathResult}(undef, np) for _ in 1:n_targets]
    for f in eachindex(flat)
        k, i = fldmod1(f, np)
        prs[k][i] = flat[f]
    end
    return [HCN._finalize_result(prs[k], np, seed, nothing) for k in 1:n_targets]
end
