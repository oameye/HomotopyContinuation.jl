## The sweep work unit, progress reporters, and distributed path tracking.

# ── Sweep work unit ─────────────────────────────────────────────────────────
#
# The solve routes use the work units in `track_paths.jl`; a sweep adds the
# target to the path index.

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

# ── Every path of a solve cache ─────────────────────────────────────────────

function HCN.track_paths(
        exec::HCN.DistributedExecutor, _, work::Union{HCN.TrackWork, HCN.PolyhedralWork},
        n::Int, progress::P, stop::HCN.EarlyStop,
    )::Vector{HCN.PathResult} where {P}
    report = progress === nothing ? NoReport() : PathReport(progress)
    return HCN._assigned_results(_distributed_map(exec, work, n, report, stop))
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
    return [HCN._finalize_result(prs[k], np, seed, HCN.ExcessCheckers()) for k in 1:n_targets]
end
