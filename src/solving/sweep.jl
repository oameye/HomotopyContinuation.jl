## Many-target sweeps: solve the same start solutions against a sequence of
## target parameters or target subspaces.
#
# One homotopy is built and retargeted per target (`target_parameters!` through
# the concrete handle stored in the worker state), so N targets cost one
# construction, not N. Threading runs over the (target, path) product so that a
# sweep with fewer targets than tasks still saturates the machine; each task owns
# its own worker state and retargets it when it crosses a target boundary.
#
# `transform_parameters(q)` and `transform_result(result, q)` are applied exactly
# once per target: the caller passes the already-transformed first target in as
# `first_q`. `transform_result` receives the untransformed `q`.

# `target_parameters!` copies into a fixed-length buffer, so a short target would
# silently leave stale values behind; check the length on every retarget.
function retarget!(
        ws::AmbientWorkerState{ParameterHomotopy}, q::AbstractVector{<:Number},
    )::Nothing
    np = length(ws.homotopy.target_p)
    length(q) == np || throw(
        ArgumentError(
            "Every target must have length $np (the number of parameters), got " *
                "$(length(q)).",
        ),
    )
    target_parameters!(ws.homotopy, q)
    return nothing
end
# The start subspace of the homotopy being retargeted; for a projective extrinsic
# problem the subspaces live one wrapper deeper.
subspace_start(H::SubspaceHomotopy)::LinearSubspace{ComplexF64} = H.start
subspace_start(H::AffineChartHomotopy{<:SubspaceHomotopy})::LinearSubspace{ComplexF64} =
    H.homotopy.start

# Every target is retargeted into the same homotopy, so each one must sit in the
# same Grassmannian as the start subspace. A mismatch would otherwise surface as a
# `BoundsError` inside the geodesic, or as a silently different problem.
function retarget_subspace!(H::AbstractHomotopy, q::LinearSubspace)::Nothing
    V = subspace_start(H)
    dim(V) == dim(q) && ambient_dim(V) == ambient_dim(q) || throw(
        ArgumentError(
            "The target subspace has dimension $(dim(q)) in ambient dimension " *
                "$(ambient_dim(q)); expected $(dim(V)) in $(ambient_dim(V)).",
        ),
    )
    target_parameters!(H, q)
    return nothing
end

retarget!(ws::Union{AmbientWorkerState, IntrinsicWorkerState}, q::LinearSubspace)::Nothing =
    retarget_subspace!(ws.homotopy, q)

retarget!(ws::PathWorker, q)::Nothing = retarget!(ws._inner[], q)::Nothing

# Retarget and track every start solution once.
function sweep_one!(cache::WorkerSolveCache{Serial}, q)::Result
    retarget!(cache.worker, q)
    prs = track_paths(
        cache.executor, cache.worker, path_work(cache),
        length(cache.start_solutions), nothing, NEVER_STOP,
    )
    return finalize_result(prs, length(prs), cache.seed, ExcessCheckers())
end

# ── Per-target Results ─────────────────────────────────────────────────────

# Threading runs over the (target, path) product, not over targets alone: with
# fewer targets than tasks, per-target threading would strand every path of a
# target on a single task and a one-target sweep would get no parallelism at all.
# Chunks are contiguous in target-major order, so a task retargets only when it
# crosses a target boundary: `n_targets + ntasks` retargets in total rather than
# one per (task, target) pair.
#
# `qs` is built outside the tasks, so a user closure stays single-threaded and a
# randomized one consumes the RNG in target order.
function sweep_results_threaded(
        cache::WorkerSolveCache{Threaded}, qs::AbstractVector, progress,
    )::Vector{Result}
    nt = cache.executor.ntasks
    starts = cache.start_solutions
    seed = cache.seed
    np = length(starts)
    n_targets = length(qs)

    prs = [Vector{PathResult}(undef, np) for _ in 1:n_targets]
    # A target's paths can straddle a chunk boundary, so no single task is
    # guaranteed to close it; count down per target and report the target as
    # solved by whichever task takes the last path.
    remaining = progress === nothing ? Threads.Atomic{Int}[] :
        [Threads.Atomic{Int}(np) for _ in 1:n_targets]
    solved = Threads.Atomic{Int}(0)
    tracked = Threads.Atomic{Int}(0)
    plock = ReentrantLock()

    @tasks for f in 1:(n_targets * np)
        @set ntasks = nt
        @local begin
            ws = cache.builder()
            targeted = Ref(0)
        end
        # Target-major: `f` walks all paths of target 1, then of target 2, ...
        k, i = fldmod1(f, np)
        if targeted[] != k
            retarget!(ws, qs[k])
            targeted[] = k
        end
        prs[k][i] = track_path!(ws, starts[i], i)
        if progress !== nothing
            n_done = Threads.atomic_add!(tracked, 1) + 1
            if Threads.atomic_sub!(remaining[k], 1) == 1
                Threads.atomic_add!(solved, 1)
            end
            @lock plock update_many_progress!(progress, solved[], n_done)
        end
    end

    return [finalize_result(prs[k], np, seed, ExcessCheckers()) for k in 1:n_targets]
end

# ── Transformed entries: one per target ────────────────────────────────────

# The element type comes from the first entry, so `transform_result` may return
# anything.
function entry_vector(first_entry::T, n::Int)::Vector{T} where {T}
    out = Vector{T}(undef, n)
    out[1] = first_entry
    return out
end

# Serial: each target is transformed before the next one is retargeted, so only
# the transformed entries are ever held.
function sweep_entries(
        cache::WorkerSolveCache{Serial}, targets::AbstractVector, idxs::Vector,
        first_q, transform_result::TR, transform_parameters::TP, progress,
    ) where {TR, TP}
    np = length(cache.start_solutions)
    out = entry_vector(
        transform_result(sweep_one!(cache, first_q), targets[idxs[1]]), length(idxs),
    )
    update_many_progress!(progress, 1, np)
    for k in 2:length(idxs)
        q = targets[idxs[k]]
        out[k] = transform_result(sweep_one!(cache, transform_parameters(q)), q)
        update_many_progress!(progress, k, k * np)
    end
    return out
end

# Threaded: every target is tracked before any is transformed, since it is the
# (target, path) product that gets chunked.
function sweep_entries(
        cache::WorkerSolveCache{Threaded}, targets::AbstractVector, idxs::Vector,
        first_q, transform_result::TR, transform_parameters::TP, progress,
    ) where {TR, TP}
    qs = [
        k == 1 ? first_q : transform_parameters(targets[idxs[k]])
            for k in eachindex(idxs)
    ]
    results = sweep_results_threaded(cache, qs, progress)
    out = entry_vector(
        transform_result(results[1], targets[idxs[1]]), length(results),
    )
    for k in 2:length(results)
        out[k] = transform_result(results[k], targets[idxs[k]])
    end
    return out
end

sweep_entries(
    cache::WorkerSolveCache{DistributedExecutor}, targets::AbstractVector,
    idxs::Vector, first_q, transform_result::TR, transform_parameters::TP,
    progress,
) where {TR, TP} = distributed_sweep_entries(
    cache, targets, idxs, first_q, transform_result, transform_parameters,
    progress,
)

# ── Flattened output: the per-target arrays concatenated ────────────────────

function flatten_first(entry::Any)
    entry isa AbstractArray || throw(
        ArgumentError(
            "`flatten_results` requires each entry to be an array, " *
                "got $(typeof(entry)).",
        ),
    )
    return collect(entry)
end

"""
    flatten_results(entries)

Concatenate the array-valued entries a [`Sweep`](@ref) returned into one vector.
A sweep returns one entry per target; every entry must be an array, and an
`ArgumentError` is thrown otherwise.

```julia
entries = solve(F, S₀, p₀, targets, Sweep(; transform_result = (r, p) -> real_solutions(r)))
all_solutions = flatten_results(entries)
```
"""
function flatten_results(entries::AbstractVector)
    isempty(entries) && throw(ArgumentError("No entries to flatten."))
    out = flatten_first(entries[1])
    for k in 2:length(entries)
        append!(out, entries[k])
    end
    return out
end

function run_sweep(
        cache::WorkerSolveCache, targets::AbstractVector, first_q,
        transform_result::TR, transform_parameters::TP,
        show_progress::Bool,
    ) where {TR, TP}
    isempty(targets) && throw(ArgumentError("No targets given."))
    idx = collect(eachindex(targets))
    entries = if show_progress
        sweep_entries(
            cache, targets, idx, first_q, transform_result, transform_parameters,
            make_many_progress(length(targets)),
        )
    else
        sweep_entries(
            cache, targets, idx, first_q, transform_result, transform_parameters,
            nothing,
        )
    end
    return entries
end

# ── Parameter sweep ────────────────────────────────────────────────────────

function init_parameter_sweep(
        F::System, starts, first_target::AbstractVector{<:Number}, exec::E,
        p_start::AbstractVector{<:Number}, alg::Sweep,
    )::WorkerSolveCache{E} where {E <: AbstractExecutor}
    check_square_or_overdetermined(F)
    np = nparameters(F)
    np > 0 || throw(
        ArgumentError(
            "a parameter sweep requires a parametric system, but the " *
                "system has no parameters.",
        ),
    )
    length(p_start) == np || throw(
        ArgumentError(
            "p_start has length $(length(p_start)), but the system has $np parameter(s).",
        ),
    )
    length(first_target) == np || throw(
        ArgumentError("each target must have length $np, the number of parameters."),
    )
    builder = ParameterRetargetBuilder(
        F, Vector{ComplexF64}(p_start), Vector{ComplexF64}(first_target),
        algorithm_tracker_options(alg), algorithm_endgame_options(alg),
    )
    return WorkerSolveCache(
        exec, builder, builder(), start_points(starts), algorithm_seed(alg), algorithm_show_progress(alg),
    )
end

"""
    solve(F::System, starts, p_start, targets::AbstractVector, alg::Sweep, exec = Threaded())
    solve(F::System, starts, L_start::LinearSubspace,
          targets::AbstractVector{<:LinearSubspace}, alg::Sweep, exec = Threaded())

Track `starts` from one start end to every target in `targets`, retargeting a
single homotopy per target. Unlike a single-target [`solve`](@ref), which returns
one [`Result`](@ref), this returns a `Vector` with one entry per target.

See [`Sweep`](@ref) for how each entry is built and how `targets` is read.

# Example
```julia
@polyvar x y a b c
F = System([x^2 + y^2 - 1, a * x + b * y + c]; variables = [x, y], parameters = [a, b, c])
p₀ = randn(ComplexF64, 3)
S₀ = solutions(solve(fix_parameters(F, p₀)))
targets = [rand(3) for _ in 1:100]
solve(F, S₀, p₀, targets, Sweep(; transform_result = (r, p) -> real_solutions(r)))
```
"""
function solve(
        F::System, starts::StartsLike, p_start::AbstractVector{<:Number},
        targets::AbstractVector,
        alg::Sweep,
        exec::AbstractExecutor = Threaded(),
    )
    transform_result = alg.transform_result
    transform_parameters = alg.transform_parameters
    show_progress = algorithm_show_progress(alg)
    isempty(targets) && throw(ArgumentError("No targets given."))
    q_first = transform_parameters(first(targets))
    cache = init_parameter_sweep(F, starts, q_first, exec, p_start, alg)
    return run_sweep(
        cache, targets, q_first, transform_result, transform_parameters,
        show_progress,
    )
end

# ── Subspace sweep ─────────────────────────────────────────────────────────

function solve(
        F::System, starts::StartsLike, L_start::LinearSubspace,
        targets::AbstractVector,
        alg::Sweep,
        exec::AbstractExecutor = Threaded(),
    )
    intrinsic = use_intrinsic(alg.coords, L_start)
    transform_result = alg.transform_result
    transform_parameters = alg.transform_parameters
    show_progress = algorithm_show_progress(alg)
    isempty(targets) && throw(ArgumentError("No targets given."))
    L_first = transform_parameters(first(targets))
    G, points, chart, gamma = subspace_solve_setup(
        F, starts, L_start, L_first, algorithm_seed(alg),
    )
    cache = if intrinsic
        init_intrinsic_subspace(G, points, L_start, L_first, chart, gamma, exec, alg)
    else
        init_extrinsic_subspace(G, points, L_start, L_first, chart, gamma, exec, alg)
    end
    return run_sweep(
        cache, targets, L_first, transform_result, transform_parameters,
        show_progress,
    )
end

# Keywords are declared only so a bad `starts` reports itself rather than tripping
# an "unsupported keyword" MethodError first. Nothing is forwarded.
solve(
    ::System, starts, ::AbstractVector{<:Number}, ::AbstractVector, ::Sweep,
    ::AbstractExecutor = Threaded(),
) = bad_starts(starts)

solve(
    ::System, starts, ::LinearSubspace, ::AbstractVector, ::Sweep,
    ::AbstractExecutor = Threaded(),
) = bad_starts(starts)
