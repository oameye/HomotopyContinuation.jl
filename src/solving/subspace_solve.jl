## Tracking a set of start solutions from one linear subspace to another.
#
# Two regimes, chosen once at `init` (`dim(V) <= codim(V)`, or forced via
# `intrinsic`):
#   * extrinsic: `[F(x); A(t)x - a(t)]`, tracked in ambient coordinates, so
#     start points and endpoints need no conversion;
#   * intrinsic: `F(A(t)v + a(t))`, tracked in the subspace, so each ambient
#     start point is converted in at t = 1 and each endpoint converted back out
#     at the `t` it was reported at.
# Both regimes report ambient solutions, so `Result` clustering compares ambient
# points.

"""
    WorkerSolveCache{E, W, B}

Solve cache for the routes that keep a concrete homotopy handle beside the
tracker: `worker` tracks the paths, `builder` produces one independent worker
state per task, and the homotopy inside `worker` can be retargeted
(`target_parameters!`) between solves.
"""
struct WorkerSolveCache{E <: AbstractExecutor, W, B}
    executor::E
    builder::B
    worker::W
    start_solutions::Vector{Vector{ComplexF64}}
    seed::UInt32
    show_progress::Bool
end

# ── One path ───────────────────────────────────────────────────────────────

# `start_solution` is copied, not aliased: the caller's `starts` vector outlives
# the result and a sweep reuses it for every target.
function _track_path!(
        ws::AmbientWorkerState, x₀::Vector{ComplexF64}, k::Int,
    )::PathResult
    track!(ws.tracker, x₀)
    return PathResult(ws.tracker; path_number = k, start_solution = copy(x₀))
end

function _track_path!(
        ws::IntrinsicWorkerState, x₀::Vector{ComplexF64}, k::Int,
    )::PathResult
    intrinsic_coordinates!(ws.u, ws.homotopy, x₀, complex(1.0))
    track!(ws.tracker, ws.u)
    pr = PathResult(ws.tracker; path_number = k, start_solution = copy(x₀))
    return _to_ambient(pr, ws.homotopy)
end

# ── All paths ──────────────────────────────────────────────────────────────

function _track_all_serial(
        ws::RetargetWorkerState, starts::Vector{Vector{ComplexF64}},
        seed::UInt32, progress,
    )::Result
    n_paths = length(starts)
    path_results = PathResult[]
    sizehint!(path_results, n_paths)
    stats = ProgressStats()
    for (k, x₀) in enumerate(starts)
        pr = _track_path!(ws, x₀, k)
        push!(path_results, pr)
        update_progress!(progress, k, stats, pr)
    end
    return _finalize_result(path_results, n_paths, seed, nothing)
end

function _track_all_threaded(cache::WorkerSolveCache{Threaded}, progress)::Result
    nt = cache.executor.ntasks
    starts = cache.start_solutions
    n_paths = length(starts)
    results = Vector{PathResult}(undef, n_paths)

    stats = ProgressStats()
    counter = Threads.Atomic{Int}(0)
    plock = ReentrantLock()

    @tasks for i in eachindex(starts)
        @set ntasks = nt
        @local ws = cache.builder()
        results[i] = _track_path!(ws, starts[i], i)
        if progress !== nothing
            k = Threads.atomic_add!(counter, 1) + 1
            @lock plock update_progress!(progress, k, stats, results[i])
        end
    end

    return _finalize_result(results, n_paths, cache.seed, nothing)
end

# ── CommonSolve.solve! ─────────────────────────────────────────────────────

function CommonSolve.solve!(cache::WorkerSolveCache{Serial})::Result
    solver = cache.show_progress ?
        _solve_worker_serial_with_progress :
        _solve_worker_serial_without_progress
    solver = Base.inferencebarrier(solver)
    return _dispatch_solve_policy(solver, cache)
end

@noinline _solve_worker_serial_without_progress(cache::WorkerSolveCache{Serial}) =
    _track_all_serial(cache.worker, cache.start_solutions, cache.seed, nothing)
@noinline _solve_worker_serial_with_progress(cache::WorkerSolveCache{Serial}) =
    _track_all_serial(
    cache.worker, cache.start_solutions, cache.seed,
    make_progress(length(cache.start_solutions), true),
)

function CommonSolve.solve!(cache::WorkerSolveCache{Threaded})::Result
    solver = cache.show_progress ?
        _solve_worker_threaded_with_progress :
        _solve_worker_threaded_without_progress
    solver = Base.inferencebarrier(solver)
    return _dispatch_solve_policy(solver, cache)
end

@noinline _solve_worker_threaded_without_progress(cache::WorkerSolveCache{Threaded}) =
    _track_all_threaded(cache, nothing)
@noinline _solve_worker_threaded_with_progress(cache::WorkerSolveCache{Threaded}) =
    _track_all_threaded(cache, make_progress(length(cache.start_solutions), true))

CommonSolve.solve!(cache::WorkerSolveCache{DistributedExecutor})::Result =
    _distributed_solve!(cache)

# ── init: subspace to subspace ─────────────────────────────────────────────

# Both subspaces must live in the ambient space of `F` and have the same
# dimension: the geodesic connects two points of one Grassmannian. Without this
# check a mismatch surfaces as a `BoundsError` inside the geodesic (intrinsic) or
# as a silently different problem (extrinsic).
function _check_subspace_pair(
        F::System, V::LinearSubspace, W::LinearSubspace,
    )::Nothing
    n = nvariables(F)
    ambient_dim(V) == n && ambient_dim(W) == n || throw(
        ArgumentError(
            "The subspaces live in dimensions $(ambient_dim(V)) and " *
                "$(ambient_dim(W)), but the system has $n variables.",
        ),
    )
    dim(V) == dim(W) || throw(
        ArgumentError(
            "The start subspace has dimension $(dim(V)) and the target subspace " *
                "$(dim(W)); tracking between subspaces requires equal dimensions.",
        ),
    )
    return nothing
end

function _check_subspace_square(H::AbstractHomotopy, regime::String)::Nothing
    m, n = size(H)
    m == n || throw(
        ArgumentError(
            "The $regime subspace homotopy has $m equation(s) in $n unknown(s). " *
                "Tracking between subspaces requires a square homotopy; check the " *
                "dimension of the subspaces against the number of equations.",
        ),
    )
    return nothing
end

function _init_intrinsic_subspace(
        G::System, starts::Vector{Vector{ComplexF64}},
        L_start::LinearSubspace, L_target::LinearSubspace,
        chart::Vector{ComplexF64}, gamma::ComplexF64,
        exec::E, seed::UInt32, tracker_options::TrackerOptions,
        endgame_options::EndgameOptions, show_progress::Bool,
    ) where {E <: AbstractExecutor}
    builder = IntrinsicSubspaceBuilder(
        G, convert(LinearSubspace{ComplexF64}, L_start),
        convert(LinearSubspace{ComplexF64}, L_target),
        chart, gamma, tracker_options, endgame_options,
    )
    worker = builder()
    _check_subspace_square(worker.homotopy, "intrinsic")
    return WorkerSolveCache(exec, builder, worker, starts, seed, show_progress)
end

function _init_extrinsic_subspace(
        G::System, starts::Vector{Vector{ComplexF64}},
        L_start::LinearSubspace, L_target::LinearSubspace,
        chart::Vector{ComplexF64}, gamma::ComplexF64,
        exec::E, seed::UInt32, tracker_options::TrackerOptions,
        endgame_options::EndgameOptions, show_progress::Bool,
    ) where {E <: AbstractExecutor}
    V = convert(LinearSubspace{ComplexF64}, L_start)
    W = convert(LinearSubspace{ComplexF64}, L_target)
    if isempty(chart)
        builder = ExtrinsicSubspaceBuilder(
            G, V, W, gamma, tracker_options, endgame_options,
        )
        worker = builder()
        _check_subspace_square(worker.homotopy, "extrinsic")
        return WorkerSolveCache(exec, builder, worker, starts, seed, show_progress)
    end
    chart_builder = ChartExtrinsicSubspaceBuilder(
        G, V, W, chart, gamma, tracker_options, endgame_options,
    )
    chart_worker = chart_builder()
    _check_subspace_square(chart_worker.homotopy, "extrinsic")
    return WorkerSolveCache(
        exec, chart_builder, chart_worker, starts, seed, show_progress,
    )
end

# Fix parameters, materialize the start points, and place them on `chart` when
# the problem is projective (the chart row is part of the tracked system, so a
# projective representative off the chart is not a solution of it).
function _subspace_solve_setup(
        F::System, starts, L_start::LinearSubspace, L_target::LinearSubspace,
        seed::UInt32,
    )
    _check_parameter_free(F, "`solve(F, starts, L_start, L_target)`")
    G = F
    _check_subspace_pair(G, L_start, L_target)
    rng = Random.MersenneTwister(seed)
    gamma = _random_gamma(rng)
    points = _start_points(starts)
    projective = is_linear(L_start) && is_linear(L_target) && is_homogeneous(G)
    chart = projective ? randn(rng, ComplexF64, nvariables(G)) : ComplexF64[]
    if projective
        for x in points
            on_chart!(x, chart)
        end
    end
    return G, points, chart, gamma
end

function CommonSolve.init(
        F::System, starts, L_start::LinearSubspace, L_target::LinearSubspace,
        exec::AbstractExecutor = Threaded();
        intrinsic::Bool = _default_intrinsic(L_start),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = true,
    )
    G, points, chart, gamma = _subspace_solve_setup(
        F, starts, L_start, L_target, seed,
    )
    return if intrinsic
        _init_intrinsic_subspace(
            G, points, L_start, L_target, chart, gamma, exec, seed,
            tracker_options, endgame_options, show_progress,
        )
    else
        _init_extrinsic_subspace(
            G, points, L_start, L_target, chart, gamma, exec, seed,
            tracker_options, endgame_options, show_progress,
        )
    end
end

"""
    solve(F::System, starts, L_start::LinearSubspace, L_target::LinearSubspace,
          exec = Threaded(); intrinsic, options...)

Track the solutions `starts` of `V(F) ∩ L_start` to `V(F) ∩ L_target`. The start
points and the returned solutions are ambient, i.e. in the coordinates of `F`.

`intrinsic` chooses the tracking coordinates: `true` tracks inside the subspace
(`F(A(t)v + a(t))`), `false` in ambient space (`[F(x); A(t)x - a(t)]`). It
defaults to `dim(L_start) <= codim(L_start)`.

`F` must be parameter-free; fix the values first with [`fix_parameters`](@ref).

# Example
```julia
@polyvar x y
F = System([x^2 + y^2 - 5])
L₁ = rand_subspace(2; codim = 1)
L₂ = rand_subspace(2; codim = 1)
S₁ = solutions(solve(F, L₁))
result = solve(F, S₁, L₁, L₂)
```
"""
function solve(
        F::System, starts, L_start::LinearSubspace, L_target::LinearSubspace,
        exec::AbstractExecutor = Threaded();
        intrinsic::Bool = _default_intrinsic(L_start),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = true,
    )::Result
    return CommonSolve.solve!(
        CommonSolve.init(
            F, starts, L_start, L_target, exec;
            intrinsic = intrinsic,
            seed = seed, tracker_options = tracker_options,
            endgame_options = endgame_options, show_progress = show_progress,
        ),
    )
end
