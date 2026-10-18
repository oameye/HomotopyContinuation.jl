update_progress!(
    progress::Nothing, stats::MonodromyStatistics;
    queued::Int = 0, solutions::Int = 0, finish::Bool = false,
) = nothing
function update_progress!(
        progress::ProgressMeter.ProgressUnknown,
        stats::MonodromyStatistics;
        queued::Int,
        solutions::Int,
        finish::Bool = false,
    )
    if finish
        ProgressMeter.update!(progress, solutions)
        showvalues = make_showvalues(stats; queued = queued, solutions = solutions)
        ProgressMeter.finish!(progress; showvalues = showvalues)
    else
        # ProgressMeter exposes these forwarded properties as `Float64` at
        # runtime, but their inferred contracts are `Float64` and `Real`.
        # Narrow them here so this optional UI branch does not introduce
        # arithmetic dispatch into monodromy's compiler graph.
        tlast = progress.tlast::Float64
        dt = progress.dt::Float64
        time() > tlast + dt || return nothing
        showvalues = make_showvalues(stats; queued = queued, solutions = solutions)
        ProgressMeter.update!(progress, solutions, showvalues = showvalues)
    end
    return nothing
end

"""
    serial_monodromy_solve!(MS, results, seed, progress)

Run the monodromy loop generation and job-tracking cycle single-threaded.
"""
function serial_monodromy_solve!(
        MS::MonodromySolver,
        results::Vector{PathResult},
        seed::UInt32,
        # `nothing` when quiet; `update_progress!` dispatches, so specializing here
        # keeps both cases concrete.
        progress::P,
    )::MonodromyCode.T where {P}
    rng = Random.MersenneTwister(seed)
    queue = LoopTrackingJob[]
    ws = MS.workers[1]
    t₀ = time()
    stats = MS.statistics
    opts = MS.options
    is_subspace = ws.base isa LinearSubspace

    retcode = MonodromyCode.IN_PROGRESS
    while retcode == MonodromyCode.IN_PROGRESS
        loop_finished!(stats, length(results))

        if opts.loop_finished_callback(results)
            retcode = MonodromyCode.TERMINATED_CALLBACK
            break
        end
        if is_subspace && nloops(MS) > 0 && opts.trace_test &&
                trace_colinearity(MS) < opts.trace_test_tol
            retcode = MonodromyCode.SUCCESS
            break
        end
        if opts.target_solutions_count == typemax(Int) &&
                length(results) >= opts.min_solutions &&
                loops_no_change(stats, length(results)) >= opts.max_loops_no_progress
            retcode = MonodromyCode.HEURISTIC_STOP
            break
        end
        if length(results) == opts.target_solutions_count
            retcode = MonodromyCode.SUCCESS
            break
        end
        if nloops(MS) > 0 && opts.single_loop_per_start_solution
            retcode = MonodromyCode.SUCCESS
            break
        end

        add_loop!(MS, rng, results)
        reset_trace!(MS)
        # schedule all jobs on the fresh loop
        new_loop_id = nloops(MS)
        for i in 1:length(results)
            push!(queue, LoopTrackingJob(i, new_loop_id))
        end

        while !isempty(queue)
            job = popfirst!(queue)
            collect_trace = opts.trace_test && nloops(MS) == job.loop_id
            res = track_loop!(
                ws, loop(MS, job.loop_id), results[job.id], collect_trace, MS,
            )
            if is_success(res) && !res.singular
                loop_tracked!(stats)

                # 1) check whether the solution already exists
                candidate = certify_candidate(MS, res, 1)
                id, got_added, accepted = add_tracked_result!(
                    MS, res, length(results) + 1, candidate, 1,
                )

                if opts.permutations
                    add_permutation!(stats, job.loop_id, job.id, id)
                end

                if got_added
                    # 2) doesn't exist, so add to results
                    push!(results, accepted)

                    # 3) schedule on the same loop again
                    if !opts.single_loop_per_start_solution
                        push!(queue, LoopTrackingJob(id, job.loop_id))
                    end

                    # 4) schedule on other loops
                    if opts.reuse_loops == ReuseLoops.ALL
                        for k in 1:nloops(MS)
                            k != job.loop_id || continue
                            push!(queue, LoopTrackingJob(id, k))
                        end
                    elseif opts.reuse_loops == ReuseLoops.RANDOM && nloops(MS) >= 2
                        k = rand(rng, 2:nloops(MS))
                        if k <= job.loop_id
                            k -= 1
                        end
                        push!(queue, LoopTrackingJob(id, k))
                    end
                end
            else
                loop_failed!(stats)
                if opts.permutations
                    add_permutation!(stats, job.loop_id, job.id, 0)
                end
            end

            update_progress!(
                progress, stats;
                solutions = length(results), queued = length(queue),
            )

            if length(results) == opts.target_solutions_count &&
                    # only terminate after a completed loop to ensure that we
                    # collect proper permutation information
                    !opts.permutations
                retcode = MonodromyCode.SUCCESS
                break
            elseif time() - t₀ > opts.timeout
                retcode = MonodromyCode.TIMEOUT
                break
            end
        end
    end

    update_progress!(
        progress, stats;
        finish = true, solutions = length(results), queued = length(queue),
    )

    return retcode
end

################
## Entrypoint ##
################

# The last two default to what monodromy-as-a-subroutine wants; `Monodromy`
# passes both, since there they are the user's to set.
function _monodromy_solve!(
        MS::MonodromySolver{H, P},
        X::AbstractVector{<:AbstractVector},
        p::P,
        seed::UInt32,
        show_progress::Bool,
        executor::E,
        catch_interrupt::Bool = true,
        warning::Bool = false,
    )::MonodromyResult{P, P} where {H, P, E <: AbstractExecutor}
    runner = show_progress ?
        _monodromy_with_progress! : _monodromy_without_progress!
    # Keeping the two bodies out of one inferred union means a quiet solve does
    # not compile ProgressMeter. The executor splits the same way one level down,
    # where `_monodromy_solve_body!` specializes on it.
    runner = Base.inferencebarrier(runner)
    return _dispatch_monodromy_policy(
        runner, MS, X, p, seed, executor, catch_interrupt, warning,
    )
end

@noinline function _dispatch_monodromy_policy(
        runner::Function, MS::MonodromySolver{H, P},
        X::AbstractVector{<:AbstractVector}, p::P, seed::UInt32,
        executor::E, catch_interrupt::Bool, warning::Bool,
    )::MonodromyResult{P, P} where {H, P, E <: AbstractExecutor}
    Base.@nospecialize runner MS X p executor
    return runner(MS, X, p, seed, executor, catch_interrupt, warning)
end


function _make_monodromy_progress(MS::MonodromySolver)::ProgressMeter.ProgressUnknown
    desc = if MS.options.equivalence_classes
        "Solutions (modulo group action) found:"
    else
        "Solutions found:"
    end
    progress = ProgressMeter.ProgressUnknown(; dt = 0.4, desc = desc, output = stdout)
    progress.tlast += 0.3
    return progress
end

@noinline function _monodromy_without_progress!(
        MS::MonodromySolver{H, P}, X, p::P, seed::UInt32,
        executor::E, catch_interrupt::Bool, warning::Bool,
    )::MonodromyResult{P, P} where {H, P, E <: AbstractExecutor}
    return _monodromy_solve_body!(
        MS, X, p, seed, nothing, executor, catch_interrupt, warning,
    )
end

@noinline function _monodromy_with_progress!(
        MS::MonodromySolver{H, P}, X, p::P, seed::UInt32,
        executor::E, catch_interrupt::Bool, warning::Bool,
    )::MonodromyResult{P, P} where {H, P, E <: AbstractExecutor}
    return _monodromy_solve_body!(
        MS, X, p, seed, _make_monodromy_progress(MS), executor,
        catch_interrupt, warning,
    )
end


"""
    threaded_monodromy_solve!(MS, results, seed, progress, ntasks)

Multithreaded variant of [`serial_monodromy_solve!`](@ref): `ntasks` long-lived
worker tasks consuming a job channel, plus a coordinator task that generates loop
generations and waits for quiescence (all workers idle, channel empty, no job in
flight).
"""
function threaded_monodromy_solve!(
        MS::MonodromySolver,
        results::Vector{PathResult},
        seed::UInt32,
        progress::P,
        nthr::Int = Threads.nthreads(),
    )::MonodromyCode.T where {P}
    # `MersenneTwister` is not thread-safe, so the loop-generating coordinator and
    # every worker task get their own stream, all derived from `seed`.
    loop_rng = Random.MersenneTwister(seed)
    queue = Channel{LoopTrackingJob}(Inf)

    # Grow the worker states to one per task via the builder (never deepcopy), and
    # the certification caches with them, so no task grows either later.
    while length(MS.workers) < nthr
        push!(MS.workers, MS.builder())
    end
    _size_certified_caches!(MS, nthr)

    data_lock = MS.unique_points_lock
    t0 = time()
    retcode = Ref(MonodromyCode.IN_PROGRESS)
    stats = MS.statistics
    opts = MS.options
    is_subspace = MS.workers[1].base isa LinearSubspace
    target_count = opts.target_solutions_count
    notify_lock = ReentrantLock()
    cond_queue_emptied = Threads.Condition(notify_lock)
    workers_idle = fill(true, nthr)
    interrupted = Ref(false)
    queued = Ref(0)
    inflight_count = Threads.Atomic{Int}(0)  # in-flight counter to check termination
    # Number of jobs sitting in the channel, tracked manually because the
    # channel's own count (Base.n_avail) is not public API. Incremented before
    # every push!, decremented when a worker receives a job.
    queued_count = Threads.Atomic{Int}(0)
    # Torn-read-free mirror of length(results). `results` is mutated only under
    # data_lock, but the progress and heuristic checks below read the count
    # without holding it; reading this atomic instead avoids racing with push!.
    n_results = Threads.Atomic{Int}(length(results))
    enqueue! = job -> begin
        Threads.atomic_add!(queued_count, 1)
        return push!(queue, job)
    end

    # Quiescence is decided by two checks: all(workers_idle) and
    # inflight_count[] == 0. One is probably enough; it is safe to have both.
    try
        for tid in 1:nthr
            let ws = MS.workers[tid], tid = tid,
                    job_rng = Random.MersenneTwister(seed + UInt32(tid))
                Threads.@spawn begin
                    for job in queue
                        Threads.atomic_add!(queued_count, -1)
                        stop_queue = false
                        Base.@lock notify_lock begin
                            if interrupted[]
                                workers_idle[tid] = true
                                if all(workers_idle)
                                    notify(cond_queue_emptied)
                                end
                                stop_queue = true
                            else
                                workers_idle[tid] = false
                            end
                        end
                        if stop_queue
                            break
                        end
                        Threads.atomic_add!(inflight_count, 1)

                        try
                            start_res = Base.@lock data_lock results[job.id]
                            collect_trace = opts.trace_test && nloops(MS) == job.loop_id
                            res = track_loop!(
                                ws, loop(MS, job.loop_id), start_res, collect_trace, MS,
                            )

                            if is_success(res) && !res.singular
                                loop_tracked!(stats)

                                candidate = n_results[] < target_count ?
                                    certify_candidate(MS, res, tid) : NoCandidate()
                                got_added = false
                                id = 0
                                Base.@lock data_lock begin
                                    accepted = res
                                    if length(results) < target_count
                                        id, got_added, accepted = add_tracked_result!(
                                            MS, res, length(results) + 1, candidate, tid,
                                        )
                                        if opts.permutations
                                            add_permutation!(
                                                stats, job.loop_id, job.id, id,
                                            )
                                        end
                                    end
                                    if got_added
                                        # 2) doesn't exist, so add to results
                                        push!(results, accepted)
                                        Threads.atomic_add!(n_results, 1)
                                    end
                                end
                                if got_added
                                    # 3) schedule on the same loop again
                                    if !opts.single_loop_per_start_solution
                                        enqueue!(LoopTrackingJob(id, job.loop_id))
                                    end
                                    # 4) schedule on other loops
                                    if opts.reuse_loops == ReuseLoops.ALL
                                        for k in 1:nloops(MS)
                                            k != job.loop_id || continue
                                            enqueue!(LoopTrackingJob(id, k))
                                        end
                                    elseif opts.reuse_loops == ReuseLoops.RANDOM &&
                                            nloops(MS) >= 2
                                        k = rand(job_rng, 2:nloops(MS))
                                        if k <= job.loop_id
                                            k -= 1
                                        end
                                        enqueue!(LoopTrackingJob(id, k))
                                    end
                                end
                            else
                                loop_failed!(stats)
                                if opts.permutations
                                    Base.@lock data_lock add_permutation!(
                                        stats, job.loop_id, job.id, 0,
                                    )
                                end
                            end

                            update_progress!(
                                progress, stats;
                                solutions = n_results[],
                                queued = max(queued_count[], 0),
                            )

                            # mark worker as idle
                            Base.@lock notify_lock begin
                                workers_idle[tid] = true
                                # if the queue is empty, check whether all
                                # others are also waiting
                                if !isready(queue) && all(workers_idle)
                                    notify(cond_queue_emptied)
                                end
                            end

                            if n_results[] >=
                                    opts.target_solutions_count &&
                                    # only terminate after a completed loop to ensure
                                    # that we collect proper permutation information
                                    !opts.permutations
                                retcode[] = MonodromyCode.SUCCESS
                                Base.@lock notify_lock begin
                                    interrupted[] = true
                                end
                            elseif time() - t0 > opts.timeout
                                retcode[] = MonodromyCode.TIMEOUT
                                Base.@lock notify_lock begin
                                    interrupted[] = true
                                end
                            end
                        finally
                            Threads.atomic_add!(inflight_count, -1)
                            Base.@lock notify_lock begin
                                if !isready(queue) && inflight_count[] == 0
                                    notify(cond_queue_emptied)
                                end
                            end
                        end
                    end
                end
            end
        end

        t = Threads.@spawn begin
            Base.@lock notify_lock begin
                while true
                    if interrupted[]
                        break
                    end
                    loop_finished!(stats, n_results[])

                    if opts.loop_finished_callback(results)
                        retcode[] = MonodromyCode.TERMINATED_CALLBACK
                        break
                    end

                    if opts.target_solutions_count == typemax(Int) &&
                            n_results[] >= opts.min_solutions &&
                            loops_no_change(stats, n_results[]) >=
                            opts.max_loops_no_progress
                        retcode[] = MonodromyCode.HEURISTIC_STOP
                        break
                    end

                    if n_results[] >=
                            opts.target_solutions_count
                        retcode[] = MonodromyCode.SUCCESS
                        break
                    end

                    if is_subspace && nloops(MS) > 0 && opts.trace_test &&
                            trace_colinearity(MS) < opts.trace_test_tol
                        retcode[] = MonodromyCode.SUCCESS
                        break
                    end

                    add_loop!(MS, loop_rng, results)
                    reset_trace!(MS)
                    # schedule all jobs
                    new_loop_id = nloops(MS)
                    for i in 1:length(results)
                        enqueue!(LoopTrackingJob(i, new_loop_id))
                    end

                    wait(cond_queue_emptied)
                    if opts.single_loop_per_start_solution
                        retcode[] = MonodromyCode.SUCCESS
                        break
                    end
                    retcode[] == MonodromyCode.IN_PROGRESS || break
                end
            end
        end

        wait(t)
    catch e
        close(queue)
        interrupted[] = true
        rethrow(e)
    finally
        queued[] = max(queued_count[], 0)
        close(queue)
    end

    update_progress!(
        progress, stats;
        finish = true, solutions = length(results), queued = queued[],
    )

    return retcode[]
end

function _run_monodromy_loop!(
        ::Serial, MS::MonodromySolver, results::Vector{PathResult},
        seed::UInt32, progress,
    )::MonodromyCode.T
    return serial_monodromy_solve!(MS, results, seed, progress)
end

function _run_monodromy_loop!(
        exec::Threaded, MS::MonodromySolver, results::Vector{PathResult},
        seed::UInt32, progress,
    )::MonodromyCode.T
    return threaded_monodromy_solve!(MS, results, seed, progress, exec.ntasks)
end

function _run_monodromy_loop!(
        executor::DistributedExecutor, MS::MonodromySolver,
        results::Vector{PathResult}, seed::UInt32, progress,
    )::MonodromyCode.T
    return _distributed_monodromy_solve!(executor, MS, results, seed, progress)
end

function _monodromy_solve_body!(
        MS::MonodromySolver{H, P},
        X::AbstractVector{<:AbstractVector},
        p::P,
        seed::UInt32,
        progress,
        executor::E,
        catch_interrupt::Bool,
        warning::Bool,
    )::MonodromyResult{P, P} where {H, P, E <: AbstractExecutor}
    MS.statistics = MonodromyStatistics()
    empty!(MS.unique_points)
    _reset_certified!(MS)
    reset_trace!(MS)
    reset_loops!(MS)
    results = check_start_solutions!(MS, X)
    if isempty(results)
        if warning
            @warn "None of the provided solutions is a valid start solution (Newton's method did not converge)."
        end
        retcode = MonodromyCode.INVALID_STARTVALUE
    else
        try
            retcode = _run_monodromy_loop!(executor, MS, results, seed, progress)
        catch e
            if !catch_interrupt || !(
                    isa(e, InterruptException) ||
                        (isa(e, TaskFailedException) && isa(e.task.exception, InterruptException))
                )
                rethrow(e)
            end
            retcode = MonodromyCode.INTERRUPTED
        end
    end

    return MonodromyResult(
        retcode,
        results,
        p,
        MS.loops,
        MS.statistics,
        MS.options.equivalence_classes,
        MS.options.duplicate_check,
        seed,
        p isa LinearSubspace ? trace_colinearity(MS) : NaN,
    )
end


"""
    solve(F, alg::Monodromy, exec = Threaded())
    solve(F, sols, p, alg::Monodromy, exec = Threaded())
    solve(F, sols, L::LinearSubspace, alg::Monodromy, exec = Threaded())

Solve a polynomial system `F(x; p)` with specified parameters and initial
solutions `sols` by monodromy techniques. This makes loops in the parameter
space of `F` to find new solutions. If the parameters occur only *linearly* in
`F`, a start pair `(x₀, p₀)` can be computed automatically; in this case `sols`
and `p` can be omitted and the generated parameters can be obtained with
[`parameters`](@ref) from the [`MonodromyResult`](@ref).

With a [`LinearSubspace`](@ref) in place of `p` the system `[F(x); L(x)] = 0` is
solved instead. If `sols` and `L` are not provided it is necessary to give
`Monodromy`'s `dim` or `codim`, the expected (co)dimension of a component of
`V(F)`. See also [`with_linear_subspace_homotopy`](@ref) for the `intrinsic` option.

`exec` is [`Serial`](@ref), [`Threaded`](@ref) or [`DistributedExecutor`](@ref).
Only the loop tracking is handed out; loop generation, deduplication and the
trace test always run in the calling process. [`Threaded`](@ref) is the default
and the faster of the two on one machine, since a loop costs about as much to
hand to another process as to track.

## Options

* `catch_interrupt = true`: If true catches interruptions (e.g. issued by
  pressing Ctrl-C) and returns the partial result.
* `check_startsolutions = true`: If `true`, track each entry of `sols` at the
  base parameters and sort out any that fail to converge. If `false`, the
  provided solutions are refined but trusted (non-converged points are kept).
* `distance = InfNorm()`: The distance function used for [`UniquePoints`](@ref).
* `duplicate_check = DuplicateCheck.HEURISTIC`: How a new solution is recognized as
  one already found. `DuplicateCheck.HEURISTIC` deduplicates by distance through
  [`UniquePoints`](@ref). `DuplicateCheck.CERTIFIED` accepts a solution only if it
  certifies as distinct from every solution found so far, and discards an endpoint
  that certifies as neither; the accepted solutions are the certified interval
  midpoints, so [`solutions`](@ref) returns certified approximations, and
  [`ncertified_distinct`](@ref) and [`ndiscarded_uncertified`](@ref) report the
  counts. Needs `using HomotopyContinuationCertification`.
* `certification_max_precision = 256`: Maximal precision used when certifying,
  under `duplicate_check = DuplicateCheck.CERTIFIED`.
* `certification_refine_solution = true`: Whether to refine an endpoint with
  Newton before certifying it, under `duplicate_check = DuplicateCheck.CERTIFIED`.
* `loop_finished_callback = always_false`: A callback called with all current
  [`PathResult`](@ref)s after a loop is exhausted. Return `true` to stop.
* `equivalence_classes = true`: Only applies with group actions: consider two
  solutions equivalent when one maps to the other under the actions, and only
  track one solution per equivalence class.
* `group_action = nothing`: A function taking one solution and returning other
  solutions obtainable constructively (e.g. by symmetry).
* `group_actions = nothing`: Several group actions chained via
  [`GroupActions`](@ref).
* `max_loops_no_progress = 5`: Maximal number of loop generations without any
  progress.
* `min_solutions`: Minimal number of solutions before a stopping heuristic
  applies.
* `parameter_sampler = independent_normal`: A function `sampler(rng, p)` taking
  a random number generator and the parameter `p`, and returning a new random
  parameter `q`. Drawing from the given `rng` is what makes `seed` determine the
  result.
* `variables`, `parameters`: split the symbols of a polynomial `F`, as in
  [`System`](@ref). Ignored when `F` is already a `System`.
* `permutations = false`: Whether to keep track of the permutations induced by
  the loops.
* `reuse_loops = ReuseLoops.ALL`: Strategy to reuse other loops for newly found
  solutions: `ReuseLoops.ALL`, `ReuseLoops.RANDOM` or `ReuseLoops.NONE`.
* `seed`: Every random choice descends from it, so the same `seed` gives the
  same loops regardless of the state of the global random number generator.
* `target_solutions_count`: Stop once this number of solutions is reached.
* `timeout`: Maximal number of seconds the computation is allowed to run.
* `trace_test = true`: Perform a trace test to check completeness (only for
  linear-subspace monodromy).
* `trace_test_tol = 1e-6`: Tolerance for the trace test.
* `unique_points_atol` / `unique_points_rtol`: tolerances for the solution
  deduplication.
"""
# The three start-data shapes, by dispatch. A parameter-free system is intersected
# with a subspace, a parameterized one is tracked in parameter space, and `alg`'s
# `SUBSPACE` parameter says which.
@noinline function _monodromy_start_pair(G, rng::Random.AbstractRNG)
    start_pair = find_start_pair(G; rng = rng)
    start_pair.found || error(
        "Cannot compute a start pair (x, p) using `find_start_pair(F)`." *
            " You need to explicitly pass a start pair.",
    )
    return start_pair
end

function solve(
        F::Union{SystemLike, PolynomialInput},
        alg::Monodromy{MO, V, P, false},
        exec::E = Threaded(),
    )::ParameterMonodromyResult where {MO, V, P, E <: AbstractExecutor}
    G = _monodromy_system(F, alg)
    # A tagged stream: `_monodromy_solve!` seeds loop generation from `seed`
    # directly, so the setup draws here must stay uncorrelated with it.
    rng = _tagged_rng(_seed(alg), 0x0000_0001)
    start_pair = _monodromy_start_pair(G, rng)
    # The intended (co)dimension is required rather than guessed, so a forgotten
    # parameter argument is caught instead of silently reinterpreted.
    is_parameterized(start_pair) || error(
        "Given system doesn't have any parameters. If you intended to intersect " *
            "with a linear subspace it is necessary to provide a " *
            "dimension (`dim`) or codimension (`codim`) of the component of interest.",
    )
    return _monodromy_parameters(G, [start_pair.x], start_pair.p, alg, exec, rng)
end

function solve(
        F::Union{SystemLike, PolynomialInput},
        alg::Monodromy{MO, V, P, true},
        exec::E = Threaded(),
    )::SubspaceMonodromyResult where {MO, V, P, E <: AbstractExecutor}
    G = _monodromy_system(F, alg)
    rng = _tagged_rng(_seed(alg), 0x0000_0001)
    start_pair = _monodromy_start_pair(G, rng)
    is_parameterized(start_pair) && error(
        "`dim` and `codim` are the expected (co)dimension of a component of a " *
            "parameter-free system, which this system is not: it has " *
            "$(nparameters(G)) parameter(s). Drop them to track loops in " *
            "parameter space, or fix the parameters first with `fix_parameters`.",
    )
    x = start_pair.x
    projective = is_homogeneous(G)
    codim_c = alg.codim < 0 ? -1 : alg.codim + Int(projective)
    # NOTE the swap: `dim`/`codim` are COMPONENT dimensions, so the subspace
    # takes the complementary ones.
    L = rand_subspace(rng, x; dim = codim_c, codim = alg.dim, affine = !projective)
    return _monodromy_subspace(G, [x], L, alg, exec, rng)
end

function solve(
        F::Union{SystemLike, PolynomialInput},
        sols::SolutionsLike,
        p::AbstractVector{<:Number},
        alg::Monodromy,
        exec::E = Threaded(),
    )::MonodromyResult where {E <: AbstractExecutor}
    G = _monodromy_system(F, alg)
    return _monodromy_parameters(
        G, _monodromy_starts(sols), p, alg, exec,
        _tagged_rng(_seed(alg), 0x0000_0001),
    )
end

function solve(
        F::Union{SystemLike, PolynomialInput},
        sols::SolutionsLike,
        L::LinearSubspace,
        alg::Monodromy,
        exec::E = Threaded(),
    )::MonodromyResult where {E <: AbstractExecutor}
    G = _monodromy_system(F, alg)
    return _monodromy_subspace(
        G, _monodromy_starts(sols), L, alg, exec,
        _tagged_rng(_seed(alg), 0x0000_0001),
    )
end

# A single solution is accepted as itself, not as a list of coordinates.
_monodromy_starts(sols::AbstractVector{<:Number})::Vector{Vector{ComplexF64}} =
    [Vector{ComplexF64}(ComplexF64.(sols))]
_monodromy_starts(sols::StartsLike)::Vector{Vector{ComplexF64}} = _start_points(sols)

"""
    solve(F::System, R::MonodromyResult, p_target, alg = Continuation(), exec = Threaded())

Track the solutions of the monodromy result `R` from its parameters to `p_target`
via a parameter homotopy. `R` supplies both the start solutions and the start
parameters, so only the target end is given.
"""
function solve(
        F::SystemLike, R::MonodromyResult, p_target::AbstractVector{<:Number},
        alg::Continuation = Continuation(), exec::E = Threaded(),
    )::Result where {E <: AbstractExecutor}
    return solve(F, solutions(R), Vector(parameters(R)), p_target, alg, exec)
end

function solve(
        F::SystemLike, R::MonodromyResult, p_target::AbstractVector{<:Number},
        exec::E,
    )::Result where {E <: AbstractExecutor}
    return solve(F, R, p_target, Continuation(), exec)
end

function _monodromy_parameters(
        F::SystemLike, S::AbstractVector{<:AbstractVector}, p, alg::Monodromy,
        exec::E, rng::Random.AbstractRNG,
    )::ParameterMonodromyResult where {E <: AbstractExecutor}
    cp = convert(Vector{ComplexF64}, p)
    return with_monodromy_solver(
        F, cp;
        options = alg.options, tracker_options = _tracker_options(alg), rng = rng,
        start_solutions = S,
    ) do MS
        _monodromy_solve!(
            MS, S, cp, _seed(alg), _show_progress(alg), exec,
            alg.catch_interrupt, alg.warning,
        )
    end
end

function _monodromy_subspace(
        F::SystemLike, S::AbstractVector{<:AbstractVector}, L, alg::Monodromy,
        exec::E, rng::Random.AbstractRNG,
    )::SubspaceMonodromyResult where {E <: AbstractExecutor}
    cp = convert(LinearSubspace{ComplexF64}, L)
    return with_monodromy_solver(
        F, cp;
        options = alg.options, tracker_options = _tracker_options(alg),
        intrinsic = _use_intrinsic(alg.coords, cp),
        rng = rng, start_solutions = S,
    ) do MS
        mH, nH = size(MS.workers[1].homotopy)
        mH < nH && throw(
            ArgumentError(
                "The homotopy for the subspace intersection is underdetermined " *
                    "($mH equations for $nH unknowns). The provided component " *
                    "dimension (dim = $(alg.dim), codim = $(alg.codim)) is likely " *
                    "overstated for this system.",
            ),
        )
        _monodromy_solve!(
            MS, S, cp, _seed(alg), _show_progress(alg), exec,
            alg.catch_interrupt, alg.warning,
        )
    end
end

## ── verify_solution_completeness ─────────────────────────────────────────────
