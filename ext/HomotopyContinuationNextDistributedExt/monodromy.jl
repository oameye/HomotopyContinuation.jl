## Multi-process monodromy. See `01_decisions.md` for the measurements behind the
## scheduling choices here.
#
# Only loop tracking is handed out. The driver keeps the job queue, the
# `UniquePoints` set, the trace matrix, the statistics and the loop list, so shared
# state stays single-writer and the dispatch order is the serial one.
#
# The queue stays on the driver rather than in the remote channel, so only a
# bounded window is in flight and a stopping criterion drops the rest of the
# generation instead of waiting it out. Jobs travel in batches because a handoff
# costs about as much as tracking one loop. An empty batch is the stop token, as
# for the path map.

# ── Worker side ─────────────────────────────────────────────────────────────

function _run_monodromy_job(
        ws::S, job::HCN.MonodromyJob,
    )::HCN.MonodromyJobResult where {S}
    sink = HCN.TraceColumns()
    res = HCN.track_loop!(
        ws, job.loop, job.x, job.ω, job.μ, job.extended_precision,
        job.collect_trace, sink,
    )
    return HCN.MonodromyJobResult(job.id, job.loop_id, res, sink.columns)
end

# A state carries fresh tapes and, under a compiled mode, freshly generated code,
# so it is built only once this task has drawn real work.
function _monodromy_task_loop(
        builder::B, jobs::Distributed.RemoteChannel,
        out::Distributed.RemoteChannel,
    )::Nothing where {B}
    batch = take!(jobs)
    isempty(batch) && return nothing
    ws = builder()
    while !isempty(batch)
        put!(out, [_run_monodromy_job(ws, job) for job in batch])
        batch = take!(jobs)
    end
    return nothing
end

# One long-lived call per process, `ntasks` consumers inside it. Only these tasks
# touch the channels, and they share nothing else.
function _run_monodromy_tasks(
        builder::B, jobs::Distributed.RemoteChannel,
        out::Distributed.RemoteChannel, ntasks::Int,
    )::Nothing where {B}
    @sync for _ in 1:ntasks
        Threads.@spawn _monodromy_task_loop(builder, jobs, out)
    end
    return nothing
end

# ── Driver ──────────────────────────────────────────────────────────────────

# `tasks_per_process == 0` resolves remotely, and the driver needs the exact count
# to end every consumer.
function _monodromy_task_counts(
        exec::HCN.DistributedExecutor, pids::Vector{Int},
    )::Vector{Int}
    exec.tasks_per_process > 0 && return fill(exec.tasks_per_process, length(pids))
    return [_transport_query(Threads.nthreads, pid) for pid in pids]
end

function HCN._distributed_monodromy_solve!(
        exec::HCN.DistributedExecutor, MS::HCN.MonodromySolver,
        results::Vector{HCN.PathResult}, seed::UInt32, progress,
    )::HCN.MonodromyCode.T
    pids = _resolve_pids(exec)
    _check_pids(pids)
    ntasks = _monodromy_task_counts(exec, pids)
    total_tasks = sum(ntasks)
    batch_size = exec.batch_size > 0 ? exec.batch_size : 8
    # Two batches per task: one being tracked, one waiting behind it.
    window = 2 * total_tasks * batch_size

    JobT = Vector{HCN.MonodromyJob{typeof(MS.workers[1].base)}}
    jobs = _transport_channel(JobT, typemax(Int))
    out = _transport_channel(Vector{HCN.MonodromyJobResult}, typemax(Int))

    retcode = HCN.MonodromyCode.IN_PROGRESS
    try
        tracking = @async begin
            try
                @sync for (k, pid) in enumerate(pids)
                    @async _transport_run(
                        _run_monodromy_tasks, pid, MS.builder, jobs, out, ntasks[k],
                    )
                end
            finally
                # Finished or failed, unblock the driver loop.
                close(out)
            end
        end

        try
            retcode = _drive_monodromy!(
                MS, results, progress, jobs, out, JobT, batch_size, window,
                total_tasks,
            )
        catch e
            # `out` closed under us: a process failed and `tracking` has its error.
            e isa InvalidStateException || rethrow()
        end
        # Buffered tokens stay takeable after `close(jobs)`.
        for _ in 1:total_tasks
            put!(jobs, JobT())
        end

        # Surface a worker failure rather than a truncated result.
        try
            wait(tracking)
        catch e
            throw(_root_cause(e))
        end
    finally
        close(jobs)
    end
    return retcode
end

function _drive_monodromy!(
        MS::HCN.MonodromySolver, results::Vector{HCN.PathResult}, progress,
        jobs::Distributed.RemoteChannel, out::Distributed.RemoteChannel,
        ::Type{JobT}, batch_size::Int, window::Int, total_tasks::Int,
    )::HCN.MonodromyCode.T where {JobT}
    queue = HCN.LoopTrackingJob[]
    stats = MS.statistics
    opts = MS.options
    is_subspace = MS.workers[1].base isa HCN.LinearSubspace
    t₀ = time()
    inflight = 0     # jobs dispatched and not yet returned
    msgs = 0         # messages dispatched and not yet returned

    dispatch! = () -> begin
        while inflight < window && !isempty(queue)
            room = min(batch_size, window - inflight)
            # Outstanding messages is what a task can be holding, so this bounds
            # how many may be out of work. Hungry tasks get what is queued, spread
            # out, so a generation smaller than the pool still reaches all of them;
            # fed ones wait for a full batch.
            hungry = total_tasks - msgs
            hungry < 1 && length(queue) < batch_size && break
            n = hungry > 0 ? clamp(cld(length(queue), hungry), 1, room) :
                min(room, length(queue))
            batch = JobT(undef, n)
            for k in 1:n
                job = popfirst!(queue)
                collect_trace = opts.trace_test && HCN.nloops(MS) == job.loop_id
                batch[k] = HCN.MonodromyJob(
                    job, HCN.loop(MS, job.loop_id), results[job.id], collect_trace,
                )
            end
            put!(jobs, batch)
            inflight += n
            msgs += 1
        end
        return nothing
    end

    retcode = HCN.MonodromyCode.IN_PROGRESS
    while retcode == HCN.MonodromyCode.IN_PROGRESS
        HCN.loop_finished!(stats, length(results))

        if opts.loop_finished_callback(results)
            retcode = HCN.MonodromyCode.TERMINATED_CALLBACK
            break
        end
        if is_subspace && HCN.nloops(MS) > 0 && opts.trace_test &&
                HCN.trace_colinearity(MS) < opts.trace_test_tol
            retcode = HCN.MonodromyCode.SUCCESS
            break
        end
        if opts.target_solutions_count === nothing &&
                length(results) >= something(opts.min_solutions, 0) &&
                HCN.loops_no_change(stats, length(results)) >=
                opts.max_loops_no_progress
            retcode = HCN.MonodromyCode.HEURISTIC_STOP
            break
        end
        if length(results) == something(opts.target_solutions_count, -1)
            retcode = HCN.MonodromyCode.SUCCESS
            break
        end
        if HCN.nloops(MS) > 0 && opts.single_loop_per_start_solution
            retcode = HCN.MonodromyCode.SUCCESS
            break
        end

        HCN.add_loop!(MS)
        HCN.reset_trace!(MS)
        # schedule all jobs on the fresh loop
        new_loop_id = HCN.nloops(MS)
        for i in 1:length(results)
            push!(queue, HCN.LoopTrackingJob(i, new_loop_id))
        end

        dispatch!()
        while inflight > 0
            batch = take!(out)
            inflight -= length(batch)
            msgs -= 1
            for r in batch
                # A decided retcode still collects what is in flight, since every
                # result counts towards the permutations, but starts no new loop.
                undecided = retcode == HCN.MonodromyCode.IN_PROGRESS
                _handle_monodromy_result!(MS, results, queue, r, undecided)

                HCN.update_progress!(
                    progress, stats;
                    solutions = length(results), queued = length(queue) + inflight,
                )

                undecided || continue
                if length(results) == something(opts.target_solutions_count, -1) &&
                        # only terminate after a completed loop to ensure that we
                        # collect proper permutation information
                        !opts.permutations
                    retcode = HCN.MonodromyCode.SUCCESS
                elseif opts.timeout !== nothing && time() - t₀ > (opts.timeout::Float64)
                    retcode = HCN.MonodromyCode.TIMEOUT
                end
                retcode == HCN.MonodromyCode.IN_PROGRESS || empty!(queue)
            end
            retcode == HCN.MonodromyCode.IN_PROGRESS && dispatch!()
        end
    end

    HCN.update_progress!(
        progress, stats;
        finish = true, solutions = length(results), queued = length(queue),
    )
    return retcode
end

# The deduplicate-and-schedule half of the serial solve loop, driven by a result
# that came back from another process.
function _handle_monodromy_result!(
        MS::HCN.MonodromySolver, results::Vector{HCN.PathResult},
        queue::Vector{HCN.LoopTrackingJob}, r::HCN.MonodromyJobResult,
        schedule_more::Bool,
    )::Nothing
    opts = MS.options
    stats = MS.statistics
    # Independent of success: the trace sums over the first two segments, which
    # can land while a later one fails.
    columns = r.trace
    columns === nothing || HCN._accumulate_trace!(MS, columns)

    res = r.result
    if res === nothing
        HCN.loop_failed!(stats)
        if opts.permutations
            HCN.add_permutation!(stats, r.loop_id, r.id, 0)
        end
        return nothing
    end
    HCN.loop_tracked!(stats)

    # 1) check whether the solution already exists
    id, got_added = HCN.add!(MS, res, length(results) + 1)

    if opts.permutations
        HCN.add_permutation!(stats, r.loop_id, r.id, id)
    end
    got_added || return nothing

    # 2) doesn't exist, so add to results
    push!(results, res)
    schedule_more || return nothing

    # 3) schedule on the same loop again
    if !opts.single_loop_per_start_solution
        push!(queue, HCN.LoopTrackingJob(id, r.loop_id))
    end

    # 4) schedule on other loops
    if opts.reuse_loops == HCN.ReuseLoops.ALL
        for k in 1:HCN.nloops(MS)
            k != r.loop_id || continue
            push!(queue, HCN.LoopTrackingJob(id, k))
        end
    elseif opts.reuse_loops == HCN.ReuseLoops.RANDOM && HCN.nloops(MS) >= 2
        k = rand(2:HCN.nloops(MS))
        if k <= r.loop_id
            k -= 1
        end
        push!(queue, HCN.LoopTrackingJob(id, k))
    end
    return nothing
end
