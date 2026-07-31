## Generic multi-process path map.
#
# Paths are handed out as index batches through a channel, so a process that
# draws a run of expensive paths does not hold up the others. Each result is
# stored at its global path index, so the output is independent of the order in
# which batches come back.

_transport_workers()::Vector{Int} = Distributed.workers()
_transport_myid()::Int = Distributed.myid()
_transport_channel(::Type{T}, n::Int) where {T} =
    Distributed.RemoteChannel(() -> Channel{T}(n))
_transport_run(f, pid::Int, args...) = Distributed.remotecall_wait(f, pid, args...)
_transport_query(f, pid::Int, args...) = Distributed.remotecall_fetch(f, pid, args...)

const BatchResult = Tuple{UnitRange{Int}, Vector{HCN.PathResult}}

# An empty range tells a process it is done; one is queued per process. A sentinel
# keeps termination independent of how the transport reports a closed channel.
const STOP_TOKEN = 1:0

# ── Preconditions ───────────────────────────────────────────────────────────

# `workers()` reports the driver itself when no worker process was started, which
# would run the whole solve here.
function _resolve_pids(exec::HCN.DistributedExecutor)::Vector{Int}
    isempty(exec.pids) || return exec.pids
    pids = _transport_workers()
    (length(pids) == 1 && only(pids) == _transport_myid()) && throw(
        ArgumentError(
            "`DistributedExecutor` found no processes to track on. Start some " *
                "with `Distributed.addprocs(n)`, or name them with `pids`.",
        ),
    )
    return pids
end

function _check_pids(pids::Vector{Int})::Nothing
    id = Base.PkgId(HCN)
    me = _transport_myid()
    for pid in pids
        pid == me && continue
        _transport_query(Base.root_module_exists, pid, id) || throw(
            ArgumentError(
                "process $pid does not have HomotopyContinuationNext loaded. Run " *
                    "`@everywhere using HomotopyContinuationNext` before solving " *
                    "with `DistributedExecutor`.",
            ),
        )
    end
    return nothing
end

# Enough batches per process that a run of expensive paths cannot hold up the
# solve, but never fewer paths than a process has tasks. The driver's own thread
# count stands in for that floor, since `tasks_per_process == 0` resolves remotely.
function _batch_ranges(
        n_paths::Int, exec::HCN.DistributedExecutor, nprocs::Int,
    )::Vector{UnitRange{Int}}
    bs = if exec.batch_size > 0
        exec.batch_size
    else
        nt = exec.tasks_per_process < 1 ? Threads.nthreads() : exec.tasks_per_process
        max(cld(n_paths, 8 * nprocs), min(nt, cld(n_paths, nprocs)))
    end
    return [i:min(i + bs - 1, n_paths) for i in 1:bs:n_paths]
end

# `@sync` wraps a worker failure in nested task/capture layers; report the error
# the user's system actually threw.
function _root_cause(e)
    e isa TaskFailedException && return _root_cause(e.task.exception)
    e isa CompositeException &&
        !isempty(e.exceptions) && return _root_cause(e.exceptions[1])
    e isa Distributed.RemoteException && return _root_cause(e.captured)
    e isa CapturedException && return _root_cause(e.ex)
    return e
end

# ── Worker side ─────────────────────────────────────────────────────────────

# A state carries fresh interpreter tapes and, under a compiled mode, freshly
# generated code, so states persist across batches and are built only as tasks
# come to need them.
function _grow_states!(states::Vector{S}, builder::B, want::Int)::Nothing where {S, B}
    have = length(states)
    have ≥ want && return nothing
    resize!(states, want)
    @sync for tid in (have + 1):want
        Threads.@spawn states[tid] = builder()
    end
    return nothing
end

function _run_batch!(
        work::W, states::Vector{S}, batch::UnitRange{Int},
        out::Distributed.RemoteChannel,
    )::Nothing where {W, S}
    n = length(batch)
    results = Vector{HCN.PathResult}(undef, n)
    # One index at a time: per-path cost varies as widely inside a batch as across.
    counter = Threads.Atomic{Int}(0)
    @sync for tid in 1:min(length(states), n)
        Threads.@spawn begin
            state = states[tid]
            while true
                j = Threads.atomic_add!(counter, 1) + 1
                j > n && break
                results[j] = work(state, batch[j])
            end
        end
    end
    put!(out, (batch, results))
    return nothing
end

# One long-lived call per process. Only this task touches the channels; the
# per-path work runs on tasks that share nothing but the results array, so all
# socket traffic stays on a single thread.
function _run_batches(
        work::W, jobs::Distributed.RemoteChannel,
        out::Distributed.RemoteChannel, tasks_per_process::Int,
        stopflag::Distributed.RemoteChannel,
    )::Nothing where {W}
    nt = tasks_per_process < 1 ? Threads.nthreads() : tasks_per_process
    batch = take!(jobs)
    # A stop token first: the queue was emptied elsewhere, so build nothing.
    isempty(batch) && return nothing
    states = [work.builder()]
    # `isready` never blocks, so an early stop costs one poll per batch and the
    # loop exits normally rather than through a closed channel.
    while !isempty(batch) && !isready(stopflag)
        _grow_states!(states, work.builder, min(nt, length(batch)))
        _run_batch!(work, states, batch, out)
        batch = take!(jobs)
    end
    return nothing
end

# ── Driver ──────────────────────────────────────────────────────────────────

# Driver-side, so a user callback never has to cross to a worker process.
# Granularity is one batch.
function _batch_trips_stop(
        early_stop::HCN.EarlyStop, prs::Vector{HCN.PathResult},
    )::Bool
    for pr in prs
        HCN.is_success(pr) && early_stop(pr) && return true
    end
    return false
end

function _distributed_map(
        exec::HCN.DistributedExecutor, work::W, n_paths::Int, report::R,
        early_stop::HCN.EarlyStop = HCN.NEVER_STOP,
    )::Vector{HCN.PathResult} where {W, R}
    results = Vector{HCN.PathResult}(undef, n_paths)
    n_paths == 0 && return results

    pids = _resolve_pids(exec)
    _check_pids(pids)
    batches = _batch_ranges(n_paths, exec, length(pids))
    n_batches = length(batches)
    # Receiving a work unit rebuilds the system's evaluator, so a process that
    # could only draw a stop token is left out.
    n_batches < length(pids) && (pids = pids[1:n_batches])

    jobs = _transport_channel(UnitRange{Int}, n_batches + length(pids))
    out = _transport_channel(BatchResult, n_batches)
    stopflag = _transport_channel(Bool, 1)
    try
        for batch in batches
            put!(jobs, batch)
        end
        for _ in pids
            put!(jobs, STOP_TOKEN)
        end

        ntasks = exec.tasks_per_process
        tracking = @async begin
            try
                @sync for pid in pids
                    @async _transport_run(
                        _run_batches, pid, work, jobs, out, ntasks, stopflag,
                    )
                end
            finally
                # Finished or failed, unblock the collector below.
                close(out)
            end
        end

        received = 0
        stopped = false
        try
            while received < n_batches
                batch, prs = take!(out)
                copyto!(results, first(batch), prs, 1, length(prs))
                received += 1
                report(batch, prs)
                if _batch_trips_stop(early_stop, prs)
                    # Tell the workers before leaving, or they block on `out`.
                    stopped = true
                    put!(stopflag, true)
                    break
                end
            end
        catch e
            e isa InvalidStateException || rethrow()
        end

        # A failure leaves batches unreceived; surface it rather than the gap.
        try
            wait(tracking)
        catch e
            throw(_root_cause(e))
        end
        stopped || received == n_batches || error(
            "distributed tracking returned $received of $n_batches batches",
        )
    finally
        # A failure leaves batches queued; drop them instead of waiting for the
        # finalizer.
        close(jobs)
    end
    return results
end
