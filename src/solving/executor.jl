## Executor types — control how paths are distributed across threads/workers.

"""
    AbstractExecutor

Abstract supertype for path-tracking execution strategies.
Subtypes dispatch `CommonSolve.solve!` to different implementations.
"""
abstract type AbstractExecutor end

"""
    Serial()

Execute path tracking sequentially in a single task.
Useful for debugging and reproducibility.
"""
struct Serial <: AbstractExecutor end

"""
    Threaded(ntasks::Int = Threads.nthreads())

Execute path tracking in parallel using `ntasks` tasks via OhMyThreads.jl.
Each task gets its own worker state (built by the builder).

`ntasks` must not exceed `Threads.nthreads()` — path tracking is CPU-bound,
so extra tasks waste memory (each holds a full WorkerState) with no throughput gain.

Default: one task per available Julia thread.
"""
struct Threaded <: AbstractExecutor
    ntasks::Int
    function Threaded(ntasks::Int)
        ntasks < 1 && throw(
            ArgumentError("ntasks ($ntasks) must be at least 1")
        )
        ntasks > Threads.nthreads() && throw(
            ArgumentError(
                "ntasks ($ntasks) exceeds available threads ($(Threads.nthreads()))"
            )
        )
        return new(ntasks)
    end
end
Threaded() = Threaded(Threads.nthreads())

# Whether an executor asks for more than one task. Used where a route chooses
# between a task fan-out and a plain loop, so `Threaded(1)` takes the plain one.
_wants_tasks(::Serial)::Bool = false
_wants_tasks(exec::Threaded)::Bool = exec.ntasks > 1
_wants_tasks(::AbstractExecutor)::Bool = true

# Tasks a fan-out inside the calling process should use. Only `Threaded` states a
# count; any other executor distributes elsewhere and leaves the local pool.
_local_ntasks(::Serial)::Int = 1
_local_ntasks(exec::Threaded)::Int = exec.ntasks
_local_ntasks(::AbstractExecutor)::Int = Threads.nthreads()

"""
    DistributedExecutor(; pids = Int[], tasks_per_process = 0, batch_size = 0)

Execute path tracking across several Julia processes, each process internally
threaded. Requires `Distributed`:

```julia
using Distributed
addprocs(4)
@everywhere using HomotopyContinuationNext
solve(F, TotalDegree(), DistributedExecutor())
```

Every process must have `HomotopyContinuationNext` loaded, hence the
`@everywhere`.

Paths are handed out in batches through a channel, so a process that draws a
run of expensive paths does not hold up the others. Each result is stored at its
global path index, so the returned [`Result`](@ref) is identical to the one
[`Serial`](@ref) produces.

- `pids`: processes to track on. Empty means `Distributed.workers()`, resolved
  when the solve starts rather than when the executor is built; if no worker
  process was started the solve errors instead of running on the driver. Naming
  the driver in `pids` is allowed.
- `tasks_per_process`: tasks each process runs. `0` means that process's
  `Threads.nthreads()`, which the driver cannot know in advance.
- `batch_size`: paths per batch. `0` picks a size from the path count, the
  process count and the task count.

[`Monodromy`](@ref) is scheduled differently, since its work is generated as
solutions are found: the calling process keeps the job queue, the deduplication
and the trace test, and hands out loops. There `batch_size` is loops per message
(`0` means 8) and two batches per task are in flight, which also bounds how many
loops still run after a stopping criterion is met.
"""
struct DistributedExecutor <: AbstractExecutor
    pids::Vector{Int}
    tasks_per_process::Int
    batch_size::Int
    function DistributedExecutor(
            pids::Vector{Int}, tasks_per_process::Int, batch_size::Int,
        )
        tasks_per_process < 0 && throw(
            ArgumentError(
                "tasks_per_process ($tasks_per_process) must be nonnegative " *
                    "(0 = one task per thread on each process)"
            )
        )
        batch_size < 0 && throw(
            ArgumentError("batch_size ($batch_size) must be nonnegative (0 = auto)")
        )
        return new(pids, tasks_per_process, batch_size)
    end
end

DistributedExecutor(;
    pids::AbstractVector{<:Integer} = Int[],
    tasks_per_process::Int = 0,
    batch_size::Int = 0,
) = DistributedExecutor(Vector{Int}(pids), tasks_per_process, batch_size)

# Hooks the Distributed extension fills in. A method the extension overwrites
# cannot be precompiled, while these untyped fallbacks are strictly less specific
# than the extension's, so both coexist.
@noinline function _distributed_ext_missing()
    throw(
        ArgumentError(
            "`DistributedExecutor` needs the Distributed extension. Run " *
                "`using Distributed` (and `@everywhere using " *
                "HomotopyContinuationNext` on the worker processes).",
        ),
    )
end

function _distributed_solve!(cache)
    Base.@nospecialize cache
    return _distributed_ext_missing()
end

function _distributed_monodromy_solve!(executor, MS, results, seed, progress)
    Base.@nospecialize executor MS results seed progress
    return _distributed_ext_missing()
end

function _distributed_sweep_entries(
        cache, targets, idxs, first_q, transform_result, transform_parameters,
        progress,
    )
    Base.@nospecialize cache targets idxs first_q transform_result
    Base.@nospecialize transform_parameters progress
    return _distributed_ext_missing()
end
