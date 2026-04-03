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
