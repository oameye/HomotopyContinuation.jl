# Executor & Threading Design

## Goal

Add a SciML-style executor abstraction to `solve()` so path tracking can run serial, threaded, or (future) distributed — selected by a positional strategy argument.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| API style | Positional 3rd argument | SciML-like, clean dispatch, no ambiguity |
| Default executor | `Threaded()` (auto-detect `nthreads()`) | Always threaded, even with 1 thread |
| `Threaded` task count | `Threaded()` = `nthreads()`, `Threaded(2)` = 2 tasks, error if > `nthreads()` | One task per thread for CPU-bound work |
| `Serial` type | Separate type, not `Threaded(1)` | Zero threading overhead, better for debugging/TTFX |
| Worker state construction | Named builder structs, rebuild from immutable inputs | Explicit, serializable for distributed, no deepcopy fragility |
| Compile mode | `_clone_system_evaluator` preserves `INTERPRETED` vs `COMPILED` backend | Serial and Threaded must run the same evaluator backend |
| Progress/interruption | Deferred — not in this design | Get threading correct first |
| Threading library | OhMyThreads.jl | Low-overhead, composable, good scheduling |

## Executor Types

```julia
abstract type AbstractExecutor end

struct Serial <: AbstractExecutor end

struct Threaded <: AbstractExecutor
    ntasks::Int
    function Threaded(ntasks::Int)
        ntasks > Threads.nthreads() && throw(ArgumentError(
            "ntasks ($ntasks) exceeds available threads ($(Threads.nthreads()))"
        ))
        return new(ntasks)
    end
end
Threaded() = Threaded(Threads.nthreads())
```

`ntasks` controls how many OhMyThreads tasks are spawned. One task = one thread for CPU-bound path tracking. Error if `ntasks > nthreads()` since extra tasks waste memory (each holds a full `WorkerState`) with zero throughput gain on compute-bound work.

`Serial` dispatches to a plain loop. `Threaded` dispatches to OhMyThreads-based parallel tracking.

Future `DistributedEx <: AbstractExecutor` lives in a package extension loaded with `using Distributed`.

## User-Facing API

All existing `solve` signatures gain an optional executor as the last positional argument before kwargs. Default is `Threaded()`.

```julia
# TotalDegree
solve(F::System)                                           # Threaded(nthreads())
solve(F::System, alg::TotalDegree)                         # Threaded(nthreads())
solve(F::System, alg::TotalDegree, exec::AbstractExecutor) # explicit

# Polyhedral
solve(F::System, alg::Polyhedral)                          # Threaded(nthreads())
solve(F::System, alg::Polyhedral, exec::AbstractExecutor)  # explicit

# Parameter homotopy
solve(F::System, starts; start_parameters=..., ...)                          # Threaded(nthreads())
solve(F::System, starts, exec::AbstractExecutor; start_parameters=..., ...)  # explicit

# Convenience
solve(F::System, exec::AbstractExecutor)  # = solve(F, TotalDegree(), exec)
```

## Mutable State Analysis

The entire evaluation chain contains mutable state that **cannot be shared across threads**:

| Layer | Mutable state | Shared-safe data |
|-------|---------------|------------------|
| `Interpreter` | `tape::V` (mutated every `execute!`) | `sequence::InstructionSequence` (immutable) |
| `SystemEvaluator` | FunctionWrapper closures capture interpreters | `_size`, `_nparameters` |
| `StraightLineHomotopy` | scratch buffers (`u_start`, `U_target`, etc.) + two `SystemEvaluator`s | `γ::ComplexF64` |
| `CoefficientHomotopy` | `coeffs`, `t_cache`, `u_scratch`, `tx1`, `tx2` + `SystemEvaluator` | `start_coeffs`, `target_coeffs`, `dt_coeffs` |
| `ToricHomotopy` | `weights`, `coeffs`, `dt_coeffs`, `d2t_coeffs`, `d3t_coeffs`, `t_cache`, `dt_cache`, `d2t_cache`, `d3t_cache`, `u_cache`, `x_scratch`, `tx1`, `tx2` + `SystemEvaluator` | `system_coeffs` |
| `HomotopyEvaluator` | FunctionWrapper closures capture the homotopy | `_size` |
| `TrackerState` | solution vectors, counters, step sizes | — |
| `Predictor` | Taylor vectors | — |
| `NewtonCorrector` | scratch buffers | — |
| `EndgameState` | samples, predictions, flags | — |

**Key insight:** `InstructionSequence` is the immutable core. Everything above it is mutable scratch. Reconstruction means: share instruction sequences, rebuild fresh interpreters (new tapes), fresh evaluators, fresh homotopy, fresh tracker.

## Prerequisite: Store compile mode in System

`System` currently does not store its `CompileMode`. `_clone_system_evaluator` needs to know whether to call `_build_system_evaluator` (interpreted) or `_build_compiled_evaluator` (compiled). Add a `compile_mode::CompileMode.T` field to `System`:

```julia
struct System{P, V}
    # ... existing fields ...
    compile_mode::CompileMode.T
end
```

This field is set once during construction in `_build_compiled_system` and never mutated.

## Worker State Bundles

Each parallel task needs a self-contained bundle of all mutable state:

```julia
struct TrackingWorkerState
    tracker::EndgameTracker
end

struct PolyhedralWorkerState
    toric_homotopy::ToricHomotopy
    toric_tracker::Tracker
    coeff_tracker::EndgameTracker
    x_buffer::Vector{ComplexF64}
end
```

## Builder Structs

Builders are concrete callable structs that store immutable reconstruction data and produce fresh worker state on each call. They replace anonymous closures — explicit, serializable for distributed, type-stable.

### Core utility: `_clone_system_evaluator`

Creates a fresh `SystemEvaluator` from a `System`'s instruction sequences, preserving the compile mode. For `INTERPRETED`: rebuilds all 6 interpreters with fresh tapes. For `COMPILED`: rebuilds 4 interpreters (df64, t1-t3) with fresh tapes plus re-generates `@RuntimeGeneratedFunction`s from the instruction sequences.

```julia
function _clone_system_evaluator(sys::System)::SystemEvaluator
    seq_eval = sys._interp_f64.sequence    # immutable, shared
    seq_jac = sys._interp_jac.sequence     # immutable, shared
    m, n = size(sys.evaluator)
    np = nparameters(sys.evaluator)

    if sys.compile_mode == CompileMode.INTERPRETED
        return _build_system_evaluator(
            Interpreter(Vector{ComplexF64}, seq_eval),
            Interpreter(Vector{ComplexDF64}, seq_eval),
            Interpreter(Vector{ComplexF64}, seq_jac),
            Interpreter(Vector{TruncatedTaylorSeries{2, ComplexF64}}, seq_eval),
            Interpreter(Vector{TruncatedTaylorSeries{3, ComplexF64}}, seq_eval),
            Interpreter(Vector{TruncatedTaylorSeries{4, ComplexF64}}, seq_eval),
            m, n, np,
        )
    else  # CompileMode.COMPILED
        return _build_compiled_evaluator(
            seq_eval, seq_jac,
            Interpreter(Vector{ComplexDF64}, seq_eval),
            Interpreter(Vector{TruncatedTaylorSeries{2, ComplexF64}}, seq_eval),
            Interpreter(Vector{TruncatedTaylorSeries{3, ComplexF64}}, seq_eval),
            Interpreter(Vector{TruncatedTaylorSeries{4, ComplexF64}}, seq_eval),
            m, n, np,
        )
    end
end
```

For `INTERPRETED`: allocates 6 fresh interpreter tapes. For `COMPILED`: allocates 4 fresh interpreter tapes + re-invokes `@RuntimeGeneratedFunction` (JIT compilation of eval/jac from instruction sequences). The `@RuntimeGeneratedFunction` call is per-worker, not per-path — acceptable overhead at task startup.

### StraightLineBuilder (TotalDegree)

```julia
struct StraightLineBuilder
    degrees::Vector{Int}            # for TotalDegreeStartSystem (stateless)
    target_system::System           # stores InstructionSequences + compile_mode
    γ::ComplexF64
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::StraightLineBuilder)()::TrackingWorkerState
    start_eval = _total_degree_startevaluator(b.degrees)
    target_eval = _clone_system_evaluator(b.target_system)
    H = StraightLineHomotopy(start_eval, target_eval; γ = b.γ)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval; options = b.tracker_options)
    eg = EndgameTracker(tracker, b.endgame_options)
    return TrackingWorkerState(eg)
end
```

`TotalDegreeStartSystem` is stateless (just `degrees::Vector{Int}`, no mutable buffers), so `_total_degree_startevaluator` safely creates a fresh evaluator every time.

### CoefficientBuilder (parameter homotopy)

```julia
struct CoefficientBuilder
    param_system::System               # parametric system with InstructionSequences
    start_coeffs::Vector{ComplexF64}   # immutable
    target_coeffs::Vector{ComplexF64}  # immutable
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::CoefficientBuilder)()::TrackingWorkerState
    sys_eval = _clone_system_evaluator(b.param_system)
    H = CoefficientHomotopy(sys_eval, b.start_coeffs, b.target_coeffs)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval; options = b.tracker_options)
    eg = EndgameTracker(tracker, b.endgame_options)
    return TrackingWorkerState(eg)
end
```

### PolyhedralBuilder (two-phase)

```julia
struct PolyhedralBuilder
    param_system::System
    start_coeffs::Vector{Vector{ComplexF64}}
    flat_start::Vector{ComplexF64}
    flat_target::Vector{ComplexF64}
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::PolyhedralBuilder)()::PolyhedralWorkerState
    # Phase 1: toric — homotopy and tracker are coupled
    toric_eval = _clone_system_evaluator(b.param_system)
    toric_H = ToricHomotopy(toric_eval, b.start_coeffs)
    toric_tracker = Tracker(HomotopyEvaluator(toric_H); options = b.tracker_options)

    # Phase 2: coefficient
    coeff_eval = _clone_system_evaluator(b.param_system)
    coeff_H = CoefficientHomotopy(coeff_eval, b.flat_start, b.flat_target)
    coeff_tracker = EndgameTracker(
        Tracker(HomotopyEvaluator(coeff_H); options = b.tracker_options),
        b.endgame_options,
    )

    n = size(toric_eval)[2]
    return PolyhedralWorkerState(toric_H, toric_tracker, coeff_tracker, Vector{ComplexF64}(undef, n))
end
```

The toric homotopy and toric tracker are built from the **same** `toric_eval` and `toric_H`, so `update_weights!(toric_H, ...)` mutates the exact homotopy captured inside the tracker's `HomotopyEvaluator` closures.

## SolveCache

Parameterized on executor type and builder type.

```julia
struct SolveCache{E<:AbstractExecutor, B}
    executor::E
    builder::B                              # callable: () -> TrackingWorkerState
    tracker::EndgameTracker                 # primary (used by Serial)
    start_solutions::Vector{Vector{ComplexF64}}
    seed::UInt32
end
```

## PolyhedralSolveCache

```julia
struct PolyhedralSolveCache{E<:AbstractExecutor, B<:PolyhedralBuilder, S<:System}
    executor::E
    builder::B                              # callable: () -> PolyhedralWorkerState
    toric_tracker::Tracker                  # primary toric (Serial)
    coeff_tracker::EndgameTracker           # primary coeff (Serial)
    toric_homotopy::ToricHomotopy           # primary toric homotopy (Serial)
    support::Vector{Matrix{Int32}}
    lifting::Vector{Vector{Int32}}
    start_solutions::Vector{Tuple{MixedSubdivisions.MixedCell, Vector{ComplexF64}}}
    seed::UInt32
    _param_system::S
end
```

## Serial Solve

Identical to current implementation — single tracker, sequential loop:

```julia
function CommonSolve.solve!(cache::SolveCache{Serial})::Result
    eg = cache.tracker
    path_results = PathResult[]
    sizehint!(path_results, length(cache.start_solutions))
    for x0 in cache.start_solutions
        track!(eg, x0)
        push!(path_results, PathResult(eg))
    end
    return Result(path_results, length(cache.start_solutions), cache.seed)
end
```

## Threaded Solve

Uses OhMyThreads `@tasks` with `@local`. Each task calls the builder once to create its own worker state, then reuses it across all paths assigned to that task.

```julia
function CommonSolve.solve!(cache::SolveCache{Threaded})::Result
    nt = cache.executor.ntasks
    starts = cache.start_solutions
    n_paths = length(starts)
    results = Vector{PathResult}(undef, n_paths)

    @tasks for i in eachindex(starts)
        @set ntasks = nt
        @local ws = cache.builder()
        track!(ws.tracker, starts[i])
        results[i] = PathResult(ws.tracker)
    end

    return Result(results, n_paths, cache.seed)
end
```

The scheduler (`:dynamic` vs `:greedy`) is left as a benchmarking decision. `:dynamic` (default) uses chunked distribution; `:greedy` uses channel-based work-stealing which may be better for unbalanced workloads (paths with varying step counts). The `Threaded` struct may gain a `scheduler` field once benchmarked.

## Threaded Polyhedral Solve

Each task gets a `PolyhedralWorkerState` bundle with coupled toric homotopy + tracker:

```julia
function CommonSolve.solve!(cache::PolyhedralSolveCache{Threaded})::Result
    nt = cache.executor.ntasks
    starts = cache.start_solutions
    n_paths = length(starts)
    results = Vector{PathResult}(undef, n_paths)

    @tasks for i in eachindex(starts)
        @set ntasks = nt
        @local ws = cache.builder()

        cell, x0 = starts[i]

        update_weights!(ws.toric_homotopy, cache.support, cache.lifting, cell; min_weight = 1.0)
        code = track!(ws.toric_tracker, x0; t1 = complex(0.0), t0 = complex(1.0))

        if code != TrackerCode.TRACKER_SUCCESS
            results[i] = PathResult(ws.toric_tracker)
            continue
        end

        copyto!(ws.x_buffer, ws.toric_tracker.state.x)
        track!(ws.coeff_tracker, ws.x_buffer)
        results[i] = PathResult(ws.coeff_tracker)
    end

    return Result(results, n_paths, cache.seed)
end
```

## Thread Safety Invariants

1. **No shared mutable state.** Each task's `WorkerState` is built from scratch by the builder — fresh interpreters (or fresh `@RuntimeGeneratedFunction`s for compiled mode), fresh homotopy scratch buffers, fresh tracker state.
2. **No locks on the hot path.** Results are written to pre-allocated array slots indexed by path number — no contention.
3. **Builder isolation via reconstruction.** Builders store only immutable data (`InstructionSequence` via `System`, coefficients, options). `_clone_system_evaluator` creates fresh interpreter tapes from shared instruction sequences and preserves compile mode. No `deepcopy` on the hot path.
4. **Coupled toric pair.** `PolyhedralBuilder` constructs `ToricHomotopy` and `Tracker` from the *same* evaluator, so `update_weights!` mutates the homotopy that the tracker actually uses.
5. **Compile mode preserved.** `_clone_system_evaluator` checks `sys.compile_mode` and calls the matching builder (`_build_system_evaluator` or `_build_compiled_evaluator`). Serial and Threaded executors always run the same evaluator backend.
6. **Read-only shared data.** `start_solutions`, `support`, `lifting`, `InstructionSequence`s are read-only during tracking.

## Dependencies

| Package | Purpose | Scope |
|---------|---------|-------|
| OhMyThreads.jl | `@tasks` / `@local` for threaded work distribution | Core dependency |

## Future: Distributed Extension

A package extension activated by `using Distributed`. Builder structs contain `System` objects which include mutable interpreter state. For true distributed transport, define an explicit serialization payload around instruction sequences, dimensions, compile mode, and coefficients — do not rely on default serialization of `System`. This is out of scope for the current design.

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/core/system.jl` | **Modify** — add `compile_mode::CompileMode.T` field to `System` |
| `src/solving/executor.jl` | **Create** — `AbstractExecutor`, `Serial`, `Threaded` |
| `src/solving/worker_state.jl` | **Create** — `TrackingWorkerState`, `PolyhedralWorkerState`, `_clone_system_evaluator` |
| `src/solving/builder.jl` | **Create** — `StraightLineBuilder`, `CoefficientBuilder`, `PolyhedralBuilder` |
| `src/solving/solve.jl` | **Modify** — parameterize `SolveCache{E,B}`, add executor dispatch, builder construction, threaded `solve!` |
| `src/solving/polyhedral.jl` | **Modify** — parameterize `PolyhedralSolveCache{E,B}`, add builder, threaded `solve!` |
| `src/HomotopyContinuationNext.jl` | **Modify** — include new files, add OhMyThreads import, export `Serial`, `Threaded` |
| `Project.toml` | **Modify** — add OhMyThreads dependency |
| `test/solve_test.jl` | **Modify** — add tests for `Serial()` and `Threaded()` executors |

## Testing Strategy

1. **`_clone_system_evaluator` correctness (INTERPRETED)**: clone evaluator produces identical outputs to original for eval, jacobian, and Taylor orders 1-3. Test with `FSVec`/`FSMat` inputs matching the actual evaluator API.
2. **`_clone_system_evaluator` correctness (COMPILED)**: same as above but for `System(...; compile=CompileMode.COMPILED)`.
3. **Clone independence**: mutating the cloned evaluator's state (by calling evaluate! with different inputs) does not affect the original.
4. **Builder correctness**: `StraightLineBuilder(...)()` produces a working tracker that solves the same system.
5. **Serial correctness**: `solve(F, TotalDegree(), Serial())` identical results to current implementation.
6. **Threaded correctness**: `solve(F, TotalDegree(), Threaded())` same solution set (unordered), same path-level accounting (`tracked_paths`, success/failure counts, multiplicities, cluster structure).
7. **Polyhedral + threading**: same as above for `Polyhedral()`.
8. **Parameter homotopy + threading**: same as above.
9. **Serial vs Threaded consistency**: compare full result structure — solution sets, multiplicities, cluster sizes — not just counts.
10. **Multi-thread execution**: tests must run with `julia -t auto` or `julia -t 4` to actually exercise parallel paths. `make test` should be updated accordingly.
11. **`Threaded` validation**: `Threaded(100)` on 4 threads throws `ArgumentError`.
12. **Default executor**: `solve(F)` uses `Threaded(nthreads())`.
