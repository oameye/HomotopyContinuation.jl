# Executor & Threading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a SciML-style executor abstraction (`Serial`, `Threaded`) to `solve()` so path tracking can run in parallel via OhMyThreads.jl, using named builder structs and worker-state bundles for explicit state construction that preserves compile mode.

**Architecture:** Builder structs store immutable reconstruction data and produce fresh `WorkerState` bundles on each call. `_clone_system_evaluator` creates fresh interpreter tapes from shared `InstructionSequence`s and preserves `CompileMode`. `SolveCache{E,B}` dispatches `solve!` on executor type. OhMyThreads `@tasks`/`@local` handles work distribution.

**Tech Stack:** OhMyThreads.jl for threading, existing CommonSolve.jl pattern for `init`/`solve!`.

**Spec:** `docs/superpowers/specs/2026-04-02-executor-threading-design.md`

---

### Task 1: Add OhMyThreads dependency

**Files:**
- Modify: `Project.toml`

- [ ] **Step 1: Add OhMyThreads to Project.toml**

Add to `[deps]`:
```toml
OhMyThreads = "67456a42-1571-4f12-9130-947374cfa36a"
```

Add to `[compat]`:
```toml
OhMyThreads = "0.7"
```

- [ ] **Step 2: Install the dependency**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project -e 'using Pkg; Pkg.instantiate()'`
Expected: Resolves and installs OhMyThreads without errors.

---

### Task 2: Add `compile_mode` field to System

**Files:**
- Modify: `src/core/system.jl`
- Modify: `test/solve_test.jl`

`System` currently does not store its `CompileMode`. The builder needs this to call the correct evaluator constructor. Add a `compile_mode::CompileMode.T` field.

- [ ] **Step 1: Write the failing test**

Add to `test/solve_test.jl`:

```julia
@testset "System stores compile_mode" begin
    @polyvar x y
    F_interp = System([x^2 - 1, y - 2])
    @test F_interp.compile_mode == CompileMode.INTERPRETED

    F_compiled = System([x^2 - 1, y - 2]; compile = CompileMode.COMPILED)
    @test F_compiled.compile_mode == CompileMode.COMPILED
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project -e 'using TestEnv; TestEnv.activate(); include("test/solve_test.jl")'`
Expected: FAIL — `System` has no field `compile_mode`.

- [ ] **Step 3: Add field to System struct**

In `src/core/system.jl`, add `compile_mode::CompileMode.T` after the existing `_interp_t3` field in the `System` struct:

```julia
struct System{P, V}
    polys::FSVec{P}
    parameters::FSVec{V}
    variables::FSVec{V}
    evaluator::SystemEvaluator
    degrees::Vector{Int}
    nvars::Int
    nparams::Int
    variable_groups::Vector{Vector{Int}}
    is_homogeneous::Bool
    support::Vector{Matrix{Int32}}
    coefficients::Vector{Vector{ComplexF64}}
    _interp_f64::Interpreter{Vector{ComplexF64}}
    _interp_df64::Interpreter{Vector{ComplexDF64}}
    _interp_jac::Interpreter{Vector{ComplexF64}}
    _interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2, ComplexF64}}}
    _interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3, ComplexF64}}}
    _interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4, ComplexF64}}}
    compile_mode::CompileMode.T
end
```

- [ ] **Step 4: Pass compile_mode in constructor**

In `_build_compiled_system` (around line 173), update the `return System(...)` call to pass `compile` as the last argument:

```julia
    return System(
        _to_fsvec(polys),
        _to_fsvec(parameters),
        _to_fsvec(variables),
        evaluator, degs, nvars, nparams,
        Vector{Int}[], is_homogeneous,
        supp, coeffs,
        interp_f64, interp_df64, interp_jac,
        interp_t1, interp_t2, interp_t3,
        compile,
    )
```

- [ ] **Step 5: Run tests to verify it passes**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project -e 'using TestEnv; TestEnv.activate(); include("test/solve_test.jl")'`
Expected: The "System stores compile_mode" testset PASSES. All other tests still pass.

---

### Task 3: Create executor types

**Files:**
- Create: `src/solving/executor.jl`
- Modify: `src/HomotopyContinuationNext.jl`
- Modify: `test/solve_test.jl`

- [ ] **Step 1: Write the failing test**

Add `Serial, Threaded` to the `using HomotopyContinuationNext:` imports at the top of `test/solve_test.jl`.

Add inside the `@testset "Solve"` block:

```julia
@testset "Executor types" begin
    @test Serial() isa HomotopyContinuationNext.AbstractExecutor
    @test Threaded() isa HomotopyContinuationNext.AbstractExecutor
    @test Threaded().ntasks == Threads.nthreads()
    @test Threaded(1).ntasks == 1
    # Cannot exceed available threads
    @test_throws ArgumentError Threaded(Threads.nthreads() + 1)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project -e 'using TestEnv; TestEnv.activate(); include("test/solve_test.jl")'`
Expected: FAIL — `Serial` and `Threaded` not defined.

- [ ] **Step 3: Create `src/solving/executor.jl`**

```julia
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
        ntasks > Threads.nthreads() && throw(ArgumentError(
            "ntasks ($ntasks) exceeds available threads ($(Threads.nthreads()))"
        ))
        return new(ntasks)
    end
end
Threaded() = Threaded(Threads.nthreads())
```

- [ ] **Step 4: Include and export in main module**

In `src/HomotopyContinuationNext.jl`:

Add `using OhMyThreads: @tasks, @set, @local` after the existing `using` block (around line 16).

Add `include("solving/executor.jl")` before `include("solving/binomial_system.jl")` (around line 76).

Add to exports:
```julia
export Serial, Threaded
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project -e 'using TestEnv; TestEnv.activate(); include("test/solve_test.jl")'`
Expected: The "Executor types" testset PASSES. All other tests still pass.

---

### Task 4: Create `_clone_system_evaluator` and worker state types

**Files:**
- Create: `src/solving/worker_state.jl`
- Modify: `src/HomotopyContinuationNext.jl`
- Modify: `test/solve_test.jl`

- [ ] **Step 1: Write the failing test**

Add `_clone_system_evaluator, TrackingWorkerState, PolyhedralWorkerState` to the `using HomotopyContinuationNext:` imports.

Add inside the `@testset "Solve"` block. Tests use `FSVec`/`FSMat` to match the actual evaluator API:

```julia
using FixedSizeArrays: FixedSizeArray

@testset "_clone_system_evaluator: interpreted" begin
    @polyvar x y
    F = System([x^2 + y - 1, x * y - 2])
    @test F.compile_mode == CompileMode.INTERPRETED

    original = F.evaluator
    cloned = _clone_system_evaluator(F)

    n = 2
    x_test = FixedSizeArray{ComplexF64, 1}(ComplexF64[1.0 + 0.5im, 2.0 - 0.3im])
    p_empty = FixedSizeArray{ComplexF64, 1}(ComplexF64[])

    # evaluate!
    u_orig = FixedSizeArray{ComplexF64, 1}(zeros(ComplexF64, n))
    u_clone = FixedSizeArray{ComplexF64, 1}(zeros(ComplexF64, n))
    original._evaluate!(u_orig, x_test, p_empty)
    cloned._evaluate!(u_clone, x_test, p_empty)
    @test u_orig ≈ u_clone

    # evaluate_and_jacobian!
    U_orig = FixedSizeArray{ComplexF64, 2}(zeros(ComplexF64, n, n))
    U_clone = FixedSizeArray{ComplexF64, 2}(zeros(ComplexF64, n, n))
    original._evaluate_and_jacobian!(u_orig, U_orig, x_test, p_empty)
    cloned._evaluate_and_jacobian!(u_clone, U_clone, x_test, p_empty)
    @test u_orig ≈ u_clone
    @test U_orig ≈ U_clone

    # Independence: calling one does not affect the other
    x_test2 = FixedSizeArray{ComplexF64, 1}(ComplexF64[3.0, 4.0])
    original._evaluate!(u_orig, x_test2, p_empty)
    cloned._evaluate!(u_clone, x_test, p_empty)  # different input
    @test !(u_orig ≈ u_clone)
end

@testset "_clone_system_evaluator: compiled" begin
    @polyvar x y
    F = System([x^2 + y - 1, x * y - 2]; compile = CompileMode.COMPILED)
    @test F.compile_mode == CompileMode.COMPILED

    original = F.evaluator
    cloned = _clone_system_evaluator(F)

    n = 2
    x_test = FixedSizeArray{ComplexF64, 1}(ComplexF64[1.0 + 0.5im, 2.0 - 0.3im])
    p_empty = FixedSizeArray{ComplexF64, 1}(ComplexF64[])

    u_orig = FixedSizeArray{ComplexF64, 1}(zeros(ComplexF64, n))
    u_clone = FixedSizeArray{ComplexF64, 1}(zeros(ComplexF64, n))
    original._evaluate!(u_orig, x_test, p_empty)
    cloned._evaluate!(u_clone, x_test, p_empty)
    @test u_orig ≈ u_clone

    U_orig = FixedSizeArray{ComplexF64, 2}(zeros(ComplexF64, n, n))
    U_clone = FixedSizeArray{ComplexF64, 2}(zeros(ComplexF64, n, n))
    original._evaluate_and_jacobian!(u_orig, U_orig, x_test, p_empty)
    cloned._evaluate_and_jacobian!(u_clone, U_clone, x_test, p_empty)
    @test u_orig ≈ u_clone
    @test U_orig ≈ U_clone
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project -e 'using TestEnv; TestEnv.activate(); include("test/solve_test.jl")'`
Expected: FAIL — `_clone_system_evaluator` not defined.

- [ ] **Step 3: Create `src/solving/worker_state.jl`**

```julia
## Worker state types and system evaluator cloning for thread-safe path tracking.

# ── Worker state bundles ─────────────────────────────────────────────────────

"""
    TrackingWorkerState

Worker-local state for single-phase homotopy tracking (TotalDegree, parameter homotopy).
"""
struct TrackingWorkerState
    tracker::EndgameTracker
end

"""
    PolyhedralWorkerState

Worker-local state for two-phase polyhedral homotopy tracking.
The `toric_homotopy` and `toric_tracker` are coupled — `update_weights!` must mutate
the exact `ToricHomotopy` captured inside the tracker's `HomotopyEvaluator` closures.
"""
struct PolyhedralWorkerState
    toric_homotopy::ToricHomotopy
    toric_tracker::Tracker
    coeff_tracker::EndgameTracker
    x_buffer::Vector{ComplexF64}
end

# ── System evaluator cloning ─────────────────────────────────────────────────

"""
    _clone_system_evaluator(sys::System) -> SystemEvaluator

Create a fresh `SystemEvaluator` from `sys`'s instruction sequences. The new evaluator
has independent interpreter tapes (mutable) but shares the instruction sequences (immutable).
Preserves the compile mode: INTERPRETED rebuilds all 6 interpreters, COMPILED rebuilds
4 interpreters + re-generates @RuntimeGeneratedFunctions for eval/jac.
"""
function _clone_system_evaluator(sys::System)::SystemEvaluator
    seq_eval = sys._interp_f64.sequence
    seq_jac = sys._interp_jac.sequence
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

- [ ] **Step 4: Include in main module**

In `src/HomotopyContinuationNext.jl`, add `include("solving/worker_state.jl")` after `include("solving/executor.jl")`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project -e 'using TestEnv; TestEnv.activate(); include("test/solve_test.jl")'`
Expected: Both "_clone_system_evaluator" testsets PASS.

---

### Task 5: Create builder structs

**Files:**
- Create: `src/solving/builder.jl`
- Modify: `src/HomotopyContinuationNext.jl`
- Modify: `test/solve_test.jl`

- [ ] **Step 1: Write the failing test**

Add `StraightLineBuilder, CoefficientBuilder, PolyhedralBuilder` to the `using HomotopyContinuationNext:` imports.

Add inside the `@testset "Solve"` block:

```julia
@testset "StraightLineBuilder produces working tracker" begin
    @polyvar x y
    F = System([x^2 - 1, y^2 - 4])
    builder = StraightLineBuilder(
        F.degrees, F, cis(2π * 0.3),
        TrackerOptions(), EndgameOptions(),
    )
    ws = builder()
    @test ws isa TrackingWorkerState

    # Second call produces independent state
    ws2 = builder()
    @test ws2.tracker !== ws.tracker
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project -e 'using TestEnv; TestEnv.activate(); include("test/solve_test.jl")'`
Expected: FAIL — `StraightLineBuilder` not defined.

- [ ] **Step 3: Create `src/solving/builder.jl`**

```julia
## Builder structs — concrete callable types that produce fresh WorkerState from immutable data.

"""
    StraightLineBuilder

Builder for TotalDegree homotopy. Stores immutable reconstruction data; each call
produces a fresh `TrackingWorkerState` with independent mutable state.
"""
struct StraightLineBuilder
    degrees::Vector{Int}
    target_system::System
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

"""
    CoefficientBuilder

Builder for parameter homotopy via CoefficientHomotopy.
"""
struct CoefficientBuilder
    param_system::System
    start_coeffs::Vector{ComplexF64}
    target_coeffs::Vector{ComplexF64}
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

"""
    PolyhedralBuilder

Builder for two-phase polyhedral homotopy. Returns `PolyhedralWorkerState` with
coupled toric homotopy + tracker.
"""
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
    return PolyhedralWorkerState(
        toric_H, toric_tracker, coeff_tracker, Vector{ComplexF64}(undef, n),
    )
end
```

- [ ] **Step 4: Include in main module**

In `src/HomotopyContinuationNext.jl`, add `include("solving/builder.jl")` after `include("solving/worker_state.jl")`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project -e 'using TestEnv; TestEnv.activate(); include("test/solve_test.jl")'`
Expected: The "StraightLineBuilder produces working tracker" testset PASSES.

---

### Task 6: Parameterize SolveCache and update solve.jl

**Files:**
- Modify: `src/solving/solve.jl`
- Modify: `test/solve_test.jl`

This task replaces `src/solving/solve.jl` entirely: new `SolveCache{E,B}` with executor and builder, updated `init` methods, and both `Serial` and `Threaded` `solve!` dispatches.

- [ ] **Step 1: Write the failing test**

Add inside the `@testset "Solve"` block:

```julia
@testset "SolveCache carries executor and builder" begin
    @polyvar x y
    cache = CommonSolve.init(System([x^2 - 1, y - 2]), TotalDegree(), Serial())
    @test cache isa SolveCache{Serial}
    result = CommonSolve.solve!(cache)
    @test nsolutions(result) >= 1

    cache2 = CommonSolve.init(System([x^2 - 1, y - 2]), TotalDegree(), Threaded())
    @test cache2 isa SolveCache{Threaded}
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project -e 'using TestEnv; TestEnv.activate(); include("test/solve_test.jl")'`
Expected: FAIL — `init` doesn't accept 3rd argument.

- [ ] **Step 3: Replace `src/solving/solve.jl`**

Replace the entire file. Key changes from current:
- `SolveCache{E<:AbstractExecutor, B}` with `executor`, `builder`, `tracker` fields
- `init` methods take executor as 3rd positional arg (default `Threaded()`)
- `solve!` dispatches on `Serial` and `Threaded`
- `solve` convenience methods pass executor through
- All `nthreads` references become `ntasks`

```julia
## solve — CommonSolve.jl integration for polynomial system solving.
#
# Pattern: solve(F, alg, exec) = solve!(init(F, alg, exec))

struct SolveCache{E<:AbstractExecutor, B}
    executor::E
    builder::B
    tracker::EndgameTracker
    start_solutions::Vector{Vector{ComplexF64}}
    seed::UInt32
end

# ── CommonSolve.init: System + TotalDegree ────────────────────────────────

function CommonSolve.init(
        F::System, alg::TotalDegree,
        exec::AbstractExecutor = Threaded(),
    )::SolveCache
    seed = alg.seed

    start_evaluator = _total_degree_startevaluator(F.degrees)
    starts = _total_degree_solutions(F.degrees)

    rng = Random.MersenneTwister(seed)
    γ = cis(2π * rand(rng))
    H = StraightLineHomotopy(start_evaluator, F.evaluator; γ = γ)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval; options = alg.tracker_options)
    eg = EndgameTracker(tracker, alg.endgame_options)

    builder = StraightLineBuilder(
        F.degrees, F, γ, alg.tracker_options, alg.endgame_options,
    )

    return SolveCache(exec, builder, eg, starts, seed)
end

# ── CommonSolve.solve!: serial ─────────────────────────────────────────────

function CommonSolve.solve!(cache::SolveCache{Serial})::Result
    eg = cache.tracker
    path_results = PathResult[]
    sizehint!(path_results, length(cache.start_solutions))

    for x₀ in cache.start_solutions
        track!(eg, x₀)
        push!(path_results, PathResult(eg))
    end

    return Result(path_results, length(cache.start_solutions), cache.seed)
end

# ── CommonSolve.solve!: threaded ───────────────────────────────────────────

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

# ── Convenience: solve(F, alg, exec) ─────────────────────────────────────

"""
    solve(F::System, alg=TotalDegree(), exec=Threaded())

Solve a polynomial system using homotopy continuation.
"""
function solve(
        F::System,
        alg::TotalDegree = TotalDegree(),
        exec::AbstractExecutor = Threaded(),
    )::Result
    return CommonSolve.solve!(CommonSolve.init(F, alg, exec))
end

function solve(F::System, exec::AbstractExecutor)::Result
    return solve(F, TotalDegree(), exec)
end

function solve(
        F::System,
        alg::Polyhedral,
        exec::AbstractExecutor = Threaded(),
    )::Result
    return CommonSolve.solve!(CommonSolve.init(F, alg, exec))
end

# ── Parameter homotopy ─────────────────────────────────────────────────────

function solve(
        F::System,
        starts::AbstractVector{<:AbstractVector{<:Number}},
        exec::AbstractExecutor = Threaded();
        start_parameters::AbstractVector{<:Number},
        target_parameters::AbstractVector{<:Number},
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
    )::Result
    return CommonSolve.solve!(
        CommonSolve.init(
            F, starts, exec;
            start_parameters = start_parameters,
            target_parameters = target_parameters,
            seed = seed,
            tracker_options = tracker_options,
            endgame_options = endgame_options,
        ),
    )
end

function CommonSolve.init(
        F::System,
        starts::AbstractVector{<:AbstractVector{<:Number}},
        exec::AbstractExecutor = Threaded();
        start_parameters::AbstractVector{<:Number},
        target_parameters::AbstractVector{<:Number},
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
    )::SolveCache
    @assert nparameters(F) > 0 "System must have parameters for parameter homotopy"
    @assert length(start_parameters) == nparameters(F) "start_parameters length must match nparameters"
    @assert length(target_parameters) == nparameters(F) "target_parameters length must match nparameters"

    sp = ComplexF64.(start_parameters)
    tp = ComplexF64.(target_parameters)
    H = CoefficientHomotopy(F.evaluator, sp, tp)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval; options = tracker_options)
    eg = EndgameTracker(tracker, endgame_options)

    builder = CoefficientBuilder(F, sp, tp, tracker_options, endgame_options)

    start_sols = [Vector{ComplexF64}(ComplexF64.(s)) for s in starts]

    return SolveCache(exec, builder, eg, start_sols, seed)
end
```

- [ ] **Step 4: Run tests to verify everything passes**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project -e 'using TestEnv; TestEnv.activate(); include("test/solve_test.jl")'`
Expected: All existing tests pass. The new "SolveCache carries executor and builder" test also passes.

---

### Task 7: Parameterize PolyhedralSolveCache and update polyhedral.jl

**Files:**
- Modify: `src/solving/polyhedral.jl`
- Modify: `test/solve_test.jl`

- [ ] **Step 1: Write the failing test**

Add inside the `@testset "Solve"` block:

```julia
@testset "PolyhedralSolveCache carries executor" begin
    @polyvar x y
    cache = CommonSolve.init(System([x^2 - 1, y^2 - 4]), Polyhedral(), Serial())
    @test cache isa PolyhedralSolveCache{Serial}
    result = CommonSolve.solve!(cache)
    @test nsolutions(result) == 4

    cache2 = CommonSolve.init(System([x^2 - 1, y^2 - 4]), Polyhedral(), Threaded())
    @test cache2 isa PolyhedralSolveCache{Threaded}
end
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `init(System, Polyhedral, Serial)` doesn't exist.

- [ ] **Step 3: Update `src/solving/polyhedral.jl`**

Replace `PolyhedralSolveCache` struct:

```julia
struct PolyhedralSolveCache{E<:AbstractExecutor, B<:PolyhedralBuilder, S<:System}
    executor::E
    builder::B
    toric_tracker::Tracker
    coeff_tracker::EndgameTracker
    toric_homotopy::ToricHomotopy
    support::Vector{Matrix{Int32}}
    lifting::Vector{Vector{Int32}}
    start_solutions::Vector{Tuple{MixedSubdivisions.MixedCell, Vector{ComplexF64}}}
    seed::UInt32
    _param_system::S
end
```

Update `CommonSolve.init` signature to accept executor:

```julia
function CommonSolve.init(
        F::System, alg::Polyhedral,
        exec::AbstractExecutor = Threaded(),
    )::PolyhedralSolveCache
```

At the end of `CommonSolve.init`, add the builder and update the return:

```julia
    builder = PolyhedralBuilder(
        param_system, start_coeffs, flat_start, flat_target,
        alg.tracker_options, alg.endgame_options,
    )

    return PolyhedralSolveCache(
        exec, builder,
        toric_tracker, coeff_tracker, toric_H,
        support, lifting,
        all_starts, seed,
        param_system,
    )
```

Replace `CommonSolve.solve!` with serial and threaded dispatches:

```julia
function CommonSolve.solve!(cache::PolyhedralSolveCache{Serial})::Result
    toric_tracker = cache.toric_tracker
    coeff_tracker = cache.coeff_tracker
    toric_H = cache.toric_homotopy
    support = cache.support
    lifting = cache.lifting

    n_paths = length(cache.start_solutions)
    path_results = PathResult[]
    sizehint!(path_results, n_paths)

    n = size(support[1], 1)
    x_buffer = Vector{ComplexF64}(undef, n)

    for (cell, x₀) in cache.start_solutions
        update_weights!(toric_H, support, lifting, cell; min_weight = 1.0)
        code = track!(toric_tracker, x₀; t₁ = complex(0.0), t₀ = complex(1.0))

        if code != TrackerCode.TRACKER_SUCCESS
            push!(path_results, PathResult(toric_tracker))
            continue
        end

        copyto!(x_buffer, toric_tracker.state.x)
        track!(coeff_tracker, x_buffer)
        push!(path_results, PathResult(coeff_tracker))
    end

    return Result(path_results, n_paths, cache.seed)
end

function CommonSolve.solve!(cache::PolyhedralSolveCache{Threaded})::Result
    nt = cache.executor.ntasks
    starts = cache.start_solutions
    n_paths = length(starts)
    results = Vector{PathResult}(undef, n_paths)

    @tasks for i in eachindex(starts)
        @set ntasks = nt
        @local ws = cache.builder()

        cell, x₀ = starts[i]

        update_weights!(ws.toric_homotopy, cache.support, cache.lifting, cell; min_weight = 1.0)
        code = track!(ws.toric_tracker, x₀; t₁ = complex(0.0), t₀ = complex(1.0))

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

- [ ] **Step 4: Run tests to verify everything passes**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project -e 'using TestEnv; TestEnv.activate(); include("test/solve_test.jl")'`
Expected: All existing polyhedral tests pass. The new "PolyhedralSolveCache carries executor" test also passes.

---

### Task 8: Add executor tests for all solve paths

**Files:**
- Modify: `test/solve_test.jl`

- [ ] **Step 1: Add Serial and Threaded tests for TotalDegree**

```julia
@testset "solve: TotalDegree + Serial" begin
    @polyvar x y
    result = solve(System([x^2 - 1, y^2 - 4]), TotalDegree(), Serial())
    @test nsolutions(result) == 4
    @test length(real_solutions(result)) == 4
end

@testset "solve: TotalDegree + Threaded" begin
    @polyvar x y
    result = solve(System([x^2 - 1, y^2 - 4]), TotalDegree(), Threaded())
    @test nsolutions(result) == 4
    @test length(real_solutions(result)) == 4
end

@testset "solve: convenience executor method" begin
    @polyvar x y
    result = solve(System([x^2 - 1, y^2 - 4]), Serial())
    @test nsolutions(result) == 4
end
```

- [ ] **Step 2: Add Serial and Threaded tests for Polyhedral**

```julia
@testset "solve: Polyhedral + Serial" begin
    @polyvar x y
    result = solve(System([x^2 - 1, y^2 - 4]), Polyhedral(), Serial())
    @test nsolutions(result) == 4
    @test length(real_solutions(result)) == 4
end

@testset "solve: Polyhedral + Threaded" begin
    @polyvar x y
    result = solve(System([x^2 - 1, y^2 - 4]), Polyhedral(), Threaded())
    @test nsolutions(result) == 4
    @test length(real_solutions(result)) == 4
end
```

- [ ] **Step 3: Add Serial and Threaded tests for parameter homotopy**

```julia
@testset "Parameter homotopy + Serial" begin
    @polyvar x y a
    F = System([x^2 - a, y^2 - a]; parameters = [a])
    F_fixed = System([x^2 - 1, y^2 - 1])
    r1 = solve(F_fixed)
    r2 = solve(
        F, solutions(r1), Serial();
        start_parameters = [1.0],
        target_parameters = [4.0],
    )
    @test nsolutions(r2) == 4
    for sol in real_solutions(r2)
        @test abs(sol[1]^2 - 4) < 1.0e-6
        @test abs(sol[2]^2 - 4) < 1.0e-6
    end
end

@testset "Parameter homotopy + Threaded" begin
    @polyvar x y a
    F = System([x^2 - a, y^2 - a]; parameters = [a])
    F_fixed = System([x^2 - 1, y^2 - 1])
    r1 = solve(F_fixed)
    r2 = solve(
        F, solutions(r1), Threaded();
        start_parameters = [1.0],
        target_parameters = [4.0],
    )
    @test nsolutions(r2) == 4
    for sol in real_solutions(r2)
        @test abs(sol[1]^2 - 4) < 1.0e-6
        @test abs(sol[2]^2 - 4) < 1.0e-6
    end
end
```

- [ ] **Step 4: Add Serial vs Threaded full consistency tests**

```julia
@testset "Serial vs Threaded: full consistency" begin
    @polyvar x y
    F = System([x^2 + y - 1, x * y - 0.5])
    r_serial = solve(F, TotalDegree(; seed = UInt32(99)), Serial())
    r_threaded = solve(F, TotalDegree(; seed = UInt32(99)), Threaded())

    # Solution counts
    @test nsolutions(r_serial) == nsolutions(r_threaded)
    @test nreal(r_serial) == nreal(r_threaded)
    @test nsingular(r_serial) == nsingular(r_threaded)
    @test nnonsingular(r_serial) == nnonsingular(r_threaded)
    @test nat_infinity(r_serial) == nat_infinity(r_threaded)

    # Path-level accounting
    @test r_serial.tracked_paths == r_threaded.tracked_paths
    n_success_serial = count(is_success, r_serial.path_results)
    n_success_threaded = count(is_success, r_threaded.path_results)
    @test n_success_serial == n_success_threaded

    # Solution sets match (as unordered sets)
    s_serial = sort(solutions(r_serial); by = s -> (real(s[1]), imag(s[1])))
    s_threaded = sort(solutions(r_threaded); by = s -> (real(s[1]), imag(s[1])))
    for (a, b) in zip(s_serial, s_threaded)
        @test a ≈ b atol = 1.0e-6
    end
end

@testset "Serial vs Threaded: cluster structure" begin
    @polyvar x y
    # Katsura-3: 4 variables, up to 8 paths — tests cluster/multiplicity accounting
    @polyvar x0 x1 x2 x3
    F = System(
        [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
    )
    r_serial = solve(F, TotalDegree(; seed = UInt32(7)), Serial())
    r_threaded = solve(F, TotalDegree(; seed = UInt32(7)), Threaded())

    @test nsolutions(r_serial) == nsolutions(r_threaded)
    @test r_serial.tracked_paths == r_threaded.tracked_paths
    @test length(r_serial.clusters) == length(r_threaded.clusters)

    m_serial = sort([length(c) for c in r_serial.clusters])
    m_threaded = sort([length(c) for c in r_threaded.clusters])
    @test m_serial == m_threaded
end
```

- [ ] **Step 5: Run all tests**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project -e 'using TestEnv; TestEnv.activate(); include("test/solve_test.jl")'`
Expected: All tests pass.

---

### Task 9: Update Makefile for multi-thread testing, run full suite, format

**Files:**
- Modify: `Makefile`
- All modified files

- [ ] **Step 1: Update Makefile to run tests with multiple threads**

Check the current `make test` command and ensure it launches Julia with `-t auto` (or `-t 4`) so threaded code paths are actually exercised under real parallelism. Update the test target accordingly.

- [ ] **Step 2: Run the full test suite with threads**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && make test`
Expected: All test files pass with multiple threads, including `explicit_imports_test.jl`, `jet_test.jl`, etc.

If `explicit_imports_test.jl` fails due to new OhMyThreads imports, add any necessary entries to the ignore lists.

- [ ] **Step 3: Format all modified files**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && make format`
Expected: Runic formats all files.

- [ ] **Step 4: Verify tests still pass after formatting**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && make test`
Expected: All tests pass.
