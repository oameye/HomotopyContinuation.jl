# Decisions & Pitfalls

## Pitfalls

### FixedSizeVector{T} is NOT concrete

`FixedSizeVector{T}` and `FixedSizeMatrix{T}` have a free `Mem` parameter (`isconcretetype` returns `false`). Using them as struct fields costs ~30x. Use `FSVec{T} = FixedSizeArray{T,1,Memory{T}}` / `FSMat{T} = FixedSizeArray{T,2,Memory{T}}`.

### Scratch slot namespace

`TapeCompiler` gives scratch slots a disjoint negative range: constants positive (`1..nconstants`), params/vars in `[-(nparams+nvars), -1]`, scratch growing downward from `_scratch_base`. Constants are registered lazily so their count is unknown at slot init; the disjoint ranges make collisions impossible. `_build_slot_remap` maps everything to the final contiguous layout.

### Variable ordering uses the public comparison operator

Canonical variable order is creation order. `_lt_variable` derives it from `isless` (DynamicPolynomials makes the first-created variable the "largest", so creation order is reversed `isless`), rather than reflecting into `variable_order.order.id`, which is not stable API. Falls back to symbol-name order for variable types without a comparison.

### SExpr hashes are NOT cached

Moshi `@data` SExpr recomputes hashes on every `hash()` call. ~260 ns per top-level cyclic-8 equation; full `System(...)` construction stays under 7 ms through cyclic-8/katsura-10. Not worth caching.

### `MP.maxdegree` counts parameters as variables

`MP.maxdegree(p)` is the total degree over *every* variable, including parameters: it reports `[3, 2]` for `[a*x^2 + y, x*y - a]` where the degrees in `[x, y]` are `[2, 2]`, giving the total-degree route 6 paths instead of 4. `_variable_degrees` sums only exponents of declared variables. Latent because every internal caller substitutes parameters first, but `degrees` is public API.

### `qr!` on FSMat returns QRCompactWY, not QR

LAPACK blocked factorization. Use `Matrix{ComplexF64}` with `LinearAlgebra.qrfactUnblocked!` for custom QR.

### LU ipiv type differs for FSMat

`lu!(FSMat)` returns `LU{ComplexF64, FSMat{ComplexF64}, FSVec{Int64}}`, not `Vector{BlasInt}`.

### `Base.n_avail` is not public

ExplicitImports rejects it. For queued-job counts (threaded monodromy progress) maintain a `Threads.Atomic{Int}`: increment before every `push!` (via an `enqueue!` closure so no call site can forget), decrement when a worker receives a job, and clamp with `max(counter[], 0)` since the decrement races with the read.

### Closures passed as `parameter_sampler` are JET-analyzed against every method

`MonodromyLoop` has methods taking `AbstractVector` and `LinearSubspace`; both call `parameter_sampler(rng, base)`. JET analyzes a sampler closure against BOTH even if only one is reachable, so `(rng, pp) -> [0; randn(rng, length(pp) - 1)]` fails the gate with `length(::LinearSubspace)`. Use a named function with two methods whose `LinearSubspace` method throws (`_zero_first_parameter_sampler`).

### Anything between a docstring and a definition breaks attachment

A comment line *or* a blank line, before any definition form. The string becomes a discarded literal and the binding is left undocumented, silently.

```julia
"""
    UniquePoints(...)
"""
# comment here  <- docstring silently attaches to nothing
struct UniquePoints{T, M, GA}
```

Confirm with `Base.Docs.doc(Base.Docs.Binding(HomotopyContinuationNext, :name))`. `@doc T` on an interpolated type object does not test this: it resolves to `DataType`'s docstring and looks fine either way.

### System{P, V} type parameters

`P` (polynomial type) and `V` (variable type) affect only the `polys`, `parameters`, `variables` fields. All evaluation goes through the type-erased `SystemEvaluator`; the parameters are invisible to the tracker.

## Design Decisions

### CSE cannot be bypassed (TESTED, REVERTED)

A 50-expression bypass threshold regressed instruction counts 30-65%. `opt_cse` is essential at every system size. ~400us for cyclic-6 Jacobian (42 exprs).

### Interpreter and codegen

- **IDENTITY elision**: non-scratch results point directly at their input-block slot.
- **MULMULSUB interleaving**: cases 3 (1 pos, N neg) and 4 (interleave pos/neg) in `_compile_sum!`.
- **`_tree_reduce!`**: shared by product and sum reduction.
- **`_cauchy_product_exprs(N)`**: shared by mul/muladd/mulsub/submul.
- **`@data ExecInstruction`**: one variant per OpType, compiled from `Instruction` at construction. A single `execute_instructions!` loop replaces the old `@generated` enum if-else chains and separate eval/jac variants.
- **`@data SExpr`**: single concrete type `SExprT`, so `Vector{SExprT}` and the CSE Dict/Set operations are concrete. The old abstract hierarchy forced dynamic dispatch in CSE.

### RuntimeGeneratedFunctions for compiled mode

`codegen.jl` generates Julia functions from `InstructionSequence` via `@RuntimeGeneratedFunction`. Three modes:
- `INTERPRETED`: everything through the tape interpreter
- `COMPILED`: RGF for eval + jacobian, interpreter for Taylor (3–6x kernel, 1.1–1.3x end-to-end; see `04_compile_modes.md`)
- `COMPILED_ALL`: RGF for eval, jacobian, and Taylor orders 1–3 (Taylor kernels 1.2–1.8x, end-to-end 1.2–1.4x)

`COMPILED_ALL` costs 1.3–1.65x build over `COMPILED` and strictly dominates it in steady state (~1.09x).

### Builder/worker-state threading pattern

Thread safety by reconstruction, not cloning. `Builder` stores immutable data (degrees, system, γ, options) and produces a fresh `WorkerState` per call. `_clone_system_evaluator` builds new interpreter tapes from shared `InstructionSequence`s and preserves `CompileMode` (re-generating RGFs). OhMyThreads `@tasks`/`@local` creates one worker state per task, not per path. `deepcopy` was rejected: it copies immutable data wastefully and mishandles RGFs.

### Distributed.jl over MPI or Dagger

Path tracking is an embarrassingly parallel flat map: no inter-path communication, and one start vector in, one `PathResult` out. Interconnect bandwidth and latency, which is MPI's advantage, buys nothing here; what matters is dynamic load balancing, because per-path cost ranges from a handful of steps to `max_steps`. Distributed.jl is a stdlib, gives `RemoteChannel` for free, and is what the cluster launchers (`SlurmClusterManager.jl`, `ClusterManagers.jl`) target, so a user whose workers are already up can use it. Dagger.jl schedules heterogeneous DAGs with data dependencies, which a one-level fan-out does not need, and its scheduler would have to be fought for reproducible result ordering. DistributedArrays.jl distributes array data, which is not the shape of this problem. `DistributedNext.jl` was considered for its multithreaded-worker fixes, but it cannot interoperate with Distributed at all (its workers and Distributed's are mutually unreachable), so choosing it would lock out anyone whose workers were launched by a Distributed-based manager. The relevant thread-safety concern is avoided by design instead: only one task per process touches the sockets.

### One socket task per process, batches over a channel

Batches of path indices go out on a `RemoteChannel`; each process takes one, runs it across its tasks, and puts the results back. Static contiguous chunks were rejected because a process that draws a run of expensive paths would hold up the whole solve. The per-process loop is sequential with respect to the channels: it takes a batch, fans the batch out to its tasks with an atomic index counter, and only then puts results back, so no worker has several threads writing to a socket. Worker states outlive the batch that created them, since a state carries fresh interpreter tapes and, under a compiled mode, freshly generated code. They are built lazily rather than up front: a process builds a state only when a task is there to use it, and a process the batch queue is too short to reach is never called at all, so a small solve on a large or highly threaded cluster does not pay to build a full pool everywhere. Results are written at their global path index, which keeps `DistributedExecutor` output identical to `Serial()` and independent of the order batches return in.

### Monodromy keeps its shared state on the calling process

Monodromy is not a flat map: the job queue grows as solutions are found, and every result has
to be deduplicated against every solution so far. Rather than make `UniquePoints` and the trace
matrix cross-process, `DistributedExecutor` hands out only `track_loop!`. The driver keeps the
queue, the dedup set, the trace matrix, the statistics and the loop list, so the shared state
stays single-writer and the dispatch order is the serial one. The algorithm is therefore
unchanged and `add!` needs no lock. Measured on Steiner (3264 solutions, ~6 ms per loop), the
driver spends 0.005 ms per result on dedup, scheduling and permutations, and 99.5% of its wall
time blocked waiting for results, so centralizing that work costs nothing.

What it does cost is a channel handoff per job. Idle, a `RemoteChannel` round trip is 0.15 ms;
with six tasks consuming, each task waits ~5 ms per handoff, the same order as tracking one
loop, so at one job per message a task spends as long waiting as working. Hence jobs travel in
batches (`batch_size`, default 8) and the driver holds a partial batch back while every task
still has work, which measured 295 to 382 loops/s on Steiner against 160 to 202 serial. Those
absolute rates move by ±30% run to run on the development machine; the ranking does not.

The gap to `Threaded(6)` (1015 to 1073 loops/s on the same six cores) is that handoff latency,
and it is neither the payload nor the driver: batching to 32, running the channels in-process
(`pids = [myid()]`), and giving each process a spare thread for its IO task all leave it where
it is. The queue is also short in steady state, since a result enqueues one or two follow-ups,
so batches average 2.3 jobs however large `batch_size` is.

So on one machine `Threaded()` wins by 2x to 4x on equal cores and stays the default.
Distributed monodromy is for what threads cannot reach: several machines, or loops expensive
enough (large systems, extended precision, big witness sets) that a fixed ~5 ms per job
disappears against them.

### Explicit serialization for the system types

`System`, `_SupportSystem` and `CompositionSystem` get `Serialization` methods (in the extension, so core gains no `Serialization` dependency) that write the two `InstructionSequence`s plus the surrounding immutable data and rebuild the evaluator through the existing `_build_mode_evaluator`. The generic serializer does survive a round trip (`Serialization` nulls `Ptr` fields and `FunctionWrappers` re-initializes lazily, and RGFs have their own hooks), but it ships every closure and leans on gensym'd closure type names agreeing across processes. It is also about 4x larger on the wire. The tape is the source of truth, so shipping it and rebuilding is both smaller and independent of closure identity.

The `System` method ships a `LoweredInput` and rebuilds through `_build_compiled_system`, the constructor's own last step, rather than listing the struct's fields positionally. A hand-written mirror of the layout is a second definition of the type that nothing checks: adding a field to `System` used to leave the extension calling an arity that no longer exists, which surfaces on a worker as `method too new to be called from this world context`. Only a new *lowered* field can break the wire format now, and `LoweredInput` is one struct in one file.

### A builder owns its homotopy construction

`init` takes the cache's own tracker from `builder()` rather than assembling the evaluator stack a second time (`_solve_cache` for `SolveCache`, `builder()` directly for `PolyhedralSolveCache`; `WorkerSolveCache` already did this for the subspace and sweep routes). Two transcriptions of one wrapper order is a divergence hazard that only the serial-vs-distributed comparison tests would catch, and it compiles the stack twice per route. On the polyhedral route it also drops a hand-maintained coupling: `update_weights!` has to mutate the exact `ToricHomotopy` the tracker's evaluator closed over, which is a property of `PolyhedralWorkerState` and was a property `init` had to reproduce. The cost is one `_clone_system_evaluator` on the serial path, measured at 2 µs / 95 µs / 361 µs for INTERPRETED / COMPILED / COMPILED_ALL against a `Threaded()` default that already pays it per task.

### Extension hooks instead of extension-defined `solve!`

A method defined in core and overwritten by an extension cannot be precompiled. So core defines `CommonSolve.solve!(::SolveCache{DistributedExecutor})` forwarding to `_distributed_solve!`, whose untyped fallback throws an actionable "load Distributed" error; the extension adds methods on the concrete cache types, which are strictly more specific. Note that `ProgressMeter` depends on `Distributed`, so in practice the extension is always active and that fallback is unreachable for normal use; it is kept as a guard, and the weakdep still keeps `Serialization` and the distributed code itself out of core.

### Task-local RNG for reproducibility

Every route that consumes randomness takes `seed::UInt32 = rand(Random.RandomDevice(), UInt32)`, builds a local `Random.MersenneTwister` from it and passes that to every `rand`/`randn` instead of mutating the global RNG. `Random.seed!` appears nowhere in `src/`, `ext/` or `lib/`. Two properties follow, and both are tested: the same seed reproduces the same result from any global-RNG state, and no route advances the caller's stream. Reproducible and safe under nested parallelism.

`seed` is `UInt32` at every boundary, field and internal positional, with no `Integer` overload. A keyword annotation asserts rather than converts, so `seed = 42` and `seed = 0x1234` are a `TypeError`: callers pass `UInt32(42)` or an eight-hex-digit literal.

Consequences worth knowing before adding a route:

- **Sub-computations take a seed drawn off the route's stream** (`rand(rng, UInt32)`), not the route's own seed. Reusing one seed for repeated calls rebuilds identical randomness: `decompose` and regeneration's `fill_up!` call `_monodromy_solve!` several times expecting different loops each time, and passing the same seed would make them generate the same loops and never converge.
- **A route that must both record `seed` verbatim and seed a sub-computation with it** uses `_tagged_rng(seed, tag)`, whose seed *vector* cannot collide with `MersenneTwister(seed)`'s. `monodromy_solve` needs this: its result records the user's seed, while its setup draws (start pair, chart) must not share the stream its loop generation derives from the same seed.
- **`MersenneTwister` is not thread-safe**, so all draws from one stream must be on one task. Where a draw sits inside a threaded region (`ReuseLoops.RANDOM` in the threaded monodromy worker) each task gets its own stream derived from the seed. Draws that only *look* threaded are fine when they precede the tasks: `membership` and the u-homotopy intersection both pre-draw in the driver, which is also what makes them bit-identical across threading modes.
- **A homotopy constructor's `gamma` default draws from the global RNG**, so a seeded route must pass `gamma` explicitly. Every worker of one solver must also get the *same* gamma, since they track the same homotopy, so it is drawn once and stored (`SubspaceMonodromyBuilder.gamma`) rather than defaulted per worker.
- **`rand`/`randn` with no result to reproduce takes an `rng` instead of a seed**, defaulting to `Random.default_rng()` the way `rand` itself does: `LA.rank`, `corank`, `find_start_pair`, `rand_subspace`, the `MonodromySolver` constructors, and the chart/gamma defaults of the homotopy constructors.

Polyhedral init passes a `_lifting_sampler` closure to `MixedSubdivisions.fine_mixed_cells`.

### Union-find solution clustering

`Result` deduplicates successful paths with O(k²) pairwise comparison plus union-find, tolerance `max(atol, rtol * max(‖s1‖, ‖s2‖))` in infinity norm. Transitive and order-independent; `multiplicity` tracks cluster sizes.

### Coefficient normalization

Systems with O(10⁸+) coefficients are scaled to O(1) at construction for tracker conditioning.

### Channel-based job queue for threaded monodromy

Monodromy does not reuse the OhMyThreads `Threaded` executor. Its workload is dynamic: a finished loop enqueues new jobs, workers update shared statistics mid-flight, and the coordinator decides termination while work is in progress. A static parallel map cannot express that, so monodromy uses a `Channel{LoopTrackingJob}` with one task per thread, each owning a `MonodromyWorkerState` built by `_clone_system_evaluator`. This is the only second threading coordinator in the codebase; a shared dynamic work-queue abstraction would serve exactly one consumer.

### Many-target sweeps thread the (target, path) product

One task per target caps speedup at the target count, and since `Threaded()` is the default a one-target sweep ran serially while paying task overhead (0.97x against its own serial path).

Threading the flattened `(target, path)` index space fixes every regime with one code path. Target-major ordering plus contiguous chunks (OhMyThreads `:batch`) means a task sees a contiguous run and retargets only at target boundaries, so retargets stay at `n_targets + ntasks` rather than `n_targets * ntasks`. That matters because retargeting a subspace homotopy recomputes a Grassmannian geodesic. Speedup versus serial, 12 threads, 125 paths per target:

| targets | 1 | 2 | 3 | 6 | 12 | 24 |
|---|---|---|---|---|---|---|
| per target | 1.36x | 1.58x | 2.26x | 3.42x | 4.52x | 5.11x |
| per (target, path) | 4.31x | 4.20x | 3.70x | 4.69x | 4.94x | 5.38x |

### A subspace homotopy's γ is applied per start, not per retarget

`set_subspaces!` rotates the start subspace by γ for genericity. `target_parameters!` used to route through it passing `H.start`, which had already been rotated, so γ was re-applied on every retarget and after `N` targets the start had been rotated `N+1` times. Monodromy never saw it (it always passes a fresh start), but a sweep did: a target's endpoint depended on how many targets its worker had reached before it, which surfaced as `Threaded` disagreeing with `Serial`. `set_subspaces!` now applies γ and delegates to `_set_subspaces!`, which takes an already-rotated start; `target_parameters!` calls the latter. Retargeting is now equivalent to constructing the homotopy for that target, pinned in `subspace_homotopy_test.jl`.

This was the whole of it: at a fixed seed, a subspace sweep is bit-identical between `Serial`, `Threaded(1..8)` and `DistributedExecutor` at any batch size, in both regimes, and reversing the target order reproduces every endpoint exactly. A residual order dependence appeared to survive the fix only because two `solve` calls that are not given a `seed` each draw their own, and the seed picks γ, so the comparison was between two different homotopies. Compare sweeps at a fixed seed or the tail digits will move for that reason alone.

A target's paths can straddle a chunk boundary, so no task is guaranteed to close a target: progress counts down a per-target atomic and reports the target solved by whichever task takes its last path. Deriving it from `paths_done ÷ n_paths` would overstate completion while tasks sit mid-target.

### `transform_parameters` is applied exactly once per target

The sweep entry point must transform the first target before `_run_sweep`, since building the homotopy needs actual parameter values. Transforming again inside the loop made N targets cost N+1 calls, and the homotopy was then built for a target never reported, so the first entry belonged to a different problem than its label. Fixed by passing the transformed value in as `first_q`; the first transform cannot be deferred. Guarded by counting closures over both executors and 1, 3, 20 targets.

The same restructure collapsed four accumulate functions (serial/threaded × nested/flatten) into two `_sweep_entries` methods plus a flatten step. The serial method transforms each `Result` before retargeting, so a long sweep holds only transformed entries; the threaded method cannot, because the whole `(target, path)` product gets chunked.

### Parameter values are positional; a route that needs the equations takes a system

`TotalDegree` and `Polyhedral` build their start system from the target's coefficients, which exist only once parameters have values. Given a parametric system both used to run against an unfilled parameter buffer (all parameters zero) and return confident wrong answers: `solve(System([x^2+y^2-a, x*y-1]; parameters=[a]))` reported four "solutions" satisfying `x^2 + y^2 = 0`. Every route now runs `_check_parameter_free` beside `_check_square_or_overdetermined` and names the fix in the message; the polyhedral `support_coefficients` guard fired too late to be the front line.

The rule the whole surface follows: **a route that evaluates `F(x; p)` takes values, positionally. A route that needs the equations themselves takes a system, and `fix_parameters(F, p)` is the one operation that produces a parameter-free one. No parameter value is ever a keyword argument.**

The evaluation-versus-construction line is what makes it exception-free. Certification, a monodromy base point and the two ends of a homotopy only evaluate `F` at a value, so they take the value: `certify(F, X, p)`, `monodromy_solve(F, S, p)`, `solve(F, starts, p_start, p_target)`, `ParameterHomotopy(F, p₁, p₀)`. Total degree and polyhedral build a start system from the target's degrees and monomial support, `slice` appends linear rows to the equations, and `witness_set`/`nid`/`regeneration` store the system and evaluate it later through `membership`, `intersect`, `decompose` and `trace_test`, so those take a system. That deleted 22 `target_parameters::Union{Nothing, AbstractVector{<:Number}}` keywords across five files, and the routes that never had the keyword (`nid`, `regeneration`) gained the capability for free.

The parameter routes went positional to match the subspace routes, which already were: `solve(F, starts, L_start, L_target)` and `solve_targets(F, starts, L_start, Ltargets)`. These are the same two operations, and the spelling used to differ only because the moving thing was a `Vector` rather than a `LinearSubspace`.

The one place a value is taken where a system would be expected is `certify`, and the reason is rigor: `p` is enclosed to `max_precision` bits, so what is certified is `F` at an enclosure of `p`. `certify(fix_parameters(F, p), X)` also works but answers a different question, certifying the substituted system whose coefficients are already-rounded `ComplexF64` products of `p`.

### `solve` returns a `Result`, `solve_targets` returns a `Vector`

The many-target route is a separate verb rather than a `solve` method, because the single-vs-many distinction cannot be carried by a positional slot's type. `solve_targets(F, S₀, p₀, 1:20; transform_parameters = i -> table[i])` passes bare numbers as targets, and `1:20` is an `AbstractVector{<:Number}`, indistinguishable from a single 20-value target. Keywords cannot dispatch, so no annotation resolves it either; v2 resolves it with a runtime `!isa(transform_parameters(first(targets)), Number)` branch, which would make the return type value-dependent. A distinct verb frees the slot, keeps metadata targets working, and removes the wart of one function returning `Result` or `Vector{Tuple}` depending on argument types. Only the public verb is new: `sweep.jl`, `_run_sweep` and `_init_parameter_sweep` keep their names.

### Fixing parameters: a `System` substitutes, a composition binds (MEASURED)

`fix_parameters(F::System, p)` substitutes the values into the equations and returns a `System`, so every downstream route sees ordinary parameter-free input and the tracked tapes carry no extra evaluator hop. `fix_parameters(C::CompositionSystem, p)` cannot: a composition has no equations to substitute into, and rebuilding one as a `System` costs 83 s on the symmetroid composition against ~0.1 s to wrap. It returns a `FixedParameterSystem`, which binds the values at the evaluator level and pays the hop knowingly.

Substituting is also what the routes reading the equations require. Polyhedral builds its start system from the *support and coefficients* of the target, and a value can cancel a term outright and remove a support column; witness sets and regeneration rewrite equations rather than only evaluating them.

Binding costs a second FunctionWrapper hop on every kernel call, measured against the substituted system on two systems (6 variables, 32 and 729 paths):

| kernel | bound vs substituted |
|---|---:|
| `evaluate_and_jacobian!` | +11.3% |
| `taylor!` K=1 | +19.4% |
| `taylor!` K=2 | +16.3% |
| `taylor!` K=3 | +11.5% |

End to end that is 3.8% of a 729-path solve (237.5 ms against 228.9 ms), against a one-off CSE pass of 0.8 ms (32-path system) to 1.9 ms (729-path system). Break-even is around 40 paths, so binding wins only where a solve costs 10 ms and loses on everything larger. An earlier version resolved the values inside `init` and chose the representation there, which forked the cache type into `Union{SolveCache{…StraightLineBuilder{System}…}, SolveCache{…StraightLineBuilder{FixedParameterSystem}…}}` and compiled `solve!` and the tracking pipeline behind it twice per route. Deciding at `fix_parameters` instead leaves each route with one cache type.

A `CompositionSystem` binds, because it has no equations to substitute into and rebuilding one as a `System` costs 83 s on the symmetroid composition against ~0.1 s to wrap. It pays the hop knowingly.

Two consequences worth knowing. Substitution renormalizes the equations per parameter value (`System` scales an equation by its coefficient scale), so tracking is not bit-identical across parameter values. And the bound and substituted routes agree only up to floating point, not exactly: 35949 against 35940 total steps on the 729-path system, same solution count and same 41 paths escalating precision, because the parametric tape reads a parameter slot where the substituted tape folds a constant.

`FixedParameterSystem` keeps the source system beside its bound evaluator, because a worker must not share the evaluator's mutable tapes: `_clone_system_evaluator` rebuilds the binding around a fresh clone of the source. It reports the degrees, shape and variable count of the system it wraps, so the shape dispatch and the square-up path treat it as any other parameter-free input. `CloneableSystem` is the union of the three things a route can clone an evaluator from and is the bound on the two straight-line builders. The `AbstractSystem` doing the actual binding is the internal `_BoundParameterSystem`.

The excess-solution checker evaluates the original overdetermined system with an empty parameter vector, so it is handed `F.evaluator`, which for a `FixedParameterSystem` is already the bound one.

### Appended linear rows: Taylor coefficient is `A x_K`, not zero

A **system** wrapper that appends affine rows (`SlicedSystem`, the chart row in `AffineChartSystem`) must report `A x_K` as the order-K Taylor coefficient of those rows, where `x_K` is the highest-order coefficient of the passed `TaylorVector`. Writing `0` is only correct when the caller zeroed that row. `StraightLineHomotopy.taylor!(Val(K))` asks systems for order `K-1` with a *prefix* of the same `TaylorVector`, whose top row is genuinely nonzero, so the shortcut feeds the predictor a wrong cross term: 101/102/80/54 steps versus 49/40/46/39 on the same paths, correct endpoints throughout. The same shortcut in `AffineChartSystem` cost 251 versus 172 steps over 8 seeds on the projective intrinsic subspace route.

A **homotopy** wrapper is the opposite case. `AffineChartHomotopy` writes `0` correctly: a homotopy is only ever the outermost wrapper, so the predictor has already zeroed the highest-order row. Computing `v'x_K` there changed nothing (190 steps either way). The distinction is *nesting depth*, not row shape, so nesting a homotopy inside a homotopy must revisit it.

`evaluate!`/`evaluate_and_jacobian!` equality does not catch this; compare `taylor!` against the rebuilt polynomial system with a nonzero top row.

### One `_endgame_tracker` for every route

`EndgameTracker(Tracker(HomotopyEvaluator(H); options = t), e)` appeared 22 times across `builder.jl`, `solve.jl`, `polyhedral.jl`, `witness_set.jl`, `regeneration.jl`, `monodromy.jl` in three line-breakings; it now lives in `endgame_tracker.jl`. The `H::AbstractHomotopy` argument costs no devirtualization: Julia specializes on concrete argument types anyway, so each caller gets its own specialization and the FunctionWrapper closures still capture the exact homotopy a worker will retarget.

The same pass unified three `_sliced_rows!` loops (ComplexF64 in/out, DF64 in with F64 out, DF64 in/out) into one method parametric in both. `A` and `b` are ComplexF64, so the accumulator follows the input and only the store rounds. The unified loop uses `muladd` where the DF64 versions used `acc += a * b`; on `Complex{DoubleF64}` those differ by at most 2.7e-31 relative (200k draws), fifteen orders below the ComplexF64 they are stored into.

### `MatrixWorkspace` does not factorize at construction

The `qr` field was initialized with `LA.qrfactUnblocked!` on an all-zero matrix purely to produce a value of the right type; `updated!`/`factorize!` overwrite it before any solve reads it. It is now built directly as `LA.QR(factors, τ)`, and a square workspace gets a 0x0 factor buffer since it can never reach the QR branch.

That drops `LA.qrfactUnblocked!` and its callees from every route (the package's own `reflector!`/`reflectorApply!` are distinct methods, so nothing was shared with the path that actually factorizes). Median first call over 4 interleaved fresh-process pairs at `-t 4`: `newton_standard` 6.248s to 6.088s, `total_degree_interpreted_serial` 9.803s to 9.540s, `polyhedral_interpreted_serial` 14.587s to 14.241s (12/12 paired differences negative). Also removes one QR factorization per workspace construction, paid once per worker in a threaded solve.

### Shape dispatch sits behind an inference barrier

`System(...)` is declared to return the unparameterized `System`, which is what keeps the three compile-mode builds from all being inferred. The side effect: `S <: SystemShape` is invisible to callers, so `system_shape(F)` union-split inside `init` and the square-up path (`_square_up`, `ExcessSolutionChecker`, `_randomize_support`, `_randomized_evaluator`) was inferred and partly codegen'd for square input that never runs it.

`init` picks the shaped initializer through `Base.inferencebarrier`, so its body specializes on the concrete `System` and shape dispatch resolves statically. The square-up subtree goes from 1009 CodeInstances (polyhedral) and 222 (total degree) to zero. The same pass fused `_variable_degrees` and `_is_homogeneous`, whose term walks differed only in max-versus-all-equal, so the generic MultivariatePolynomials machinery compiles once.

Median first call over 4 interleaved pairs: `total_degree_interpreted_serial` 9.486s to 9.230s, `polyhedral_interpreted_serial` 14.115s to 13.625s, `witness_set_build` 15.969s to 15.794s. Cost: one dynamic dispatch per `init`.

### Non-polynomial unary interpreter variants are retained (DECIDED)

`_EXEC_INSTRUCTION_SPECS` drives one generated if-chain per tape element type (`execute_instructions!`, `execute_taylor_instructions!`, the Acb interpreter), so every entry contributes to compilation for all tape element types. Removing `OP_SIN`, `OP_COS`, `OP_SQRT` recovers ~0.26s from a polynomial-only first solve.

Rejected: the planned non-polynomial expression frontend must support `sqrt(γ) * x₁ + x₂^2` and its contract includes `sin`/`cos`. Keeping `ExecInstruction` aligned with `OpType` means that frontend can add lowering and differentiation without rebuilding every interpreter backend. Remaining work is tracked under "Non-polynomial expression input" in `02_status.md`.

**The spec order must follow the `@data ExecInstruction` declaration order.** Reordering by measured op frequency looks like a win (`OP_MUL`/`OP_ADD` are 86 and 43 of 234 instructions across katsura-3 and cyclic-7 tapes, yet sit 11th and 9th behind `Cb`, `Inv`, `InvNotZero`, `InvSqr`, which never appear) but made the cyclic-7 Jacobian 387ns to 710ns (+83%, 3/3 pairs). LLVM turns the tag tests into a dense switch, and that only holds while the tested order matches the variant order.

### Extended-precision wrappers are installed on first use

A `FunctionWrapper` compiles its target at construction (`gen_fptr` is `@generated`), so holding `_evaluate_df64!`/`_evaluate_df64_out!` as ordinary fields compiled a second evaluation kernel over a `ComplexDF64` tape for every evaluator. Only `extended_prec_refinement_step!` reaches it, and only near a singular solution: 478ms of unexecuted `DoubleF64` CodeInstances in a `total_degree_interpreted_serial` first call.

Both fields are replaced by `_df64::Base.RefValue{SysEvalDF64Pair}` plus an installer `FunctionWrapper{Nothing, Tuple{}}`; `_df64_evaluators` fills the cache on the first extended-precision call. The installer reaches its builder through `Base.inferencebarrier`, without which compiling the installer would compile the very kernel the laziness defers.

The cache is unsynchronized, for the same reason as `CertificationCache.arb`: every worker-state builder starts from `_clone_system_evaluator`, so a `SystemEvaluator` reached from a task is owned by that task. `test/alloc_check_test.jl` covers the read path (`newton!` reaches `evaluate!` on the extended-precision branch).

### The support is extracted on demand

Only `Polyhedral` reads `support_coefficients(F)`, but `System` computed it for every parameter-free build, walking every term through generic MultivariatePolynomials code: 86ms of a total-degree first call. The `support`/`coefficients` fields are now one `Base.RefValue` filled by the accessor.

The accessor reads a private `_support_input` snapshot rather than the exposed `polys`/`variables`, whose elements can be overwritten. Every other derived quantity is fixed at construction, so reading them lazily would let `Polyhedral` and `TotalDegree` disagree about which system they are solving.

### Row scaling takes the matrix, not the workspace

`MatrixWorkspace <: AbstractMatrix{ComplexF64}`, so passing a workspace to `skeel_row_scaling!(d, A::AbstractMatrix{ComplexF64}, c)` type-checked and compiled the scaling body a second time, reading through `getindex(::MatrixWorkspace, i, j)` at runtime. The two endgame call sites pass `ws.A`, collapsing 112ms of duplicate specialization to 57ms.

### The QR path is erased behind a shape-chosen FunctionWrapper

`factorize!` and `LA.ldiv!` branch on `m == n`, so every square route compiled `qr!`, `qr_ldiv!`, `reflector!`, `ldiv_upper!` for a branch it never takes (156ms exclusive on `total_degree_interpreted_serial`). A bare `Base.inferencebarrier` recovers it but puts a dynamic dispatch on the tracker hot path, which `test/alloc_check_test.jl` rejects.

`MatrixWorkspace` carries `qr_factorize::QRFactorizeFW` and `qr_solve::QRSolveFW`, chosen by shape in `_make_matrix_workspace`. The wrapper call is a `ccall` through a function pointer, so the hot path keeps static resolution, and the tall pair is built via `_tall_qr_ops` through `Base.inferencebarrier`, so nothing about QR is statically reachable from a square workspace. A square-only session compiles zero specializations of `qr!`, `qr_ldiv!`, `reflector!`, `lmul_Q_adj!`, and `ldiv_upper!` drops from 4 to 2. Interleaved fresh-process pairs at `-t 4` favour the wrapper 5/5, ~0.15s median, first-call spread 0.64s to 0.11s. The first tall solve pays ~0.35s to compile the path on demand.

Two accepted costs: `QRSolveFW` needs a concrete `Args` and `ldiv!` takes `x::AbstractVector{ComplexF64}`, so the QR branch solves into a dedicated `qr_x` buffer (length 0 when square) and copies out (`O(n)` against an `O(mn)` solve); and every square workspace allocates two no-op wrappers whose `gen_fptr` thunk is generated once per session.

### The Taylor order stays a type parameter, and its compile cost is accepted

The generated execute loop is compiled once per tape element type, so `TruncatedTaylorSeries{2,3,4}` means three copies: ~1.15s of the `total_degree_interpreted_serial` first call, the largest remaining block of compilation. That cost buys the arithmetic: with `N` in the type every `taylor_op_*` is `@generated` into an unrolled Cauchy product with no loop bounds or length checks, and `TTS{N,T}` is an `NTuple`, so tape slots are inline `isbits` storage.

Two ways to collapse the copies are rejected.

Running every order on one order-4 tape is correct (coefficient `k` depends only on input coefficients up to `k`, so zero-padding is harmless) and would save at most two thirds of the 1.15s. But `predict!` runs `Val(1)`, `Val(2)`, `Val(3)` in the same call on the Padé path, so no low-order prediction is left alone to absorb it: widening takes Cauchy-product multiplications per tape multiply from 3+6+10 to 10+10+10 and doubles memory traffic for the order-1 pass. The singular branch is worst, returning before orders 2 and 3 and so running `Val(1)` alone, 3 multiplications to 10.

Making the order a runtime value shares the loop without widening the arithmetic, but gives up unrolling: convolutions become runtime-bounded loops over 3 to 10 terms, too small to amortize loop overhead, and the coefficient count stops being visible to the optimizer.

Both trade a permanent per-step cost for a one-time 0.7s of compilation, the wrong direction for a workload of thousands of paths per solve. No steady-state number was taken for either; the multiplication counts decide it, since the trade only pays off if the steady-state hit is near zero. Revisit only for a representation that shares the loop while keeping per-order arithmetic unrolled. The support-only executor tried earlier is not it (it duplicated the graph, `05_ttfx_invalidations.md`). Most of the cost was never that the loop is compiled three times but that it was never cached at all, which the next entry recovers.

### Tape executors are precompiled by signature

Right after `using`, `execute_taylor!` and `execute!` had zero specializations: every executor is reached through the `@cfunction` inside a `FunctionWrapper`, which leaves no backedge for the serializer, so no workload can pull them into the package image. Declaring them is sufficient because their signatures name no user type (a tape is data, not a type parameter, so `Interpreter{Vector{TruncatedTaylorSeries{2,ComplexF64}}}` is known at build time) and the wrapper thunks only forward arguments, so the cached out-of-line copy is exactly what they call.

`src/precompile_signatures.jl` declares ten signatures: eval and Jacobian over the `ComplexF64` tape, two extended-precision eval variants, and all three Taylor orders in scalar-parameter and Taylor-parameter form. Each target carries `@noinline` (`execute_taylor!`, `_execute_eval_fw!`, `_execute_jac_fw!`), without which `Base.@propagate_inbounds` force-inlined the executor into every thunk and the cached copy went unused. Every indexing site in those bodies already has an explicit `@inbounds`, and the added call happens once per pass over a whole tape.

On `total_degree_interpreted_serial`, six samples per variant interleaved against a control: 9.766s baseline, 8.819s with the Taylor signatures alone, 8.483s with all ten (-1.28s, -13.1%). Baseline minimum 9.473s exceeds the full variant's maximum 8.780s. Load pays ~25ms. This is not PrecompileTools: `src/precompile.jl` remains disabled, and `precompile` is a Base builtin that adds no dependency and runs no workload.

The `@noinline` annotations sit on the tracker hot path, so they were checked against an interleaved steady-state probe on a cooled machine: `track!`, `step!`, `update!(::Predictor, …)`, `evaluate!`, `evaluate_and_jacobian!` and all three `taylor!` orders on Katsura-3. Nothing regressed beyond run-to-run spread (~±3% at two samples per cell; `track!` alone varies 114.6µs to 122.5µs regardless of variant).

### Dependency invalidations are largely inert (MEASURED)

Loading the package invalidates ~3150 method instances, none triggered by the package's own methods: the worst offenders are broad dependency signatures (`convert(Type{String}, Any)`, `|(Any, Bool)` from InitialValues via BangBang via OhMyThreads, `step(OrdinalRange{Int64,Int64})` from MutableArithmetics, StaticArrays' `similar`). MixedSubdivisions accounts for 2133 instances, DynamicPolynomials 1806, OhMyThreads 523.

This does not translate into first-call time. A probe build that never loads OhMyThreads (`@tasks` bodies discarded by a stub macro) shrank load by 58ms and left the first call unchanged: `total_degree_interpreted_serial` 8.741s to 8.740s, `newton_standard` 5.453s to 5.494s over 3 interleaved pairs. The ~245ms of `Compiler.inferiterate_2arg` re-inference in every first call survives the probe, so dropping OhMyThreads would buy load time only.

### Explicit kwargs in monodromy

Every forwarding boundary (`monodromy_solve`, `verify_solution_completeness`, internal solves) enumerates its keywords explicitly, since the repo forbids kwargs splatting (blocks inference). This is the main reason `monodromy.jl` is ~280 lines larger than its v2 counterpart.

### Fresh `@polyvar` for augmented variables

`verify_solution_completeness` augments the system with `t, v[1:m], a[1:n], λ`. DynamicPolynomials variables are identity-distinct by construction, so a plain `@polyvar` in the function body suffices (no `@unique_var` equivalent needed). Substitution uses `MP.subs(f, p => p .+ λ .* v)`.

### Two solution-dedup mechanisms exist (known duplication)

`Result` clustering (`_cluster_solutions` in `result.jl`, union-find with transitive closure) predates the monodromy port, which added `UniquePoints`/`multiplicities` backed by `VoronoiTree` (first-match dedup, O(n log n), group-action aware). The one place they meet is `_orbit_merge!`: the sweep's sort key `Re(x₁) + Im(x₁)` is not preserved by a group action, so symmetry-aware clustering indexes one representative per proximity cluster in a `VoronoiTree` and unions on a hit. It indexes every representative before querying any of them, and unions a representative with every representative its images land on rather than stopping at the first hit. Both are needed for transitivity: an action is only required to be a generating set, so a full orbit is connected through a chain of single applications, and a chain is only complete if every edge is discovered against the full index. Querying while building would find only the edges pointing back at already-indexed points, which splits an orbit under, for instance, a single generator of a cyclic group. Consolidating the proximity sweep itself onto the VoronoiTree is tracked in `02_status.md`, but it touches solution-count semantics of every `solve()`.

### Overdetermined parameter homotopy stays rectangular

Square-up and excess filtering apply only to total-degree/polyhedral start systems; the parameter-homotopy path tracks the rectangular system directly with least-squares QR Newton. No randomization means no excess solutions, and squaring up would introduce them. Tradeoff: a least-squares stationary point can be reported as success.

### Certification is a separate subpackage, not a package extension

Certification is the only consumer of Arblib; loading it into core cost ~0.34s of load time and thousands of extra invalidations (core load ~1.6s to ~0.77s after the move). An extension was rejected: every certificate type embeds an `AcbMatrix`, so the types cannot be defined without Arblib, and extensions can only add methods to existing functions, never define or export new types. See `00_architecture.md`.

### Certificate builders behind a type-parameter barrier

`certify_solution` used to return `Union{SolutionCertificate,ExtendedSolutionCertificate}` because `extended_certificate::Bool` chose the type at runtime. The flag is resolved once at the `_certify` boundary and the concrete type is threaded as a type parameter through `certify_solution`/`extended_prec_certify_solution`, with `_float64_certificate`/`_arb_certificate`/`_uncertified_certificate` dispatching on `::Type{CertT}`. Every function on the path infers a concrete return type.

### Arb fallback state is built lazily

`AcbCertCache` (several KB of Arb buffers) is only needed when the Float64 Krawczyk test fails. `CertificationCache` is therefore a `mutable struct` with `const` on every field except `arb`, which the inner constructor leaves undefined; `_arb(cache)` builds it on first use from the stored instruction sequences and size, so no reference to `F` is held. Each cache is used by a single task.

### Certification allocation hygiene

The approximate inverse `C ≈ J⁻¹` is computed in place via `inv!(lu!(copyto!(C_C64, J_C64)))` instead of `inv(J)` (a matrix per solution). The threaded certify driver reuses the caller-provided `CertificationCache` as one entry of a `Channel`-based cache pool rather than discarding it.

### Interval-arithmetic boundary fixes

`0 * Interval` returns the zero *interval* (not a scalar, which broke type stability); `Interval(0)/Interval(0)` (and any `0 ∈ denominator`) returns a NaN interval instead of `[0,0]`; `x^0` returns `one(x)` for every `x`, including intervals containing zero.

### Front-end lowering happens before the inference barrier

`System` accepts MultivariatePolynomials input and `Expression` input, and each needs a
different lowering to an `InstructionSequence`. Dispatching on that inside the builder chain
looks natural, but the chain is deliberately `@nospecialize`d (see "Shape dispatch sits behind
an inference barrier"), so `polys` is only known there to be an `AbstractVector`. Both
`_lower_input` methods then apply, and inference walks both: a plain polynomial build was
inferring `_differentiate`, `_eadd`, `_emul`, `_epow`, `expression_to_sexpr` and their callees,
+488 MethodInstances it never executes.

Measured cost of getting this wrong, median of 3 interleaved pairs against the same commit
without the expression front-end: `total_degree_interpreted_serial` 7.46s to 8.14s,
`monodromy_serial` 11.87s to 12.40s, `large_symbolic_interpreted_build` 5.65s to 6.20s. A pure
`System(...)` build regressed as much as a full solve, which is what pointed at construction
rather than tracking.

The fix is to call `_lower_input` from the typed `System` frame, where the input type is still
known, and pass the result through the barrier as a concretely typed `LoweredInput`. Adding a
front-end is therefore adding a `_lower_input` method, never a branch further down.

### Rectangular intervals implement sqrt, sin and cos

v2 has no `sqrt`/`sin`/`cos` for its rectangular interval type and certifies any system using
them with Arb from the start. v3 implements them, so those systems take the Float64 Krawczyk
path and only escalate when the test actually fails.

The complex `sqrt` needs care. The textbook form `√z = u + i·sign(Im z)·v` with
`u = √((|z| + Re z)/2)` and `v = √((|z| - Re z)/2)` is sound but useless near the positive real
axis, which is exactly where a box around a real solution sits: `|z| - Re z` cancels to nothing
and `v` comes out as the square root of the box width. A 1e-8-wide box around 4 gave an
imaginary radius of 1e-4, far too wide for the inclusion test. Computing only the
well-conditioned root and recovering the other from `2uv = Im z` restores a radius proportional
to the box width (2.5e-9 for that box).

`sin`/`cos` bound the endpoints with two ulps of slack and saturate to `[-1, 1]` when an
extremum can lie inside, tested against a lattice built from an enclosure of `π`. The test
answers "true" when unsure, which only widens the result. Soundness is checked by sampling
random boxes in `interval_arithmetic_test.jl`.


### `det` on expressions expands by cofactors

`LinearAlgebra.det` factors the matrix: it calls `abs` to choose a pivot and divides to
eliminate. Neither works on a symbolic entry, and dividing would build a rational expression
where a polynomial one exists. `det(::AbstractMatrix{Expression})` therefore expands along the
first row, skipping zero entries. It is only ever called on the small matrices that show up in
modeling (tangency conditions, minors), never on a numeric hot path.

Without it, models that go through a determinant have to inline their own cofactor expansion,
which is what v2's tangency tests and the ported certification regression used to do.


### Canonicalization cancels rational expressions, and that is the contract

`_emul` collects repeated bases and adds their exponents, so `x/x` folds to `1`, `x^2/x` to `x`
and `(x*y)/x` to `y`, at construction time and before any evaluator exists. `System([x/x - y, x])`
is therefore the system `[1 - y, x]`: it evaluates at `x = 0`, and `certify` will happily
certify `(0, 1)` even though the expression the user typed is undefined there.

This is what every canonicalizing front-end does (v2's SymEngine layer folds the same three
examples identically) and the folded system is what `show` prints, so the simplification is
visible rather than hidden. Tracking the domain of each denominator through the tree would mean
carrying a side condition on every expression and giving certification a second obligation to
discharge, for a class of input that is a modeling mistake rather than a solving problem. The
front-end's guarantee is about the canonical form it builds, not about the poles of the literal
input.

### `conj` conjugates literals, `transpose` is the identity

`Expression <: Number`, so `conj` has to conjugate a numeric literal: leaving `Expression(2im)`
alone breaks the `Number` contract outright. It recurses into the tree and conjugates the
`ENum` leaves, which makes `adjoint((1+2im)*x)` agree with what DynamicPolynomials produces for
the same input, so a model written against either front-end conjugates its coefficients the
same way.

Variables are left alone: the tape has no conjugation instruction, so a variable stands for a
real symbol and `'` on a matrix of expressions is a transpose of conjugated coefficients. This
differs from v2, which left the coefficients unconjugated too.

### `sin`/`cos` on `DoubleF64` sum the limbs for large arguments

Reducing `a` modulo `2π` in double-double arithmetic costs about `|a|·2⁻¹⁰⁶` radians of absolute
accuracy, because the product `2π·round(a/2π)` is rounded at the double-double epsilon. Past
`|a| = 2⁵³` the reduced angle is worth less than a Float64 ulp, and by `2¹⁰⁶` it has no correct
bits at all: `sin(DoubleF64(1e32))` used to return exactly `0` against a true value of `0.585`.

Above `2⁵³`, `sincos` therefore evaluates `sin(hi + lo)` through the angle-addition formula
instead. `hi + lo` is exact by construction and `Base.sin`/`Base.cos` reduce each limb with a
full Payne-Hanek reduction, so the result keeps Float64 accuracy at any magnitude rather than
degrading to noise. Full double-double accuracy above `2⁵³` would need a Payne-Hanek reduction
against a bit table of `2/π`, which no caller has asked for.

`sinh`/`cosh` take a similar branch: above `|a| = 40` the term `e^-|a|` is below the
double-double ulp of `e^|a|`, so both functions are `exp(|a| - log 2)`. Forming `e ± 1/e` there
divides by a zero or infinite `exp` and yields NaN where the true value is finite (`sinh(710)`)
or infinite (`sinh(1000)`).
