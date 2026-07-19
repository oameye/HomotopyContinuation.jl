# Decisions & Pitfalls

Things you'll forget when you come back to this codebase.

## Pitfalls

### FixedSizeVector{T} is NOT concrete

`FixedSizeVector{T}` and `FixedSizeMatrix{T}` have a free `Mem` parameter. Using them as struct fields causes ~30x performance loss.

```julia
# BAD — not concrete, getindex returns Any
field::FixedSizeVector{Float64}

# GOOD — fully concrete
const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}
field::FSVec{Float64}
```

Verify: `isconcretetype(FixedSizeVector{Float64})` → `false`.

### Scratch slot namespace

`TapeCompiler` gives scratch slots their own disjoint negative range during compilation: constants are positive (`1..nconstants`), params/vars occupy `[-(nparams+nvars), -1]`, and scratch grows downward from `-(nparams+nvars+1)` (`_scratch_base`). Collisions are impossible for any number of constants (registered lazily, so their count is unknown at slot init) or scratch slots. Everything is remapped to the final contiguous layout in `_build_slot_remap`.

### Variable ordering uses the public comparison operator

Canonical variable order is creation order. `_lt_variable` obtains it from the public comparison operator (`isless`), since DynamicPolynomials variables compare with the first-created variable as the "largest" (creation order is the reverse of `isless`). This avoids reflecting into DP-internal fields (`variable_order.order.id`), which is not a stable API. Falls back to symbol-name order for variable types without a comparison.

### SExpr hashes are NOT cached (since Moshi refactor)

Old code cached `_hash::UInt` at construction; Moshi `@data` SExpr recomputes hashes on every `hash()` call. Measured negligible: ~260 ns per top-level cyclic-8 equation hash, and full `System(...)` construction (CSE, hashing, tape compile, FunctionWrappers) stays under 7 ms through cyclic-8/katsura-10, far below the ~16 s first-solve TTFX. Not worth caching.

### `qr!` on FSMat returns QRCompactWY, not QR

LAPACK blocked factorization. Use `Matrix{ComplexF64}` with `LinearAlgebra.qrfactUnblocked!` for custom QR if needed.

### LU ipiv type differs for FSMat

`lu!(FSMat)` returns `LU{ComplexF64, FSMat{ComplexF64}, FSVec{Int64}}` — not `Vector{BlasInt}`.

### `Base.n_avail` is not public

ExplicitImports rejects it. To report the number of jobs waiting in a `Channel`
(threaded monodromy progress display), maintain a manual
`Threads.Atomic{Int}` counter: increment before every `push!` (via an
`enqueue!` closure so no call site can forget), decrement when a worker
receives a job. Clamp with `max(counter[], 0)` when reading, since the
decrement races with the read.

### Closures passed as `parameter_sampler` are JET-analyzed against every method

`MonodromyLoop` has one method taking `AbstractVector` and one taking
`LinearSubspace`; both call `parameter_sampler(base)`. JET analyzes a sampler
closure against BOTH branches even if only one is reachable at runtime, so an
anonymous `pp -> [0; randn(length(pp) - 1)]` fails the JET gate with
`length(::LinearSubspace)`. Fix: use a named function with two methods, where
the unreachable `LinearSubspace` method throws an `ArgumentError`
(see `_zero_first_parameter_sampler` in `monodromy.jl`).

### A comment between a docstring and a struct breaks attachment

```julia
"""
    UniquePoints(...)
"""
# some comment here  <- docstring silently attaches to nothing
struct UniquePoints{T, M, GA}
```

Move the comment inside the struct body or above the docstring.

### System{P, V} type parameters

The `System` struct is parameterized on `P` (polynomial type) and `V` (variable type) from DynamicPolynomials. These only affect the `polys`, `parameters`, and `variables` fields — all evaluation goes through the type-erased `SystemEvaluator`. The parameters are invisible to the tracker.

## Design Decisions

### CSE cannot be bypassed — TESTED, REVERTED

Threshold of 50 expressions tested. Instruction counts regressed 30-65% without `opt_cse`. It is essential regardless of system size. ~400us for cyclic-6 Jacobian (42 exprs).

### IDENTITY elision — SHIPPED

Non-scratch results point directly to their input-block slot instead of emitting IDENTITY copies.

### MULMULSUB interleaving — SHIPPED

Case 3 (1 pos, N neg) and Case 4 (interleave pos/neg for MULMULSUB pairing) in `_compile_sum!`.

### Tree-reduce helper — SHIPPED

`_tree_reduce!` shared by product and sum reduction. Eliminates duplicated pop-4/3/2 pattern.

### Cauchy product deduplication — SHIPPED

`_cauchy_product_exprs(N)` helper shared by mul/muladd/mulsub/submul. Saved ~60 lines.

### ExecInstruction variants replace @generated dispatch — SHIPPED

Old approach: `@generated` functions with enum if-else chain and op-order tuning. New approach: Moshi `@data ExecInstruction` with one variant per OpType, compiled from `Instruction` at construction. Single `execute_instructions!` loop replaces separate eval/jac variants.

### Moshi @data for SExpr — SHIPPED

Old: abstract type hierarchy with 9 concrete subtypes → `Vector{SExpr}` was abstractly typed, forcing dynamic dispatch in CSE. New: `@data SExpr` gives single concrete type `SExprT`. All Dict/Set operations use `SExprT` directly.

### RuntimeGeneratedFunctions for compiled mode — SHIPPED

`codegen.jl` (299 lines) generates Julia functions from `InstructionSequence` at runtime via `@RuntimeGeneratedFunction`. Three modes:
- `INTERPRETED`: all operations go through the tape interpreter
- `COMPILED`: RGF for eval + jacobian, interpreter for Taylor (3–6x kernel speedup, but only 1.1–1.3x end-to-end; see `04_compile_modes.md`)
- `COMPILED_ALL`: RGF for eval, jacobian, and Taylor orders 1–3 (Taylor kernels 1.2–1.8x, end-to-end 1.2–1.4x vs `INTERPRETED`)

Build overhead for `COMPILED_ALL` is 1.3–1.65x vs `COMPILED`, but it strictly dominates `COMPILED` in steady state (a consistent extra ~1.09x).

### Builder/worker-state threading pattern — SHIPPED

Thread-safe parallel path tracking via reconstruction, not cloning. Each `Builder` struct stores immutable data (degrees, system, γ, options) and produces a fresh `WorkerState` with independent mutable state per call. `_clone_system_evaluator` creates new interpreter tapes from shared `InstructionSequence`s and preserves `CompileMode` (re-generates RGFs for COMPILED/COMPILED_ALL). OhMyThreads `@tasks`/`@local` creates one worker state per task, not per path.

Alternative considered: `deepcopy` — rejected because it copies immutable data wastefully and doesn't handle RGFs correctly.

### Task-local RNG for reproducibility — SHIPPED

`solve()` uses `Random.MersenneTwister(seed)` passed explicitly to all `rand`/`randn` calls instead of mutating global RNG with `Random.seed!`. Polyhedral init passes a custom `_lifting_sampler` closure to `MixedSubdivisions.fine_mixed_cells` that draws from the local rng. Same seed → same RNG stream → full reproducibility, but thread-safe for nested parallel calls.

### Union-find solution clustering — SHIPPED

`Result` constructor automatically deduplicates successful paths using O(k²) pairwise comparison with union-find. Tolerance: `max(atol, rtol * max(‖s1‖, ‖s2‖))` in infinity norm. Transitive and order-independent. `multiplicity` tracks cluster sizes.

### Coefficient normalization — SHIPPED

Systems with O(10⁸+) coefficients are automatically scaled to O(1) at construction time to improve numerical conditioning in the tracker.

### Channel-based job queue for threaded monodromy (SHIPPED)

Threaded `monodromy_solve` does not reuse the OhMyThreads `Threaded` executor
from `solve()`. The monodromy workload is dynamic: a finished loop enqueues new
loop jobs, workers update shared statistics mid-flight, and the coordinator
decides termination (target count reached, heuristic stop) while work is in
progress. OhMyThreads' static parallel-map shape cannot express this, so the
port keeps v2's design: a `Channel{LoopTrackingJob}` with one task per thread,
each owning a `MonodromyWorkerState` (fresh evaluator + tracker via the same
`_clone_system_evaluator` mechanism the solve executor uses). This is the one
place with two threading coordinators in the codebase; a shared dynamic work
queue abstraction would currently serve exactly one consumer.

### Explicit kwargs replace v2 option-bag splatting in monodromy (SHIPPED)

v2 forwards `kwargs...` through `MonodromyOptions` and into nested solves. The
repo rule forbids kwargs splatting (blocks inference), so every forwarding
boundary (`monodromy_solve`, `verify_solution_completeness`, internal solves)
enumerates its keywords explicitly. This is the main reason v3's
`monodromy.jl` is ~280 lines larger than v2's.

### Fresh `@polyvar` replaces v2's `@unique_var` (SHIPPED)

`verify_solution_completeness` augments the system with new variables
`t, v[1:m], a[1:n], λ`. v2 uses ModelKit's `@unique_var`; DynamicPolynomials
variables are identity-distinct by construction, so a plain `@polyvar` inside
the function body is sufficient. Substitution uses
`MP.subs(f, p => p .+ λ .* v)` in place of v2's callable `System`.

### Two solution-dedup mechanisms exist (known duplication)

`Result` clustering (`_cluster_solutions` in `result.jl`, union-find with
transitive closure) predates the monodromy port. The port added
`UniquePoints`/`multiplicities` backed by `VoronoiTree` (first-match dedup,
O(n log n), group-action aware), which is what v2 builds Result clustering on.
Consolidating result clustering onto the VoronoiTree is a candidate follow-up
(tracked in `02_status.md` open items) but touches solution-count semantics of
every `solve()`, so it was kept out of the port.

### Overdetermined parameter homotopy stays rectangular (SHIPPED, matches v2)

Square-up and excess filtering apply only to total-degree/polyhedral start systems; the parameter-homotopy path tracks the rectangular system directly with least-squares QR Newton, exactly like v2. No randomization means no excess solutions, and squaring up would introduce them. Shared tradeoff with v2: a least-squares stationary point can be reported as success.

### Certification is a separate subpackage, not a package extension (SHIPPED)

Certification is the only consumer of Arblib (a heavy binary dependency), and
loading Arblib into core cost about 0.34s of load time and thousands of extra
method invalidations. Moving certification to `lib/HomotopyContinuationNextCertification`
takes Arblib out of core entirely (core load about 1.6s to about 0.77s). A
package extension was considered and rejected: every certificate type embeds an
`AcbMatrix` field, so the types cannot be defined without Arblib, and Julia
extensions can only add methods to existing functions, never define or export
new types into the parent. A subpackage is the only isolation that preserves the
full exported certificate API. See `00_architecture.md` for the layout.

### Certificate builders behind a type-parameter barrier (SHIPPED)

`certify_solution` used to return `Union{SolutionCertificate,ExtendedSolutionCertificate}`
because the `extended_certificate::Bool` flag chose the type at runtime. The flag
is now resolved once at the `_certify` boundary and the concrete certificate type
is threaded as a type parameter through `certify_solution` and
`extended_prec_certify_solution`; small `_float64_certificate`/`_arb_certificate`/`_uncertified_certificate`
builders dispatch on `::Type{CertT}`. Every function on the path now infers a
concrete return type.

### Arb fallback state is built lazily (SHIPPED)

The arbitrary-precision `AcbCertCache` (several KB of Arb buffers) is only needed
when the Float64 Krawczyk test fails, which is the exception. `CertificationCache`
is therefore a `mutable struct` (justified) with `const` on every field except
`arb`, which an inner constructor leaves undefined; `_arb(cache)` builds it on
first use. The cache stores the two instruction sequences plus the size so it can
build the fallback without holding a reference to `F`. Each cache is used by a
single task, so the lazy init is not shared across threads.

### Certification allocation hygiene (SHIPPED)

The approximate inverse `C ≈ J⁻¹` is computed in place into a preallocated buffer
via `inv!(lu!(copyto!(C_C64, J_C64)))` instead of `inv(J)` (which allocated a
matrix per solution). The threaded certify driver reuses the caller-provided
`CertificationCache` as one entry of a `Channel`-based cache pool rather than
discarding it.

### Interval-arithmetic boundary fixes (SHIPPED)

Three interval boundary cases were corrected: `0 * Interval` now returns the zero
*interval* (not a scalar, which broke type stability); `Interval(0)/Interval(0)`
(and any `0 ∈ denominator`) returns a NaN interval instead of `[0,0]`; and
`x^0` returns `one(x)` for every `x`, including intervals containing zero.
