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

### Magic constant 10000 (scratch slot base)

`TapeCompiler` uses offset 10001 for scratch slots during compilation, remapped to final layout afterward. Systems with >10000 constants+params+vars would collide. Unlikely in practice but not guarded.

### DynamicPolynomials introspection is fragile

`_variable_creation_id` uses `hasfield`/`getfield` reflection into `variable_order.order.id` for deterministic variable ordering. This can break across DynamicPolynomials versions.

### SExpr hashes are NOT cached (since Moshi refactor)

Old code cached `_hash::UInt` at construction. Moshi `@data` SExpr recomputes hashes on every `hash()` call. For deeply nested expressions used as Dict/Set keys in CSE, this could regress build time on large systems. Not yet benchmarked on cyclic-7/8 Jacobians.

### `_stable_sort!` is insertion sort (O(n²))

Replaces `sort!(..., alg=MergeSort)`. Fine for small arrays (<50) but used on `vertex_list` in `_optimize_instruction_order` which can be hundreds of elements for Jacobian tapes. Consider `sort!(..., alg=InsertionSort)` from Base.

### `qr!` on FSMat returns QRCompactWY, not QR

LAPACK blocked factorization. Use `Matrix{ComplexF64}` with `LinearAlgebra.qrfactUnblocked!` for custom QR if needed.

### LU ipiv type differs for FSMat

`lu!(FSMat)` returns `LU{ComplexF64, FSMat{ComplexF64}, FSVec{Int64}}` — not `Vector{BlasInt}`.

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
- `COMPILED`: RGF for eval + jacobian, interpreter for Taylor (3–6x kernel speedup)
- `COMPILED_ALL`: RGF for eval, jacobian, and Taylor orders 1–3 (~1.3x additional Taylor speedup, ~1.2x end-to-end)

Build overhead for `COMPILED_ALL` is 1.3–1.65x vs `COMPILED`, but the Taylor kernel wins compound over many path steps.

### Builder/worker-state threading pattern — SHIPPED

Thread-safe parallel path tracking via reconstruction, not cloning. Each `Builder` struct stores immutable data (degrees, system, γ, options) and produces a fresh `WorkerState` with independent mutable state per call. `_clone_system_evaluator` creates new interpreter tapes from shared `InstructionSequence`s and preserves `CompileMode` (re-generates RGFs for COMPILED/COMPILED_ALL). OhMyThreads `@tasks`/`@local` creates one worker state per task, not per path.

Alternative considered: `deepcopy` — rejected because it copies immutable data wastefully and doesn't handle RGFs correctly.

### Task-local RNG for reproducibility — SHIPPED

`solve()` uses `Random.MersenneTwister(seed)` passed explicitly to all `rand`/`randn` calls instead of mutating global RNG with `Random.seed!`. Polyhedral init passes a custom `_lifting_sampler` closure to `MixedSubdivisions.fine_mixed_cells` that draws from the local rng. Same seed → same RNG stream → full reproducibility, but thread-safe for nested parallel calls.

### Union-find solution clustering — SHIPPED

`Result` constructor automatically deduplicates successful paths using O(k²) pairwise comparison with union-find. Tolerance: `max(atol, rtol * max(‖s1‖, ‖s2‖))` in infinity norm. Transitive and order-independent. `multiplicity` tracks cluster sizes.

### Coefficient normalization — SHIPPED

Systems with O(10⁸+) coefficients are automatically scaled to O(1) at construction time to improve numerical conditioning in the tracker.

### Overdetermined parameter homotopy stays rectangular (SHIPPED, matches v2)

Square-up and excess filtering apply only to total-degree/polyhedral start systems; the parameter-homotopy path tracks the rectangular system directly with least-squares QR Newton, exactly like v2. No randomization means no excess solutions, and squaring up would introduce them. Shared tradeoff with v2: a least-squares stationary point can be reported as success.
