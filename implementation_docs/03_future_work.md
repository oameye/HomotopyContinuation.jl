# Future Work — Features Not Covered in v1

This document lists all features from the current HomotopyContinuation.jl (v2) that are
**not part of the initial implementation**. Each entry explains what it does, why it's
deferred, and what's needed to add it.

---

## Tier 1: Add After Core Solve Works

These extend `solve()` with additional start system strategies and user-facing APIs.

### ~~Parameter homotopy (user API)~~ — DONE

Implemented. Uses `CoefficientHomotopy` (which interpolates `F(x; t·p_start + (1-t)·p_target)`):

```julia
solve(F, starts; start_parameters=p₁, target_parameters=p₀)
```

### Multi-homogeneous (variable groups)

**What:** Exploit multi-homogeneous structure for tighter Bézout bounds. Uses `variable_groups` keyword in `solve()`.

**Why deferred:** Single-homogeneous and polyhedral cover most use cases. Multi-homogeneous requires `MultiBezoutIterator` and multi-projective affine charts.

**What's needed:** `MultiBezoutSolutionsIterator`, multi-projective `AffineChartHomotopy`, degree matrix computation.

### Compiled evaluation mode (`:mixed`, `:all`)

**What:** Use `Symbolics.build_function` or `RuntimeGeneratedFunctions` to compile polynomial evaluation into native code instead of interpreting the tape.

**Why deferred:** Interpreter is within ~4% of compiled performance. This is a pure optimization.

**What's needed:** Package extension for Symbolics.jl. Convert DynamicPolynomials → Symbolics Num, call `build_function(; expression=Val{false}, cse=true)`, wrap in FunctionWrapper.

---

## Tier 2: Advanced Algorithms

These are standalone algorithms that build on the core tracker.

### Monodromy solving

**What:** `monodromy_solve(F, solutions, parameters)` — discover solutions by tracking around loops in parameter space. Includes trace test for completeness certification.

**Why deferred:** Requires parameter homotopy, `UniquePoints` deduplication, loop management, and threading. Large feature surface.

**What's needed:** `MonodromySolver` struct, `MonodromyLoop`, `MonodromyOptions` (with typed callback fields — no `Any`), `MonodromyResult`, `find_start_pair` algorithm, trace test.

### Certification (Krawczyk interval method)

**What:** `certify(F, result)` — rigorously certify that computed solutions are close to true solutions using the Krawczyk operator with interval arithmetic.

**Why deferred:** Requires interval arithmetic infrastructure (`Interval{T}`, `IComplex{T}`), Arblib integration for escalating precision, `CertificationCache`, `SolutionCertificate` types.

**What's needed:** `interval_arithmetic.jl` (or use IntervalArithmetic.jl), Arblib.jl as optional dependency, `CertificationCache` with dual-precision workspace, `DistinctSolutionCertificates` using interval trees.

### Witness sets

**What:** `witness_set(F; dim=k)` — compute a witness set for a variety by intersecting with a random linear subspace. Includes `membership` test and `trace_test` for irreducibility.

**Why deferred:** Requires monodromy (for `fill_up!`), linear subspace homotopies, and the full `LinearSubspace` geodesic machinery.

**What's needed:** `WitnessSet` struct, `MembershipCache`, subspace homotopy types (extrinsic, intrinsic, projective), `GrassmannianGeodesic`.

### Numerical irreducible decomposition (NID)

**What:** `nid(F)` — decompose a variety into irreducible components across all dimensions using u-regeneration.

**Why deferred:** Requires witness sets, monodromy, and the full regeneration algorithm (Duff, Leykin & Rodriguez).

**What's needed:** `RegenerationCache`, `WitnessPoints`, `decompose_with_monodromy!`, u-regeneration variable augmentation.

---

## Tier 3: Extensions and Integrations

### Subspace homotopies

**What:** `ExtrinsicSubspaceHomotopy`, `IntrinsicSubspaceHomotopy`, `IntrinsicSubspaceProjectiveHomotopy` — track along Grassmannian geodesics between linear subspaces.

**Why deferred:** Only needed for witness sets and NID.

**What's needed:** `GrassmannianGeodesic` struct, `grassmannian_svd`, geodesic interpolation `γ(t)`, Stiefel coordinate computation.

### SemialgebraicSets integration

**What:** `SemialgebraicSetsHCSolver` — adapter so HomotopyContinuationNext can be used as a solver backend for SemialgebraicSets.jl.

**Why deferred:** Package extension, not core functionality.

**What's needed:** Package extension defining `SemialgebraicSets.solve(V, hcsolver)`. Fix the `options::Any` field from the current codebase — use a properly typed options struct.

### Symbolics.jl input (non-polynomial systems)

**What:** Accept `Symbolics.Num` expressions as input, including non-polynomial functions (sin, cos, exp).

**Why deferred:** Package extension. Requires Symbolics.jl for code generation of non-polynomial evaluation.

**What's needed:** Package extension that converts Symbolics expressions to a `SystemEvaluator` via `build_function`. The abstract `AbstractSystem` interface already supports non-polynomial systems — this is just a convenience layer.

### Sparse Jacobian support

**What:** For large systems (hundreds of variables), use sparse Jacobian computation with matrix coloring to reduce evaluation cost.

**Why deferred:** Only matters for very large systems. Dense Jacobian is fine for typical use.

**What's needed:** `Symbolics.jacobian_sparsity` for sparsity detection, SparseDiffTools.jl for coloring, sparse LU factorization in `MatrixWorkspace`.

---

## Tier 4: Numeric Primitives (add when needed)

### Arblib (arbitrary-precision ball arithmetic)

**What:** `Acb`, `AcbRef`, `AcbRefVector` — arbitrary-precision complex ball arithmetic for certification with escalating precision.

**Why deferred:** Only needed for certification. Heavy dependency.

**What's needed:** `Interpreter{AcbRefVector}` tape backend, `acb_op_*!` in-place operations, Arblib.jl dependency.

### Interval arithmetic

**What:** `Interval{T}`, `IComplex{T}` — rigorous interval arithmetic with outward rounding for certification.

**Why deferred:** Only needed for certification.

**What's needed:** Either port the current `interval_arithmetic.jl` or depend on IntervalArithmetic.jl.

### CompositionSystem

**What:** `g ∘ f` — compose two systems with chain rule for Jacobian and Taylor.

**Why deferred:** Not needed for basic polynomial solving.

**What's needed:** `CompositionSystem` struct with pre-allocated scratch for intermediate evaluation, chain rule in `evaluate_and_jacobian!` and `taylor!`.

---

## Superseded — Same Functionality, Better Design

These v2 features are **not missing** — they are replaced by better alternatives in the new architecture.

| v2 Feature | Replaced By | Why |
|------------|-------------|-----|
| SymEngine FFI (`Expression`, `Variable`) | DynamicPolynomials + `MP.differentiate` | Pure Julia, no C FFI, no reinitialization bug (#643), precompilable |
| `CompiledSystem{ID}` hash-based type parameter | `SystemEvaluator` with FunctionWrapper | Eliminates 28s TTFX — no unique type per system |
| `MixedSystem` (compiled eval + interpreted Taylor) | Interpreter-only (within 4% of compiled) | Single code path, no dual-track complexity |
| LoopVectorization `@avx` in ToricHomotopy | Standard loops | Marginal benefit, causes ~5,000 invalidations |
| StructArrays for Jacobian storage | `FSMat{ComplexF64}` | Benchmarked slower than Matrix; StructArray adds complexity |
| `ResultIterator` with `S::AbstractSolver` field | Concrete `Result` with `Vector{PathResult}` | `AbstractSolver` field was type-unstable |
| Mutable `PathResult` | Immutable `PathResult` | Results should not be mutated after creation |
| Symbol-based return codes (`:success`, etc.) | `EnumX.@enumx` scoped enums | Type-safe, faster comparison, scoped namespace |
| `Parameters.@unpack` | Direct field access or destructuring | One less dependency |
| `Reexport.@reexport` | Explicit re-exports | One less dependency, more transparent |
| `LRUCache` for Grassmannian geodesics | No cache (recompute) | Complexity for marginal benefit in monodromy loops |
| `SimpleGraphs` for DAG reorder | Inline topological sort | Avoids dependency for ~30 lines of code |
