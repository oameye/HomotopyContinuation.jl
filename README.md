# HomotopyContinuationNext.jl

This is a fork of [HomotopyContinuation.jl](https://github.com/JuliaHomotopyContinuation/HomotopyContinuation.jl), with the goals to make the package more type-stable and reduce inference time (TTFX). The eventual goal would be for the package to become HomotopyContinuation v3.

## Feature Parity with HC v2

### Done
- [x] Total degree homotopy
- [x] Polyhedral homotopy (two-phase: toric + coefficient)
- [x] Parameter homotopy (`solve(F, starts; start_parameters, target_parameters)`)
- [x] `System` type (cached interpreter pipeline, replaces v2's `System` + compilation modes)
- [x] `SystemEvaluator` / `HomotopyEvaluator` type firewall (FunctionWrappers)
- [x] `StraightLineHomotopy`, `CoefficientHomotopy`, `ToricHomotopy`
- [x] Predictor-corrector tracker with adaptive stepping
- [x] Extended precision (DoubleF64) path tracking
- [x] Newton corrector with alpha-theory convergence
- [x] Ill-conditioning termination
- [x] `Result` / `PathResult` with `solutions`, `real_solutions`, `nsolutions`, `nreal`
- [x] CommonSolve.jl integration (`init` / `solve!`)
- [x] Seed-based reproducibility

### Tier 1: Core features (needed for general use)
- [ ] **Endgame tracker**: Singular solution handling via Cauchy integral / power series. Required for systems with singular solutions. Without this, paths to singular points fail or return inaccurate results.
- [ ] **Threading**: Parallel path tracking. v2 uses `Threads.@spawn` with dynamic load balancing. Infrastructure is ready (`SolveCache` is parallelizable).
- [ ] **Overdetermined systems**: `RandomizedSystem` to square-up via random linear combinations, plus `excess_solution_check!` to filter false solutions.
- [ ] **Affine charts / projective tracking**: `AffineChartHomotopy` for homogeneous systems. Needed for multi-projective variable groups.
- [ ] **Standalone Newton**: Expose `newton(F, x₀)` as a public API (internal `newton!` already exists).
- [ ] **Path diagnostics**: `path_info(tracker, x₀)` returning per-step data (step sizes, condition numbers, accuracy).
- [ ] **Multiplicity / singularity detection**: Compute multiplicities from clustered solutions, detect singular vs nonsingular paths.
- [ ] **Progress bars**: `ProgressMeter.jl` integration for `solve`.

### Tier 2: Advanced algorithms (research features)
- [ ] **Monodromy solving**: `monodromy_solve(F, solutions, parameters)` — discover solutions via parameter loops. Requires: parameter homotopy (done), `UniquePoints` deduplication, loop management, trace test, group actions.
- [ ] **Certification**: `certify(F, result)` — Krawczyk interval method with Arblib.jl for rigorous solution enclosures.
- [ ] **Witness sets**: `witness_set(F; dim=k)` — compute via random linear section intersection. Requires: linear subspace homotopies, monodromy.
- [ ] **Numerical irreducible decomposition**: `nid(F)` — decompose variety into irreducible components by dimension. Requires: witness sets, monodromy.

### Tier 3: Extensions & optimizations
- [ ] **Linear subspace homotopies**: `ExtrinsicSubspaceHomotopy`, `IntrinsicSubspaceHomotopy` for Grassmannian tracking.
- [ ] **Compiled evaluation mode**: Symbolics.jl package extension to compile polynomial evaluation to native code (v2's `:all` mode). Currently interpreter-only is within ~4% of compiled.
- [ ] **Direct monomial evaluator for polyhedral**: Build `InstructionSequence` directly from support matrices, bypassing the DynamicPolynomials roundtrip in `_build_parametric_system`.
- [ ] **Sparse Jacobian**: Exploit sparsity structure for large systems.
- [ ] **SemialgebraicSets.jl adapter**: Package extension for algebraic set solving backend.

## Development TODO

- [ ] Benchmark interpreter vs v2 compiled mode on standard benchmarks (katsura, cyclic)
- [ ] No-allocation tests for hot paths (tracker step, Newton, predictor)
- [x] ~~Add ConcreteStructs tests~~ Done: `test/concrete_structs_test.jl`
- [x] ~~Review type system and API~~ Done: `System` type, flattened algorithm kwargs, parameter homotopy
