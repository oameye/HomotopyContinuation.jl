# Status

Last updated: 2026-04-01.

**Reproduce:** `make benchmark` (timings), `make compare` (v3/v2 ratios), `make test` (instruction counts)

## Feature Checklist

### Done

- [x] `solve(F)` with CommonSolve.jl `init`/`solve!`
- [x] Total-degree start system (Bezout bound)
- [x] Polyhedral start system (two-phase: toric + coefficient, MixedSubdivisions.jl)
- [x] Parameter homotopy (`solve(F, starts; start_parameters, target_parameters)`)
- [x] `System` type (caches compiled interpreters for all eval modes)
- [x] `SystemEvaluator` / `HomotopyEvaluator` type firewall (FunctionWrapper)
- [x] StraightLineHomotopy, CoefficientHomotopy, ToricHomotopy
- [x] Predictor-corrector tracker (Pade 2,1 + adaptive step)
- [x] Newton corrector (alpha-theory, up to 11 iterations)
- [x] Extended precision (DoubleF64) iterative refinement
- [x] Weighted norms, custom LU with Skeel scaling
- [x] Binomial system solver (HNF)
- [x] `Result` / `PathResult` with `solutions`, `real_solutions`, `nsolutions`, `nreal`
- [x] Seed reproducibility
- [x] Tape interpreter for eval, jacobian, Taylor orders 1-3, DF64
- [x] CSE optimizer (SymEngine port)
- [x] Moshi @data SExpr + ExecInstruction ADTs

### Not Done

**Tier 1 — needed for general use:**

- [ ] **Endgame tracker** — singular solutions fail without this
- [ ] **Threading** — parallel path tracking (infrastructure ready, just needs `@spawn`)
- [ ] **Overdetermined systems** — `RandomizedSystem` + excess solution filtering
- [ ] **Standalone `newton(F, x0)`** — internal `newton!` exists, not exposed
- [ ] **Progress bars** — ProgressMeter.jl integration
- [ ] **Affine charts** — multi-projective variable groups
- [ ] **Multi-homogeneous** — `MultiBezoutIterator` for tighter bounds
- [ ] **Path diagnostics** — per-step data export
- [ ] **Multiplicity detection** — from clustered solutions

**Tier 2 — advanced algorithms:**

- [ ] Monodromy solving (requires UniquePoints, loop management, trace test)
- [ ] Certification (Krawczyk + Arblib.jl interval arithmetic)
- [ ] Witness sets (requires monodromy + subspace homotopies)
- [ ] NID (requires witness sets)

**Tier 3 — extensions:**

- [ ] Compiled eval mode (Symbolics.jl package extension — interpreter matches compiled, see `benchmark/compare/v2_modes.jl`)
- [ ] Sparse Jacobian (matrix coloring, only for large systems)
- [ ] Linear subspace homotopies (Grassmannian)
- [ ] SemialgebraicSets.jl adapter (package extension)

## Performance

Re-run `make benchmark` for absolute timings, `make compare` for v3/v2 ratios.

### Steady-state execution (raw interpreter, zero allocation)

From `benchmarks_output.json` (pre-Moshi refactor). Re-run `make benchmark` for current numbers.

| Benchmark | Time |
|-----------|------|
| eval katsura-3 (4x4) | 57 ns |
| eval cyclic-7 (7x7) | 131 ns |
| jac katsura-3 (4x4) | 104 ns |
| jac cyclic-7 (7x7) | 389 ns |
| taylor katsura-3 (order 3) | 101 ns |
| track one path katsura-3 | 551 us |
| build katsura-3 (full System()) | 664 us |
| build cyclic-7 (full System()) | 2.37 ms |

### v3/v2 interpreter ratios (> 1.0 = v3 faster)

From `benchmark/compare/interpreter.jl` (post-Moshi refactor, 2026-04-01).

| Metric | Min | Median | Max |
|--------|----:|-------:|----:|
| Eval | 0.70x | 1.25x | 1.55x |
| Jacobian | 1.21x | 1.39x | 1.82x |
| Build | 1.18x | 1.31x | 2.17x |

### TTFX (from `benchmark/compare/ttfx.jl`, fresh session)

| Metric | Time |
|--------|------|
| v3 first solve | ~13s |
| v2 first solve [:mixed] | ~44s |
| v2 first solve [:none] | ~11s |

### Instruction counts (regression test limits)

| System | Eval instrs |
|--------|------------:|
| cyclic-3/4/5/6/7 | 6 / 11 / 19 / 28 / 38 |
| chain-3/4/5/6/7 | 17 / 23 / 28 / 34 / 39 |
| dense-quad-3/4/5/6 | 28 / 52 / 92 / 138 |
| sparse 6x6 (random) | 82-96 |
| sparse 8x8 (random) | 162-170 |

## Open Items

1. **Benchmark CSE build time** on cyclic-7/8 after Moshi refactor (hash caching removed)
2. **Replace `_stable_sort!`** insertion sort with Base `InsertionSort` or restore `MergeSort` for larger inputs
3. **Endgame** is the single most impactful missing feature — blocks correctness on singular systems
4. **Threading** is required for competitive performance on large systems (cyclic-7: 924 paths)
5. **No benchmark CI** — comparison benchmarks are manual, regressions can go unnoticed
6. **No-allocation tests** for hot paths (tracker step, Newton, predictor) — should be enforced

## Dependencies

| Package | Purpose |
|---------|---------|
| CommonSolve | `init`/`solve!` interface |
| DynamicPolynomials | `@polyvar`, user-facing polynomial input |
| EnumX | Scoped enums (TrackerCode, PathResultCode, etc.) |
| FixedSizeArrays | FSVec/FSMat (size not in type parameter) |
| FunctionWrappers | Type erasure for SystemEvaluator/HomotopyEvaluator |
| LinearAlgebra | stdlib LU/QR |
| MixedSubdivisions | BKK mixed volume for polyhedral start system |
| Moshi | `@data` ADT for SExpr and ExecInstruction |
| MultivariatePolynomials | Abstract polynomial interface, differentiation |
