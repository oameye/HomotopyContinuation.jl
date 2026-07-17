# Status

Last updated: 2026-07-16.

**Reproduce:**
- `make benchmark` — steady-state timings
- `make compare` — v3/v2 ratios
- `julia --project=benchmark benchmark/compare/tracking.jl` — end-to-end solve comparison
- `make test` — test suite (28 test files, parallel via ParallelTestRunner)

## Summary

v3 is a credible replacement for v2's core solve pipeline. Total-degree solving is 1.8–3.6x faster
than v2. Polyhedral solving matches v2 within noise (0.95–1.01x). TTFX (load + first solve) is
22.5s vs v2's 45s. Endgame is at full v2 result parity using v2's default parameters. Threading
and overdetermined systems are done. The main remaining gaps are advanced features (monodromy,
certification, NID) and a distributed executor.

## Feature Checklist

### Done

- [x] `solve(F)` with CommonSolve.jl `init`/`solve!`
- [x] Total-degree and polyhedral start systems
- [x] Parameter homotopy (CoefficientHomotopy with linear parameter interpolation)
- [x] `System{P,V}` type (caches compiled interpreters for all eval modes, stores original MP polys)
- [x] `SystemEvaluator` / `HomotopyEvaluator` type firewall (FunctionWrapper, 10 wrappers each,
  including a DF64-output evaluate for extended-precision residual combining)
- [x] StraightLineHomotopy, CoefficientHomotopy, ToricHomotopy
- [x] Cauchy product Taylor convolution for parametric homotopies
- [x] Two-stage toric reparameterization (weight renormalization when max_weight ≥ 10)
- [x] Predictor-corrector tracker (Padé 2,1 + adaptive step + s-plane Hermite for winding > 1)
- [x] Multi-round iterative refinement in predictor (weighted-norm orders 2–3, inf-norm order 1)
- [x] Newton corrector (α-theory) with DoubleF64 extended precision
- [x] Step control: ω extrapolation, convergence-rate rejection, near-target scaling
- [x] Endgame tracker — Puiseux valuation, winding number estimation, singular Cauchy endgame
  (geometric stepping λ=0.25), at-infinity/at-zero detection, cubic Hermite endpoint prediction,
  jump-to-zero gating for m=1 paths
- [x] Solution deduplication (union-find clustering), multiplicity tracking
- [x] Binomial system solver (HNF), weighted norms, custom LU with Skeel scaling
- [x] Tape interpreter for eval, jacobian, Taylor 1–3, DF64
- [x] CSE optimizer, Moshi ADTs for SExpr/ExecInstruction
- [x] RGF compiled eval+jac backend (`CompileMode.COMPILED`, 3–6x kernel speedup)
- [x] RGF compiled Taylor backend (`CompileMode.COMPILED_ALL`, 1.3–1.7x Taylor kernel speedup, ~1.2x end-to-end)
- [x] Automatic coefficient normalization (scales polynomials with O(10^8+) coefficients to O(1))
- [x] AllocCheck zero-allocation enforcement on all hot paths
- [x] Integration tests from v2 with exact result parity
- [x] Threading via OhMyThreads.jl — `Serial`/`Threaded` executor types, builder/worker-state
  pattern for thread-safe evaluator cloning, `@tasks`/`@local` work distribution
- [x] Overdetermined systems: `RandomizedSystem` square-up (identity block plus random fold of
  the lowest-degree equations, permutation keeps degrees exact), wired into total-degree
  (squared-up evaluator) and polyhedral (merged support/coefficients), excess-solution
  filtering post-pass (Newton on the original system for nonsingular endpoints, residual
  comparison for singular ones), `PATH_EXCESS_SOLUTION` result code and `nexcess_solutions`

### Not Done

- [ ] **Distributed executor** — extend `AbstractExecutor` with a `Distributed` type for multi-process path tracking (Distributed.jl / MPI)
- [ ] Direct polynomial compiler (`polynomial_compiler.jl` exists, deferred)
- [ ] Standalone `newton(F, x0)`, progress bars, path diagnostics
- [ ] Monodromy, certification, witness sets, NID
- [ ] Benchmark CI

## Test Suite

26 test files run in parallel via ParallelTestRunner (`make test`, default 10 workers):

| Category | Files | Purpose |
|----------|-------|---------|
| Quality gates | `aqua_test.jl`, `jet_test.jl`, `explicit_imports_test.jl` | Static analysis, type inference, import hygiene |
| Type safety | `concrete_structs_test.jl` | Verify all struct fields are concretely typed |
| Allocation | `alloc_check_test.jl` | Zero-allocation hot paths (norms, LA, predictor, Newton, tracker, endgame) |
| Primitives | `double_f64_test.jl`, `norms_test.jl`, `linear_algebra_test.jl`, `operations_test.jl` | DoubleF64, WeightedNorm, MatrixWorkspace, op_* functions |
| Model kit | `interpreter_test.jl`, `codegen_test.jl`, `instruction_count_test.jl`, `taylor_test.jl`, `polynomial_input_test.jl` | Tape execution, RGF codegen, instruction regression, Taylor series |
| Core | `core_test.jl` | System/Homotopy construction and evaluation |
| Tracking | `tracking_test.jl`, `endgame_test.jl` | Newton, predictor, path tracking, valuation, winding |
| Solving | `solve_test.jl`, `binomial_system_test.jl`, `polyhedral_regression_test.jl`, `overdetermined_test.jl` | End-to-end solving, executor dispatch, serial/threaded consistency, binomial HNF, polyhedral regression, square-up + excess-solution filtering |
| v2 parity | `compare_v2_primitives_test.jl`, `compare_v2_solve_counts_test.jl`, `compare_v2_solve_match_test.jl`, `v2_parity_test.jl` | Primitive matching, solution counts, solution values, overall parity |
| Misc | `utils_test.jl` | SegmentStepper, stable_sort, etc. |

JET test filters known false positives: MP.variables dispatch (construction-time), Moshi `@match`/`@derive` generated code.

## Performance

Measured 2026-04-03, Julia 1.12.5, single-threaded.

### End-to-end solve vs v2

`CompileMode.COMPILED`, fixed seed `0x4567`, endgame enabled.

#### Total-degree (katsura, chain)

| System | v3/v2 ratio | v3 steps/path | v2 steps/path |
|--------|------------:|--------------:|--------------:|
| katsura-3 | **3.07x** | 36.2 (290 acc, 0 rej) | 87.0 (696 acc, 0 rej) |
| katsura-4 | **2.06x** | 47.3 (757 acc, 0 rej) | 78.8 (1260 acc, 0 rej) |
| katsura-5 | **2.01x** | 56.4 (1806 acc, 0 rej) | 98.5 (3137 acc, 15 rej) |
| chain-3 | **3.62x** | 23.0 (184 acc, 0 rej) | 46.2 (370 acc, 0 rej) |
| chain-4 | **2.35x** | 34.4 (550 acc, 0 rej) | 64.2 (1028 acc, 0 rej) |
| chain-5 | **1.82x** | 40.2 (1288 acc, 0 rej) | 65.0 (2081 acc, 0 rej) |

#### Polyhedral (cyclic, random sparse)

| System | v3/v2 ratio | v3 steps/path | v2 steps/path |
|--------|------------:|--------------:|--------------:|
| cyclic-4 | 0.96x | 77.1 (1188 acc, 46 rej) | 74.3 (1158 acc, 31 rej) |
| cyclic-5 | 1.00x | 50.8 (3556 acc, 0 rej) | 50.8 (3553 acc, 0 rej) |
| sparse-3x3 | 0.95x | 45.0 (1924 acc, 9 rej) | 45.0 (1924 acc, 9 rej) |
| sparse-4x4 | 1.01x | 75.1 (12575 acc, 47 rej) | 75.0 (12551 acc, 47 rej) |
| sparse-5x5 | 0.98x | 83.3 (34080 acc, 85 rej) | 83.4 (34121 acc, 90 rej) |

#### Compiled vs interpreted kernels

| System | Eval speedup | Jac speedup |
|--------|-------------:|------------:|
| katsura-3 | 3.35x | 3.82x |
| katsura-5 | 3.65x | 5.04x |
| katsura-7 | 3.29x | 6.74x |

#### COMPILED_ALL (compiled Taylor) vs INTERPRETED

Scalar-parameter Taylor kernel (parameter-free systems, speedup = interp/all):

| System | Taylor 1 speedup | Taylor 2 speedup | Taylor 3 speedup | Solve speedup | Build overhead |
|--------|------------------:|------------------:|------------------:|--------------:|---------------:|
| katsura-3 | 1.66x | 1.72x | 1.62x | 1.09x | 1.62x |
| katsura-5 | 1.73x | 1.59x | 1.61x | 1.09x | 1.39x |
| katsura-7 | 1.74x | 1.65x | 1.76x | — | 1.29x |

TaylorVector-parameter Taylor kernel (production path — CoefficientHomotopy/ToricHomotopy):

| System | Taylor 1 speedup | Taylor 2 speedup | Taylor 3 speedup |
|--------|------------------:|------------------:|------------------:|
| katsura-3 | 1.48x | 1.72x | 1.48x |
| katsura-5 | 1.51x | 1.65x | 1.25x |
| katsura-7 | 1.30x | 1.63x | 1.20x |

#### TTFX (fresh session)

| Metric | Time |
|--------|------|
| v3 package load | 6.25s |
| v3 first solve() | 16.26s |
| **v3 total (load + solve)** | **22.51s** |
| v2 package load | 1.30s |
| v2 first solve() [:mixed] | 43.78s |
| **v2 total [:mixed]** | **45.08s** |
| v2 first solve() [:none] | 10.79s |
| v2 second solve() (different system) | 5.72s |

v3 is 2x faster than v2[:mixed] on first solve. v2[:none] (interpreter-only) is faster for
first solve because it skips SymEngine compilation, but v3 wins on subsequent solves.
v3's higher package load time (6.25s vs 1.30s) is due to precompiling more code upfront.

### Endgame result parity

| System | v2 result | v3 result |
|--------|-----------|-----------|
| (x-10)^2 | nresults=1, nsingular=1 | same |
| at-infinity | 2 success + 2 at_infinity | same |
| winding family d=2,4,6 | d+1 success each | same |
| Hyperbolic 6,6 | nresults=2, nsingular=2 | same |
| singular multiplicity 3 | nresults=2, nsingular=1, nnonsingular=1 | same |

## Open Items

### Architecture debt

3. Direct polynomial compiler — validate and promote (`polynomial_compiler.jl:6` TODO)
4. Fragile DynamicPolynomials introspection (`_variable_creation_id`)
5. Uncached SExpr hashes (Moshi refactor removed `_hash` fields)
6. O(n²) `_stable_sort!` on potentially large vertex lists
7. Magic constant 10000 (scratch slot placeholder base)

### Infrastructure

8. No benchmark CI — regressions go unnoticed
