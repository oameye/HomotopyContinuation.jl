# Status

Last updated: 2026-04-01.

**Reproduce:**
- `make benchmark` — steady-state timings
- `make compare` — v3/v2 ratios
- `julia --project=benchmark benchmark/compare/trace_tracker.jl 3` — raw tracker trace on katsura-3
- `make test` — test suite

## Honest Assessment

This is still not a v2 replacement, but the core tracker is in much better shape than it was a
few days ago.

- **Endgame is still missing**. This is now the clearest correctness and performance blocker.
  Raw tracker traces still show large late-path gaps on hard paths, and v2's endgame is part of
  why it terminates those cases much earlier.
- **Core regular-path tracking improved substantially.** The straight-line homotopy Taylor
  formulas were wrong at orders 2 and 3 when `x` and `t` varied together. Fixing those cross
  terms, plus the Newton/predictor parity work, reduced the current katsura compare from roughly
  `234 / 197 / 406` steps per path to about `77 / 105 / 154` accepted steps per path on
  katsura-3/4/5.
- **The current end-to-end katsura compare is around parity or better in wall time**, but those
  numbers are still provisional: `benchmark/compare/tracking.jl` is currently unseeded and
  reports accepted steps only.
- **No threading, no overdetermined systems, no progress bars.**

The strongest honest claim is now:

**v3 has a solid monomorphic core and a much better regular-path tracker, but it is still missing
the late-path/endgame machinery required to call it a real v2 replacement.**

## Feature Checklist

### Done

- [x] `solve(F)` with CommonSolve.jl `init`/`solve!`
- [x] Total-degree and polyhedral start systems
- [x] Parameter homotopy
- [x] `System` type (caches compiled interpreters for all eval modes)
- [x] `SystemEvaluator` / `HomotopyEvaluator` type firewall (FunctionWrapper)
- [x] StraightLineHomotopy, CoefficientHomotopy, ToricHomotopy
- [x] Predictor-corrector tracker (Pade 2,1 + adaptive step)
- [x] Newton corrector (alpha-theory), DoubleF64 refinement
- [x] Extended precision support in Newton corrector and tracker (v2 parity)
- [x] Step control matching v2: ω extrapolation, convergence-rate rejection, near-target scaling, β_a
- [x] Iterative refinement in predictor (accurate Taylor coefficients)
- [x] StraightLineHomotopy Taylor formula fix (cross-derivative terms)
- [x] Raw tracker trace tooling against v2 (`benchmark/compare/trace_tracker.jl`)
- [x] Binomial system solver (HNF), weighted norms, custom LU
- [x] Result types, seed reproducibility in solve APIs
- [x] Tape interpreter for eval, jacobian, Taylor 1-3, DF64
- [x] CSE optimizer (SymEngine port), Moshi ADTs
- [x] RGF compiled eval+jac backend (`CompileMode.COMPILED`, opt-in, 3-6x kernel speedup)

### Not Done — Top Priority

- [ ] **Endgame tracker / late-path handoff** — correctness blocker and likely the main remaining tracker-performance gap
- [ ] **Threading** — performance blocker for large systems
- [ ] **Overdetermined systems** — applicability blocker
- [ ] **Tracking benchmark cleanup** — fix seed handling in `benchmark/compare/tracking.jl` and report total steps, not only accepted steps

### Not Done — Later

- [ ] Direct polynomial compiler — `polynomial_compiler.jl` exists, deferred
- [ ] Compiled Taylor backend — RGF codegen for Taylor (only if profiling justifies it)
- [ ] Standalone `newton(F, x0)`, progress bars, path diagnostics
- [ ] Monodromy, certification, witness sets, NID
- [ ] Benchmark CI, no-allocation enforcement tests

## Performance

Treat the current `benchmark/compare/tracking.jl` numbers as **indicative**, not final. That
script currently:

- uses fresh random seeds for `solve`
- prints **accepted** steps/path, not total predictor-corrector attempts

It is still useful as a trend check, but not yet good enough for stable headline claims.

### End-to-end solve vs v2 (> 1.0 = v3 faster)

From the current `benchmark/compare/tracking.jl` with `CompileMode.COMPILED`:

| System | v3/v2 ratio | v3 accepted steps/path | v2 accepted steps/path |
|--------|------------:|-----------------------:|-----------------------:|
| katsura-3 | ~1.86x | ~76.9 | ~62.9 |
| katsura-4 | ~1.33x | ~104.6 | ~75.6 |
| katsura-5 | ~1.01x | ~153.5 | ~89.8 |

This is a large improvement over the earlier tracker state. The remaining gap is no longer
"v3 is generically 2-3x worse everywhere"; it is concentrated in harder late-path cases.

### Raw tracker trace: where the remaining gap lives

From `benchmark/compare/trace_tracker.jl 3` using the raw tracker on katsura-3 with `γ = 1`:

- current worst traced raw path: Next `155 / 115 / 270` accepted/rejected/total
- same path in HC v2 raw tracker: `27 / 0 / 27`

The large remaining mismatch shows up near the target / late-path regime. That is exactly where
the missing endgame becomes relevant.

### v3 compiled vs v3 interpreted (`make benchmark` → compile_modes group)

| Metric | Eval speedup | Jac speedup | Build overhead |
|--------|-------------:|------------:|---------------:|
| katsura-3 | 3.2x | 3.9x | 1.42x |
| katsura-5 | 3.5x | 5.6x | 1.29x |
| katsura-7 | 3.3x | 6.3x | 1.22x |

End-to-end solve speedup remains modest because eval is only part of total step cost.

### TTFX (fresh session)

| Mode | Time |
|------|------|
| v3 total | ~15s |
| v2 total `[:mixed]` | ~47s |
| v2 solve-only `[:none]` | ~11s |
| v2 second system `[:mixed]` | ~6s |

v3 still wins clearly against v2 default `:mixed`, which is the main architectural motivation for
the rewrite.

### Tracker component breakdown

The old conclusion still holds: raw eval kernels are not the bottleneck. Predictor updates,
Newton correction, and step acceptance dominate.

## Open Items

### Performance / correctness

1. **Endgame / late-path handoff**
   The raw trace tooling now makes this visible: the hardest remaining step-count gaps occur
   near the target, where v2's overall pipeline has endgame machinery and v3 does not.

2. **Benchmark cleanup**
   `benchmark/compare/tracking.jl` should use fixed seeds and should report accepted, rejected,
   and total steps consistently. Right now it is fine for trend tracking, not for documentation
   claims.

3. **Invalid-start metadata leak**
   Invalid starts now fail correctly, but `PathResult` can still expose stale tracker metadata
   (`condition_jacobian`, machine-epsilon `accuracy`) from earlier state unless the invalid-start
   path resets those fields explicitly.

### Architecture debt

4. **Direct polynomial compiler** — validate and promote for polynomial input
5. **Fragile DynamicPolynomials introspection** — `_variable_creation_id` uses reflection
6. **Uncached SExpr hashes** — Moshi refactor removed `_hash` fields
7. **O(n^2) `_stable_sort!`** — insertion sort on potentially large vertex lists
8. **Magic constant 10000** — scratch slot placeholder base, unguarded

### Infrastructure debt

9. **No benchmark CI** — regressions go unnoticed
10. **No-allocation tests** — should be enforced for tracker step, Newton, predictor

## Dependencies

| Package | Purpose |
|---------|---------|
| CommonSolve | `init`/`solve!` interface |
| DynamicPolynomials | `@polyvar`, user-facing polynomial input |
| EnumX | Scoped enums (TrackerCode, PathResultCode, CompileMode, etc.) |
| FixedSizeArrays | FSVec/FSMat (size not in type parameter) |
| FunctionWrappers | Type erasure for SystemEvaluator/HomotopyEvaluator |
| LinearAlgebra | stdlib LU/QR |
| MixedSubdivisions | BKK mixed volume for polyhedral start system |
| Moshi | `@data` ADT for SExpr and ExecInstruction |
| MultivariatePolynomials | Abstract polynomial interface, differentiation |
| RuntimeGeneratedFunctions | Compiled eval/jac backend (`CompileMode.COMPILED`) |
