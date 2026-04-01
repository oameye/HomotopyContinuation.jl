# Status

Last updated: 2026-04-01.

**Reproduce:** `make benchmark` (timings), `make compare` (v3/v2 ratios), `make test` (instruction counts)

## Honest Assessment

This is a robust monomorphic core, not yet a v2 replacement. The architecture is right (FunctionWrapper firewall, pure Julia, no per-system recompilation), but major gaps remain:

- **Endgame missing** — singular/at-infinity solutions will fail. This blocks real use.
- **End-to-end solve is slower than v2** — 0.47x–0.69x on katsura-3/4/5 (`benchmark/compare/tracking.jl`).
- **Raw interpreter is 3–8x slower than v2 compiled** for eval, 7–11x for Jacobian (`benchmark/compare/v2_modes.jl`). LU/Newton dominate end-to-end time, so the impact is smaller, but it's real.
- **No threading, no overdetermined, no progress bars.**
- **Low-level debt:** uncached SExpr hashes, O(n^2) insertion sort, fragile DP introspection, magic constants, no benchmark CI.

The strongest claim: **better foundation** (predictable compilation, pure Julia, extensible via AbstractSystem). The weakest claim: **already the right replacement**.

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
- [x] Binomial system solver (HNF), weighted norms, custom LU
- [x] Result types, seed reproducibility
- [x] Tape interpreter for eval, jacobian, Taylor 1-3, DF64
- [x] CSE optimizer (SymEngine port), Moshi ADTs
- [x] RGF compiled eval+jac backend (`CompileMode.COMPILED`, opt-in, 3-6x kernel speedup)

### Not Done — Top Priority

- [ ] **Endgame tracker** — correctness blocker
- [ ] **Threading** — performance blocker for large systems
- [ ] **Overdetermined systems** — applicability blocker
- [ ] **Direct polynomial compiler** — `polynomial_compiler.jl` exists, deferred. For polynomial input (the common case), this path could replace the SExpr→CSE→tape pipeline with less complexity.

### Not Done — Later

- [ ] Compiled Taylor backend — RGF codegen for Taylor evaluation (only if profiling shows interpreter Taylor is a bottleneck; v2 found interpreted Taylor efficient)
- [ ] Standalone `newton(F, x0)`, progress bars, path diagnostics
- [ ] Monodromy, certification, witness sets, NID
- [ ] Benchmark CI, no-allocation enforcement tests

## Performance

Re-run `make benchmark` for absolute timings, `make compare` for v3/v2 ratios.

### End-to-end solve vs v2 (> 1.0 = v3 faster)

From `benchmark/compare/tracking.jl` + inline comparison (2026-04-01). **v3 is still slower.**

| System | v3 INTERPRETED/v2 | v3 COMPILED/v2 |
|--------|------------------:|---------------:|
| katsura-3 | 0.75x | 0.85x |
| katsura-4 | 0.57x | 0.61x |
| katsura-5 | 0.51x | 0.54x |

Note: v2 includes endgame; v3 does not. The compiled backend closes ~15-18% of the gap
but v2 remains faster. The remaining gap is in LU, Newton, and tracker overhead.

### v3 compiled vs v3 interpreted (`make benchmark` → compile_modes group)

| Metric | Eval speedup | Jac speedup | Build overhead |
|--------|-------------:|------------:|---------------:|
| katsura-3 | 3.2x | 3.9x | 1.42x |
| katsura-5 | 3.5x | 5.6x | 1.29x |
| katsura-7 | 3.3x | 6.3x | 1.22x |

End-to-end solve speedup: 7-18% (eval is a fraction of total time).

### TTFX (fresh session)

| Mode | Time |
|------|------|
| v3 INTERPRETED | ~14s |
| v3 COMPILED | ~15s |
| v2 [:mixed] | ~44s |
| v2 [:none] | ~11s |

v3 wins against v2 default (:mixed). COMPILED adds ~1s TTFX overhead vs INTERPRETED.

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

### Instruction counts (regression test limits, `test/instruction_count_test.jl`)

| System | Eval instrs |
|--------|------------:|
| cyclic-3/4/5/6/7 | 6 / 11 / 19 / 28 / 38 |
| chain-3/4/5/6/7 | 17 / 23 / 28 / 34 / 39 |
| dense-quad-3/4/5/6 | 28 / 52 / 92 / 138 |
| sparse 6x6 (random) | 82-96 |
| sparse 8x8 (random) | 162-170 |

## Open Items

### Architecture debt
1. **Direct polynomial compiler** — validate `polynomial_compiler.jl` and promote to default for polynomial input. The SExpr→CSE path is more machinery than needed for the common case.
2. **Fragile DynamicPolynomials introspection** — `_variable_creation_id` uses reflection
3. **Uncached SExpr hashes** — Moshi refactor removed `_hash` fields, may regress CSE build time on large systems
4. **O(n^2) `_stable_sort!`** — insertion sort on potentially large vertex lists
5. **Magic constant 10000** — scratch slot placeholder base, unguarded

### Infrastructure debt
6. **No benchmark CI** — regressions go unnoticed
7. **No-allocation tests** — should be enforced for tracker step, Newton, predictor
8. **Steady-state benchmarks stale** — pre-Moshi numbers, need re-running

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
