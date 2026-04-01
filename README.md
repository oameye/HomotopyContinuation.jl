# HomotopyContinuationNext.jl

A ground-up rewrite of [HomotopyContinuation.jl](https://github.com/JuliaHomotopyContinuation/HomotopyContinuation.jl) for solving polynomial systems via homotopy continuation.

**Status:** Robust monomorphic core. Not yet a v2 replacement — missing endgame (singular solutions), threading, and overdetermined support. End-to-end solve is currently slower than v2 on nonsingular systems (`make compare`). The win is architectural: predictable compilation, no TTFX pathology, pure Julia throughout.

## What this solves

v2's `CompiledSystem{ID}` creates a unique type per polynomial system, forcing full recompilation of the tracker pipeline on every new system (~44s first solve, ~6s each new system). This is a fundamental type-system design problem, not fixable by tuning.

v3 inserts a `FunctionWrapper` type firewall at the evaluator boundary (`SystemEvaluator`). The tracker is monomorphic — compiled once, reused for all systems. First solve ~13s (vs v2 :mixed ~44s). No per-system recompilation cost.

The secondary goal: replace SymEngine (C FFI) with a pure-Julia symbolic pipeline. Fully precompilable, debuggable, no FFI boundary.

See `implementation_docs/03_v3_vs_v2.md` for the full comparison.

## Current Limitations

- **No endgame tracker** — singular or at-infinity solutions will fail. This is the top priority.
- **No threading** — solve loop is sequential. Infrastructure is ready but not wired.
- **No overdetermined support** — more equations than variables not handled.
- **End-to-end solve is slower than v2** on nonsingular systems (0.47x–0.69x on katsura-3/4/5). The FunctionWrapper indirection and lack of compiled evaluation backends are the main costs.
- **Raw interpreter is 3–8x slower than v2 compiled mode** for polynomial evaluation. End-to-end impact is smaller (LU/Newton dominate), but it's real.

## What works well

```julia
using HomotopyContinuationNext

@polyvar x y
F = System([x^2 + y - 1, x*y - 2])
result = solve(F)
solutions(result)
real_solutions(result)
```

- Total degree and polyhedral (BKK-optimal) start systems
- Parameter homotopy: `solve(F, starts; start_parameters=p₁, target_parameters=p₀)`
- Predictor-corrector tracker with adaptive stepping and DoubleF64 refinement
- Pure-Julia tape interpreter for eval, Jacobian, Taylor orders 1-3
- CommonSolve.jl integration, seed reproducibility
- All hot-path types concrete, monomorphic tracker, immutable results

## Near-term priorities

1. **Endgame** — correctness blocker for any real use
2. **Threading** — performance blocker for large systems
3. **Overdetermined systems** — extends applicability
4. **Direct polynomial compiler** — the SExpr→CSE→tape path may be more machinery than needed for polynomial input. A direct polynomial→tape compiler exists (`polynomial_compiler.jl`) but is deferred. Promoting it could reduce complexity and build cost.
5. **Benchmark CI** — no regression tracking, claims rot quickly

See `implementation_docs/02_status.md` for the full status.

## Development

```sh
make test          # run all tests in parallel
make benchmark     # steady-state timings
make compare       # v3/v2 comparison (interpreter, tracking, TTFX, v2 modes)
make format        # format with Runic.jl
```

See `CLAUDE.md` for coding rules and conventions.
