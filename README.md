# HomotopyContinuationNext.jl

A ground-up rewrite of [HomotopyContinuation.jl](https://github.com/JuliaHomotopyContinuation/HomotopyContinuation.jl) for solving polynomial systems via homotopy continuation.

**Status:** Robust monomorphic core. Not yet a v2 replacement — missing endgame (singular solutions), threading, and overdetermined support. The regular-path tracker is much closer to v2 now, but the remaining gap is concentrated in late-path/endgame behavior. The main win is still architectural: predictable compilation, no TTFX pathology, pure Julia throughout.

## What this solves

v2's `CompiledSystem{ID}` creates a unique type per polynomial system, forcing full recompilation of the tracker pipeline on every new system (~47s first solve, ~6s each new system). This is a fundamental type-system design problem, not fixable by tuning.

v3 inserts a `FunctionWrapper` type firewall at the evaluator boundary (`SystemEvaluator`). The tracker is monomorphic — compiled once, reused for all systems. First solve is about ~15s (vs v2 `:mixed` ~47s). No per-system recompilation cost.

The secondary goal: replace SymEngine (C FFI) with a pure-Julia symbolic pipeline. Fully precompilable, debuggable, no FFI boundary.

See `implementation_docs/03_v3_vs_v2.md` for the full comparison.

## Current Limitations

- **No endgame tracker** — singular or at-infinity solutions will fail. This is the top priority.
- **No threading** — solve loop is sequential. Infrastructure is ready but not wired.
- **No overdetermined support** — more equations than variables not handled.
- **Late-path tracker gap remains on hard paths** — raw tracker traces still show large step-count blowups near the target compared to v2. This now looks more like missing endgame / late-path handling than a generic regular-path issue.
- **Current compare benchmark needs cleanup** — `benchmark/compare/tracking.jl` is useful for trend tracking, but it still uses fresh random seeds and reports accepted steps only.

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
- Parameter homotopy: `solve(F, starts, p₁, p₀)`, many targets: `solve(F, starts, p₁, targets, Sweep())`
- One verb, one shape: `solve(problem..., algorithm, executor)`. The executor says where the work
  runs, the algorithm carries every other option, and `solve` takes no keyword arguments
- Parameter values are always positional; `fix_parameters(F, p)` makes a parametric system parameter-free
- Predictor-corrector tracker with adaptive stepping and DoubleF64 refinement
- Straight-line homotopy Taylor formulas fixed for coupled `x,t` variation
- Raw tracker trace tooling against v2 (`benchmark/compare/trace_tracker.jl`)
- Pure-Julia tape interpreter for eval, Jacobian, Taylor orders 1-3
- Optional RGF-compiled eval+jac backend via `CompileMode.COMPILED`
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
