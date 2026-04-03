# Status

Last updated: 2026-04-03.

**Reproduce:**
- `make benchmark` — steady-state timings
- `make compare` — v3/v2 ratios
- `julia --project=benchmark benchmark/compare/tracking.jl` — end-to-end solve comparison
- `make test` — test suite

## Summary

v3 is a credible replacement for v2's core solve pipeline. Total-degree solving is 2–3.6x faster
than v2. Polyhedral solving matches v2 within noise. Endgame is at full v2 result parity using
v2's default parameters. The main remaining gaps are threading, overdetermined systems, and
advanced features (monodromy, certification, NID).

## Feature Checklist

### Done

- [x] `solve(F)` with CommonSolve.jl `init`/`solve!`
- [x] Total-degree and polyhedral start systems
- [x] Parameter homotopy
- [x] `System` type (caches compiled interpreters for all eval modes)
- [x] `SystemEvaluator` / `HomotopyEvaluator` type firewall (FunctionWrapper)
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
- [x] Automatic coefficient normalization (scales polynomials with O(10^8+) coefficients to O(1))
- [x] AllocCheck zero-allocation enforcement on all hot paths
- [x] Integration tests from v2 with exact result parity

### Not Done — Top Priority

- [ ] **Threading** — performance blocker for large systems
- [ ] **Overdetermined systems** — applicability blocker

### Not Done — Later

- [ ] Direct polynomial compiler (`polynomial_compiler.jl` exists, deferred)
- [ ] Compiled Taylor backend (RGF codegen, only if profiling justifies)
- [ ] Standalone `newton(F, x0)`, progress bars, path diagnostics
- [ ] Monodromy, certification, witness sets, NID
- [ ] Benchmark CI

## Performance

### End-to-end solve vs v2

`CompileMode.COMPILED`, fixed seed `0x4567`, endgame enabled.

#### Total-degree (katsura, chain)

| System | v3/v2 ratio | v3 steps/path | v2 steps/path |
|--------|------------:|--------------:|--------------:|
| katsura-3 | **3.05x** | 36.2 (290 acc, 0 rej) | 87.0 (696 acc, 0 rej) |
| katsura-4 | **2.05x** | 47.3 (757 acc, 0 rej) | 78.8 (1260 acc, 0 rej) |
| katsura-5 | **1.95x** | 56.4 (1806 acc, 0 rej) | 98.5 (3137 acc, 15 rej) |
| chain-3 | **3.65x** | 23.0 (184 acc, 0 rej) | 46.2 (370 acc, 0 rej) |
| chain-4 | **2.36x** | 34.4 (550 acc, 0 rej) | 64.2 (1028 acc, 0 rej) |
| chain-5 | **1.82x** | 40.2 (1288 acc, 0 rej) | 65.0 (2081 acc, 0 rej) |

#### Polyhedral (cyclic, random sparse)

| System | v3/v2 ratio | v3 steps/path | v2 steps/path |
|--------|------------:|--------------:|--------------:|
| cyclic-4 | 0.94x | 77.1 (1188 acc, 46 rej) | 74.3 (1158 acc, 31 rej) |
| cyclic-5 | 0.98x | 50.8 (3556 acc, 0 rej) | 50.8 (3553 acc, 0 rej) |
| sparse-3x3 | 0.94x | 45.0 (1924 acc, 9 rej) | 45.0 (1924 acc, 9 rej) |
| sparse-4x4 | 0.99x | 75.1 (12575 acc, 47 rej) | 75.0 (12551 acc, 47 rej) |
| sparse-5x5 | 0.98x | 83.3 (34080 acc, 85 rej) | 83.4 (34121 acc, 90 rej) |

#### Compiled vs interpreted kernels

| System | Eval speedup | Jac speedup |
|--------|-------------:|------------:|
| katsura-3 | 3.2x | 3.9x |
| katsura-5 | 3.5x | 5.6x |
| katsura-7 | 3.3x | 6.3x |

#### TTFX (fresh session)

| Mode | Time |
|------|------|
| v3 | ~15s |
| v2 `[:mixed]` | ~47s |
| v2 `[:none]` | ~11s |

### Endgame result parity

| System | v2 result | v3 result |
|--------|-----------|-----------|
| (x-10)^2 | nresults=1, nsingular=1 | same |
| at-infinity | 2 success + 2 at_infinity | same |
| winding family d=2,4,6 | d+1 success each | same |
| Hyperbolic 6,6 | nresults=2, nsingular=2 | same |
| singular multiplicity 3 | nresults=2, nsingular=1, nnonsingular=1 | same |

## Open Items

### Performance

1. **Threading** — main blocker for large systems (cyclic-7 has 924 paths)
2. **Overdetermined systems** — `RandomizedSystem` + excess solution check

### Architecture debt

3. Direct polynomial compiler — validate and promote
4. Fragile DynamicPolynomials introspection (`_variable_creation_id`)
5. Uncached SExpr hashes (Moshi refactor removed `_hash` fields)
6. O(n²) `_stable_sort!` on potentially large vertex lists
7. Magic constant 10000 (scratch slot placeholder base)

### Infrastructure

8. No benchmark CI — regressions go unnoticed
