# Compile Modes: v3 vs v2

Last updated: 2026-07-17. Verified against `src/core/system.jl` (v3) and
v2.18.0 `src/systems/mixed_system.jl` / `src/homotopies.jl`.

Both libraries expose a three-mode ladder controlling which evaluation kernels
are compiled versus interpreted.

## Mode table

| Mode | eval / Jacobian | Taylor (predictor) | Codegen backend | First-solve cost | Steady-state |
|---|---|---|---|---|---|
| v3 `INTERPRETED` (default) | tape interpreter | tape interpreter | none | lowest (16.3s incl. load, no codegen) | baseline |
| v3 `COMPILED` | compiled | tape interpreter | RuntimeGeneratedFunctions from the CSE tape | + RGF build per system | eval 3.3-3.7x, jac 3.8-6.7x kernel speedup; 1.1-1.3x end-to-end |
| v3 `COMPILED_ALL` | compiled | compiled | RGF, Taylor kernels too | ~1.3-1.6x more build time than `COMPILED` | Taylor kernels 1.2-1.8x; 1.08-1.10x over `COMPILED`, 1.2-1.4x end-to-end |
| v2 `:none` | `InterpretedSystem` | `InterpretedSystem` | none | lowest for v2 (10.8s first solve) | baseline |
| v2 `:mixed` (default) | `CompiledSystem` | `InterpretedSystem` | ModelKit-generated Julia code (Julia JIT) | highest (43.8s first solve) | v2's recommended production mode |
| v2 `:all` | `CompiledSystem` | `CompiledSystem` | ModelKit-generated Julia code | worst; compile time grows badly with system size | fastest v2 kernels |

Timings from `02_status.md` (Julia 1.12.5, single-threaded).

## Why 3-7x kernels give only 1.1-1.4x solves

Measured 2026-07-17 with `benchmark/compile_modes_e2e.jl` and
`benchmark/profile_compile_modes.jl` (serial executor, progress off):

| System | INTERPRETED | COMPILED | COMPILED_ALL |
|--------|------------:|---------:|-------------:|
| katsura-3 | 0.91ms | 0.83ms (1.10x) | 0.77ms (1.18x) |
| katsura-5 | 8.79ms | 7.65ms (1.15x) | 7.04ms (1.25x) |
| katsura-7 | 64.98ms | 53.01ms (1.23x) | 48.91ms (1.33x) |
| katsura-9 | 482.94ms | 378.89ms (1.27x) | 343.30ms (1.41x) |

The kernel speedup dilutes three ways:

1. The tracker calls kernels through the `StraightLineHomotopy`, which adds
   start-system evaluation and the γ blend on every call. Through the
   `HomotopyEvaluator` at katsura-5, eval goes 164ns to 95ns (1.7x, not 3.65x)
   and eval+jac 349ns to 176ns (2.0x, not 5x).
2. The most expensive kernels per step are Taylor orders 2 and 3 (574ns and
   757ns at katsura-5, vs 176ns for the compiled jac), and `COMPILED` leaves
   them interpreted. That gap is exactly what `COMPILED_ALL` closes.
3. A large share of step time is mode-independent: custom LU, triangular
   solves, iterative refinement, row scaling, weighted norms, predictor
   trust-region logic, and buffer copies dominate both flat profiles.

Practical consequence: the compile ladder is not the lever the kernel numbers
suggest. `COMPILED_ALL` strictly dominates `COMPILED` for steady-state use
(same asymptote, one more ~1.09x step, modest extra build time), and the
default `INTERPRETED` gives up at most ~30% on katsura-sized systems.

## Correspondence

The pairing is exact by construction: v2's `MixedSystem` holds a
`CompiledSystem` for `evaluate!`/`evaluate_and_jacobian!` and an
`InterpretedSystem` for `taylor!`, the same split as v3's `COMPILED`.

- v3 `INTERPRETED` ↔ v2 `:none`
- v3 `COMPILED` ↔ v2 `:mixed` (the pairing used by `benchmark/compare/tracking.jl`)
- v3 `COMPILED_ALL` ↔ v2 `:all` (never benchmarked head to head)

## Asymmetries

1. **Default philosophy.** v2 defaults to its middle mode (`:mixed`) and pays
   43.8s on first solve. v3 defaults to its bottom mode (`INTERPRETED`) and
   pays 16.3s, favoring time-to-first-solve.
2. **Codegen cost structure.** v2's `CompiledSystem` routes ModelKit-generated
   expressions through Julia's JIT, which is what makes `:all` impractical for
   large systems. v3's RGF path compiles the already-CSE-optimized tape, which
   is cheaper and makes `COMPILED_ALL` actually usable.
3. **Extended precision.** In v3 the DF64 extended-precision evaluator stays on
   the interpreter in every mode, so the extended-precision residual path is
   identical across all three modes.

## Benchmark fairness

The existing comparison (`benchmark/compare/tracking.jl`) is fair for the pair
it tests: v3 `COMPILED` vs v2 `:mixed`, both compiled-eval modes with the same
kernel split, system construction outside the timed region on both sides,
same seed, JIT warmup, endgame included. Note that the reported 1.8-3.6x
total-degree wins are not only kernel speed: v3 also takes far fewer tracker
steps per path (e.g. katsura-3: 36 vs 87), which no v2 compile mode would
recover.

## TODO: genuine top-mode benchmark

The v3 side of the matrix is now measured (`benchmark/compile_modes_e2e.jl`
reports steady-state solve and construction time for all three v3 modes).
Still missing:

- [ ] v3 `COMPILED_ALL` vs v2 `:all`, end-to-end solve, katsura-5 and one
      larger system (e.g. katsura-8) to expose v2's `:all` compile-time blowup.
- [ ] First-solve wall time under each v3 default candidate (`INTERPRETED` vs
      `COMPILED`) on a fresh session, to settle the default `CompileMode`
      decision (see `03_v2_improvement_opportunities.md` item 9) with data.
      Given the 1.1-1.3x steady-state gap, `INTERPRETED` as default looks
      safe unless fresh-session RGF cost turns out negligible.
- [ ] The v2 rows of the six-mode matrix, measured rather than collated.
