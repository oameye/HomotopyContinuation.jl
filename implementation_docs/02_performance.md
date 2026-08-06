

## Performance

Measured 2026-04-03, Julia 1.12.5, single-threaded, except where a section says otherwise.

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

Raw `SystemEvaluator` kernels. These do not carry to end-to-end solves: through the
`StraightLineHomotopy` they shrink to 1.7–2.0x, and Taylor plus linear algebra dominate
the tracker step.

| System | Eval speedup | Jac speedup |
|--------|-------------:|------------:|
| katsura-3 | 3.35x | 3.82x |
| katsura-5 | 3.65x | 5.04x |
| katsura-7 | 3.29x | 6.74x |

#### End-to-end solve per compile mode

Measured 2026-07-17, serial executor, speedup vs `INTERPRETED`
(`benchmark/compile_modes_e2e.jl`; analysis in `04_compile_modes.md`):

| System | INTERPRETED | COMPILED | COMPILED_ALL |
|--------|------------:|---------:|-------------:|
| katsura-3 | 0.91ms | 1.10x | 1.18x |
| katsura-5 | 8.79ms | 1.15x | 1.25x |
| katsura-7 | 64.98ms | 1.23x | 1.33x |
| katsura-9 | 482.94ms | 1.27x | 1.41x |

#### COMPILED_ALL (compiled Taylor) vs INTERPRETED

Scalar-parameter Taylor kernel (parameter-free systems, speedup = interp/all):

| System | Taylor 1 speedup | Taylor 2 speedup | Taylor 3 speedup | Solve speedup vs COMPILED | Build overhead |
|--------|------------------:|------------------:|------------------:|--------------:|---------------:|
| katsura-3 | 1.66x | 1.72x | 1.62x | 1.09x | 1.62x |
| katsura-5 | 1.73x | 1.59x | 1.61x | 1.09x | 1.39x |
| katsura-7 | 1.74x | 1.65x | 1.76x | 1.08x | 1.29x |

(The solve column is `COMPILED_ALL` vs `COMPILED`, not vs `INTERPRETED`; the ratio holds at
1.08–1.10x across katsura 3/5/7/9.)

TaylorVector-parameter Taylor kernel (production path — CoefficientHomotopy/ToricHomotopy):

| System | Taylor 1 speedup | Taylor 2 speedup | Taylor 3 speedup |
|--------|------------------:|------------------:|------------------:|
| katsura-3 | 1.48x | 1.72x | 1.48x |
| katsura-5 | 1.51x | 1.65x | 1.25x |
| katsura-7 | 1.30x | 1.63x | 1.20x |

#### TTFX (fresh session)

Latest v3 pass measured 2026-07-25; v2 numbers from 2026-04-03.

The v3 rows predate the tape-executor precompilation of 2026-07-26, which cut the
common path by 1.28s in a paired comparison (`01_decisions.md`, "Tape executors are
precompiled by signature"). They are not restated here because the machine was
contended when the change landed and no clean absolute sweep was taken; re-run
`make ttfx` on a quiet machine before trusting the numbers below.

| Metric | Time |
|--------|------|
| v3 package load | 0.77s |
| v3 construction + init + first solve! | 8.44--8.77s |
| **v3 total (load + construction + solve)** | **9.21--9.54s** |
| v2 package load | 1.30s |
| v2 first solve() [:mixed] | 43.78s |
| **v2 total [:mixed]** | **45.08s** |
| v2 first solve() [:none] | 10.79s |
| v2 second solve() (different system) | 5.72s |

The root-cause pass cut build/init/solve from 14.44s to ~9.81s by isolating mutually exclusive
compiler branches, using adaptive direct polynomial lowering, and stabilizing constructor types.
A later pass took it to ~9.2--9.4s by not factorizing the `MatrixWorkspace` QR field at
construction and by putting shape dispatch behind an inference barrier. A third pass reached
~8.4--8.8s while testing four changes: temporarily dropping three non-polynomial unary
interpreter variants, installing the extended-precision wrappers on first use, extracting the
support on demand, and handing `skeel_row_scaling!` a matrix instead of a `MatrixWorkspace`
(see `01_decisions.md`). The unary-variant removal recovered about 0.26s but was rejected because
those operations belong to the planned expression frontend. The table below therefore records a
historical experiment, not the current absolute TTFX; remeasure with the variants retained.
Over 3 interleaved fresh-process pairs at `-t 4`, the experimental build had 21 of 21 paired
differences negative:

| workload | before | after | |
|---|---:|---:|---:|
| `total_degree_interpreted_serial` | 9.339s | 8.437s | −9.7% |
| `newton_standard` | 6.031s | 5.259s | −12.8% |
| `singular_endgame` | 9.333s | 8.505s | −8.9% |
| `overdetermined_total_degree` | 10.009s | 9.180s | −8.3% |
| `total_degree_compiled_all_serial` | 10.243s | 9.532s | −6.9% |
| `witness_set_build` | 16.046s | 14.984s | −6.6% |
| `polyhedral_interpreted_serial` | 13.745s | 13.170s | −4.2% |

Polyhedral gains least because it still extracts the support and is dominated by
MixedSubdivisions. Full SnoopCompile/JET/Cthulhu/invalidation report: `05_ttfx_invalidations.md`.

A fourth pass took the untaken `MatrixWorkspace` QR branch out of the square routes by
erasing both its entry points behind shape-chosen `FunctionWrapper` fields, which keeps the
tracker hot path free of runtime dispatch (`01_decisions.md`, "The QR path is erased behind a
shape-chosen FunctionWrapper"). Over 5 interleaved fresh-process pairs at `-t 4` on
`total_degree_interpreted_serial`, all 5 paired differences favour it: 8.427--9.062s before
against 8.312--8.426s after, ~0.15s median. A square-only session now compiles zero
specializations of `qr!`, `qr_ldiv!`, `reflector!` and `lmul_Q_adj!`; the first tall solve pays
~0.35s to compile the path on demand.

SnoopCompile v3.2.5 does not load on Julia 1.12 (`UndefVarError: Compiler.Params`), so that pass
used `SnoopCompileCore`'s `@snoop_inference` tree directly: on 1.12 the per-node costs live on the
`CodeInstance` as `time_infer_self` / `time_compile`, `UInt16` fields holding `Float16` seconds
(`reinterpret(Float16, ci.time_infer_self)`). Codegen outweighs inference on a first solve, 9.97s
against 6.20s for `polyhedral_interpreted_serial`, so a change that removes generated code counts
for more than one that only removes inference.

Core load stays at 0.77s because certification is a separate `lib/` subpackage. Adding Arblib to
core raised load to ~1.6s and ~50% more invalidation descendants (one Arblib `show` method alone
accounted for ~2700), which is what motivated the split.

### Cost of the transcendental ops and the non-integer power

Measured 2026-08-03 against a worktree at the previous commit, alternating runs.

Eight `OpType` entries take `ExecInstruction` from 25 variants to 33, and the switch in
`execute_instructions!` and `execute_taylor_instructions!` grows with it. Nothing moved:

| | before | after |
|---|---|---|
| `sizeof(ExecInstructionT)` | 24 B | 24 B |
| cyclic-7 eval, 200k-iteration loop | 83 to 86 ns | 83 to 88 ns |
| cyclic-7 Jacobian, same | 411 to 422 ns | 396 to 451 ns |
| package load | 1.026 to 1.047 s | 1.028 to 1.038 s |
| first `total_degree_interpreted_serial` | 8.08 to 8.16 s | 7.96 to 8.11 s |
| `Base.compilecache` of the package | 9.6 / 9.9 s | 9.8 / 9.6 s |

Moshi had tag room, so the instruction vector did not grow, and the precompile image absorbs the
larger switch. Cross-process A/B cannot resolve an end-to-end solve here: three fresh processes
on the *same* tree gave 1.07, 1.49 and 1.56 ms for the 18-solution intro system, so the
in-process loop above is the only reliable form. `@benchmarkable` is also too coarse, quantizing
to 10 ns and reporting a spurious 700 → 750 ns for the cyclic-7 Jacobian.

One real regression, in `Expression` construction only. `_emul` now tallies exponents as
`ComplexF64` rather than `Int`, which is what collapses `x^1.5 * x^-1` to one node instead of
leaving CSE two. A 4000-factor chained product costs 5.55 to 5.65 ms and 17180 KiB before
against 5.93 to 6.64 ms and 18677 KiB after (+7% time, +8.7% allocation, the tally vector being
16 bytes per element instead of 8). The heaviest real build in the suite, the Fano quintic
restricted to a line, is unchanged at 0.6 ms. An `Int` fast path would be a second code path in
`_emul` for 7% of a construction-only microbenchmark, so it was left alone.

A first solve on a route that uses the new ops (a transcendental parameter homotopy) costs
0.82 s load + 4.19 s `System` construction + 4.52 s first solve, against 8.1 s for the plain
total-degree first solve.

### Certification of a `ResultIterator`

Measured 2026-08-06, `-t auto` on 12 threads.

The 3264-conic instance through this route (27072 polyhedral paths, `leaf_size_bound = 200`,
`boundaries = -10:0.5:10`) takes 1m23s for 3264 distinct certified solutions, tracking
30336 paths: 27072 to place the enclosures plus 3264 to certify the terminal leaves. That is one
pass over the iterator plus one pass over the certified subset, which is what the route costs when
refinement does no tracking of its own. Peak certificates alive is one leaf's worth, at most 200
of 3264.

The pass count is what the design turns on, and the tests pin it: `ntracked` is `27072 + 3264`
there, `2 × 25` on a 25-solution leaf refined to a bound of 3, and 8 on a 4-solution leaf refined
to 1 (measured at 17 with the earlier refinement, which re-tracked a leaf per proposed cut). The
growth of that earlier form is derived rather than measured, since it was replaced before the
large instance ran: a cut taken from a single enclosure peels that one off, so a leaf of N
enclosures takes O(N) splits at a pass each, and leaves ~N terminal leaves to pay another pass
each.

The pass count does not change what is tracked (each terminal leaf re-tracks only its own
entries, so the total is the same however the leaves are cut) but it does multiply the per-pass
fixed cost, and that is what the default configuration exposes: `leaf_size_bound = 50_000` with
`boundaries = -100:0.1:100` means refinement never triggers, so there is a pass per *occupied
grid cell* rather than one per `leaf_size_bound` worth of solutions. Worker construction was the
bulk of that cost (a cloned system evaluator plus an endgame tracker, ~93 KiB and ~18 µs each,
built per task per pass), so the workers are now built once per run and handed to every pass. On a
400-solution bivariate system, default `boundaries` against a single leaf: 2002 leaves vs 1,
identical output (400 distinct, `ntracked` 800 both ways), 5.5 MiB vs 3.9 MiB and 11 ms vs 8 ms.
The residual ~1.6 MiB over ~37 passes is the per-pass channels, leaf mask and payload vector.
Coalescing adjacent under-full leaves before the certification walk would remove that too, at the
cost of changing what `nleaves` reports; not done.

### Endgame result parity

| System | v2 result | v3 result |
|--------|-----------|-----------|
| (x-10)^2 | nresults=1, nsingular=1 | same |
| at-infinity | 2 success + 2 at_infinity | same |
| winding family d=2,4,6 | d+1 success each | same |
| Hyperbolic 6,6 | nresults=2, nsingular=2 | same |
| singular multiplicity 3 | nresults=2, nsingular=1, nnonsingular=1 | same |