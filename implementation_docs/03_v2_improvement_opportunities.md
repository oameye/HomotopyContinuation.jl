# v2 Improvement Opportunities

Prioritized list of concrete ways to make the v3 core solver better than
HomotopyContinuation.jl v2 (reference version 2.18.0). Compiled from a two-sided
audit: v2's numerical/algorithmic techniques vs. v3's current implementation, and
an internal quality audit of v3's own hot paths.

## Outcomes (2026-07-17)

All items were worked through; per-item status:

- **#1 binomial BigInt fallback: DONE.** `H_big`/`U_big` buffers, `_hnf_big!`,
  BigFloat angular solve with entry-size-scaled precision, `_validate_result`,
  and a direct `solve_binomial!(X, BSS, A, b)` entry. Regression test uses a
  unimodular 5x5 matrix (entries up to ~6e6) that overflows the Int64 HNF.
- **#2 dedup scaling: DONE** (sort-and-window sweep, 20k paths cluster in ~15 ms;
  new `result_clustering_test.jl` cross-checks against a brute-force reference).
  **Group-action symmetry: DONE** via `recluster(::Result; group_action, atol, rtol)`,
  which re-runs `_cluster_solutions` with an orbit-merge pass (`_orbit_merge!`)
  layered on the proximity sweep. `multiplicity` stays proximity-based, so
  collapsing an orbit never inflates it. Note there is nothing to port here: v2's
  `compute_multiplicities` discards its `kwargs...`, so v2's `Result` dedup is not
  symmetry-aware either.
- **#3 dispatch reorder: REFUTED by measurement**, see the item below. Kept
  declaration order.
- **#4 hot-path kwargs: DONE.** Positional `iterative_refinement!` cores with
  kwarg wrappers at the edge; positional `incremental` on the evaluator
  `taylor!`.
- **#5 `execute_taylor!` fill guard: DONE** (mirrors `_extract_u!`).
- **#6 `_stable_sort!`: DONE** (stable `MergeSort` above a 32-element cutoff).
- **#7 underdetermined `newton`: DONE.** `m < n` solves each step via
  column-pivoted QR (`_solve_wide!`), `NewtonCache` handles both shapes with
  empty-size sentinels.
- **#8 singular-start classification: DONE.**
  `TERMINATED_INVALID_STARTVALUE_SINGULAR_JACOBIAN` across
  TrackerCode/EndgameCode/PathResultCode, corank via `_start_jacobian_corank`.
- **#9 CompileMode default: kept INTERPRETED**, tradeoff now documented in the
  `System` docstring. Flipping the default is a user decision.
- **#10 SoA LU: SKIPPED** (conflicts with the monomorphic-tracker design; would
  need a `MatrixWorkspace` type parameter and a second compiled tracker body).
- **#11 tracker presets: DONE** (`DEFAULT/FAST/CONSERVATIVE_TRACKER_OPTIONS`
  const presets with v2's β values).
- **#12 DoubleF64 transcendentals: DEFERRED TO EXPRESSION INPUT.** The interpreter
  retains `SIN`, `COS`, and `SQRT`; complete extended-precision transcendental
  support with the planned non-polynomial frontend (`02_status.md`).

## Context: the core loop is already a faithful port

The path-tracking inner loop is a near-line-for-line port of v2 and has
essentially no numerical gaps.

- **Tracker step control** (`tracking/tracker.jl`): ω-extrapolation, β_τ/β_ω
  logic, strict-β_τ near target, θ-based rejection, accuracy-limit tolerance
  `a³·h(a)`. Identical to v2.
- **Newton corrector α-theory** (`tracking/newton_corrector.jl`): convergence
  test, ω/θ estimation, extended-precision residual, `init_newton!` perturbation.
  Identical.
- **Predictor** (`tracking/predictor.jl`): Padé(2,1) plus trust region plus
  s-plane Hermite. Identical.
- **Endgame**: v2's Cauchy endgame is entirely commented out; both versions use
  the valuation plus s-plane cubic-Hermite singular endgame. v3 ported it
  faithfully and added some gating improvements.
- **WeightedNorm**: adaptive weights, Higham inverse-inf-norm condition estimate,
  Skeel row scaling, mixed/fixed-precision iterative refinement. All present.

v3 is already 1.8x to 3.6x faster than v2 on total-degree solving, and on one
point it is *more* aggressive than v2: it enables the ill-conditioned termination
check (`tracker.jl:251`) that v2 leaves commented out.

So the genuine opportunities are narrow and specific, listed below by impact.

## Tier 1: correctness / robustness (highest leverage)

### 1. Binomial (polyhedral start) HNF has no BigInt overflow fallback

- **v2**: `binomial_system.jl` keeps both Int64 (`H`, `U`) and BigInt (`H_big`,
  `U_big`) Hermite-normal-form buffers. `solve!` computes the HNF in Int64 and, on
  overflow or a failed `validate_result` (`binomial_system.jl:173-190,217`),
  recomputes in BigInt.
- **v3**: `solving/binomial_system.jl:119` `_hnf!` works in Int64 (`H`, `U`
  buffers, exponents come in as `Matrix{Int32}`) via `gcdx`, with no BigInt path
  and no `validate_result` equivalent. It does use `Base.checked_mul` /
  `Base.checked_add` (`binomial_system.jl:109-110`), so overflow throws an
  `OverflowError` rather than silently corrupting results, and nothing in
  `src/solving/` catches it.
- **Gain**: v3 currently aborts the entire solve with an uncaught
  `OverflowError` on mixed cells with large exponents/volumes, where v2 recovers
  via BigInt and keeps tracking. So this is a hard-failure robustness gap (loud,
  not silent), and the fix is to catch the overflow and redo the HNF in BigInt,
  plus optionally port `validate_result`.
- **Note**: validate with a large-support system that overflows Int64 HNF, before
  and after the fix.

### 2. Solution deduplication is O(k²) and has no symmetry support

- **v2**: `unique_points.jl` / `voronoi_tree.jl`. `UniquePoints` wraps a
  `VoronoiTree` with `search_in_radius`/`add!` (sub-quadratic nearest-neighbor),
  plus optional `GroupActions`/`SymmetricGroup` symmetry-aware clustering.
  `result.jl` provides `MultiplicityInfo`/`compute_multiplicities`.
- **v3**: `solving/result.jl:19` `_cluster_solutions` does a full O(k²) pairwise
  `inf_distance` scan with union-find, collecting into a `Dict{Int,Vector{Int}}`
  with a fresh `Int[]` per cluster. No spatial structure, no group-action handling.
- **Gain**: dedup/multiplicity that scales to large solution counts (thousands of
  paths), fewer allocations, and correct unique counts for symmetric systems.
- **Resolved**: the scan became a sort-and-window sweep, and symmetry is opt-in via
  `recluster`. The sweep's sort key `Re(x₁) + Im(x₁)` is not preserved by a group
  action, so orbit images cannot be found by the sweep; `_orbit_merge!` indexes one
  representative per proximity cluster in a `UniquePoints` tree instead, which
  already walks orbits in `search_in_radius`.

## Tier 2: cheap, low-risk performance wins

### 3. Interpreter dispatch chain ordered by declaration, not frequency

**REFUTED by measurement (2026-07-17), do not implement.** The hypothesis was
that reordering `_EXEC_INSTRUCTION_SPECS` (`model_kit/interpreter.jl:158-184`) by
op frequency would cut tag comparisons per instruction. Measured on katsura-8
(Julia 1.12.6, same machine, fresh sessions): declaration order runs the jacobian
interpreter in 489 to 535 ns and the Taylor-2 kernel in 316 to 356 ns; the
frequency order (measured hot-op ranking: Mul 241, MulMulAdd 189, Add 165,
Add4 107, MulAdd 79, Sqr 75, Sub 65 across katsura-5/8 and cyclic-5/7 tapes) runs
them in about 680 ns and 400 ns, roughly 30 percent SLOWER. Explanation: with the
`isa` conditions in ADT declaration order, the conditions test type tags in
ascending order and LLVM lowers the chain to a jump table; any permuted order
breaks the switch recognition and produces an actual linear scan. The
declaration-ordered chain is already optimal. Keep it.

### 4. kwargs on hot-path calls (violates the stated "no kwargs in hot paths" rule)

- `tracking/predictor.jl:106,141,159`: `iterative_refinement!(...; tol, max_iters)`
  called up to three times per prediction (once per Taylor order).
- `core/homotopy_evaluator.jl:102-118`: `taylor!` `Val{2}`/`Val{3}` take
  `incremental::Bool = false` as a keyword; called per step from
  `predictor.jl:135,153`.
- **Fix**: positional `tol, max_iters` inner methods for `iterative_refinement!`
  and a positional `incremental` on the `taylor!` evaluator methods; keep kw
  wrappers at the API edge.
- **Impact**: TTFX and minor perf (no allocation, but kwarg-sort overhead and
  extra specialization surface).

### 5. `execute_taylor!` always zero-fills output even when fully assigned

- `model_kit/interpreter.jl:458` unconditionally `fill!(u, zero(eltype(u)))`,
  whereas scalar `_extract_u!` (line 353) skips the fill when
  `I.sequence.all_u_assigned`. Runs every predictor step.
- **Fix**: mirror the `all_u_assigned` guard.
- **Impact**: minor hot-path (full-length fill of a Taylor-series vector per call).

### 6. `_stable_sort!` insertion sort is O(n²) on large vertex lists

- `utils.jl:9-20` (`_stable_sort_by!` at 22-34 has the same shape), used at `instruction_sequence.jl:143` on `vertex_list` and
  throughout CSE (`cse.jl`). Already flagged as architecture-debt item 6 in
  `02_status.md`.
- **Fix**: dispatch on length; use `sort!(v; alg = MergeSort)` (stable) above a
  small cutoff, keep insertion sort for tiny inputs.
- **Impact**: TTFX (`System` construction) for large-support systems.

## Tier 3: harder-system reach

### 7. No column-pivoted QR for underdetermined (m < n) Jacobians

- **v2**: `newton.jl:140` uses column-pivoted QR (`qr_col_norm!`, which wraps
  `LA.qr!(M, LA.ColumnNorm())`, `linear_algebra.jl:916`) for the m < n case.
- **v3**: `primitives/linear_algebra.jl:38` throws on m < n; the only QR
  (`qr!`, line 219) is unpivoted, tall-only.
- **Gain**: numerically sound minimum-norm solves for excess-variable and
  rank-deficient systems.

### 8. Singular-Jacobian invalid-start classification

- **v2**: `tracker.jl:712-723` computes Jacobian corank via `LA.rank` and returns
  `terminated_invalid_startvalue_singular_jacobian` vs. plain
  `terminated_invalid_startvalue`.
- **v3**: `tracking/tracker.jl:473-475` returns a single generic
  `TERMINATED_INVALID_STARTVALUE`.
- **Gain**: distinguishes "start point off the path" from "start Jacobian
  rank-deficient", enabling smarter retry/reporting.

## Tier 4: deliberate tradeoffs (decide explicitly)

### 9. Default `CompileMode.INTERPRETED`

`core/system.jl:52,74`. Every headline benchmark in `02_status.md` uses
`COMPILED`, so an out-of-the-box `solve(System(polys))` silently gets the
interpreter. This is legitimate TTFX-vs-steady-state tension, not a bug. Either
flip the default to `COMPILED` or document the trade at the `solve`/`System`
docstring boundary. Benchmark first-`solve` TTFX under each default before
deciding.

### 10. Struct-of-arrays LU layout for n ≳ 25

- **v2**: `linear_algebra.jl:36-38` wraps `A` in a `StructArrays.StructArray`
  when `m > 25` (the LU factors inherit the layout via `copy(A)`; the
  `MatrixWorkspace{M}` type parameter exists for this), vectorizing the LU inner
  loops.
- **v3**: `primitives/linear_algebra.jl:17-20` hardcodes AoS `FSMat{ComplexF64}`.
- **Tradeoff**: SoA conflicts with the monomorphic-tracker design; a real design
  call, not a free win.

### 11. Tracker tuning surface

- **v2** exposes `DEFAULT`/`FAST`/`CONSERVATIVE` `TrackerParameters` presets
  (`tracker.jl:58-62`), `min_rel_step_size`, and an `automatic_differentiation`
  order knob.
- **v3** hardcodes these (`TrackerOptions`, `tracking/tracker.jl:14-26`); always
  AD order 3. (v3's interpreter makes always-AD cheap, so the AD knob itself is
  not a robustness gap.)
- **Gain**: user control over the speed/robustness tradeoff.

### 12. DoubleF64 transcendentals (low priority)

- **v2**: `DoubleDouble.jl:612-1023` implements exp/log/sin/cos/tan/atan.
- **v3**: `primitives/double_f64.jl` implements only algebraic ops plus `sqrt`.
- **Gain**: extended-precision residuals for non-polynomial homotopies. This is
  required before `sin`/`cos` expression nodes can claim parity across ordinary,
  DF64, Taylor, and certification evaluation. `sqrt` already has the required
  extended-precision arithmetic.
- **Validation**: use parameter-dependent unary expressions in the common
  expression-input matrix described in `02_status.md`; do not validate only the
  scalar helpers in isolation.

## Recommended order

1. **#1 (binomial BigInt)**: robustness gap; large-support polyhedral solves
   that v2 handles abort with an uncaught `OverflowError` in v3.
2. **#2 (dedup scaling plus symmetry)**: scaling plus a genuine v2 feature gap.
3. **#4 to #6**: quick follow-on wins.

(#3, the dispatch reorder, was refuted by measurement; see above.)

## What is already solid (do not touch)

The tracker/Newton/predictor hot paths are genuinely allocation-free
(pre-allocated `FSVec`/`FSMat`/`TaylorVector` buffers, positional args,
`@inbounds` column-major loops), consistent with the AllocCheck gate. The
FunctionWrapper firewall is correctly monomorphic; all inspected struct fields are
concrete (`SolveCache{E,B,C}`, `System{P,V,M,S}`, `Interpreter{V}` properly
parametrized; `Union{ExcessSolutionChecker,Nothing}` only appears in the non-hot
`_finalize_result`). The `@generated` Taylor ops and custom LU/QR are
well-targeted.
