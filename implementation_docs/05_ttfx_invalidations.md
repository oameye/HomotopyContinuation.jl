# Whole-package TTFX, inference, and invalidation report

Last updated: 2026-07-18. Measurements use Julia 1.12.6 and
HomotopyContinuationNext. `src/precompile.jl` is disabled. No result in this
report uses PrecompileTools; precompilation remains a last-mile option after
root-cause work.

## Executive result

The package-wide pass changes the earlier conclusion in an important way: the
small interpreted total-degree solve is not representative of the package's
largest cold paths. Polyhedral solving, subspace tracking, monodromy, and
solution-completeness verification compile substantially broader graphs.

The complete 28-workflow matrix found:

- zero call-time invalidation trees in every workflow;
- zero SnoopCompile precompile blockers in every workflow;
- package-level JET correctness analysis with zero errors;
- first-execution cost dominated by genuine inference and LLVM generation,
  not by runtime invalidation;
- avoidable policy leakage: quiet calls compiled ProgressMeter, serial
  monodromy compiled the threaded scheduler, and automatic start-pair search
  compiled mutually exclusive parameter-free and joint-Newton paths.

The retained policy barriers remove those unused compiler branches. Selected
fresh-process improvements are:

| Cold workflow | Before | After | Reduction |
|---|---:|---:|---:|
| Interpreted total degree, serial, quiet | 11.063 s | 9.840 s | 11.1% |
| Interpreted polyhedral, serial, quiet | 20.870 s | 20.252 s | 3.0% |
| Monodromy, serial, quiet | 17.056 s | 14.216 s | 16.7% |
| Monodromy, threaded, quiet | 17.232 s | 14.517 s | 15.8% |
| Subspace monodromy with trace test | 25.849 s | 19.842 s | 23.2% |
| Solution-completeness verification | 30.790 s | 25.217 s | 18.1% |
| Total degree with visible progress | 11.045 s | 10.742 s | 2.7% |

The largest remaining costs are real feature breadth: evaluator and Taylor
FunctionWrapper construction, MixedSubdivisions traversal, tracker/Newton
compilation, subspace homotopies, and repeated monodromy/parameter-homotopy
construction in completeness verification.

## Reproducible scope

The workload catalog is implemented in
`benchmark/ttfx_workloads.jl`. It covers:

1. total degree in all three compile modes, serial and threaded;
2. polyhedral solving in all three compile modes, serial and threaded;
3. parameter homotopy, serial and threaded;
4. square, overdetermined, underdetermined-Newton, and singular systems;
5. large symbolic construction in interpreted and compiled-all modes;
6. ordinary and extended-precision Newton;
7. intrinsic and extrinsic subspace tracking;
8. affine-chart construction;
9. serial, threaded, group-action, and subspace-trace monodromy;
10. completeness verification;
11. unique-point group actions and result clustering;
12. the visible-progress path.

Each cold wall measurement ran after package load in an otherwise fresh Julia
session. The analysis environments load Revise automatically, so package load
was about 1.24--1.31 s rather than the earlier startup-disabled 0.78 s. The
tables therefore compare feature execution consistently within this matrix;
they should not be combined arithmetically with the older startup-disabled
load number.

SnoopCompile compiler totals are hierarchical accounting: nested inference
and LLVM times are summed and can be much larger than wall time. They rank
compiler breadth; they are not elapsed time.

## Whole-feature cold matrix

This is the complete pre-policy-refactor matrix used to identify the next
root causes. `Compiler` includes inference plus LLVM. `Exclusive` is the sum of
flattened exclusive compiler time. `Nodes` is the flattened inference-node
count. Every row had zero runtime invalidation trees.

| Workflow | Wall | Allocated | Compiler | Exclusive | Nodes |
|---|---:|---:|---:|---:|---:|
| Total degree, interpreted, serial | 11.063 s | 1.61 GB | 50.49 s | 12.29 s | 6,660 |
| Total degree, compiled, serial | 11.747 s | 1.76 GB | 54.47 s | 13.61 s | 7,069 |
| Total degree, compiled-all, serial | 11.295 s | 1.61 GB | 49.38 s | 12.50 s | 7,096 |
| Total degree, interpreted, threaded | 10.544 s | 1.64 GB | 54.24 s | 11.94 s | 7,649 |
| Polyhedral, interpreted, serial | 20.870 s | 3.26 GB | 90.78 s | 22.73 s | 13,475 |
| Polyhedral, compiled, serial | 20.633 s | 3.20 GB | 88.98 s | 22.25 s | 13,184 |
| Polyhedral, compiled-all, serial | 21.058 s | 3.24 GB | 88.45 s | 22.63 s | 13,454 |
| Polyhedral, interpreted, threaded | 21.165 s | 3.36 GB | 97.68 s | 23.24 s | 14,474 |
| Parameter homotopy, interpreted, serial | 9.986 s | 1.55 GB | 41.38 s | 11.35 s | 5,909 |
| Parameter homotopy, compiled-all, threaded | 9.871 s | 1.51 GB | 44.58 s | 11.16 s | 7,193 |
| Overdetermined total degree | 11.403 s | 1.60 GB | 46.90 s | 12.76 s | 6,475 |
| Overdetermined polyhedral | 21.637 s | 3.27 GB | 88.36 s | 23.17 s | 13,243 |
| Singular endgame | 10.953 s | 1.61 GB | 46.78 s | 12.46 s | 6,763 |
| Large symbolic interpreted build | 6.896 s | 0.95 GB | 30.12 s | 8.49 s | 5,069 |
| Large symbolic compiled-all build | 7.531 s | 0.99 GB | 25.87 s | 8.52 s | 5,439 |
| Newton, standard | 6.269 s | 1.04 GB | 25.54 s | 7.50 s | 4,890 |
| Newton, extended underdetermined | 6.349 s | 1.05 GB | 25.45 s | 7.58 s | 4,883 |
| Intrinsic subspace track | 16.032 s | 2.36 GB | 55.64 s | 17.08 s | 9,295 |
| Extrinsic subspace track | 15.574 s | 2.13 GB | 53.44 s | 16.63 s | 9,129 |
| Affine chart | 10.534 s | 1.59 GB | 39.05 s | 10.97 s | 7,243 |
| Monodromy, serial | 17.056 s | 2.44 GB | 70.96 s | 18.67 s | 10,396 |
| Monodromy, threaded | 17.232 s | 2.45 GB | 70.66 s | 18.79 s | 10,422 |
| Monodromy with group action | 17.891 s | 3.36 GB | 91.51 s | 19.21 s | 12,829 |
| Monodromy with subspace trace | 25.849 s | 3.84 GB | 103.46 s | 26.66 s | 14,331 |
| Verify solution completeness | 30.790 s | 5.02 GB | 138.87 s | 32.39 s | 21,370 |
| Unique points with group action | 0.551 s | 0.10 GB | 1.94 s | 0.57 s | 661 |
| Result clustering | 0.387 s | 0.06 GB | 0.75 s | 0.39 s | 309 |
| Total degree with visible progress | 11.045 s | 1.62 GB | 50.31 s | 12.35 s | 6,709 |

### What the matrix says

- Compile mode is not the main first-call determinant. Interpreted remains the
  best default overall; compiled modes move work between construction and
  execution without a consistent cold win.
- Polyhedral solving roughly doubles total-degree compiler breadth. The
  additional graph is primarily MixedSubdivisions traversal and start-system
  construction, not an invalidation storm.
- Intrinsic/extrinsic subspace paths add about 2,400--2,600 inference nodes
  over the basic solve and compile distinct homotopy/Taylor closures.
- Monodromy adds evaluator cloning, start-pair construction, loop scheduling,
  deduplication, and repeated Newton/tracker paths.
- Completeness verification is the widest workload because it composes an
  initial monodromy solve, another monodromy phase, and multiple parameter
  homotopies. Its 30.79 s was cumulative feature breadth, not one bad method.
- Unique-point deduplication and clustering are small, concrete leaf features;
  they are not current TTFX priorities.

## Root attribution

Across the broad workloads, the shared leading exclusive costs were:

| Compiler root | Typical exclusive cost | Interpretation |
|---|---:|---|
| `execute_taylor_instructions!` | 0.25--0.30 s | Taylor interpreter specialization |
| `FunctionWrappers.FunctionWrapper` | about 0.26 s | evaluator type firewall construction |
| FunctionWrappers generated thunk | about 0.23 s | call-wrapper generation |
| `_optimize_instruction_order` | about 0.15 s | symbolic tape ordering |
| `init_newton!` / `newton!` | about 0.16--0.17 s each | numerical tracker stack |

Feature-specific additions were:

- polyhedral: `MixedSubdivisions.normalize_supports`, traverser construction,
  regeneration, and `exchange_column!`;
- subspaces: intrinsic/extrinsic Taylor closures, `LinearSubspace`, affine
  chart, and coordinate transformations;
- monodromy: `MonodromySolver`, `_clone_system_evaluator`,
  `find_start_pair`, parameter Taylor wrappers, and loop schedulers;
- completeness: two monodromy construction chains, parameter solves,
  polynomial merge, SVD, and the verification orchestration itself.

Module-exclusive accounting for the representative polyhedral workload was
led by HomotopyContinuationNext (8.80 s), Base (5.73 s), MixedSubdivisions
(1.62 s), FunctionWrappers (1.06 s), DynamicPolynomials (0.87 s), and Printf
(0.81 s). For serial monodromy it was HomotopyContinuationNext (9.52 s), Base
(3.56 s), FunctionWrappers (1.05 s), and DynamicPolynomials (0.82 s).

## Retained root-cause changes

### Construction policy barriers

Earlier retained work selects compile mode, system shape, and polynomial
lowering strategy at hard inference boundaries. This prevents interpreted
construction from traversing compiled RuntimeGeneratedFunctions and prevents
square solves from traversing overdetermined randomization. Adaptive direct
polynomial lowering removes the S-expression/CSE frontend for small systems,
while large systems retain it for compact tapes.

Parameter-free systems also reuse ordinary Taylor wrappers instead of building
three unused parameter-Taylor convolution kernels. Coefficient normalization
always produces a stable floating polynomial type.

### Progress and executor policy split

`SolveCache` and `PolyhedralSolveCache` retain the public `show_progress::Bool`
API, but `solve!` now selects a concrete quiet/progress body behind one hard
outer dispatch. The quiet body receives `nothing` directly and never infers
ProgressMeter or Printf.

Monodromy selects one of four concrete bodies: serial/threaded crossed with
quiet/progress. A quiet serial run no longer compiles the threaded task graph
or progress display.

Fresh SnoopCompile comparisons validate the mechanism:

| Metric | Total degree before | Total degree after | Monodromy before | Monodromy after |
|---|---:|---:|---:|---:|
| Hierarchical compiler accounting | 50.49 s | 43.29 s | 70.96 s | 63.03 s |
| Inference accounting | 43.07 s | 36.40 s | 59.09 s | 52.97 s |
| Exclusive accounting | 12.29 s | 11.35 s | 18.67 s | 16.26 s |
| Flattened nodes | 6,660 | 6,236 | 10,396 | 9,598 |
| ProgressMeter exclusive work | present | zero | present | zero |
| `threaded_monodromy_solve!` instances in serial run | n/a | n/a | present | zero |
| Runtime invalidation trees | 0 | 0 | 0 | 0 |

This intentionally trades a small, explicit outer runtime dispatch for a much
smaller compiled graph. Concrete result annotations preserve `Result` and
`MonodromyResult{P,P}` contracts after the firewall.

### Start-pair strategy split

`find_start_pair` formerly inferred all of these for every automatic call:

1. Newton on a parameter-free system;
2. the linear-in-parameters fast path;
3. joint Newton in `(x,p)` as a nonlinear fallback.

It now selects parameter-free versus parameterized construction behind a hard
policy boundary. The joint-Newton fallback has a second barrier and compiles
only if three linear attempts actually fail. This is a major contributor to
the 2.7--6.0 s monodromy/completeness reductions above.

### ProgressMeter numeric narrowing

JET found two arithmetic dispatches in `ProgressUnknown`, not in the trace
test as first suspected. ProgressMeter exposes `dt` through an inferred `Real`
property although its runtime value is `Float64`. Concrete local assertions
remove the `Float64 + Real` and `Float64 > Any` reports.

## Invalidations

### Package load

The isolated package-load capture recorded:

- 62 nontrivial invalidation trees;
- 6,170 summed descendants;
- zero invalidated target MethodInstances owned by HomotopyContinuationNext;
- seven inserted roots owned by HomotopyContinuationNext or its
  `ExecInstruction` submodule, with 52 summed descendants.

`filtermod(HomotopyContinuationNext, trees)` answers which invalidated targets
belong to the package. Root ownership must instead be read from
`tree.method.module`; these are different questions.

| Module owning inserted roots | Trees | Summed descendants |
|---|---:|---:|
| DataStructures | 10 | 2,546 |
| MultivariatePolynomials | 11 | 2,187 |
| InitialValues | 4 | 341 |
| StaticArrays | 13 | 311 |
| MutableArithmetics | 4 | 227 |
| Jieko | 1 | 190 |
| StarAlgebras | 3 | 167 |
| HomotopyContinuationNext, including ExecInstruction | 7 | 52 |

DataStructures and MultivariatePolynomials account for about 77% of the
descendant-impact sum. The largest roots are broad `merge!`, `==`, and
`isequal` method insertions. Their correct fixes are upstream signature
narrowing, not package-local precompile workloads.

The local roots are narrow numeric/array/generated-enum interfaces involving
package-owned types:

| Local inserted root | Descendants |
|---|---:|
| `isnan(::DoubleF64)` | 16 |
| `convert(::Type{DoubleF64}, ::DoubleF64)` | 14 |
| `zero(::Type{DoubleF64})` | 10 |
| generated `variants(::Type{ExecInstruction})` | 5 |
| `convert(::Type{DoubleF64}, ::Integer)` | 4 |
| `one(::Type{DoubleF64})` | 2 |
| `eltype(::Type{TaylorVector})` | 1 |

Removing the apparently redundant same-type `convert` was tested. It removed
that 14-descendant root, but the same backedges redistributed to the broader
`Integer` and `AbstractFloat` conversion roots. The capture remained exactly
seven local trees and 52 descendants, so the method was restored. This was a
neutral experiment, not an invalidation improvement.

### Runtime workflows

All 28 workload captures produced:

```text
runtime invalidation trees = 0
runtime invalidation descendants = 0
precompile blockers = 0
```

Some captures had stale instances, but with zero trees and zero blockers they
were pre-existing load/dependency state, not invalidations caused by executing
the feature. Remaining TTFX should therefore be attacked as compiler breadth,
not as runtime invalidation.

## Current JET report

Package-level correctness analysis passes with zero reports. The complete
concrete optimization matrix is summarized below. Reports classified as
`policy` or `builder` are intentional hard firewalls; `shape` is the bounded
runtime system-shape construction boundary.

| Workflow family | Reports per workload | Current interpretation |
|---|---:|---|
| Total degree, all modes/executors | 4 | 2 policy, 1 builder, 1 shape |
| Polyhedral, all modes/executors | 5--6 | 2 policy, 2--3 builders, 1 shape |
| Parameter homotopy | 3 | 1 policy, 1 builder, 1 shape |
| Overdetermined/singular solve | 4--5 | policy/builders/shape only |
| Large symbolic construction | 1 | selected system builder |
| Newton, ordinary/extended | 1 | selected system builder |
| Intrinsic/extrinsic subspace | 1 | selected system builder |
| Affine chart | 1 | selected system builder; value-dependent homotopy union |
| Monodromy, serial/threaded | 12 | 1 policy, 2 builders, 9 evaluator-wrapper reports |
| Monodromy with group action | 10 | 2 builders, 8 evaluator-wrapper reports |
| Monodromy with subspace trace | 12 | same plus value-dependent result union |
| Completeness verification | 15 | composed monodromy/solve boundaries |
| Unique points/group action | 0 | fully concrete |
| Result clustering | 0 | fully concrete |
| Visible progress | 4 | policy/builder/shape only; no ProgressMeter report |

The nine recurring monodromy reports are concentrated in abstract polynomial
variable discovery and construction of `SysEvalFW`, `SysEvalJacFW`, and the
ordinary/parameter Taylor FunctionWrappers behind the deliberate evaluator
type firewall. The earlier ProgressMeter arithmetic reports are gone.

Current inferred return contracts include:

```text
total degree / polyhedral / parameter solve => Result
Newton                                    => NewtonResult
intrinsic subspace track                  => Tuple{EndgameCode.T, Vector{ComplexF64}}
extrinsic subspace track                  => EndgameCode.T
vector-parameter monodromy                => MonodromyResult{Vector{ComplexF64},Vector{ComplexF64}}
automatic subspace monodromy              => union of vector/subspace MonodromyResult
verify_solution_completeness              => Union{Nothing,Bool}
unique points / clustering                => concrete container/tuple types
```

The `Union{Nothing,Bool}` completeness return is its documented failure-or-
answer contract, not an inference defect.

## Cthulhu report

Programmatic `find_method_instance` plus `generate_code_instance` confirmed
the same concrete contracts for representative total-degree, polyhedral,
parameter, Newton, intrinsic/extrinsic subspace, vector monodromy, unique-point,
and clustering workflows.

Two limitations remain visible:

- automatic subspace monodromy conservatively returns a union of vector- and
  subspace-parameter `MonodromyResult`, because the automatic start-pair result
  is selected from runtime parameter count;
- Cthulhu 3.0.2 on Julia 1.12.6 hits compiler assertion
  `info === NoCallInfo()` while generating the completeness workflow. JET and
  `Core.Compiler.return_type` succeed and report `Union{Nothing,Bool}`. This is
  a tool/compiler-API limitation, not a package runtime failure.

## Verification

- Full suite: 40 files, 3,524/3,524 tests passed in 1m57.8s.
- Final focused suite after concrete monodromy result annotations: 322/322.
- Coverage includes JET, Aqua, ExplicitImports, CheckConcreteStructs,
  AllocCheck, all compile modes, total-degree/polyhedral/parameter solves,
  overdetermined and singular systems, Newton, subspaces, monodromy parity,
  group actions, clustering, progress, and endgames.
- `git diff --check` passes.

## Remaining root-cause priorities

Continue without PrecompileTools in this order:

1. **Polyhedral start construction.** Profile and reduce
   MixedSubdivisions `normalize_supports`, traverser, regeneration, and support
   conversion. This is now the largest isolated feature addition.
2. **Evaluator/Taylor wrapper architecture.** Consolidate or stage repeated
   FunctionWrapper thunk construction without exposing large interpreter types
   to tracker specialization. The previous fully parametric evaluator attempt
   regressed cold solve time and remains rejected.
3. **Monodromy evaluator reuse.** Reduce `_clone_system_evaluator` and repeated
   MonodromySolver/parameter-Taylor construction, especially across
   completeness verification's multiple phases.
4. **Automatic subspace dispatch.** Move vector-parameter versus
   `LinearSubspace` result selection to a concrete outer method boundary if it
   improves both caller inference and cold time; do not merely exchange the
   current union for another speculative branch.
5. **Large symbolic input.** Lower directly into a package-owned canonical
   representation where possible, avoiding DynamicPolynomials arithmetic used
   only for normalization while retaining CSE for genuinely large systems.
6. **Dependency invalidations upstream.** Narrow the broad DataStructures and
   MultivariatePolynomials methods responsible for most load-time impact.
7. **Precompile only the irreducible remainder.** If the root-cause items above
   plateau, add the smallest safe representative workload and re-run every
   allocation-sensitive Taylor test for every compile mode.

The current evidence does not justify using PrecompileTools yet: substantial
latency was still removable by architecture, and the remaining hotspots have
specific source-level owners.
