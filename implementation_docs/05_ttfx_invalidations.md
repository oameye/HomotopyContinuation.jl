# Whole-package TTFX, inference, and invalidation report

Last updated: 2026-07-19. Measurements use Julia 1.12.6 and
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
- zero precompile blockers from call-time invalidations in every workflow;
- eight small workload overlaps with package-load invalidation trees in the
  representative polyhedral capture (one package-owned, seven dependency-owned);
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
| Interpreted polyhedral, serial, quiet | 20.870 s | 14.767 s | 29.2% |
| Monodromy, serial, quiet | 17.056 s | 14.179 s | 16.9% |
| Monodromy, threaded, quiet | 17.232 s | 14.517 s | 15.8% |
| Subspace monodromy with trace test | 25.849 s | 19.842 s | 23.2% |
| Solution-completeness verification | 30.790 s | 25.217 s | 18.1% |
| Total degree with visible progress | 11.045 s | 10.742 s | 2.7% |

The largest remaining costs are real feature breadth: the two parameter-Taylor
FunctionWrappers actually used by polyhedral tracking, essential
MixedSubdivisions regeneration and cell traversal, tracker/Newton compilation,
subspace homotopies, and repeated monodromy/parameter-homotopy construction in
completeness verification.

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

- polyhedral: essential traverser construction, regeneration,
  `exchange_column!`, parameter-Taylor wrappers, and the two tracker stacks;
- subspaces: intrinsic/extrinsic Taylor closures, `LinearSubspace`, affine
  chart, and coordinate transformations;
- monodromy: `MonodromySolver`, `_clone_system_evaluator`,
  `find_start_pair`, parameter Taylor wrappers, and loop schedulers;
- completeness: two monodromy construction chains, parameter solves,
  polynomial merge, SVD, and the verification orchestration itself.

After direct support lowering, module-exclusive accounting for the
representative polyhedral workload is led by HomotopyContinuationNext (7.60 s),
Base (3.69 s), MixedSubdivisions (1.04 s), FunctionWrappers (0.99 s),
DynamicPolynomials (0.68 s), and LinearAlgebra (0.59 s). The remaining
DynamicPolynomials work belongs to construction of the user's input `System`,
not to a synthetic polyhedral parameter system. For the earlier serial
monodromy capture, HomotopyContinuationNext (9.52 s), Base (3.56 s),
FunctionWrappers (1.05 s), and DynamicPolynomials (0.82 s) led the accounting.

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

### Polyhedral canonical-support fast path

`System` already caches parameter-free supports as dense, nonnegative
`Matrix{Int32}` values and coefficients as `Vector{ComplexF64}` values. The
public MixedSubdivisions matrix entry point accepts arbitrary integer supports,
so it recompiles and executes `normalize_supports` even though polynomial
exponents cannot be negative here.

The retained `_fine_mixed_cells_canonical` path constructs the same
`MixedCellIterator` state directly from the trusted support cache. It keeps the
regeneration traverser, target traverser, fine-cell checks, retry behavior, and
overflow/singularity handling. It removes only redundant normalization plus
the unused progress and deprecated one-argument lifting-sampler branches. It
does not extend a MixedSubdivisions function, so it adds no type piracy or new
method-table invalidation root. MixedSubdivisions 1.2.0 does not expose an
equivalent public constructor; the six internal names are explicitly recorded
in the ExplicitImports allowlist and covered by an equivalence regression test.
Compatibility is capped to MixedSubdivisions 1.2.x until that private coupling
is replaced or revalidated against a newer release.

An exact before/after SnoopCompile capture of the interpreted serial workload
showed:

| Metric | Before fast path | After fast path | Reduction |
|---|---:|---:|---:|
| Compiler inclusive total | 86.126 s | 73.095 s | 15.1% |
| Flattened exclusive total | 22.506 s | 20.700 s | 8.0% |
| Inference nodes | 13,246 | 12,067 | 8.9% |
| `normalize_supports` instances | 30 | 0 | 100% |
| `normalize_supports` exclusive time | 0.243 s | 0 | 100% |

Three independent direct cold runs were 18.695 s, 18.574 s, and 18.656 s
(median 18.656 s, 2.71 GB allocated). The immediately preceding post-policy
measurement was 20.252 s, so this isolated change removes another 7.9% of wall
time. Cold validation across the other polyhedral entry points produced:

| Workflow | Cold wall after fast path |
|---|---:|
| Compiled, serial | 18.509 s |
| Compiled-all, serial | 18.913 s |
| Interpreted, one-task threaded | 19.464 s |
| Overdetermined, serial | 19.467 s |

The exact typed public `MixedCellIterator` method was already avoiding its
generic integer conversion overload; support conversion was therefore not a
remaining cost. Traverser construction also cannot be skipped on a first
solve: it is the algorithm state needed to enumerate mixed cells. The remaining
polyhedral gap was therefore led by genuine regeneration (`exchange_column!`,
`regeneration_stage_carry_over!`), the synthetic parametric start-system
frontend, Taylor wrappers, and the two tracker stacks.

### Direct support-backed polyhedral evaluator

The next pass removes that synthetic frontend. The old cold path rebuilt a
coefficient-parametric system in the following order:

```text
cached Matrix{Int32} support
  -> DynamicPolynomials coefficient variables and monomials
  -> general System normalization/lowering policy
  -> eval and Jacobian instruction sequences
  -> RuntimeGeneratedFunctions eval/Jacobian kernels
  -> System fields unused by polyhedral tracking
```

Polyhedral tracking only needs the coefficient-linear evaluator
`F_i(x; p) = sum_j p_ij * x^A_ij`, so `_SupportSystem` now stores only an
evaluator plus immutable eval/Jacobian instruction sequences. A narrow support
lowerer emits the tapes directly from `Vector{Matrix{Int32}}`. It caches powers
and repeated monomials, emits analytic derivatives in column-major Jacobian
order, and preserves the exact flattened coefficient-parameter ordering used
by `ToricHomotopy` and `CoefficientHomotopy`. Worker-local clones rebuild only
mutable interpreter tapes and FunctionWrappers from those immutable sequences.

Coefficient/monomial pairs go through the tape compiler's fused product-sum
reducer rather than emitting separate multiply and add instructions. On the
representative two-equation parity system this reduced the first direct tapes
from 10/20 to 8/14 eval/Jacobian instructions; the old symbolic path produced
8/15. A fully warm 1,000-solve comparison was 98.5 microseconds per direct
solve versus 94.1 microseconds for the former compiled evaluator (4.7% slower),
while allocations fell from about 7.07 KB to 5.81 KB per solve (17.9% lower).
The small steady-state trade is documented rather than hidden; restoring
runtime-generated eval/Jacobian code would give back part of the cold graph
this pass intentionally removes.

This removes `_build_parametric_system`, all synthetic DynamicPolynomials
arithmetic, the general small-versus-symbolic lowerer decision, CSE entry,
`System` normalization/metadata construction, and runtime-generated eval and
Jacobian code from the polyhedral path. The evaluator remains behind the
existing concrete `SystemEvaluator` FunctionWrapper firewall; the rejected
fully-parametric tracker design was not reintroduced.

Three independent direct cold interpreted-serial runs after tape fusion were
15.996 s, 15.845 s, and 16.132 s (median 15.996 s, about 2.36 GB allocated).
Relative to the canonical-support result of 18.656 s and 2.71 GB, this is
another 14.3% wall reduction and roughly 13% fewer allocated bytes. Relative
to the original pre-policy 20.870 s measurement, the cumulative polyhedral
reduction is 23.4%.

| Workflow | Before direct evaluator | After direct evaluator | Reduction |
|---|---:|---:|---:|
| Interpreted, serial (median) | 18.656 s | 15.996 s | 14.3% |
| Compiled, serial | 18.509 s | 16.952 s | 8.4% |
| Compiled-all, serial | 18.913 s | 17.519 s | 7.4% |
| Interpreted, one-task threaded | 19.464 s | 16.474 s | 15.4% |
| Overdetermined, serial | 19.467 s | 16.796 s | 13.7% |

A fresh SnoopCompile capture of the representative interpreted-serial workload
showed the same structural reduction:

| Metric | Canonical-support path | Direct evaluator | Reduction |
|---|---:|---:|---:|
| Compiler inclusive total | 73.095 s | 65.983 s | 9.7% |
| Flattened exclusive total | 20.700 s | 16.178 s | 21.8% |
| Inference nodes | 12,067 | 9,474 | 21.5% |

The entire new support construction subtree is about 1.16 s inclusive in that
capture; direct tape lowering is about 0.19 s inclusive. No
`_build_parametric_system`, RuntimeGeneratedFunctions expression generation,
or synthetic coefficient-polynomial construction remains. The highest
polyhedral entries are now Taylor instruction execution/wrappers, tracker
Newton/predictor inference, FunctionWrappers calls, and genuine
MixedSubdivisions regeneration.

### Capability-specific polyhedral Taylor evaluator

A follow-up call-graph audit found that the general interpreted evaluator was
still eagerly constructing six Taylor tapes/wrappers for `_SupportSystem`.
Polyhedral tracking has a narrower contract:

- first-order homotopy data comes from ordinary value/Jacobian evaluation;
- `ToricHomotopy` and `CoefficientHomotopy` request parameter-Taylor orders 2
  and 3;
- they never request scalar-parameter Taylor orders 1--3 or
  parameter-Taylor order 1.

The support evaluator now constructs only the order-2 and order-3 parameter
tapes and wrappers. Its four unreachable `SystemEvaluator` slots contain a
shared fail-fast function, preserving the concrete, monomorphic evaluator
layout without hiding an optional-value union in the tracker or compiling
unused kernels. This is an internal capability restriction: ordinary public
`System` evaluators retain all six Taylor modes. Regression tests assert both
the two supported modes against a symbolic `System` and the explicit errors
for unsupported internal modes.

The first implementation duplicated the four ordinary evaluation closure
types and erased most of the win. The retained implementation factors those
wrappers into `_build_interpreted_evaluation_fws`, shared by ordinary and
support-backed evaluators, so FunctionWrappers sees one callable type per
signature.

Fresh cold results are:

| Workflow | Before capability split | After capability split | Reduction |
|---|---:|---:|---:|
| Interpreted, serial (median of three) | 15.996 s | 14.351 s | 10.3% |
| Compiled, serial | 16.952 s | 15.353 s | 9.4% |
| Compiled-all, serial | 17.519 s | 15.563 s | 11.2% |
| Interpreted, one-task threaded | 16.474 s | 14.961 s | 9.2% |
| Overdetermined, serial | 16.796 s | 14.972 s | 10.9% |

The three interpreted-serial samples were 14.339 s, 14.351 s, and 14.462 s.
Allocated bytes fell from about 2.36 GB to 2.20 GB. An immediate fresh
SnoopCompile comparison moved hierarchical compiler accounting from 66.178 s
to 65.369 s (1.2% lower), exclusive accounting from 16.481 s to 16.634 s
(0.9% higher), and flattened nodes from 9,604 to 9,632 (0.3% higher). The
small exclusive/node movement is retained and reported rather than presented
as a compiler-graph reduction: the strong wall/allocation result comes from
not materializing four tapes/wrappers, while the two real parameter kernels
and the explicit shared-wrapper boundary still have to compile.

Because the factored evaluation-wrapper constructor is also used by ordinary
interpreted systems, total degree was checked separately in three fresh
sessions: 9.933 s, 9.936 s, and 10.053 s (median 9.936 s versus the preceding
9.840 s). The roughly 1% movement is within cold-run noise and shows no
material cross-feature regression.

### Positional tracker Taylor boundary

The next tracker-construction audit separated honest concrete compilation from
removable dispatch overhead. `Tracker`, `Predictor`, `NewtonCorrector`, and the
two polyhedral `HomotopyEvaluator` instances all infer concrete return types;
their remaining kernels are executed by tracking and cannot be removed or
deferred without changing the work performed. The removable boundary was the
order-2/order-3 `HomotopyEvaluator` closures: each called the underlying
homotopy through `incremental=` keyword dispatch even though the wrapper had
already received a concrete `Bool`. An intermediate implementation moved this
work into positional `_taylor!(..., incremental)` kernels while retaining
keyword compatibility wrappers.

A complete call-graph and history audit showed that this compatibility was not
useful. HomotopyContinuation v2's predictor passed `true` for orders 2 and 3,
apparently reserving the argument for incremental tape reuse, but the generic
`AbstractHomotopy` method immediately discarded it and every relevant concrete
implementation ignored it. The current predictor has no incremental caller or
state-reuse protocol. Retaining the argument would therefore preserve an
unimplemented intention rather than behavior.

Orders 2 and 3 now follow the order-1 contract directly: each built-in
homotopy implements one positional `taylor!(u, Val(K), H, tx, t)` method. The
`HomotopyEvaluator` closures call those methods directly; their
`FunctionWrapper` signatures no longer contain the unused `Bool`; and the
generic `_taylor!` bridge and generated keyword methods are gone. This was
carried through `StraightLineHomotopy`, both `LinearParameterHomotopy` modes,
`ToricHomotopy`, intrinsic and extrinsic subspace homotopies, and
`AffineChartHomotopy`. External `AbstractHomotopy` implementations use the
same five-argument contract.

In fresh sessions, direct Parameter/Toric methods and their type-erased
`HomotopyEvaluator` wrappers infer exactly `Nothing` for both orders. Targeted
JET optimization reports are empty and Cthulhu independently reports exact
`Nothing`. A repeated fresh interpreted-polyhedral SnoopCompile capture using
SnoopCompile 3.2.5/SnoopCompileCore 3.1.2 produced 9,550 flattened nodes and
zero `#taylor!#` roots. The earlier 9,355-node capture used the preceding
analysis environment, so it is retained as historical evidence for removal of
the keyword roots but is not presented as a paired node comparison with the
current tool environment.

Fresh cold samples were 14.777 s, 14.717 s, and 14.767 s (median 14.767 s,
about 2.196 GB). This does not reproduce the preceding 14.351-second median,
so no wall-time improvement is claimed for this pass. The current headline
uses the latest median, reducing the cumulative measured polyhedral gain from
31.2% to 29.2% rather than hiding the adverse sample.

A support-only Taylor interpreter with 13 polynomial operations was also
tested. It lowered the local support-evaluator inference root from about
0.95 s to 0.62 s, but the general Taylor kernels remained necessary elsewhere
in the same workflow. The additional executor therefore increased total nodes
from 9,355 to 9,424. It was reverted: locally smaller inference is not useful
when it duplicates the whole-workflow compiler graph.

### Concrete monodromy builders and evaluator ownership

The first vector-parameter monodromy worker was previously created by the same
anonymous closure used to build additional threaded workers. Two independent
problems followed:

- the closure captured `F` at its declared `System` abstraction, so inference
  traversed `_clone_system_evaluator` for interpreted, compiled, and
  compiled-all systems even when the concrete input was interpreted;
- the closure cloned `F.evaluator` for worker 1 even though no other owner uses
  that evaluator during the solve. Only simultaneous additional workers need
  independent mutable interpreter tapes.

`ParameterMonodromyBuilder{S}` and `ChartParameterMonodromyBuilder{S}` now keep
the concrete system type. Worker 1 consumes `F.evaluator` directly, while each
additional threaded worker calls its concrete builder and receives a fresh
clone. This is evaluator ownership reuse, not capability removal or deferred
serial work: serial execution never needs the clone, and threaded execution
still allocates one independent tape set per concurrently active worker.

The paired fresh serial SnoopCompile capture changed from 9,594 to 8,885 nodes
(7.4% lower) and from 16.464 s to 15.973 s of exclusive accounting (3.0%
lower). All serial `_clone_system_evaluator` roots disappeared; before the
change their inclusive subtree totaled 1.305 s and included all three compile
modes. In the threaded capture exactly one concrete interpreted clone root
remains (0.012 s inclusive), demonstrating that cloning is staged only where
worker isolation requires it. The threaded builder has the exact inferred and
Cthulhu return type
`MonodromyWorkerState{ParameterHomotopy,Vector{ComplexF64}}`; its sole targeted
JET report is the already-documented parameter-Taylor construction policy
barrier.

Fresh serial wall samples were 14.191 s, 14.179 s, and 14.158 s (median
14.179 s, about 1.788 GB). The wall improvement over the preceding 14.216 s is
only 0.3% and is not material; the retained win is the causal 709-node
inference-graph reduction and removal of speculative compile-mode clones.

## Invalidations

### Package load

The isolated package-load capture recorded:

- 62 nontrivial invalidation trees;
- 6,170 summed descendants;
- zero invalidated target MethodInstances owned by HomotopyContinuationNext;
- seven inserted roots owned by HomotopyContinuationNext or its
  `ExecInstruction` submodule, with 52 summed descendants.

The post-fast-path recapture kept the aggregate exactly at 62 trees and 6,170
descendants. It observed six package-owned roots and 47 descendants because the
five-descendant generated `ExecInstruction.variants` root was already satisfied
by the active precompile cache. The numeric roots and their descendant counts
were unchanged. The fast path itself inserts no external methods.

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
precompile blockers from runtime invalidations = 0
```

The fresh post-positional-boundary polyhedral capture again produced zero
runtime trees and zero descendants. The new lowering and shared wrapper constructor
define methods only on package-owned functions and package-owned
`_SupportSystem`/compiler types; neither extends a dependency function nor
inserts a method on an external type.

Some captures had stale instances, but with zero trees and zero blockers they
were pre-existing load/dependency state, not invalidations caused by executing
the feature. Remaining TTFX should therefore be attacked as compiler breadth,
not as runtime invalidation.

For completeness, pairing the representative polyhedral inference capture
with the earlier package-load capture reports eight load-time precompile
blockers and 118 stale instances:

| Inserted root owner | Blocker paths | Blocked workload edge |
|---|---:|---|
| HomotopyContinuationNext | 1 | `isnan(::DoubleF64)` to `isunordered` |
| StarAlgebras | 1 | broadcast `similar` to `restart_copyto_nonleaf!` |
| MutableArithmetics | 1 | `MutatingStepRange.step` to `_collect` |
| DataStructures | 5 | broad `merge!` methods to the same `_collect` edge |

The blocked inclusive timings are tiny (about 0.0004 s, 0.0007 s, and 0.0119 s
for the shared collection edge). They are real and should not be called zero,
but they do not explain the roughly 14.4-second first polyhedral execution.
The dependency-owned roots require upstream narrowing. The package-owned
`isnan(::DoubleF64)` specialization is part of the required `AbstractFloat`
interface and is the only local load-time blocker reached by this workload.

## Current JET report

Package-level correctness analysis passes with zero reports. The complete
concrete optimization matrix is summarized below. Reports classified as
`policy` or `builder` are intentional hard firewalls; `shape` is the bounded
runtime system-shape construction boundary.

| Workflow family | Reports per workload | Current interpretation |
|---|---:|---|
| Total degree, all modes/executors | 4 | 2 policy, 1 builder, 1 shape |
| Polyhedral, all modes/executors | 4 | 2 policy, 1 builder, 1 shape |
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

After direct support lowering, the targeted interpreted polyhedral report is
down from six deliberate reports to four: the input `System` builder, two solve
policy barriers, and the input system-shape boundary. JET reports zero for
`_build_support_instruction_sequence`, `_support_evaluator`, and
`_support_system`; the synthetic parameter-system builder barrier is gone.
The post-capability-split targeted correctness and optimization reports for
`_support_evaluator` and `_support_system` are also empty. Package-level
correctness analysis still has zero reports.

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

For the new path specifically, Cthulhu reports
`MixedCellIterator{RegenerationTraverser}` for
`_canonical_mixed_cell_iterator` and
`Tuple{Vector{MixedCell},Vector{Vector{Int32}}}` for
`_fine_mixed_cells_canonical`. Exhausted retries and the two recoverable
MixedSubdivisions failures throw at this internal boundary, so neither a union
nor `Any` escapes into `CommonSolve.init`.

For direct evaluator construction, including the capability split and shared
evaluation-wrapper constructor, it reports exact `_SupportSystem`,
`InstructionSequence`, and `SystemEvaluator` returns. The representative square
`CommonSolve.init` return is exactly
`PolyhedralSolveCache{Serial,PolyhedralBuilder{_SupportSystem},_SupportSystem,Nothing}`.
`Core.Compiler.return_type` agrees on every contract.

Two limitations remain visible:

- automatic subspace monodromy conservatively returns a union of vector- and
  subspace-parameter `MonodromyResult`, because the automatic start-pair result
  is selected from runtime parameter count;
- Cthulhu 3.0.2 on Julia 1.12.6 hits compiler assertion
  `info === NoCallInfo()` while generating the completeness workflow. JET and
  `Core.Compiler.return_type` succeed and report `Union{Nothing,Bool}`. This is
  a tool/compiler-API limitation, not a package runtime failure.

## Verification

- Full post-Taylor-contract source suite: 40 files, 3,571/3,571 tests passed
  in 2m19.6s, including the evaluator-ownership assertions and all affected
  homotopy families.
- Canonical/public MixedSubdivisions equivalence plus direct/symbolic evaluator
  value, Jacobian, clone, supported Taylor-order parity, and unsupported-mode
  contract: polyhedral regression 72/72.
- Coverage includes JET, Aqua, ExplicitImports, CheckConcreteStructs,
  AllocCheck, all compile modes, total-degree/polyhedral/parameter solves,
  overdetermined and singular systems, Newton, subspaces, monodromy parity,
  group actions, clustering, progress, and endgames.
- `git diff --check` passes.

## Follow-up pass, 2026-07-25

SnoopCompile v3.2.5 does not load on Julia 1.12 (`UndefVarError: Compiler.Params`),
so this pass drove `SnoopCompileCore` directly. On 1.12 each `CodeInstance` carries
`time_infer_self` and `time_compile` as `UInt16` fields holding `Float16` seconds
(`reinterpret(Float16, ci.time_infer_self)`); `@snoop_invalidations` returns an
`InvalidationLists` with separate `logmeths` and `logedges` vectors, and the
`logmeths` layout is the documented `[(tree, sig)..., method, reason]` grouping.

An inference barrier makes the callee a *new root* of the `@snoop_inference` tree
rather than a child, so subtree attribution must be read per root. For
`total_degree_interpreted_serial` the roots are:

| Root | Cost | CIs |
|---|---:|---:|
| `_solve_total_degree_serial_without_progress` | 3010 ms | 987 |
| `FunctionWrappers` thunks (32 roots) | 2473 ms | |
| `Main.Workload.run` (the `System` call chain) | 1064 ms | 759 |
| `_build_interpreted_system` | 884 ms | 1157 |
| `_build_instruction_sequence_direct` | 852 ms | 386 |
| `_init_total_degree_shaped` | 848 ms | 704 |
| `Compiler.inferiterate_2arg` (3 roots) | 248 ms | 6 |

The `FunctionWrappers` block is the single largest, and within it the three Taylor
thunks for the target system dominate (682 ms, 276 ms, 190 ms for tape element types
`TruncatedTaylorSeries{2,3,4}`). Their cost is the generated execute loop, not the
`taylor_op_*` kernels: `execute_taylor_instructions!` alone accounts for 279 ms of
inference on the order-1 tape because it inlines every op branch.

Four changes were A/B'd over interleaved fresh-process pairs at `-t 4`
(see `01_decisions.md` for the reasoning):

| Change | Recovered | Decision |
|---|---:|---|
| Drop `OP_SIN`/`OP_COS`/`OP_SQRT` from the interpreter dispatch table | ~0.26 s | Rejected: retain for the planned expression frontend |
| Install extended-precision wrappers on first use | ~0.25 s | Shipped |
| Extract the polynomial support on demand | ~0.09 s | Shipped |
| Pass `ws.A` to `skeel_row_scaling!` instead of the workspace | ~0.06 s | Shipped |

Four hypotheses or candidate changes were rejected. The unary interpreter variants
remain part of the execution contract for expressions such as
`sqrt(γ) * x₁ + x₂^2`; see the implementation and validation checklist in
`02_status.md`. Deferring the untaken QR branch with a
bare inference barrier recovers 156 ms but fails `test/alloc_check_test.jl`; the
follow-up pass below recovers the same time without the dynamic dispatch.
Removing OhMyThreads (and
with it the InitialValues invalidations, 523 of the ~3150 instances invalidated at
load) saves 58 ms of load time and changes the first call by less than the
run-to-run spread. Ordering `_EXEC_INSTRUCTION_SPECS` by measured op frequency
instead of `@data` declaration order cost 83% on the cyclic-7 Jacobian tape.

Steady state was measured for the rejected interpreter experiment because it
touched the hot tape loop. The first comparison against the 2026-07-24 baseline
showed everything ~1.5x
slower, including `inf_norm_4` (4.02ns -> 7.29ns) and `lu_ldiv_4` in files this
pass never touched: the machine was throttled after 45 minutes of A/B. After it
settled the untouched primitives returned to baseline (3.96ns, 172ns) and the
touched paths came out ahead: `track_one_path_katsura3` 151.9us -> 116.5us,
`build_jac_katsura3` 496us -> 406us, `build_katsura3` 460us -> 407us,
`taylor_katsura3` 107.9ns -> 103.0ns. Never compare a steady-state number against
a baseline from another session without an untouched control in the same run.

## The untaken QR branch, 2026-07-26

The 156 ms that the previous pass could only recover with a hot-path dynamic
dispatch is now recovered without one. `MatrixWorkspace` gained
`qr_factorize::QRFactorizeFW` and `qr_solve::QRSolveFW`, chosen by shape in
`_make_matrix_workspace`; the tall pair is constructed by calling `_tall_qr_ops`
through `Base.inferencebarrier`, the square pair by two no-op targets.

Why the wrapper is what makes the deferral legal: `FunctionWrappers` generates
its thunk in `gen_fptr`, which is `@generated` on the target's type, so putting
the *construction* of the tall pair behind the barrier is what keeps the QR
kernels out of the session. The *call* is then a `ccall` through a function
pointer, which `test/alloc_check_test.jl` already filters as a FunctionWrappers
boundary rather than a dynamic dispatch. All 46 assertions pass.

Verified by specialization count rather than by timing alone. Counting
`Base.specializations` over `methods(f)` after `total_degree_interpreted_serial`:

| method | before | after |
|---|---:|---:|
| `qr!` | 1 | 0 |
| `qr_ldiv!` | 1 | 0 |
| `reflector!` | 1 | 0 |
| `lmul_Q_adj!` | 1 | 0 |
| `ldiv_upper!` | 4 | 2 |

`ldiv_upper!` keeps the two `FSMat` specializations the LU path needs and loses
the two `Matrix` ones only `qr_ldiv!` reached.

Timing, 5 interleaved fresh-process pairs at `-t 4`, all 5 paired differences
favouring the wrapper: 8.427--9.062 s before, 8.312--8.426 s after, ~0.15 s
median. The spread matters as much as the median here: removing the branch takes
the first-call range from 0.64 s to 0.11 s. Forcing the deferred path in a
session that has already run a square solve costs 0.345--0.363 s, and the second
tall solve costs 54 us, so the deferral is complete rather than partial.

## Remaining root-cause priorities

Continue without PrecompileTools in this order:

1. **Automatic subspace dispatch.** Move vector-parameter versus
   `LinearSubspace` result selection to a concrete outer method boundary if it
   improves both caller inference and cold time; do not merely exchange the
   current union for another speculative branch.
2. **Completeness evaluator reuse.** The initial monodromy worker clone is now
   gone. Audit whether the two sequential trace homotopies can safely retarget
   and reuse one parameter tracker without retaining or sharing tapes from the
   completed auxiliary monodromy phase.
3. **Large symbolic input.** Lower directly into a package-owned canonical
   representation where possible, avoiding DynamicPolynomials arithmetic used
   only for normalization while retaining CSE for genuinely large systems.
4. **Taylor interpreter architecture.** The used order-2/order-3 kernels are
   honest work. Revisit them only with a representation that replaces the
   general executor across all consumers; the tested support-only executor
   duplicated the graph and is rejected. The three execute-loop copies this
   leaves are a settled cost, and so no longer a priority: see
   `01_decisions.md`, "The Taylor order stays a type parameter".
5. **Dependency invalidations upstream.** Narrow the broad DataStructures and
   MultivariatePolynomials methods responsible for most load-time impact, and
   request a public MixedSubdivisions already-normalized iterator constructor
   so the current private 1.2.x integration can be removed.
6. **Precompile only the irreducible remainder.** If the root-cause items above
   plateau, add the smallest safe representative workload and re-run every
   allocation-sensitive Taylor test for every compile mode.

The current evidence does not justify using PrecompileTools yet: substantial
latency was still removable by architecture, and the remaining hotspots have
specific source-level owners.

That verdict is unchanged by `src/precompile_signatures.jl`, which is a different
mechanism. `precompile` is a Base builtin taking a signature, so it adds no
dependency and executes no workload; it exists because the tape executors sit
behind a `@cfunction` and therefore cannot be reached by any workload at all. See
`01_decisions.md`, "Tape executors are precompiled by signature".
