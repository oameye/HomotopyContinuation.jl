## Test Suite

Last updated: 2026-09-14.

`make test` runs the core suite through ParallelTestRunner and then the certification subpackage;
`make test-cert` runs the certification subpackage alone. `test/extensive/` remains outside the
normal test discovery and is run with `make test-extensive`.

The exact assertion count is intentionally not treated as an invariant: several loops assert over
discovered solutions, so the number of assertions can vary when a stochastic solve finds a
different number of endpoints. Correctness is pinned through solution counts, classifications,
numerical comparisons and dedicated deterministic regressions instead.

## Core coverage

| Category | Representative files | Coverage |
|---|---|---|
| Quality gates | `aqua_test.jl`, `jet_test.jl`, `explicit_imports_test.jl`, `concrete_structs_test.jl` | API/package hygiene, inference/static analysis, concrete fields |
| Allocation | `alloc_check_test.jl` | zero-allocation hot paths for norms, linear algebra, predictor/Newton/tracker/endgame and wrapper-system paths |
| Primitives | `double_f64_test.jl`, `norms_test.jl`, `linear_algebra_test.jl`, `operations_test.jl`, `taylor_test.jl` | DoubleF64/ComplexDF64, norms, custom LA, scalar op and Taylor recurrences |
| Model/evaluation | `interpreter_test.jl`, `codegen_test.jl`, `instruction_count_test.jl`, `polynomial_input_test.jl`, `expression_test.jl`, `symbolic_utils_test.jl` | tape execution, RGF lowering, CSE/instruction regressions, polynomial/expression front ends |
| Non-polynomial | `nonpolynomial_test.jl`, `log_test.jl` | rational/transcendental evaluation, Jacobian/DF64/Taylor paths, principal-branch `log`, branch behavior |
| Core systems | `core_test.jl`, `composition_test.jl`, `fixed_parameter_test.jl`, `fixed_parameter_homotopy_test.jl`, `linear_subspace_test.jl` | system wrappers, composition, fixed parameters, custom fixed-parameter homotopies, subspace endpoint dimensions/promotion |
| Homotopies | `parameter_homotopy_test.jl`, `symbolic_homotopy_test.jl`, `homotopy_solve_test.jl`, `subspace_homotopy_test.jl`, `affine_chart_test.jl` | parameter/symbolic/custom homotopies, cloning, explicit solve routes, affine/projective wrappers |
| Tracking | `tracking_test.jl`, `endgame_test.jl`, `newton_test.jl`, `tracker_warmstart_test.jl`, `valuation_test.jl`, `tracker_regression_test.jl` | predictor/corrector, endgame, precision escalation, valuation, trust-region regressions |
| Solving | `solve_test.jl`, `binomial_system_test.jl`, `polyhedral_regression_test.jl`, `overdetermined_test.jl`, `result_clustering_test.jl`, `path_diagnostics_test.jl`, `progress_test.jl` | algorithms, path/result classification, excess filtering, clustering, progress and early stop |
| Subspaces/sweeps | `sliced_solve_test.jl`, `subspace_solve_test.jl`, `many_targets_test.jl`, `result_iterator_test.jl` | slices, subspace continuation, many targets, lazy tracking |
| Distributed | `distributed_test.jl` | process execution against serial, batching, serialization and failure propagation |
| Monodromy | `monodromy_test.jl`, `monodromy_v2_parity_test.jl`, `voronoi_tree_test.jl`, `group_actions_test.jl` | serial/threaded/distributed semantics, endpoint admission, permutations, trace, group actions, v2 cases |
| Witness/NID | `witness_set_test.jl`, `nid_test.jl` | affine/projective, zero-dimensional, parametric/rational input, regeneration, decomposition, unresolved outputs |
| v2 comparison | `compare_v2_primitives_test.jl`, `compare_v2_solve_counts_test.jl`, `compare_v2_solve_match_test.jl`, `v2_parity_test.jl` | primitive, count and solution-value parity |
| Integration | `semialgebraic_sets_test.jl`, `utils_test.jl` | SemialgebraicSets and solution/parameter I/O |

## Final parity-tranche regressions

The September 2026 catch-up PRs added or strengthened regressions for the late-v2 changes. These
are now part of the ordinary suite and should be treated as release invariants.

### Predictor trust radius

`tracker_regression_test.jl` covers a zero crossing where no derivative-derived trust radius is
available. The fallback must be finite and positive, while an available derivative-based radius
remains authoritative. Regular and Hermite predictor modes reject non-finite/non-positive radii.

### LinearSubspace endpoint dimensions and promotion

`linear_subspace_test.jl` covers:

- `dim == 0` and `dim == ambient_dim`;
- `codim == 0` and `codim == ambient_dim`;
- affine point slices;
- rejection of a zero-dimensional linear subspace through a nonzero point;
- `rand_subspace!` with a zero point and `affine = false`;
- translating a real-coefficient subspace by a complex displacement without mutating the source.

### Certification candidates from Result

`lib/HomotopyContinuationNextCertification/test/result_candidate_certification_test.jl` pins the
semantic distinction between tracker classification and rigorous certification: a successful
regular root manually labelled numerically singular is hidden by `solutions(result)` but still
appears as one successful cluster representative to `certify(F, result)` and certifies.

Failed paths remain excluded and raw duplicates remain collapsed by the `Result` cluster
partition.

### Principal-branch log

`test/log_test.jl` and `lib/.../test/log_test.jl` cover `log` through:

- `Expression` / `SExpr` lowering;
- interpreted and compiled evaluation;
- symbolic differentiation;
- DoubleF64 / ComplexDF64;
- generated Taylor recurrences;
- interval and Acb certification;
- principal-branch/negative-real-axis rejection where a box or ball crosses the cut.

### Witness ambient-coordinate contract

Witness intersection tests require exact ordered ambient variables. Two witness sets that use a
different variable set or a different variable order are rejected instead of being silently
renamed positionally.

### Regeneration ownership of point identity

`nid_test.jl` covers the final regeneration semantics:

- `EquationSorting.BY_DEGREE` means increasing degree;
- `EquationSorting.RANDOMIZED` retains its decreasing-degree pre-order before triangular
  randomization;
- outer `atol`/`rtol` propagate through internal monodromy and final deduplication;
- internal monodromy runs with `equivalence_classes = false`;
- the final `unique_points` pass does not receive group actions, so symmetry cannot quotient
  witness cardinality.

### Lossless unresolved decomposition

NID regressions verify that witness points discovered during trace-test monodromy are absorbed into
persistent point identity, and that an iteration-budget exhaustion returns unresolved orbits as
`WitnessSet(...; irreducibility = Irreducibility.UNKNOWN)` rather than dropping them.

The public result surface includes `irreducible_components`, `unresolved_witness_sets` and
`unresolved_degree`; `ncomponents` and `degrees` describe proven irreducible components only.

### Hardened monodromy endpoint admission

`monodromy_test.jl` covers:

- rejecting singular loop endpoints before they become future starts;
- base-problem revalidation of genuinely new heuristic endpoints;
- no redundant revalidation of already-known duplicates;
- preservation of the certified-duplicate path;
- the same driver-owned identity semantics across serial/threaded/distributed execution;
- zero-padding incomplete permutation histories when the solution set grows.

### Fixed-parameter custom homotopies

`fixed_parameter_homotopy_test.jl` defines a custom parameterized `AbstractHomotopy` and checks
binding through `FixedParameterHomotopy`: value, Jacobian, DoubleF64, Taylor coefficients,
metadata, cloning and an actual solve all pass through the ordinary v3 homotopy evaluator path.

## Certification subpackage

The certification package has its own test environment and covers:

- rectangular interval arithmetic and sampled soundness;
- the Acb interpreter and precision refinement;
- Krawczyk certification and Arb fallback;
- `DistinctCertifiedSolutions` and collision/dedup behavior;
- certified monodromy duplicate admission;
- low-memory `ResultIterator` certification;
- principal-branch transcendental operations, including `log`;
- the candidate-selection semantics of `certify(F, ::Result)`.

The extensive iterator-certification test and the 3264-conic data remain outside the ordinary
suite.

## Extensive suite

`make test-extensive` covers the large reference problems in `test/extensive/`:

- lines on a quintic surface;
- 3264 conics tangent to five conics;
- the low-memory iterator-certification route on the large conic problem.

These runs are intentionally separated from the ordinary CI-sized suite because they are long
numerical workloads rather than unit/regression tests.

## Permanent CI

The development CI introduced in PRs #10/#11 is now part of the parity claim. On pushes to `v3`
and relevant pull requests it supplies:

- **Core tests** on Julia 1.10 and current Julia;
- **Certification tests** on Julia 1.10 and current Julia;
- **JET** package-level static analysis;
- **Runic** formatting;
- **Benchmark tracking** on pinned Julia 1.13 with a cached baseline and fatal regression
  threshold;
- **TTFX tracking** for representative fresh-process workloads;
- **TTFX full inventory** on `v3` pushes.

After the final catch-up PRs were merged, the exact integrated `v3` tree passed all of these gates.
This is important: feature completion is certified on the combined tree rather than inferred from
individually green PR branches.

## v2 suite coverage

Every v2 feature-bearing test file has a v3 counterpart or a documented architecture-specific
reason why the test itself is inapplicable. The principal intentionally unported test family is
v2's compiled-cache locking: v3 has no global system-specific compile table, so there is no cache
whose locks need the corresponding test.

The remaining differences are name/API architecture differences and release-surface decisions,
not missing numerical capabilities. See `02_status.md`, `03_v3_vs_v2.md`, and
`09_v2_parity_freeze.md`.