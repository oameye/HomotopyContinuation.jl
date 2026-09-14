## Test Suite

Last updated: 2026-09-14.

`make test` runs the core suite through ParallelTestRunner and then the
`HomotopyContinuationCertification` package; `make test-cert` runs certification alone.
`test/extensive/` remains outside normal discovery and is run with `make test-extensive`.

The exact assertion count is not an invariant: several loops assert over discovered solutions, so
the number can vary with stochastic endpoint discovery. Correctness is pinned through counts,
classifications, numerical comparisons and deterministic regressions.

## Package-identity transition

PR #12 restores the production package from the temporary rewrite identity
`HomotopyContinuationNext` to the registered `HomotopyContinuation` name and UUID.

The package-level gates exercise the renamed production module directly:

- `aqua_test.jl` -> `Aqua.test_all(HomotopyContinuation)`;
- `jet_test.jl` -> `JET.report_package(HomotopyContinuation)`;
- `explicit_imports_test.jl` -> imports/public-access checks on `HomotopyContinuation` and
  `HomotopyContinuationDistributedExt`;
- `concrete_structs_test.jl` -> types whose parent module is `HomotopyContinuation`;
- TTFX package-load timing -> `using HomotopyContinuation`.

The large behavioral test corpus still contains many pre-rename import prefixes. Temporary
forwarding packages under `test/compat/` make those tests execute the real renamed implementation
without adding an old-name module to production. They are transitional test infrastructure only.

Three historical direct comparison files are deliberately excluded from ordinary discovery after
the identity restoration:

- `compare_v2_primitives_test.jl`;
- `compare_v2_solve_counts_test.jl`;
- `compare_v2_solve_match_test.jl`.

Those files relied on installing/loading v2 and v3 as distinct packages in one Julia environment.
That model becomes invalid once both correctly share the registered name and UUID; naively keeping
them could turn the oracle into a v3-v3 self-comparison. Live comparisons must be rebuilt around
isolated environments/processes. Fixed parity regressions such as `v2_parity_test.jl` and
`monodromy_v2_parity_test.jl` remain active.

## Core coverage

| Category | Representative files | Coverage |
|---|---|---|
| Quality gates | `aqua_test.jl`, `jet_test.jl`, `explicit_imports_test.jl`, `concrete_structs_test.jl` | package/API hygiene on the renamed production module, inference, concrete fields |
| Allocation | `alloc_check_test.jl` | zero-allocation hot paths for norms, LA, predictor/Newton/tracker/endgame and wrappers |
| Primitives | `double_f64_test.jl`, `norms_test.jl`, `linear_algebra_test.jl`, `operations_test.jl`, `taylor_test.jl` | DoubleF64/ComplexDF64, norms, custom LA, scalar ops and Taylor recurrences |
| Model/evaluation | `interpreter_test.jl`, `codegen_test.jl`, `instruction_count_test.jl`, `polynomial_input_test.jl`, `expression_test.jl`, `symbolic_utils_test.jl` | tape execution, RGF lowering, CSE/instruction regressions, polynomial/expression front ends |
| Non-polynomial | `nonpolynomial_test.jl`, `log_test.jl` | rational/transcendental evaluation, Jacobian/DF64/Taylor, principal-branch `log` |
| Core systems | `core_test.jl`, `composition_test.jl`, `fixed_parameter_test.jl`, `fixed_parameter_homotopy_test.jl`, `linear_subspace_test.jl` | wrappers, composition, fixed parameters/homotopies, subspace endpoints/promotion |
| Homotopies | `parameter_homotopy_test.jl`, `symbolic_homotopy_test.jl`, `homotopy_solve_test.jl`, `subspace_homotopy_test.jl`, `affine_chart_test.jl` | parameter/symbolic/custom homotopies, cloning and affine/projective wrappers |
| Tracking | `tracking_test.jl`, `endgame_test.jl`, `newton_test.jl`, `tracker_warmstart_test.jl`, `valuation_test.jl`, `tracker_regression_test.jl` | predictor/corrector, endgame, precision escalation, valuation, trust-region regressions |
| Solving | `solve_test.jl`, `binomial_system_test.jl`, `polyhedral_regression_test.jl`, `overdetermined_test.jl`, `result_clustering_test.jl`, `path_diagnostics_test.jl`, `progress_test.jl` | algorithms, classifications, excess filtering, clustering, progress/early stop |
| Subspaces/sweeps | `sliced_solve_test.jl`, `subspace_solve_test.jl`, `many_targets_test.jl`, `result_iterator_test.jl` | slices, subspace continuation, many targets, lazy tracking |
| Distributed | `distributed_test.jl` | process execution against serial, batching, serialization and failure propagation |
| Monodromy | `monodromy_test.jl`, `monodromy_v2_parity_test.jl`, `voronoi_tree_test.jl`, `group_actions_test.jl` | executor semantics, endpoint admission, permutations, trace, group actions, v2 cases |
| Witness/NID | `witness_set_test.jl`, `nid_test.jl` | affine/projective, zero-dimensional, parametric/rational input, regeneration/decomposition/unresolved output |
| Fixed v2 parity | `v2_parity_test.jl`, `monodromy_v2_parity_test.jl` and ported cases throughout suite | known v2 numerical/count behavior without loading a second package identity |
| Integration | `semialgebraic_sets_test.jl`, `utils_test.jl` | SemialgebraicSets and solution/parameter I/O |

## Final parity-tranche regressions

The September 2026 catch-up PRs added or strengthened regressions that are release invariants.

### Predictor trust radius

`tracker_regression_test.jl` covers a zero crossing with no derivative-derived trust radius. The
fallback must be finite and positive, while an available derivative radius remains authoritative.

### LinearSubspace endpoint dimensions and promotion

`linear_subspace_test.jl` covers `dim/codim == 0/n`, affine point slices, rejection of a
zero-dimensional linear subspace through a nonzero point, safe `rand_subspace!` at zero, and
real->complex coefficient promotion under translation without source mutation.

### Certification candidates from Result

`lib/HomotopyContinuationCertification/test/` executes the candidate-selection regression: a
successful regular root manually labelled numerically singular is hidden by `solutions(result)`
but remains one successful cluster representative for `certify(F, result)` and certifies.
Failed paths stay excluded and raw duplicates remain clustered.

### Principal-branch log

Core and certification log tests cover expression lowering, interpreted/compiled evaluation,
symbolic differentiation, DoubleF64/ComplexDF64, Taylor recurrences, interval/Acb certification,
and negative-real-axis branch-cut rejection.

### Witness ambient-coordinate contract

Witness intersection tests require exact ordered ambient variables; different variable identities
or order are rejected rather than silently renamed positionally.

### Regeneration ownership of point identity

`nid_test.jl` pins increasing `BY_DEGREE`, randomized pre-order, outer tolerance ownership,
`equivalence_classes = false` for internal monodromy, and final deduplication without group-action
quotienting.

### Lossless unresolved decomposition

NID regressions verify persistent identity for points discovered during trace-test monodromy and
return unresolved orbits as `Irreducibility.UNKNOWN` rather than dropping them.

### Hardened monodromy endpoint admission

`monodromy_test.jl` covers singular-endpoint rejection, base-problem revalidation of genuinely new
heuristic endpoints, certified duplicate admission, cross-executor identity semantics and
zero-padded incomplete permutation histories.

### Fixed-parameter custom homotopies

`fixed_parameter_homotopy_test.jl` exercises a parameterized `AbstractHomotopy` through
`FixedParameterHomotopy`: values, Jacobian, DoubleF64, Taylor, metadata, cloning and an actual solve.

## Certification package

`HomotopyContinuationCertification` has its own environment and covers rectangular interval
arithmetic, Acb precision refinement, Krawczyk/Arb fallback, distinct-certificate deduplication,
certified monodromy admission, low-memory iterator certification, transcendental branch behavior,
and `certify(F, ::Result)` candidate semantics.

The new test runner explicitly loads both `HomotopyContinuation` and
`HomotopyContinuationCertification` before executing the retained certification test bodies, so a
packaging/module rename failure cannot be hidden solely by the temporary forwarding modules.

## Extensive suite

`make test-extensive` covers:

- lines on a quintic surface;
- 3264 conics tangent to five conics;
- low-memory iterator certification on the large conic problem.

These are long numerical reference workloads rather than ordinary CI-sized regressions.

## Permanent CI

The permanent matrix supplies:

- core tests on Julia 1.10 and current Julia;
- certification tests on Julia 1.10 and current Julia;
- JET package-level static analysis;
- Runic formatting;
- benchmark tracking on pinned Julia 1.13 with a cached baseline and fatal regression threshold;
- representative fresh-process TTFX tracking;
- full TTFX inventory on `v3` pushes.

The September parity tree passed all of these after integration. PR #12 must independently make the
same standard true for the restored package identity: only the exact final rename head counts.

## v2 suite coverage

Every v2 feature-bearing test family has a v3 counterpart or a documented architecture-specific
reason why the original test is inapplicable. The principal intentionally unported family is v2's
global compiled-cache locking because v3 has no corresponding global compile table.

The three disabled direct-v2 files are a tooling limitation caused by restoring the correct package
identity, not a reduction in fixed regression coverage. A future live comparison harness must run
v2 and v3 in isolated environments.

See `02_status.md`, `03_v3_vs_v2.md`, and `09_v2_parity_freeze.md`.
