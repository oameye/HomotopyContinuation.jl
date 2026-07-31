
## Test Suite

53 test files run in parallel via ParallelTestRunner (`make test`, default 10 workers),
about 5700 passing assertions. The total is not exactly reproducible: 25 assertion
loops iterate over *discovered* solutions (`for s in solutions(res)`), so a run that
finds a different number of endpoints reports a different number of assertions.
Observed 5676 and 5738 on one commit:

| Category | Files | Notes |
|----------|-------|-------|
| Quality gates | `aqua_test.jl`, `jet_test.jl`, `explicit_imports_test.jl` | Static analysis, type inference, import hygiene |
| Type safety | `concrete_structs_test.jl` | All struct fields concretely typed |
| Allocation | `alloc_check_test.jl` | Zero-alloc norms, LA, predictor, Newton, tracker, endgame, and the three system wrappers whose appended rows run on the predictor's hot path (`RandomizedSystem`, `AffineChartSystem`, `SlicedSystem`) |
| Primitives | `double_f64_test.jl`, `norms_test.jl`, `linear_algebra_test.jl`, `operations_test.jl` | |
| Model kit | `interpreter_test.jl`, `codegen_test.jl`, `instruction_count_test.jl`, `taylor_test.jl`, `polynomial_input_test.jl`, `expression_test.jl` | Tape execution, RGF codegen, instruction-count regression; every `taylor_op_*` against a Cauchy-integral oracle; `expression_test.jl` covers the `Expression` ADT, `@var`, canonicalization, `differentiate`, `subs` (pairs and dicts), folding to a number, the cofactor `det`, conjugation, degrees, MP conversion, and three whole models built out of expression algebra (bottleneck, Steiner, reach of a plane curve) |
| Non-polynomial input | `nonpolynomial_test.jl` (+ `test_systems.jl`) | Sweep of 3 systems (`small_rational`, `sqrt_parameters`, `trig`) x 3 compile modes against a plain-Julia reference: eval, Jacobian vs central differences, DF64, Taylor 1--3 with Taylor-valued parameters against a Cauchy-integral oracle. Each system also runs its tape over `Expression` values, which has to rebuild the input, and goes through a `StraightLineHomotopy` sweep (eval, Jacobian, `Val(1)` t-derivative, Taylor 2--3). Plus parameter tracking through `sqrt`, both v2 rational monodromy testsets in two compile modes, `verify_solution_completeness`, the `TotalDegree`/`Polyhedral` rejection paths (plain and sliced), `Regeneration` and `Decomposition` over polynomial and rational expression input plus their rejection of `sqrt` of a variable, and the two routes that rewrite equations instead of evaluating them: polyhedral support extraction (dense, sparse where BKK beats Bezout, and squared-up overdetermined) and parameter fixing under a slice, both checked against the same system through DynamicPolynomials |
| Evaluation sweep | `system_sweep_test.jl` (+ `test_systems.jl`) | 9 real systems (cyclic5/7, bacillus, cyclo, moments3, six_revolute, steiner, four_bar, tritangents) × 3 compile modes: eval, jacobian, DF64, Taylor 1–3 with constant and Taylor-valued parameters, plus the straight-line homotopy, all against exact symbolic ground truth |
| Core | `core_test.jl`, `linear_subspace_test.jl`, `parameter_homotopy_test.jl`, `subspace_homotopy_test.jl`, `affine_chart_test.jl` | `affine_chart_test.jl` checks the chart row's Taylor coefficient `c·x_K` for a nonzero and a zeroed top row |
| Tracking | `tracking_test.jl`, `endgame_test.jl`, `newton_test.jl`, `tracker_warmstart_test.jl`, `valuation_test.jl`, `tracker_regression_test.jl` | `valuation_test.jl` checks asymptotic valuations (finite, diverging, fractional); `tracker_regression_test.jl` covers the four-bar and Steiner near-singular paths |
| Solving | `solve_test.jl`, `binomial_system_test.jl`, `polyhedral_regression_test.jl`, `overdetermined_test.jl`, `result_clustering_test.jl`, `path_diagnostics_test.jl`, `progress_test.jl` | Executor dispatch, serial/threaded consistency, excess-solution filtering |
| Subspaces / sweeps | `sliced_solve_test.jl`, `subspace_solve_test.jl`, `many_targets_test.jl`, `result_iterator_test.jl` | `slice`, subspace→subspace, many-target sweeps, lazy iteration. `many_targets_test.jl` asserts threaded == serial exactly, at a fixed seed, for target counts on both sides of `ntasks`, since `(target, path)` threading splits one target's paths across tasks. `ntasks` is passed explicitly (`Threaded(nt)` for `nt` in 1, 2, 3, 8, clamped to `nthreads()` since `Threaded` rejects more tasks than threads), so one run covers several splits of the same work. Run it under `-t N`: the parallel runner gives each worker one thread, where every case collapses to the single-task split |
| Distributed | `distributed_test.jl` | `addprocs(2)` (the parallel runner uses Malt, not Distributed, so nesting works), then every path-map route against `Serial()` field by field, at several `batch_size`s including one that puts a boundary inside a sweep target. Monodromy is compared by return code and solution set instead, since only its tracking is distributed and its loops are generated on the driver: vector parameters, `permutations`, a subspace run whose trace columns come back from another process, and a `timeout = 0.0` stop. Also `Serialization` round trips for `System` (all three compile modes), `_SupportSystem` and `CompositionSystem`, and the error paths (worker without the package, worker-side error unwrapped from `RemoteException`). Every comparison passes an explicit `seed`, which is what makes it exact: for subspace routes the seed picks γ, so two `solve` calls left to draw their own differ in the tail digits whatever the executor |
| Monodromy | `monodromy_test.jl`, `voronoi_tree_test.jl`, `group_actions_test.jl` | Serial + threaded, permutations, trace, `verify_solution_completeness` |
| Witness/NID | `witness_set_test.jl`, `nid_test.jl` | Affine + projective, zero-dim, parametric, serial + threaded |
| v2 parity | `compare_v2_primitives_test.jl`, `compare_v2_solve_counts_test.jl`, `compare_v2_solve_match_test.jl`, `v2_parity_test.jl`, `monodromy_v2_parity_test.jl` | Primitives, counts, values |
| Misc | `utils_test.jl` | |

The certification subpackage adds 541 assertions (`make test-cert`):
`interval_arithmetic_test.jl` (including sampled soundness checks on random boxes for the
`sqrt`/`sin`/`cos` enclosures), `acb_interpreter_test.jl` (the Arb tape interpreter against
Float64 ground truth, ball containment and precision refinement, on polynomial tapes and on
rational/`sqrt`/`sin`/`cos` ones), `certification_test.jl`
(which certifies a `sqrt` and a `sin`/`cos` system at 53 bits, so a regression that pushed
those to Arb would show up), `export_surface_test.jl`, `quality_test.jl`.

JET test filters known false positives: MP.variables dispatch (construction-time), Moshi `@match`/`@derive` generated code.

The default `JOBS=10` can drive a worker to SIGTERM under memory pressure (observed on
`endgame_test`, which passes in 15s on its own and reports
`Malt.TerminatedWorkerException` when killed). `make test JOBS=6` is the reliable
setting. A `TerminatedWorkerException` is a resource symptom, not a test failure;
re-run the file alone before believing it.

### Not covered from v2's suite

Every v2 test file has a v3 counterpart except `semialgebraic_sets_test.jl` (no
SemialgebraicSets integration), plus the composition / `mixed_volume` / `stop_early_cb` /
start-target-`solve` testsets, whose APIs v3 does not have. Of v2's "paths to track" testset
the total-degree half is ported (`solve_test.jl` and `variable_groups_test.jl`); the
polyhedral half waits on `mixed_volume`.

`model_kit/symbolic_test.jl` is covered by `expression_test.jl` except where v2's ModelKit
carries machinery v3 puts elsewhere or does not have:

| v2 testset | why not ported |
|------------|----------------|
| SymEngine, Convert | `Expression` coefficients are `ComplexF64`; there is no `BigFloat`/`Rational`/`Int128` tower to round-trip and no conversion back to a Julia number type |
| Expand | expressions are canonicalized on construction, so there is no separate expansion step (and no distributed normal form to expand *to*) |
| Horner, to_dict, Rand / dense poly, exponents_coefficients | polynomial utilities; v3's polynomial layer is DynamicPolynomials, which provides them |
| System (show), Homotopy | `System` has no custom `show` and there is no symbolic `Homotopy` type |

v2's `get_num_den` is `num_den`, covered by `expression_test.jl`'s `num_den` testset.