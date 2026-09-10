# v2.22.4 semantic parity ledger

Upstream: `JuliaHomotopyContinuation/HomotopyContinuation.jl@0cbf1e27d062d7eb01d962c3b948c39444c9a62c`.
V3 baseline: `oameye/HomotopyContinuation.jl@b5c7c2e81f372c22a29ff2963743ab5fad1fa9ed`.

This ledger classifies every mechanically discovered upstream `@testset` occurrence.
It is intentionally separate from source/API compatibility: implementation types removed by the v3 rewrite are not resurrected merely to retain names.

## Classification counts

- `architecture-obsolete`: **4**
- `equivalent-test`: **23**
- `replacement-test`: **179**
- `upstream-disabled`: **2**

## Review gate

- `unaccounted` must be zero before the ledger PR can be ready.
- `gap` must be zero before functional v2.22.4 parity can be claimed.
- `architecture-obsolete` entries require an explicit rationale and replacement evidence where applicable.
- Public/exported API compatibility is audited separately.

## unaccounted

None.

## architecture-obsolete

- `homotopies_test.jl:212` — `@testset "MixedHomotopy" begin`
  Evidence: `test/codegen_test.jl; test/homotopy_solve_test.jl`. MixedHomotopy as a wrapper type is replaced by the v3 evaluator/compile-mode design.
- `model_kit/compiled_cache_test.jl:14` — `@testset "Compiled cache thread safety" begin`
  Evidence: `test/core_test.jl; test/distributed_test.jl`. V3 has no process-global compiled cache; evaluator builders/clones provide task-local state, so the upstream cache race cannot occur.
- `model_kit/symbolic_test.jl:2` — `@testset "SymEngine" begin`
  Evidence: `test/expression_test.jl; test/polynomial_input_test.jl`. V3 deliberately removed the SymEngine frontend; supported symbolic input is Expression or MultivariatePolynomials.
- `systems_test.jl:89` — `@testset "MixedSystem" begin`
  Evidence: `test/codegen_test.jl; test/system_sweep_test.jl`. MixedSystem as a wrapper type is replaced by CompileMode and the evaluator firewall.

## upstream-disabled

- `endgame_test.jl:71` — `# @testset "Bacillus Subtilis" begin`
  Evidence: `-`. The upstream testset occurrence is commented out and is not an active v2.22.4 release gate.
- `endgame_test.jl:109` — `# @testset "Ill-conditioned solution - look's almost diverging" begin`
  Evidence: `-`. The upstream testset occurrence is commented out and is not an active v2.22.4 release gate.
