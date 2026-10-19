# Developer guide

HomotopyContinuation.jl is organized around a simple rule: **problem data stays data; only bounded algorithmic choices may become static compiler state.** The numerical core should remain concrete, allocation-free where required, and independent of the identity or dimension of a user's polynomial system.

The public API is intentionally higher level than the internal execution machinery. Flexible inputs are normalized before they reach tracking, and internal planning/evaluator types are not part of the user-facing vocabulary.

## Data flow

```text
DynamicPolynomials / Expression input
                ↓
        canonical SExpr trees
                ↓
              CSE
                ↓
       InstructionSequence
                ↓
 interpreter or generated evaluator
                ↓
         SystemEvaluator
                ↓
        HomotopyEvaluator
                ↓
 Predictor → Newton → Tracker → EndgameTracker
                ↓
        solver orchestration
                ↓
              Result
```

`System` retains source metadata needed by setup and symbolic/introspection APIs. Numerical evaluation crosses the concrete `SystemEvaluator` boundary before tracking, so user-specific symbolic types do not parameterize `Tracker` or `EndgameTracker`.

`CompileMode.INTERPRETED` executes the package-owned tape. `COMPILED` generates evaluation/Jacobian kernels while retaining interpreted Taylor kernels. `COMPILED_ALL` also generates Taylor kernels. Generated code is an evaluator concern; solver and tracking state must not become specialized on the identity of a user system.

## Mutable numerical state

Interpreter tapes, predictor/Newton workspaces, linear-algebra buffers, and endgame state are mutable and belong to one worker. Threaded and distributed execution reconstruct worker-local state from shared immutable/data-only inputs rather than sharing scratch buffers.

Hot numerical code should use concrete fields and preallocated storage. Dimensions, path counts, coefficients, tolerances, seeds, and other problem data remain runtime values rather than type parameters.

## Package layout

```text
src/primitives/      numeric primitives, norms, custom linear algebra
src/model_kit/       expressions, CSE, tape compiler, interpreter, code generation, Taylor
src/core/            systems, evaluators, homotopies, subspace geometry
src/tracking/        predictor, Newton, tracker, valuation, endgame
src/solving/         solver orchestration, execution, results, monodromy, NAG workflows
ext/                 optional distributed and SemialgebraicSets integration
lib/HomotopyContinuationCertification/
                     interval/Arb evaluation and rigorous certification
```

The deeper current architecture and deliberate boundaries are in `implementation_docs/00_architecture.md`; non-obvious implementation constraints are in `implementation_docs/01_decisions.md`. Future architecture belongs in roadmap issues, not in current-state documentation.

## Quality contracts

Normal tests and instrumented/static analysis are separate measurements:

- the normal suite checks package semantics, allocation contracts, JET, Aqua, imports, and concrete layouts;
- the strict test environment enables the package-wide DispatchDoctor `@stable` contract;
- `quality/strict_mode.jl` proves compiled guarantees for classified code;
- benchmarks and fresh-process TTFX workloads guard runtime and first-use behavior.

DispatchDoctor instrumentation changes allocation behavior, so allocation assertions are measured in the production configuration rather than treated as meaningful under instrumentation.

The long-term rule is stronger than "type stable": hot static kernels should be concrete, directly dispatched, unboxed, and allocation-free where their contract requires it. Dynamic compatibility boundaries must be narrow and explicit.

## Development workflow

Use the Makefile rather than reproducing CI commands by hand:

```sh
make test
make test-strict
make test-cert
make test-extensive
make benchmark
make ttfx
make format
make deps
```

During development, run the affected test files first and reserve a complete suite for the end of a tranche. Test and quality runs already preserve their useful output; diagnose those logs rather than immediately paying for a duplicate run.

v2/v3 differential measurements must use isolated Julia environments/processes because both versions have the same registered package name and UUID.
