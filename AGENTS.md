# AGENTS.md — HomotopyContinuation.jl

HomotopyContinuation.jl is a numerical homotopy-continuation package. Optimize for numerical correctness, robust solver semantics, bounded compilation, high steady-state performance, low first-use latency, and a small readable codebase.

## Architecture

Read `implementation_docs/00_architecture.md` for the current implementation and `implementation_docs/01_decisions.md` for non-obvious constraints. Future architecture belongs in roadmap issues rather than current-state docs.

Core rule: **problem data stays data.** User-system identity, dimensions, path counts, coefficients, tolerances, and seeds must not become unbounded type parameters. Bounded package-owned algorithmic choices may be specialized when measurement justifies it.

The numerical tracking path should be concrete, unboxed, directly dispatched, and allocation-free where its contract requires it. Dynamic compatibility/extension boundaries must be narrow and explicit.

## Git policy

Agents may create commits, push branches, open pull requests, and close pull requests when useful for completing requested work. **Never merge a pull request unless the user explicitly authorizes that specific merge.** Keep commits scoped/reviewable and report exact branch/commit/PR state after mutations.

## Development workflow

Use the Makefile:

```sh
make test           # core + certification
make test-cert      # certification only
make jet            # JET package and workload analysis
make test-extensive # long reference solves
make benchmark      # runtime benchmarks
make ttfx           # fresh-process first-use workloads
make format         # Runic
make deps           # instantiate development environments
make update         # update development environments
```

Run affected test files while iterating; reserve complete suites for tranche completion. Test commands already preserve useful output, so diagnose their logs instead of immediately rerunning a long suite.

v2/v3 live comparisons must run in isolated Julia environments/processes because both versions have the same registered package name and UUID.

## PR goal audit

Green CI is necessary but not sufficient. Before freezing or proposing any PR for merge, compare the final tree against its parent issue and original motivation.

- Restate the specific goal the PR is supposed to solve.
- Account for every relevant scope item and acceptance criterion as **done**, **already satisfied**, **deliberately retained**, or **explicitly deferred**.
- Verify that stated non-goals were not crossed accidentally.
- Search for residual instances of the problem instead of stopping after the first local fix.
- Give every deferred item a clear owning issue or later tranche.
- Record why any retained wrapper, boundary, command, or duplicate-looking path is intentional rather than forgotten cleanup.
- Check that the final diff serves the stated goal without unrelated churn.
- For umbrella issues implemented by several PRs, do not imply completion until the umbrella acceptance criteria have explicit completion accounting.

Only after this goal audit should correctness, quality, performance, TTFX, specialization, and code-size evidence be used to freeze the exact tree.

## Quality gates

Before a PR is considered ready:

1. relevant numerical/API tests pass, then the complete core/certification suites pass;
2. JET reports no unexpected package issues;
3. StrictMode guarantees for classified code pass;
4. concrete-layout, inference and allocation contracts remain satisfied;
5. Runic/ExplicitImports/Aqua remain clean;
6. runtime benchmarks and representative TTFX show no material regression.

## Coding rules

### Types and dispatch

- Struct fields on hot paths must be concrete.
- Use `FSVec{T}` / `FSMat{T}` for runtime-sized numerical buffers; never encode arbitrary system dimensions in the type.
- Use scoped `EnumX` enums for semantic finite state. Keep `Val` internal and bounded.
- Prefer concrete internal signatures. Use abstract collection types at genuine public/extension boundaries.
- A return annotation states a contract. Avoid body assertions that merely hide an upstream inference defect; fix the source unless the value crosses an intentional erased/external boundary.
- Do not specialize tracking/solver infrastructure on user symbolic types or generated function identity.
- Mutable buffers belong to one worker/task/process. Share immutable/data-only setup products instead.

### Performance

- Preallocate and mutate in hot numerical loops; zero steady-state allocations where contracted.
- Keep keyword processing and flexible input normalization at API/setup boundaries, not inside hot loops.
- Follow column-major access and avoid accidental temporaries.
- Use `@views` for non-copying slices where appropriate.
- Use `@inbounds` only with a clear bounds argument; prefer `eachindex`.
- Use `@fastmath` only when IEEE edge cases are handled deliberately; never in interval/certification code.
- Optimize measured end-to-end behavior, not isolated compiler aesthetics. Runtime, allocations, TTFX, MethodInstances/native code, and numerical work all matter.

### API and structure

- Julia privacy is export/module based; do not use a leading underscore mechanically as a substitute for API design.
- Prefer one canonical representation per concept and shallow, descriptive control flow.
- Do not add wrapper layers/macros merely to reduce line count.
- Public defaults should make the common path obvious; internal planning/compiler types should not leak into user vocabulary.
- Preserve API improvements already present in v3; avoid churn without a concrete consistency/usability gain.

### Imports, formatting, comments

- Use explicit imports or qualified module access for implementation dependencies.
- Format Julia code with Runic before declaring a tranche complete.
- Comments/docstrings describe current numerical or architectural reasons, never development history.
- A public docstring must attach directly to its definition; do not insert comments or blank lines between them.

## Tests

Tests run via ParallelTestRunner and each test file must be self-contained. Test files are grouped in folders that mirror `src/` (`model_kit/`, `primitives/`, `core/`, `tracking/`, `solving/`, `monodromy/`, `geometry/`), plus `scenarios/` and `extensions/`. Shared systems live in `test/helpers/`, which the runners skip; extensive reference solves live under `test/extensive/`.

`make test` runs the ordinary suite and the certification suite. The ordinary suite is the one test suite: every numerical and behavioural test lives there and uses public API alone, meaning exported names plus the names declared with `@public`. A primitive worth testing directly gets a sensible `@public` declaration rather than a private test. The static contracts live in `test/quality/`, one file per kind of check, and are excluded from it: `aqua`, `jet`, `explicit_imports`, `concrete_types`, `inference`, `allocations` (AllocCheck plus the measured zero-allocation checks) and `instruction_count`. Run one as `julia --project=test test/quality/<file>.jl`, as `.github/workflows/Quality.yml` and `JET.yml` do (`make jet` covers JET), and `test/quality/strict_mode/strict_mode.jl` with `--project=test/quality/strict_mode`. Test files carry no `_test` suffix, and a slow file is split along its testsets so ParallelTestRunner can balance the workers.

`test/quality/strict_mode/strict_mode.jl` provides the compiled-code proof layer for type stability. Do not weaken one analyzer merely because another analyzer already covers a nearby property.
