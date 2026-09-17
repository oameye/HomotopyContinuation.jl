# CLAUDE.md — HomotopyContinuation.jl v3

## What is this?

A ground-up rewrite of HomotopyContinuation.jl for solving polynomial systems via homotopy continuation. The design prioritizes type stability, minimal TTFX, and zero runtime dispatch on hot paths. The temporary `HomotopyContinuationNext` development identity has been retired; v3 uses the registered `HomotopyContinuation` package name and UUID.

## Architecture

Read `implementation_docs/00_architecture.md` for the full architecture, `implementation_docs/01_decisions.md` for pitfalls and design choices, and `implementation_docs/02_status.md` for feature/performance status.

Key design decisions:
- **Interpreter-first**: tape-based evaluator handles eval, jacobian, Taylor, DF64 — no Symbolics.jl in core
- **FunctionWrapper type firewall**: `SystemEvaluator`/`HomotopyEvaluator` wrap any system into a single concrete type — the tracker is monomorphic
- **FixedSizeArrays**: all pre-allocated scratch buffers use `FSVec`/`FSMat` (size is runtime, not a type parameter). **CRITICAL:** `FixedSizeVector{T}` is NOT concrete — the `Mem` parameter is free. Use `FixedSizeArray{T,N,Memory{T}}` for struct fields (see type aliases in main module).
- **Moshi ADTs**: `SExpr` and `ExecInstruction` use Moshi.jl `@data` for tagged unions — all variants are one concrete type, eliminating dynamic dispatch in CSE and interpreter
- **DynamicPolynomials input**: users provide polynomials via `@polyvar`, internal pipeline uses `MP.differentiate` for Jacobian
- **Expression input** for anything not polynomial (division, negative and non-integer powers, and the unary functions `sqrt`, `exp`, `sin`, `cos`, `tan`, `asin`, `acos`, `sinh`, `cosh`, `tanh`): `@var` builds a canonicalizing `Expression <: Number` tree, lowered by `expression_to_sexpr` with Jacobians from symbolic `differentiate`. `MP.RationalPoly` converts to `Expression` automatically
- **Immutable by default**: mutable structs require justification, use `const` fields for buffer references

## Package layout

```
src/HomotopyContinuation.jl         # Main module (core, Arblib-free)
src/primitives/                      # DoubleF64, norms, linear algebra
src/model_kit/                       # SExpr, Expression, CSE, tape compiler, interpreter, Taylor
src/core/                            # AbstractSystem/Homotopy, SystemEvaluator, homotopy types
src/tracking/                        # Predictor, Newton, Tracker
src/solving/                         # solve(), total degree, polyhedral, subspaces, sweeps, result types
src/utils.jl                         # SegmentStepper, _stable_sort!, fast_abs, etc.

lib/HomotopyContinuationCertification/      # Certification package
  src/interval_arithmetic.jl, interval_arblib.jl   # Interval / IComplexF64 + Arb bridge
  src/acb_interpreter.jl                            # arbitrary-precision tape interpreter
  src/certification.jl, certification_arb.jl        # certify(), Krawczyk + Arb fallback
```

**Certification is a separate package.** It depends on Arblib (a heavy binary
dep), so keeping it out of core is what makes core TTFX minimal. Every certificate
type embeds an `AcbMatrix`, so a package extension is not possible (extensions cannot
define/export types); a separate package is the correct isolation. To certify:

```julia
using HomotopyContinuation, HomotopyContinuationCertification
certify(F, solutions)
```

## Git policy

**Never commit or push.** Neither Claude nor any subagent may run `git commit`, `git push`, or any git command that modifies history. All commits are made by the user. Claude's job is to write code, run tests, and report results — the user decides when to commit.

## Development workflow

All common tasks go through the Makefile:

```sh
make test           # core tests in parallel, then certification
make test-cert      # certification package only
make test-strict    # same files with DispatchDoctor in error mode
make test-extensive # large reference solves
make test-serial    # serial debugging run
make benchmark      # steady-state benchmarks
make ttfx           # first-use workload inventory
make format         # Runic
make deps           # instantiate environments
make update         # update environments
make help           # list targets
```

The old same-process `make compare` harness is intentionally disabled after restoring the registered HomotopyContinuation identity. v2 and v3 have the same package name and UUID; live comparisons must run in isolated Julia environments/processes. Never re-enable the old harness through aliases, because that can silently compare v3 against itself.

### Test structure

Tests run via ParallelTestRunner — each file is self-contained and runs in its own worker.

- `test/aqua_test.jl` — Aqua package hygiene on `HomotopyContinuation`
- `test/jet_test.jl` — JET `report_package(HomotopyContinuation)`
- `test/explicit_imports_test.jl` — ExplicitImports checks on the production package and extensions
- `test/concrete_structs_test.jl` — concrete-field audit on production types

`test/test_systems.jl` and `test/minors_polys.jl` hold shared polynomial system data; they define no tests and are `include`d by the files that need them. Add new systems to `TEST_SYSTEM_COLLECTION` to get them covered by the evaluation sweep in `test/system_sweep_test.jl`.

`test/extensive/` holds solves that take minutes each (the 15625-path Fano quintic, the 27072-path 3264 problem). It has its own environment because it certifies, runs threaded through a plain `runtests.jl`, and is filtered out of `make test` discovery.

`test/strict/` holds no test files of its own. It is an environment whose `LocalPreferences.toml` sets `dispatch_doctor_mode = "error"`, and its `runtests.jl` reruns the files in `test/` under that instrumentation. Two contracts need two runs because they cannot be measured in one process: `@stable` heap-allocates the closures the evaluators call through, so an instrumented run cannot measure allocations, and it adds wrapper methods and generated names, so an instrumented `report_package` describes the wrappers. `alloc_check_test.jl` and `jet_test.jl` are therefore excluded from the strict run, and the `@allocated` assertions in `core_test.jl`, `tracking_test.jl` and `endgame_test.jl` skip themselves when the preference is set.

The historical same-process direct-v2 files `compare_v2_primitives_test.jl`, `compare_v2_solve_counts_test.jl`, and `compare_v2_solve_match_test.jl` are excluded after package-identity restoration. Fixed parity regressions remain active. A future live oracle must use isolated environments.

**Never pipe a suite run through `tail`/`head`/`grep` as its only sink.** A full run takes minutes. `make test` and `make test-serial` already `tee` to logs; diagnose those logs instead of paying for a second run.

**Run the affected test files, not the whole suite.** Iterate with `julia --project=test -t 8 test/runtests.jl NAME...`; use the quality gates plus touched tests, then reserve one full `make test` for the end.

### Quick debugging with Julia MCP

Use the `julia-mcp` MCP server (`julia_eval`, `julia_list_sessions`, `julia_restart`) for quick debugging and small snippets when available.

### Formatting

Code is formatted with Runic.jl:

```sh
make format
```

### Quality gates

Before merging any PR:
1. `make test` passes
2. JET reports zero issues on `HomotopyContinuation`
3. benchmark/TTFX workloads complete without regression failures
4. no `Any`-typed fields in structs
5. every `mutable struct` has documented justification and `const` on fixed fields
6. for package-identity changes, Aqua/import/extension loading must exercise the real production module, not only a compatibility alias
7. the package-wide `@stable` contract holds. `make test-strict` checks it: `test/strict/LocalPreferences.toml` sets `dispatch_doctor_mode = "error"`, so an instability throws a `TypeInstabilityError` and fails the file. `make test` cannot check it, because it runs the production configuration where the contract compiles to nothing.
   The gate only bites where `DispatchDoctor.JULIA_OK` is true. It was false on 1.13 until DispatchDoctor 0.4.29 (upstream issue #126), which is why compat requires `0.4.29` and must not be relaxed, and why `test/strict/runtests.jl` errors out rather than running when the flag is false. A run that resolves an older DispatchDoctor compiles `@stable` to nothing and passes vacuously.
   To enumerate every unstable site instead of stopping at the first, set `dispatch_doctor_mode = "warn"` in a copy of that environment and grep for `Instability detected`. Instrumenting `distributed_test.jl` that way needs the preference on each worker, so a warn sweep that skips it does not cover the extension's code paths; the error-mode run does.
8. the two StrictMode layers in `quality/strict_mode.jl` hold. They prove strictly more than DispatchDoctor, which only checks that a call's return type is concrete: StrictMode reads compiled output, so it also catches dynamic dispatch *inside* a body. That is why it names files rather than covering the package. `GUARANTEED` carries `:typestable`; `STATIC_CORE`, a subset, carries `:noalloc` and `:trim_compatible`, so the numeric core is proven allocation-free and juliac-trim compatible as well.
   The type-erasure layer can never join either list. `PathWorker` holds a `Base.RefValue{Any}` and the evaluators dispatch through a `FunctionWrapper`, both deliberate, so anything calling through them fails `:typestable` by construction. Taking an erased argument is not by itself disqualifying: measured, `tracking/newton.jl` passes 1 of 8 signatures and `tracking/newton_corrector.jl` 1 of 11, but `tracking/predictor.jl` passes 9 of 9 though all three take `H::HomotopyEvaluator`. Growing the lists means finding leaf code that crosses no erasure boundary; run `proof_audit(HomotopyContinuation; sweep = true, guarantees = (:typestable,), only = <file predicate>)` after warming the TTFX workloads to see the current surface. Pass `only` — an unfiltered sweep analyses every signature in the package and needs more than 25 GB.

## Coding rules

### Function signatures

- **Use the most restrictive signature type possible.** This lets JET catch unintended errors.
- **Explicit `;` for keyword arguments.** Always use an explicit semicolon before keyword arguments.

### Type system

- **No abstract-typed fields on hot paths.** Every struct field must be concretely typed.
- **`const` on buffer fields in mutable structs.** If a field holds a pre-allocated buffer that is never reassigned, mark it `const`.
- **`RefValue` for cache scalars in immutable structs.** Use `Base.RefValue{T}` for cached values that need mutation.
- **`NTuple{N,T}` for small fixed-size collections.**
- **Enums over Symbols.** Use `EnumX.@enumx` for return codes and state-machine states.
- **`FSVec{T}` / `FSMat{T}` for pre-allocated buffers.** Never use `FixedSizeVector{T}` / `FixedSizeMatrix{T}` directly as struct-field types because their memory parameter is free.
- **`AbstractVector` / `AbstractMatrix` only where truly needed.** Use them in public extension contracts; prefer concrete types internally.
- **Moshi `@data` for tagged unions.** Use `Moshi.Match.@match` to bind a variant's fields, and `isa_variant` for a single-variant test. Never name the storage union: `variant_storage(expr)` returns it by construction, and holding it makes the caller type unstable.
- **Continuations for a type chosen from runtime data.** Branch into a concrete call rather than returning the value: `with_system_shape`, `with_polyhedral_system`, `with_linear_subspace_homotopy`, `with_monodromy_solver`. Every arm must agree on what `f` returns, so a caller passing `identity` defeats the pattern and a test must run its assertions inside the continuation.
- **Erase a choice a struct only carries; do not lift it into a parameter.** `PathBuilder{W}` and `PathWorker` hold the chosen builder/worker in a `Base.RefValue{Any}` and assert the type at the call site, as `SystemEvaluator` does. A `FunctionWrapper` is the tighter erasure and belongs on a per-path call, but it carries a raw pointer into the process that made it, so anything crossing the wire to a distributed worker uses the box.
- **An empty collection, not `Union{Nothing,T}`, for "there is none".** `ExcessCheckers`, an empty `chart`, an empty `perm`.
- **A return annotation states a contract; an assertion in a body hides a defect.** `f(...)::T` on a definition is welcome. `x::T` inside a body means a value arrived untyped, so fix where it came from: type the struct field, annotate the callee's return, or restructure so the value is never read back out of an untyped container. `nested_ifs` needed `expr.args[end]::Expr` only because it walked `Expr.args`, a `Vector{Any}`; building the tree inside out removed the assertion and the dispatch with it.
  The exceptions are boundaries the package does not own or erased on purpose, and there are eleven left in `src/`: the `factory[]` and `_inner[]` unboxes behind the erasure, `make()` past its `@nospecialize` barrier, the `inferencebarrier` that keeps `qr!` out of a square-only session, caller-supplied callbacks in `polyhedral.jl` and `result_iterator.jl`, ProgressMeter's forwarded `Real` properties, and one `Union{Nothing,CertifiedEndpoint}` narrowed after its status code already ruled out `nothing`. `Serialization.deserialize` in `ext/DistributedExt/` is the same case: the wire hands back `Any`.

### Performance

For full reference, see `.claude/skills/julia-perf/` and `.claude/skills/julia-ttfx/`.

- **Zero allocations on hot paths.** Pre-allocate and mutate via `!` functions.
- **Column-major access.** Inner loops over rows, outer loops over columns.
- **No kwargs in hot paths.** Expose kwargs at API boundaries and forward to positional inner functions.
- **No kwargs splatting.** Explicitly name and forward each keyword.
- **Fuse broadcasts.** Avoid temporaries.
- **`@views` for slices.**
- **`@inbounds` only when provably valid; prefer `eachindex`.**
- **`@fastmath` only where IEEE edge cases are handled separately; never in certification/interval arithmetic.**
- **`abs2(z)` over `abs(z)^2`.**
- **Avoid string interpolation in hot/structured I/O.**

### Imports and style

- **No broad `using X` for implementation dependencies.** Use explicit imports or `import X`.
- **Format with Runic.** Run `make format` before merging.

### Comments

- **Comments/docstrings describe the code as it is, never its development history.** v2 parity/history belongs in `implementation_docs/`.
- **A docstring is public. No private internals in public docstrings.**
- **Nothing between a docstring and its definition.** Check with `Base.Docs.doc(Base.Docs.Binding(HomotopyContinuation, :name))` when needed.
- **Default to no comment; be terse when a comment is necessary.** Record design rationale in `implementation_docs/`.

## Dependencies

| Package | Purpose |
|---------|---------|
| CommonSolve | `init`/`solve!` interface |
| DynamicPolynomials | `@polyvar`, concrete polynomial types |
| EnumX | Scoped enums |
| FixedSizeArrays | Non-resizable vectors/matrices with runtime size |
| FunctionWrappers | Type-stable function erasure for evaluator firewalls |
| LinearAlgebra | stdlib |
| MixedSubdivisions | BKK mixed-volume computation |
| Moshi | `@data` tagged unions for SExpr/ExecInstruction |
| MultivariatePolynomials | Abstract polynomial interface, differentiation, exponent access |
