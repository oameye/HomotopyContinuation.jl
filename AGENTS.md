# AGENTS.md — HomotopyContinuation.jl v3

A ground-up rewrite for solving polynomial systems via homotopy continuation, prioritising type stability, minimal TTFX, and zero runtime dispatch on hot paths.

## Architecture

`implementation_docs/00_architecture.md` for the design, `01_decisions.md` for pitfalls, `02_status.md` for feature and performance status.

- **Interpreter-first**: a tape-based evaluator handles eval, Jacobian, Taylor and DF64. No Symbolics.jl in core.
- **FunctionWrapper type firewall**: `SystemEvaluator`/`HomotopyEvaluator` wrap any system into one concrete type, so the tracker is monomorphic.
- **Moshi ADTs**: `SExpr` and `ExecInstruction` are `@data` tagged unions, so CSE and the interpreter carry one concrete type.
- **DynamicPolynomials input** via `@polyvar`, with Jacobians from `MP.differentiate`.
- **Expression input** for anything not polynomial (division, negative and non-integer powers, `sqrt`, `exp`, `sin`, `cos`, `tan`, `asin`, `acos`, `sinh`, `cosh`, `tanh`): `@var` builds a canonicalising `Expression <: Number` tree, lowered by `expression_to_sexpr`. `MP.RationalPoly` converts automatically.
- **Immutable by default**: a `mutable struct` needs justification.

**Certification is a separate package**, `lib/HomotopyContinuationCertification/`. It depends on Arblib, and keeping that out of core is what makes core TTFX minimal. Every certificate type embeds an `AcbMatrix`, so an extension cannot work (extensions cannot define types).

## Git policy

**Never commit or push.** Neither Claude nor any subagent may run `git commit`, `git push`, or any git command that modifies history. All commits are made by the user. Claude's job is to write code, run tests, and report results — the user decides when to commit.

## Development workflow

All common tasks go through the Makefile; `make help` lists the targets.

**Run the affected test files, not the whole suite.** Iterate with `julia --project=test -t 8 test/runtests.jl NAME...`, then reserve one full `make test` for the end.

**Diagnose the logs, not a second run.** A full run takes minutes, and `make test` and `make test-serial` already `tee` to logs. Piping a suite through `tail`/`head`/`grep` as its only sink throws the run away.

v2 and v3 share the registered package name and UUID, so live comparisons need isolated environments and processes. The old same-process `make compare` harness is disabled because it can silently compare v3 against itself; do not re-enable it through an alias.

### Test structure

Tests run via ParallelTestRunner — each file is self-contained and runs in its own worker.

`test/test_systems.jl` and `test/minors_polys.jl` hold shared polynomial data and define no tests. Add new systems to `TEST_SYSTEM_COLLECTION` to get them into the evaluation sweep in `test/system_sweep_test.jl`.

`test/extensive/` holds solves that take minutes each and is filtered out of `make test` discovery.

`test/strict/` holds no test files. It is an environment whose `LocalPreferences.toml` sets `dispatch_doctor_mode = "error"` and reruns the files in `test/` under that instrumentation. Two runs are irreducible: `@stable` heap-allocates the closures the evaluators call through, so an instrumented run cannot measure allocations, and it adds wrapper methods, so an instrumented `report_package` describes the wrappers. Hence `alloc_check_test.jl` and `jet_test.jl` are excluded there, and the `@allocated` assertions in `core_test.jl`, `tracking_test.jl` and `endgame_test.jl` skip themselves when the preference is set.

## Quality gates

Before merging any PR:

1. `make test` passes. It covers Aqua, JET `report_package`, ExplicitImports, and the concrete-field audit.
2. benchmark/TTFX workloads complete without regression failures.
3. every `mutable struct` has documented justification and `const` on fixed fields.
4. `make test-strict` passes, holding the package-wide `@stable` contract. `make test` cannot check it: it runs the production configuration where the contract compiles to nothing. The gate only bites where `DispatchDoctor.JULIA_OK` is true, so compat requires DispatchDoctor `0.4.29` and must not be relaxed — an older resolve compiles `@stable` to nothing and passes vacuously. To enumerate every unstable site rather than stop at the first, set `dispatch_doctor_mode = "warn"` in a copy of that environment and grep for `Instability detected`.
5. both StrictMode layers in `quality/strict_mode.jl` pass. StrictMode reads compiled output, so it also catches dynamic dispatch *inside* a body, which is why it names files rather than covering the package. `GUARANTEED` carries `:typestable`; `STATIC_CORE`, a subset, carries `:noalloc`, `:noboxing` and `:trim_compatible`.
   Growing either list means finding leaf code that unboxes no erased value; `PathWorker`'s `Base.RefValue{Any}` and the evaluator `FunctionWrapper`s are deliberate, and a body reading through one fails. Survey with `proof_audit(HomotopyContinuation; sweep = true, guarantees = (:typestable,), only = <file predicate>)` after warming the TTFX workloads. Pass `only`: an unfiltered sweep analyses every signature and needs more than 25 GB.

## Coding rules

### Function signatures

- **Use the most restrictive signature type possible.** This lets JET catch unintended errors.
- **Explicit `;` for keyword arguments.**

### Type system

- **No abstract-typed fields on hot paths.** Every struct field must be concretely typed.
- **`const` on buffer fields in mutable structs.**
- **`RefValue` for cache scalars in immutable structs.**
- **`NTuple{N,T}` for small fixed-size collections.**
- **Enums over Symbols.** `EnumX.@enumx` for return codes and state-machine states.
- **`FSVec{T}` / `FSMat{T}` for pre-allocated buffers.** `FixedSizeVector{T}` / `FixedSizeMatrix{T}` are not concrete — their memory parameter is free — so struct fields use the aliases.
- **`AbstractVector` / `AbstractMatrix` only in public extension contracts.** Prefer concrete types internally.
- **Moshi `@data` for tagged unions.** `Moshi.Match.@match` binds a variant's fields, `isa_variant` tests a single variant. Never name the storage union: `variant_storage(expr)` returns it by construction, and holding it makes the caller type unstable.
- **Continuations for a type chosen from runtime data.** Branch into a concrete call rather than returning the value: `with_system_shape`, `with_polyhedral_system`, `with_linear_subspace_homotopy`, `with_monodromy_solver`. Every arm must agree on what `f` returns, so a caller passing `identity` defeats the pattern and a test must run its assertions inside the continuation.
- **Erase a choice a struct only carries; do not lift it into a parameter.** `PathBuilder{W}` and `PathWorker` hold the chosen builder/worker in a `Base.RefValue{Any}` and assert at the call site, as `SystemEvaluator` does. A `FunctionWrapper` is the tighter erasure and belongs on a per-path call, but it carries a raw pointer into the process that made it, so anything crossing the wire to a distributed worker uses the box.
- **An empty collection, not `Union{Nothing,T}`, for "there is none".** `ExcessCheckers`, an empty `chart`, an empty `perm`.
- **A return annotation states a contract; an assertion in a body hides a defect.** `f(...)::T` on a definition is welcome. `x::T` inside a body means a value arrived untyped, so fix where it came from: type the struct field, annotate the callee's return, or restructure so the value is never read back out of an untyped container. The exceptions are boundaries the package does not own or erased on purpose — the unboxes behind the erasure, `make()` past its `@nospecialize` barrier, the `inferencebarrier` keeping `qr!` out of a square-only session, caller-supplied callbacks, and `Serialization.deserialize` in `ext/DistributedExt/`, where the wire hands back `Any`.

### Performance


- **Zero allocations on hot paths.** Pre-allocate and mutate via `!` functions.
- **Column-major access.** Inner loops over rows, outer loops over columns.
- **Keyword arguments at API boundaries only.** Forward them to positional inner functions, naming each one explicitly rather than splatting.
- **Fuse broadcasts.** Avoid temporaries.
- **`@views` for slices.**
- **`@inbounds` only when provably valid; prefer `eachindex`.**
- **`@fastmath` only where IEEE edge cases are handled separately.** Never in certification or interval arithmetic.
- **`abs2(z)` over `abs(z)^2`.**
- **Plain I/O in hot and structured paths.** String interpolation belongs elsewhere.

### Imports and style

- **Explicit imports or `import X` for implementation dependencies.**
- **Format with Runic.** Run `make format` before merging.

### Comments

- **Comments and docstrings describe the code as it is, never its development history.** v2 parity and history belong in `implementation_docs/`.
- **A docstring is public. No private internals in public docstrings.**
- **Nothing between a docstring and its definition.** Check with `Base.Docs.doc(Base.Docs.Binding(HomotopyContinuation, :name))`.
- **Default to no comment; be terse when a comment is necessary.** Design rationale goes in `implementation_docs/`.
